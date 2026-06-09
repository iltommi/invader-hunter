#!/usr/bin/env python3
"""
Fine-tune the CLIP visual encoder on Space Invader reference images using
multi-view contrastive learning (3 views per image per step):

  view A — raw augmented photo
  view B — raw augmented photo (independent)
  view S — grid-snapped canonical pixel art (or a third raw view if snap fails)

All three views of the same invader are pulled together; views from different
invaders are pushed apart. This teaches the model to bridge raw photos and
clean pixel art representations.

Output: clip_finetuned.pt

Run:
  python3 train_clip.py
  python3 extract_features_clip.py   # rebuild the index with fine-tuned weights
"""

import json
import random
import time
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

IMAGES_DIR       = Path('images')
OUT_FILE         = Path('clip_finetuned.pt')
SNAP_LABELS_FILE = Path('snap_labels.json')

def load_snap_labels():
    try:
        return json.loads(SNAP_LABELS_FILE.read_text())
    except Exception:
        return {}
BATCH_SIZE  = 48          # reduced from 64 — 3 forward passes need more memory
EPOCHS      = 10
LR          = 1e-5
TEMPERATURE = 0.07
SNAP_CONF   = 0.12

device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)
print(f'Device: {device}')

# ── Grid snap (Python/numpy mirror of the JS implementation) ──────────────────

def _gs_wavelet_period(profile, min_p=2.0, max_p=None):
    """Period detection via Morlet CWT power spectrum (pure numpy).
    Evaluates CWT power at a dense grid of periods using Parseval's theorem:
    P(T) = Σ_f |X(f)|² · exp(−(ω₀·T·f − ω₀)²)
    Continuous scale axis gives sub-pixel accuracy and handles small images
    with few tiles better than FFT.
    """
    n = len(profile)
    if max_p is None:
        max_p = n / 3.0
    if max_p < min_p or n < min_p * 3:
        return 0, 0.0

    w0   = 6.0
    step = 0.1
    periods = np.arange(min_p, max_p + step, step)
    if len(periods) < 3:
        return 0, 0.0

    p     = profile - profile.mean()
    X     = np.fft.rfft(p)
    Xpow  = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n)

    power = np.array([
        float((Xpow * np.exp(-(w0 * (freqs * T - 1.0)) ** 2)).sum())
        for T in periods
    ])

    best_idx = int(power.argmax())

    if 0 < best_idx < len(periods) - 1:
        p0, p1, p2 = power[best_idx - 1], power[best_idx], power[best_idx + 1]
        denom = 2 * p1 - p2 - p0
        delta = (0.5 * (p2 - p0) / denom) if abs(denom) > 1e-12 else 0.0
        best_period = float(periods[best_idx] + delta * step)
    else:
        best_period = float(periods[best_idx])

    bg   = float(np.median(power))
    peak = float(power[best_idx])
    conf = max(0.0, min(1.0, (peak / (bg + 1e-9) - 1.0) / 2.0))

    return best_period, conf

def _gs_phase(profile, period):
    T = max(1.0, float(period))
    T_int = max(1, round(T))
    n = len(profile)
    best_phase, best_score = 0, np.inf
    for p in range(T_int):
        score, i = 0.0, 0
        while True:
            s = p + round(i * T)
            if s >= n:
                break
            e = p + round((i + 1) * T)
            block = profile[s:min(e, n)]
            if len(block) > 1:
                score += float(block.var()) * len(block)
            i += 1
        if score < best_score:
            best_score = score
            best_phase = p
    return best_phase

def grid_snap_pil(img, force_T=None):
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    h, w = arr.shape[:2]
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    col_profile = np.abs(np.diff(gray.mean(axis=0)))
    row_profile = np.abs(np.diff(gray.mean(axis=1)))
    if force_T is not None:
        T, confidence = float(force_T), 1.0
    else:
        Tx, cx = _gs_wavelet_period(col_profile)
        Ty, cy = _gs_wavelet_period(row_profile)
        if cx < SNAP_CONF and cy < SNAP_CONF:
            return None, 0.0, 0
        T = Tx if cx >= cy else Ty
        confidence = max(cx, cy)
    px = _gs_phase(col_profile, T)
    py = _gs_phase(row_profile, T)
    margin = min(max(0, round(T * 0.15)), max(0, int(T) // 2 - 1))

    out = np.zeros_like(arr)
    n_cells = 0
    yi = 0
    while True:
        gy = py + round(yi * T)
        if gy >= h:
            break
        y1 = min(py + round((yi + 1) * T), h)
        xi = 0
        while True:
            gx = px + round(xi * T)
            if gx >= w:
                break
            x1 = min(px + round((xi + 1) * T), w)
            patch = arr[gy + margin:y1 - margin, gx + margin:x1 - margin]
            if patch.size > 0:
                out[gy:y1, gx:x1] = patch.reshape(-1, 3).mean(axis=0)
                n_cells += 1
            xi += 1
        yi += 1

    if n_cells < 9:
        return None, 0.0, 0
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)), confidence, T

# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(img):
    img = img.rotate(random.uniform(-20, 20), expand=False, fillcolor=(128, 128, 128))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 1.5))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.6, 1.4))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 2.0)))
    w, h   = img.size
    margin = int(min(w, h) * 0.15)
    img    = img.crop((
        random.randint(0, margin), random.randint(0, margin),
        w - random.randint(0, margin), h - random.randint(0, margin),
    ))
    w, h  = img.size
    new_w = int(w * random.uniform(0.7, 1.3))
    new_h = int(h * random.uniform(0.7, 1.3))
    img   = img.resize((max(new_w, 1), max(new_h, 1)), Image.BILINEAR)
    return img

def augment_snap(img):
    img = img.rotate(random.uniform(-10, 10), expand=False, fillcolor=(0, 0, 0))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
    return img

def synth_subpixel(img, force_T=None):
    snapped, _, T = grid_snap_pil(img, force_T=force_T)
    if snapped is None or T < 2:
        return None

    w, h   = snapped.size
    n_tx   = max(1, round(w / T))
    n_ty   = max(1, round(h / T))

    art    = snapped.resize((n_tx, n_ty), Image.NEAREST)
    padded = Image.new('RGB', (n_tx + 2, n_ty + 2), (0, 0, 0))
    padded.paste(art, (1, 1))

    T_new  = T * random.uniform(0.75, 1.25)
    ox     = random.random()   # sub-tile phase offset [0, 1)
    oy     = random.random()

    big_w  = max(4, round((n_tx + 2) * T_new))
    big_h  = max(4, round((n_ty + 2) * T_new))
    big    = padded.resize((big_w, big_h), Image.BILINEAR)

    # Crop: skip the 1-tile border plus the random phase offset
    x0     = round(T_new * (1 + ox))
    y0     = round(T_new * (1 + oy))
    result = big.crop((x0, y0, x0 + round(n_tx * T_new), y0 + round(n_ty * T_new)))

    return result if result.width >= 4 and result.height >= 4 else None

# ── Dataset ───────────────────────────────────────────────────────────────────

class InvaderDataset(Dataset):
    def __init__(self, paths, processor, snap_labels):
        self.paths       = paths
        self.processor   = processor
        self.snap_labels = snap_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))

        pv = lambda view: self.processor(images=view, return_tensors='pt')['pixel_values'][0]

        view_a = pv(augment(img))
        view_b = pv(augment(img))

        label   = self.snap_labels.get(path.stem, {})
        verdict = label.get('verdict', '')
        force_T = label.get('T') if label.get('forced_T') else None

        if verdict == 'bad':
            view_s = pv(augment(img))
        else:
            synth = synth_subpixel(img, force_T=force_T)
            if synth is not None:
                view_s = pv(augment_snap(synth))
            else:
                snapped, _, _ = grid_snap_pil(img, force_T=force_T)
                view_s = pv(augment_snap(snapped) if snapped is not None else augment(img))

        return view_a, view_b, view_s

# ── Multi-view NT-Xent loss ───────────────────────────────────────────────────

def nt_xent(z1, z2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N  = z1.shape[0]
    z  = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / TEMPERATURE
    sim.masked_fill_(torch.eye(2 * N, dtype=torch.bool, device=z.device), float('-inf'))
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(N)]).to(z.device)
    return F.cross_entropy(sim, labels)

def multi_view_loss(za, zb, zs):
    return (nt_xent(za, zb) + nt_xent(za, zs) + nt_xent(zb, zs)) / 3.0

# ── Setup ─────────────────────────────────────────────────────────────────────

print('Loading CLIP...')
model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

for p in model.text_model.parameters():       p.requires_grad = False
for p in model.text_projection.parameters(): p.requires_grad = False

for layer in list(model.vision_model.encoder.layers)[:-4]:
    for p in layer.parameters():
        p.requires_grad = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable: {trainable:,} params')

paths       = sorted(IMAGES_DIR.rglob('*.png'))
snap_labels = load_snap_labels()
print(f'Snap labels: {len(snap_labels)} entries  '
      f'({sum(1 for v in snap_labels.values() if v.get("verdict")=="good")} good, '
      f'{sum(1 for v in snap_labels.values() if v.get("verdict")=="bad")} bad)')
loader = DataLoader(
    InvaderDataset(paths, processor, snap_labels),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=1e-4,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * len(loader),
)

# ── Training loop ─────────────────────────────────────────────────────────────

def encode(pv):
    return model.visual_projection(model.vision_model(pixel_values=pv).pooler_output)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (pv_a, pv_b, pv_s) in enumerate(loader, 1):
        pv_a = pv_a.to(device)
        pv_b = pv_b.to(device)
        pv_s = pv_s.to(device)

        za = encode(pv_a)
        zb = encode(pv_b)
        zs = encode(pv_s)

        loss = multi_view_loss(za, zb, zs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        print(f'  epoch {epoch}/{EPOCHS}  step {step}/{len(loader)}  loss={loss.item():.4f}', end='\r', flush=True)

    print(f'\nEpoch {epoch}/{EPOCHS}  avg_loss={total_loss/len(loader):.4f}  {time.time()-t0:.0f}s')

# ── Save ──────────────────────────────────────────────────────────────────────

torch.save({
    'vision_model':      model.vision_model.state_dict(),
    'visual_projection': model.visual_projection.state_dict(),
}, OUT_FILE)
print(f'Saved → {OUT_FILE}')
