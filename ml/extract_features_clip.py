#!/usr/bin/env python3
"""
Extract CLIP embeddings from all reference images.
CLIP generalises far better to out-of-domain images than EfficientNet.

TTA strategy: each reference image is embedded as:
  - TTA_N raw augmented views  (always)
  - SNAP_N grid-snapped views  (when snap confidence ≥ SNAP_CONF)

All views are averaged to produce the final embedding.

Output: embeddings_clip.npz  — {ids, embeddings}
"""

import json
import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import random

import torch
from transformers import CLIPProcessor, CLIPModel

IMAGES_DIR       = Path('images')
OUT_FILE         = Path('embeddings_clip.npz')
SNAP_LABELS_FILE = Path('snap_labels.json')
BATCH_SIZE       = 32
TTA_N            = 8     # raw augmented views per image
SNAP_N           = 4     # grid-snapped views per image (when snap succeeds)
SNAP_CONF        = 0.12

def load_snap_labels():
    try:
        return json.loads(SNAP_LABELS_FILE.read_text())
    except Exception:
        return {}

device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)
print(f'Device: {device}')

# ── CLIP model ────────────────────────────────────────────────────────────────

FINETUNED_FILE = Path('clip_finetuned.pt')

print('Loading CLIP...')
model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

if FINETUNED_FILE.exists():
    ckpt = torch.load(FINETUNED_FILE, map_location=device)
    model.vision_model.load_state_dict(ckpt['vision_model'])
    model.visual_projection.load_state_dict(ckpt['visual_projection'])
    print(f'Loaded fine-tuned weights from {FINETUNED_FILE}')

# ── Grid snap (mirrors train_clip.py) ────────────────────────────────────────

def _gs_wavelet_period(profile, min_p=2.0, max_p=None):
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
    conf = max(0.0, min(1.0, (float(power[best_idx]) / (bg + 1e-9) - 1.0) / 2.0))
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
            return None, 0.0
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
        return None, 0.0
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)), confidence

# ── Augmentation ──────────────────────────────────────────────────────────────

def augment(img):
    img = img.rotate(random.uniform(-15, 15), expand=False, fillcolor=(128, 128, 128))
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.4))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
    w, h = img.size
    margin = int(min(w, h) * 0.1)
    img = img.crop((
        random.randint(0, margin), random.randint(0, margin),
        w - random.randint(0, margin), h - random.randint(0, margin),
    ))
    w, h  = img.size
    new_w = int(w * random.uniform(0.7, 1.3))
    new_h = int(h * random.uniform(0.7, 1.3))
    return img.resize((max(new_w, 1), max(new_h, 1)), Image.BILINEAR)

def augment_snap(img):
    img = img.rotate(random.uniform(-10, 10), expand=False, fillcolor=(0, 0, 0))
    return ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))

# ── Embed a list of PIL images ────────────────────────────────────────────────

def embed_images(imgs):
    inputs = processor(images=imgs, return_tensors='pt').to(device)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs['pixel_values'])
        feats = model.visual_projection(vision_out.pooler_output).cpu().numpy()
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return feats

# ── Main ──────────────────────────────────────────────────────────────────────

paths       = sorted(IMAGES_DIR.rglob('*.png'))
snap_labels = load_snap_labels()
print(f'{len(paths)} images  ×  ({TTA_N} raw + up to {SNAP_N} snapped) views')
print(f'Snap labels: {len(snap_labels)} entries  '
      f'({sum(1 for v in snap_labels.values() if v.get("verdict")=="good")} good, '
      f'{sum(1 for v in snap_labels.values() if v.get("verdict")=="bad")} bad)')

ids        = [p.stem for p in paths]
embeddings = np.zeros((len(paths), 512), dtype=np.float32)

snap_hit = snap_total = 0
t0 = time.time()

for batch_start in range(0, len(paths), BATCH_SIZE):
    batch_paths = paths[batch_start:batch_start + BATCH_SIZE]
    n = len(batch_paths)

    # Load images and pre-compute snapped versions once per batch
    raw_imgs     = []
    snapped_imgs = []
    for p in batch_paths:
        try:
            img = Image.open(p).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))
        raw_imgs.append(img)
        label   = snap_labels.get(p.stem, {})
        verdict = label.get('verdict', '')
        force_T = label.get('T') if label.get('forced_T') else None
        if verdict == 'bad':
            snapped_imgs.append(None)
        else:
            snapped, _ = grid_snap_pil(img, force_T=force_T)
            snapped_imgs.append(snapped)

    snap_total += n
    snap_hit   += sum(1 for s in snapped_imgs if s is not None)

    # view counts per image: TTA_N raw + SNAP_N if snapped
    view_counts = np.array(
        [TTA_N + (SNAP_N if s is not None else 0) for s in snapped_imgs],
        dtype=np.float32,
    )
    batch_emb = np.zeros((n, 512), dtype=np.float32)

    # Raw TTA passes
    for i in range(TTA_N):
        imgs = [augment(img) if i > 0 else img for img in raw_imgs]
        batch_emb += embed_images(imgs)

    # Snapped TTA passes
    for i in range(SNAP_N):
        imgs, idxs = [], []
        for j, snapped in enumerate(snapped_imgs):
            if snapped is not None:
                imgs.append(augment_snap(snapped) if i > 0 else snapped)
                idxs.append(j)
        if imgs:
            embs = embed_images(imgs)
            for k, j in enumerate(idxs):
                batch_emb[j] += embs[k]

    batch_emb /= view_counts[:, None]
    norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
    batch_emb /= np.where(norms == 0, 1, norms)
    embeddings[batch_start:batch_start + n] = batch_emb

    done    = batch_start + n
    elapsed = time.time() - t0
    rate    = done / elapsed
    eta     = (len(paths) - done) / rate if rate > 0 else 0
    print(f'  {done:4d}/{len(paths)}  snap={snap_hit}/{snap_total}  {rate:.1f} img/s  ETA {eta:.0f}s', end='\r', flush=True)

print()
np.savez_compressed(OUT_FILE, ids=np.array(ids), embeddings=embeddings)
print(f'Saved → {OUT_FILE}  ({embeddings.shape}, {OUT_FILE.stat().st_size/1e6:.1f} MB)')
