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

import random
import time
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor

IMAGES_DIR  = Path('images')
OUT_FILE    = Path('clip_finetuned.pt')
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

def _gs_fft_period(profile, min_p=5):
    n = len(profile)
    k_min, k_max = 3, n // min_p
    if k_max < k_min:
        return 0, 0.0
    ps = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
    k_max = min(k_max, len(ps) - 1)
    sub = ps[k_min:k_max + 1]
    total = sub.sum()
    if total == 0:
        return 0, 0.0
    best_idx = int(sub.argmax())
    best_k = k_min + best_idx
    return round(n / best_k), float(sub[best_idx] / total)

def _gs_phase(profile, period):
    T = max(1, round(period))
    n = len(profile)
    best_phase, best_score = 0, np.inf
    for p in range(T):
        score = 0.0
        s = p
        while s < n:
            block = profile[s:min(s + T, n)]
            if len(block) > 1:
                score += float(block.var()) * len(block)
            s += T
        if score < best_score:
            best_score = score
            best_phase = p
    return best_phase

def _gs_kmeans(colors, k=4, iters=25):
    n = len(colors)
    idx = np.linspace(0, n - 1, k, dtype=int)
    centroids = colors[idx].astype(np.float64).copy()
    assign = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        dists = np.stack([((colors - c) ** 2).sum(axis=1) for c in centroids], axis=1)
        new_assign = dists.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            mask = assign == c
            if mask.any():
                centroids[c] = colors[mask].mean(axis=0)
    return centroids, assign

def _gs_kmeans_adaptive(colors, k_min=2, k_max=8, threshold=0.10, iters=25):
    k_max = min(k_max, len(colors))
    best_centroids, best_assign = None, None
    prev_wcss = None
    first_improvement = None
    for k in range(k_min, k_max + 1):
        centroids, assign = _gs_kmeans(colors, k=k, iters=iters)
        wcss = sum(float(((colors[assign == c] - centroids[c]) ** 2).sum())
                   for c in range(k) if (assign == c).any())
        if prev_wcss is not None:
            imp = prev_wcss - wcss
            if first_improvement is None:
                first_improvement = imp
            elif first_improvement > 0 and imp < threshold * first_improvement:
                break
        best_centroids, best_assign = centroids, assign
        prev_wcss = wcss
    return best_centroids, best_assign

def _gs_morph_clean(cells, assign):
    gx_vals = sorted(set(c[0] for c in cells))
    gy_vals = sorted(set(c[1] for c in cells))
    to_col  = {gx: i for i, gx in enumerate(gx_vals)}
    to_row  = {gy: i for i, gy in enumerate(gy_vals)}
    grid    = {(to_col[c[0]], to_row[c[1]]): i for i, c in enumerate(cells)}
    out     = assign.copy()
    for i, c in enumerate(cells):
        ci, ri = to_col[c[0]], to_row[c[1]]
        nbrs = [grid.get((ci-1, ri)), grid.get((ci+1, ri)),
                grid.get((ci, ri-1)), grid.get((ci, ri+1))]
        nbrs = [j for j in nbrs if j is not None]
        if len(nbrs) < 3:
            continue
        votes = {}
        for j in nbrs:
            v = assign[j]; votes[v] = votes.get(v, 0) + 1
        top, cnt = max(votes.items(), key=lambda x: x[1])
        if cnt >= 3 and top != assign[i]:
            out[i] = top
    return out

def grid_snap_pil(img):
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    h, w = arr.shape[:2]
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    col_profile = np.abs(np.diff(gray.mean(axis=0)))
    row_profile = np.abs(np.diff(gray.mean(axis=1)))
    Tx, cx = _gs_fft_period(col_profile)
    Ty, cy = _gs_fft_period(row_profile)
    if cx < SNAP_CONF and cy < SNAP_CONF:
        return None, 0.0
    T = Tx if cx >= cy else Ty
    confidence = max(cx, cy)
    px = _gs_phase(col_profile, T)
    py = _gs_phase(row_profile, T)
    margin = max(1, round(T * 0.15))

    cells = []
    gy = py
    while gy < h:
        y1 = min(gy + T, h)
        gx = px
        while gx < w:
            x1 = min(gx + T, w)
            patch = arr[gy + margin:y1 - margin, gx + margin:x1 - margin]
            if patch.size > 0:
                cells.append((gx, gy, x1, y1, patch.reshape(-1, 3).mean(axis=0)))
            gx += T
        gy += T

    if len(cells) < 9:
        return None, 0.0

    colors = np.array([c[4] for c in cells], dtype=np.float32)
    centroids, raw_assign = _gs_kmeans_adaptive(colors)
    assign = _gs_morph_clean(cells, raw_assign)

    out = np.zeros_like(arr)
    for i, (gx, gy, x1, y1, _) in enumerate(cells):
        out[gy:y1, gx:x1] = centroids[assign[i]]

    return Image.fromarray(out.clip(0, 255).astype(np.uint8)), confidence

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

# ── Dataset ───────────────────────────────────────────────────────────────────

class InvaderDataset(Dataset):
    def __init__(self, paths, processor):
        self.paths     = paths
        self.processor = processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224))

        pv = lambda view: self.processor(images=view, return_tensors='pt')['pixel_values'][0]

        view_a = pv(augment(img))
        view_b = pv(augment(img))

        snapped, _ = grid_snap_pil(img)
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

paths  = sorted(IMAGES_DIR.rglob('*.png'))
loader = DataLoader(
    InvaderDataset(paths, processor),
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
