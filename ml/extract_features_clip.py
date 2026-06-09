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

import time
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import random

import torch
from transformers import CLIPProcessor, CLIPModel

IMAGES_DIR = Path('images')
OUT_FILE   = Path('embeddings_clip.npz')
BATCH_SIZE = 32
TTA_N      = 8     # raw augmented views per image
SNAP_N     = 4     # grid-snapped views per image (when snap succeeds)
SNAP_CONF  = 0.4   # peak-spacing consistency threshold (0=chaotic, 1=perfect)

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

def _gs_peak_period(profile, min_tiles=8, min_spacing=4):
    n = len(profile)
    if n < min_tiles * min_spacing:
        return 0, 0.0
    w = max(3, n // (min_tiles * 3))
    smooth = np.convolve(profile, np.ones(w) / w, mode='same')
    threshold = smooth.mean()
    peaks = [i for i in range(1, n - 1)
             if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > threshold]
    if len(peaks) < min_tiles:
        return 0, 0.0
    spacings = np.diff(peaks).astype(float)
    spacings = spacings[spacings >= min_spacing]
    if len(spacings) < min_tiles - 1:
        return 0, 0.0
    period = max(min_spacing, int(round(float(np.median(spacings)))))
    cv = float(spacings.std() / (spacings.mean() + 1e-9))
    return period, max(0.0, 1.0 - cv)

def _gs_sobel_profiles(arr):
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    sx = (-gray[:-2, :-2] + gray[:-2, 2:]
          - 2*gray[1:-1, :-2] + 2*gray[1:-1, 2:]
          - gray[2:, :-2] + gray[2:, 2:])
    sy = (-gray[:-2, :-2] - 2*gray[:-2, 1:-1] - gray[:-2, 2:]
          + gray[2:, :-2] + 2*gray[2:, 1:-1] + gray[2:, 2:])
    return np.abs(sx).mean(axis=0), np.abs(sy).mean(axis=1)

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

def _gs_kmeans_once(colors, k, iters, seed):
    n = len(colors)
    rng = np.random.default_rng(seed)
    centroids = [colors[int(rng.integers(0, n))].astype(np.float64)]
    for _ in range(k - 1):
        d2 = np.array([min(float(((c - ctr)**2).sum()) for ctr in centroids) for c in colors])
        probs = d2 / (d2.sum() + 1e-9)
        centroids.append(colors[rng.choice(n, p=probs)].astype(np.float64))
    centroids = np.array(centroids)
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
    wcss = sum(float(((colors[assign == c] - centroids[c])**2).sum()) for c in range(k) if (assign == c).any())
    return centroids, assign, wcss

def _gs_kmeans(colors, k=4, iters=25, restarts=3):
    best = min((_gs_kmeans_once(colors, k, iters, s) for s in range(restarts)), key=lambda x: x[2])
    return best[0], best[1]

def _gs_kmeans_adaptive(colors, k_min=2, k_max=16, target_var=300, iters=25):
    k_max = min(k_max, len(colors))
    best_centroids, best_assign = None, None
    for k in range(k_min, k_max + 1):
        centroids, assign = _gs_kmeans(colors, k=k, iters=iters)
        wcss = sum(float(((colors[assign == c] - centroids[c]) ** 2).sum())
                   for c in range(k) if (assign == c).any())
        best_centroids, best_assign = centroids, assign
        if wcss / len(colors) < target_var:
            break
    return best_centroids, best_assign

def _gs_merge_close(centroids, assign, min_dist=50):
    centroids = centroids.copy().tolist()
    assign = assign.copy()
    changed = True
    while changed:
        changed = False
        k = len(centroids)
        for i in range(k):
            for j in range(i + 1, k):
                d = float(sum((centroids[i][c] - centroids[j][c])**2 for c in range(3))**0.5)
                if d < min_dist:
                    ci_n = int((assign == i).sum())
                    cj_n = int((assign == j).sum())
                    tot = ci_n + cj_n or 1
                    centroids[i] = [(centroids[i][c]*ci_n + centroids[j][c]*cj_n)/tot for c in range(3)]
                    centroids.pop(j)
                    assign[assign == j] = i
                    assign[assign > j] -= 1
                    changed = True
                    break
            if changed:
                break
    return np.array(centroids, dtype=np.float64), assign

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
    col_profile, row_profile = _gs_sobel_profiles(arr)
    Tx, cx = _gs_peak_period(col_profile, min_tiles=8)
    Ty, cy = _gs_peak_period(row_profile, min_tiles=6)
    x_ok = cx >= SNAP_CONF and Tx >= 4
    y_ok = cy >= SNAP_CONF and Ty >= 4
    if x_ok and y_ok:
        T = max(4, round((Tx * cx + Ty * cy) / (cx + cy)))
        confidence = min(cx, cy)
    elif x_ok:
        T = Tx; confidence = cx
    elif y_ok:
        T = Ty; confidence = cy
    else:
        return None, 0.0
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
    centroids, raw_assign = _gs_merge_close(centroids, raw_assign, min_dist=50)
    assign = _gs_morph_clean(cells, raw_assign)

    out = np.zeros_like(arr)
    for i, (gx, gy, x1, y1, _) in enumerate(cells):
        out[gy:y1, gx:x1] = centroids[assign[i]]

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

paths = sorted(IMAGES_DIR.rglob('*.png'))
print(f'{len(paths)} images  ×  ({TTA_N} raw + up to {SNAP_N} snapped) views')

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
        snapped, _ = grid_snap_pil(img)
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
