#!/usr/bin/env python3
"""
Extract CNN embeddings from all reference images using EfficientNet-B0.

Output:
  ml/embeddings.npz  — {ids: [N], embeddings: [N x 1280]}

Run once (or after adding new images):
  python3 extract_features.py
"""

import json, time
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as T

IMAGES_DIR = Path('images')
OUT_FILE   = Path('embeddings.npz')
BATCH_SIZE = 32

# ── Device ───────────────────────────────────────────────────────────────────

device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)
print(f'Using device: {device}')

# ── Model: EfficientNet-B0, features only (drop classifier) ──────────────────

weights = models.EfficientNet_B0_Weights.DEFAULT
model   = models.efficientnet_b0(weights=weights)
model.classifier = torch.nn.Identity()  # output: [B, 1280]
model = model.to(device).eval()

transform = weights.transforms()  # standard ImageNet preprocessing

# ── Collect image paths ───────────────────────────────────────────────────────

paths = sorted(IMAGES_DIR.rglob('*.png'))
print(f'Found {len(paths)} images')

ids        = [p.stem for p in paths]   # filename without extension = POI ID
embeddings = []

# ── Extract in batches ────────────────────────────────────────────────────────

def load_image(path):
    try:
        img = Image.open(path).convert('RGB')
        return transform(img)
    except Exception as e:
        print(f'  ⚠ failed to load {path.name}: {e}')
        return None

t0 = time.time()
for batch_start in range(0, len(paths), BATCH_SIZE):
    batch_paths = paths[batch_start:batch_start + BATCH_SIZE]
    tensors = []
    valid_indices = []
    for i, p in enumerate(batch_paths):
        t = load_image(p)
        if t is not None:
            tensors.append(t)
            valid_indices.append(i)

    if not tensors:
        continue

    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        feats = model(batch).cpu().numpy()

    # Insert zeros for failed images to keep alignment
    batch_emb = np.zeros((len(batch_paths), feats.shape[1]), dtype=np.float32)
    for out_i, vi in enumerate(valid_indices):
        batch_emb[vi] = feats[out_i]
    embeddings.append(batch_emb)

    done = batch_start + len(batch_paths)
    elapsed = time.time() - t0
    rate = done / elapsed
    eta  = (len(paths) - done) / rate if rate > 0 else 0
    print(f'  {done:4d}/{len(paths)}  {rate:.1f} img/s  ETA {eta:.0f}s', end='\r')

print()
embeddings = np.vstack(embeddings)

# L2-normalise so cosine similarity = dot product
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms = np.where(norms == 0, 1, norms)
embeddings /= norms

np.savez_compressed(OUT_FILE, ids=np.array(ids), embeddings=embeddings)
print(f'\nSaved {len(ids)} embeddings → {OUT_FILE}')
print(f'Shape: {embeddings.shape}  ({embeddings.nbytes/1e6:.1f} MB)')
