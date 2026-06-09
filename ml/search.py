#!/usr/bin/env python3
"""
Similarity search: given a query image, find the closest invaders.

Usage:
  python3 search.py path/to/photo.jpg [--top 5] [--city PA]

The query image should be a perspective-corrected crop of the invader.
"""

import sys, argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torchvision.models as models

EMB_FILE = Path('embeddings.npz')

# ── Load index ────────────────────────────────────────────────────────────────

data       = np.load(EMB_FILE)
all_ids    = data['ids']
all_emb    = data['embeddings']   # already L2-normalised

# ── Load model (same as extract_features.py) ─────────────────────────────────

device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)

weights   = models.EfficientNet_B0_Weights.DEFAULT
model     = models.efficientnet_b0(weights=weights)
model.classifier = torch.nn.Identity()
model     = model.to(device).eval()
transform = weights.transforms()

# ── Query ─────────────────────────────────────────────────────────────────────

def embed(img_path):
    img    = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy()[0]
    feat /= (np.linalg.norm(feat) or 1)
    return feat

def search(img_path, top=5, city_filter=None):
    q    = embed(img_path)
    sims = all_emb @ q             # cosine similarity (dot product of L2-normalised vectors)

    if city_filter:
        mask = np.array([i.split('_')[0] == city_filter.upper() for i in all_ids])
        sims = np.where(mask, sims, -1)

    idx  = np.argsort(sims)[::-1][:top]
    return [(all_ids[i], float(sims[i])) for i in idx]

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--top',  type=int, default=5)
    parser.add_argument('--city', default=None, help='Restrict to city code (e.g. PA)')
    args = parser.parse_args()

    results = search(args.image, top=args.top, city_filter=args.city)
    print(f'\nTop {args.top} matches for {args.image}:')
    for rank, (pid, score) in enumerate(results, 1):
        print(f'  {rank}. {pid:<12}  similarity={score:.4f}')
