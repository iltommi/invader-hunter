#!/usr/bin/env python3
"""
Similarity search with CLIP: given a query image, find the closest invaders.

Usage:
  python3 search_clip.py path/to/photo.jpg [--top 5] [--city PA]

The query image should be a perspective-corrected crop of the invader.
Requires embeddings_clip.npz produced by extract_features_clip.py.
"""

import argparse
import subprocess
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from transformers import CLIPProcessor, CLIPModel

EMB_FILE   = Path('embeddings_clip.npz')
IMAGES_DIR = Path('images')

# ── Load index ────────────────────────────────────────────────────────────────

data    = np.load(EMB_FILE)
all_ids = data['ids']
all_emb = data['embeddings']   # already L2-normalised

# ── Load model ────────────────────────────────────────────────────────────────

device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available()         else
    torch.device('cpu')
)

FINETUNED_FILE = Path('clip_finetuned.pt')

model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

if FINETUNED_FILE.exists():
    ckpt = torch.load(FINETUNED_FILE, map_location=device)
    model.vision_model.load_state_dict(ckpt['vision_model'])
    model.visual_projection.load_state_dict(ckpt['visual_projection'])
    print(f'Using fine-tuned weights')

# ── Query ─────────────────────────────────────────────────────────────────────

def embed(img_path):
    img    = Image.open(img_path).convert('RGB')
    inputs = processor(images=img, return_tensors='pt').to(device)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs['pixel_values'])
        feat = model.visual_projection(vision_out.pooler_output).cpu().numpy()[0]
    feat /= (np.linalg.norm(feat) or 1)
    return feat

def search(img_path, top=5, city_filter=None):
    q    = embed(img_path)
    sims = all_emb @ q             # cosine similarity (dot product of L2-normalised vectors)

    if city_filter:
        mask = np.array([i.split('_')[0] == city_filter.upper() for i in all_ids])
        sims = np.where(mask, sims, -1)

    idx = np.argsort(sims)[::-1][:top]
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

    if results:
        top_id = results[0][0]
        city   = top_id.split('_')[0]
        img    = IMAGES_DIR / city / f'{top_id}.png'
        subprocess.run(['open', str(img)])
