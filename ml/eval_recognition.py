#!/usr/bin/env python3
"""
Measure recognition accuracy on the held-out eval split of confirmed flash
photos (see flash_split.py) -- the 1-in-5 subset train_clip.py never trains
on, so this reflects generalization to unseen real photos, not memorization.

Reports top-1 / top-5 / top-16 accuracy and mean reciprocal rank (MRR) for
both the whole-image embed and the multi-crop search, against the current
embeddings_clip.npz + clip_finetuned.pt (if present). Run this before and
after a retrain to see whether it actually helped.

Also appends one summary line per run to eval_history.log, so results stay
comparable across retrains over time, and overwrites eval_latest.json with
the same numbers in machine-readable form (used by auto_retrain.py to decide
whether a retrain actually improved things).

Usage (from ml/ directory):
  python3 eval_recognition.py
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from flash_split import is_eval
from label_flashes import load_model, search_single, search_multi_crop

IDENTITY_LABELS_FILE = Path('flashes/identity_labels.json')
FLASH_IMAGES_DIR     = Path('flashes/images')
EMB_FILE             = Path('embeddings_clip.npz')
HISTORY_FILE         = Path('eval_history.log')
LATEST_FILE          = Path('eval_latest.json')

def load_eval_set():
    try:
        labels = json.loads(IDENTITY_LABELS_FILE.read_text())
    except Exception:
        return []
    items = []
    for flash_id, label in labels.items():
        if label.get('verdict') != 'confirmed' or not label.get('poi_id'):
            continue
        if not is_eval(flash_id):
            continue
        path = FLASH_IMAGES_DIR / f'{flash_id}.jpg'
        if path.exists():
            items.append((flash_id, label['poi_id'], path))
    return items

def rank_of(sims, all_ids, poi_id):
    positions = np.where(all_ids == poi_id)[0]
    if len(positions) == 0:
        return None
    true_sim = sims[positions].max()
    return int((sims > true_sim).sum()) + 1

def summarize(ranks, n):
    ranks = np.array(ranks)
    return {
        'top1':  float((ranks <= 1).sum()) / n,
        'top5':  float((ranks <= 5).sum()) / n,
        'top16': float((ranks <= 16).sum()) / n,
        'mrr':   float((1.0 / ranks).mean()),
    }

def fmt(m):
    return f"top1={m['top1']:.1%} top5={m['top5']:.1%} top16={m['top16']:.1%} mrr={m['mrr']:.3f}"

def main():
    eval_set = load_eval_set()
    if not eval_set:
        print('No held-out eval examples yet -- label more flashes with label_flashes.py first.')
        return

    print(f'{len(eval_set)} held-out confirmed photos (never trained on)')
    print('Loading CLIP...')
    model, processor, device = load_model()

    data    = np.load(EMB_FILE)
    all_ids = data['ids']
    all_emb = data['embeddings']

    ranks_single, ranks_multi = [], []
    missing = 0
    t0 = time.time()

    for i, (flash_id, poi_id, path) in enumerate(eval_set, 1):
        img = Image.open(path).convert('RGB')

        sims_single = search_single(model, processor, device, img, all_emb)
        sims_multi  = search_multi_crop(model, processor, device, img, all_emb)

        r_single = rank_of(sims_single, all_ids, poi_id)
        r_multi  = rank_of(sims_multi, all_ids, poi_id)
        if r_single is None or r_multi is None:
            missing += 1
            print(f'  warn: {poi_id} (flash {flash_id}) not found in embeddings index, skipping')
            continue

        ranks_single.append(r_single)
        ranks_multi.append(r_multi)
        print(f'  [{i:3d}/{len(eval_set)}] {poi_id:<10} rank(single)={r_single:<5} rank(multi)={r_multi}',
              end='\r', flush=True)

    print()
    n = len(ranks_single)
    if n == 0:
        print('No eval examples had a matching entry in the embeddings index.')
        return

    m_single = summarize(ranks_single, n)
    m_multi  = summarize(ranks_multi, n)

    print(f'\n{n} eval examples ({missing} skipped, not in index)  [{time.time()-t0:.0f}s]\n')
    print(f'{"":13s} {"top-1":>7s} {"top-5":>7s} {"top-16":>7s} {"MRR":>7s}')
    print(f'{"single-embed":13s} {m_single["top1"]:>6.1%} {m_single["top5"]:>6.1%} '
          f'{m_single["top16"]:>6.1%} {m_single["mrr"]:>7.3f}')
    print(f'{"multi-crop":13s} {m_multi["top1"]:>6.1%} {m_multi["top5"]:>6.1%} '
          f'{m_multi["top16"]:>6.1%} {m_multi["mrr"]:>7.3f}')

    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with HISTORY_FILE.open('a') as f:
        f.write(f'{ts}  n={n}  single: {fmt(m_single)}  multi-crop: {fmt(m_multi)}\n')
    print(f'\nAppended to {HISTORY_FILE}')

    LATEST_FILE.write_text(json.dumps({
        'timestamp': ts, 'n': n, 'single': m_single, 'multi_crop': m_multi,
    }, indent=2))

if __name__ == '__main__':
    main()
