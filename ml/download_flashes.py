#!/usr/bin/env python3
"""
Download live invader photos from the FlashInvaders public feed.

The site itself polls this same JSON endpoint every ~2s to drive the live
feed at https://www.space-invaders.com/flashinvaders/. Each response holds
the last ~300 flashes (several minutes of history), so we can poll less
aggressively than the browser does and still catch every new photo.

Images are saved to:   ml/flashes/images/{flash_id}.jpg
Per-image metadata (city, player, timestamp) is appended to:
                       ml/flashes/metadata.jsonl

Safe to stop (Ctrl+C) and re-run: already-downloaded flash_ids are skipped,
and --max only counts new images downloaded in this run.
"""

import argparse
import json
import time
from pathlib import Path

import requests

API_URL    = 'https://www.space-invaders.com/flashinvaders/flashes/'
BASE_URL   = 'https://www.space-invaders.com'
OUT_DIR    = Path(__file__).parent / 'flashes'
LINE_WIDTH = 100  # pad printed lines to this so \r-overwrites never leave stale trailing text
HEADERS    = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0 Safari/537.36',
    'Referer': 'https://www.space-invaders.com/flashinvaders/',
}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_seen(images_dir):
    """flash_ids already on disk, so re-runs pick up where they left off."""
    return {int(p.stem) for p in images_dir.glob('*.jpg')}

def poll(session):
    r = session.get(API_URL, headers=HEADERS, timeout=10)
    r.raise_for_status()
    return r.json()['with_paris']

def download_image(session, flash, images_dir):
    url = BASE_URL + flash['img']
    dest = images_dir / f"{flash['flash_id']}.jpg"
    r = session.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    dest.write_bytes(r.content)

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max', type=int, default=1000,
                         help='number of new images to download before exiting (default: 1000)')
    parser.add_argument('--interval', type=float, default=2.0,
                         help="seconds between polls (default: 2.0, matches the site's own refresh rate)")
    args = parser.parse_args()

    images_dir = OUT_DIR / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT_DIR / 'metadata.jsonl'

    seen = load_seen(images_dir)
    downloaded = 0
    session = requests.Session()

    print(f'{len(seen)} images already on disk. Downloading up to {args.max} new images...')

    try:
        with metadata_path.open('a') as meta_file:
            while downloaded < args.max:
                try:
                    flashes = poll(session)
                except requests.RequestException as e:
                    print(f'poll failed ({e}), retrying in {args.interval:g}s')
                    time.sleep(args.interval)
                    continue

                # Oldest-first so images/metadata land in chronological order.
                new_this_poll = 0
                for flash in sorted(flashes, key=lambda f: f['flash_id']):
                    if flash['flash_id'] in seen:
                        continue
                    seen.add(flash['flash_id'])
                    new_this_poll += 1

                    try:
                        download_image(session, flash, images_dir)
                    except requests.RequestException as e:
                        print(f"  x {flash['flash_id']} ({e})".ljust(LINE_WIDTH))
                        continue

                    meta_file.write(json.dumps(flash) + '\n')
                    meta_file.flush()
                    downloaded += 1
                    print(f"[{downloaded:5d}/{args.max}] + {flash['flash_id']} {flash['city']} / {flash['player']}".ljust(LINE_WIDTH))

                    if downloaded >= args.max:
                        break

                if new_this_poll == 0:
                    # Padded to LINE_WIDTH so this \r-updated line always fully
                    # overwrites whatever was printed here before (otherwise a
                    # shorter line leaves the previous one's tail visible).
                    print(f'  (poll: 0 new / {len(flashes)} already had - still watching...)'.ljust(LINE_WIDTH),
                          end='\r', flush=True)

                if downloaded < args.max:
                    time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f'\nInterrupted. downloaded={downloaded}')
        return

    print(f'\nDone. downloaded={downloaded} total_on_disk={len(seen)}')

if __name__ == '__main__':
    main()
