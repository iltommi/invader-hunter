#!/usr/bin/env python3
"""
Download reference images from invader-spotter.art for all known POIs.

Images are saved to:  ml/images/{city_code}/{POI_ID}.png

Skips already-downloaded files so the script is safe to re-run.
Rate-limited to ~3 req/s to be polite to the server.
"""

import json, time, urllib.request, urllib.error
from pathlib import Path

SPOTTER_BASE  = 'https://www.invader-spotter.art/grosplan'
SPOTTER_FILE  = Path('../spotter_pois.json')
OUT_DIR       = Path('images')
DELAY         = 0.35   # seconds between requests (~3 req/s)
HEADERS       = {'User-Agent': 'InvaderHunter-ML/1.0'}

# ── helpers ──────────────────────────────────────────────────────────────────

def city_padding(pois):
    """Return dict {city_code: zero-pad width} derived from max ID number per city."""
    max_num = {}
    for p in pois:
        code, num = p['id'].split('_', 1)
        n = int(num)
        if code not in max_num or n > max_num[code]:
            max_num[code] = n
    return {code: len(str(n)) for code, n in max_num.items()}

def image_url(poi_id, padding):
    code, num = poi_id.split('_', 1)
    width = padding.get(code, len(num))
    return f'{SPOTTER_BASE}/{code}/{code}_{num.zfill(width)}-grosplan.png'

def download(url, dest):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r:
        dest.write_bytes(r.read())

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    pois    = json.loads(SPOTTER_FILE.read_text())
    padding = city_padding(pois)

    total = len(pois)
    done = skipped = failed = 0

    for i, p in enumerate(pois, 1):
        poi_id   = p['id']
        code     = poi_id.split('_')[0]
        city_dir = OUT_DIR / code
        city_dir.mkdir(parents=True, exist_ok=True)
        dest = city_dir / f'{poi_id}.png'

        if dest.exists():
            skipped += 1
            continue

        url = image_url(poi_id, padding)
        try:
            download(url, dest)
            done += 1
            print(f'[{i:4d}/{total}] ✓ {poi_id}', flush=True)
            time.sleep(DELAY)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                failed += 1
                print(f'[{i:4d}/{total}] ✗ {poi_id} (404)', flush=True)
            else:
                failed += 1
                print(f'[{i:4d}/{total}] ✗ {poi_id} ({e.code})', flush=True)
                time.sleep(1)
        except Exception as e:
            failed += 1
            print(f'[{i:4d}/{total}] ✗ {poi_id} ({e})', flush=True)
            time.sleep(1)

    print(f'\nDone.  downloaded={done}  skipped={skipped}  failed={failed}')

if __name__ == '__main__':
    main()
