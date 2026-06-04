#!/usr/bin/env python3
"""
Fetches Space Invader coordinates from OpenStreetMap via Overpass API,
merges with spotter_pois.json (IDs + statuses from invader-spotter.art),
and writes a combined world_invaders.json ready to use as the app data source.

Run periodically as new POIs appear on OSM or spotter.

Output fields: id, city, lat, lng, status, points
"""

import json, re, time, urllib.request, urllib.parse, sys
from pathlib import Path

OVERPASS   = 'https://overpass-api.de/api/interpreter'
GITHUB_URL = 'https://raw.githubusercontent.com/goguelnikov/SpaceInvaders/main/world_space_invaders_V05.json'
SPOTTER    = Path('spotter_pois.json')
OUTPUT     = Path('world_invaders.json')

OVERPASS_QUERY = '''
[out:json][timeout:120];
node["artist_name"="Invader"];
out body;
'''

# ── helpers ──────────────────────────────────────────────────────────────────

def fetch_json(url, data=None, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=data,
                headers={'User-Agent': 'InvaderHunter/1.0',
                         'Content-Type': 'application/x-www-form-urlencoded'}
            )
            return json.loads(urllib.request.urlopen(req, timeout=130).read())
        except Exception as e:
            print(f'  attempt {attempt+1} failed: {e}')
            if attempt < retries - 1:
                time.sleep(5)
    raise RuntimeError(f'Failed to fetch {url}')

ID_RE = re.compile(r'[A-Z]{1,6}_\d+', re.I)

def extract_id(tags):
    """Pull the invader ID out of OSM tags (ref preferred, fallback to name)."""
    for field in ('ref', 'name', 'description'):
        val = tags.get(field, '')
        m = ID_RE.search(val)
        if m:
            return m.group(0).upper()
    return None

def normalise_status(s):
    if not s:
        return 'ok'
    l = s.lower()
    if 'destroy' in l or 'gone' in l or 'missing' in l: return 'destroyed'
    if 'hidden' in l or 'covered' in l:                  return 'hidden'
    if 'damage' in l or 'little' in l or 'partial' in l: return 'damaged'
    return 'ok'

def parse_coord(v):
    try:
        return float(str(v).replace(',', '.'))
    except Exception:
        return None

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Load spotter data (IDs, statuses, points — no coords)
    if not SPOTTER.exists():
        sys.exit('spotter_pois.json not found — run scrape_spotter.py first')
    spotter = {p['id']: p for p in json.loads(SPOTTER.read_text())}
    print(f'Spotter: {len(spotter)} POIs')

    # 2. Load GitHub source (has coords for most POIs)
    print('Fetching GitHub source…')
    raw = fetch_json(GITHUB_URL)
    items = raw if isinstance(raw, list) \
        else raw.get('invaders', list(raw.values())[0] if isinstance(raw, dict) else [])

    github = {}
    for p in items:
        pid  = str(p.get('id') or p.get('name') or '').strip()
        lat  = parse_coord(p.get('lat'))
        lng  = parse_coord(p.get('lng'))
        if pid and lat is not None and lng is not None and lat != 0 and lng != 0:
            github[pid] = {
                'id':     pid,
                'city':   p.get('city', ''),
                'lat':    lat,
                'lng':    lng,
                'status': normalise_status(p.get('status')),
                'points': int(p['points']) if str(p.get('points','')).isdigit() else None,
            }
    print(f'GitHub:  {len(github)} geolocated POIs')

    # 3. Fetch OSM coords
    print('Querying Overpass (worldwide, may take ~30s)…')
    osm_raw = fetch_json(
        OVERPASS,
        data=urllib.parse.urlencode({'data': OVERPASS_QUERY}).encode()
    )
    osm = {}
    for node in osm_raw.get('elements', []):
        pid = extract_id(node.get('tags', {}))
        if pid:
            osm[pid] = {'lat': node['lat'], 'lng': node['lon']}
    print(f'OSM:     {len(osm)} geolocated POIs')

    # 4. Merge: spotter is the source of truth for IDs/status/points;
    #    coords come from GitHub first, OSM as fallback.
    result   = []
    no_coord = []

    for pid, sp in spotter.items():
        entry = {
            'id':     pid,
            'city':   sp.get('city', ''),
            'status': sp.get('status', 'ok'),
            'points': sp.get('points'),
        }
        if pid in github:
            entry['lat'] = github[pid]['lat']
            entry['lng'] = github[pid]['lng']
            # Prefer spotter status (more up-to-date), keep github coords
        elif pid in osm:
            entry['lat'] = osm[pid]['lat']
            entry['lng'] = osm[pid]['lng']
        else:
            no_coord.append(pid)
            continue   # skip until coords are found
        result.append(entry)

    # 5. Also include any github POIs not in spotter (shouldn't happen, but safe)
    spotter_ids = set(spotter.keys())
    for pid, gh in github.items():
        if pid not in spotter_ids:
            result.append(gh)

    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'\nOutput:  {len(result)} geolocated POIs → {OUTPUT}')
    print(f'Skipped: {len(no_coord)} POIs with no coords yet')
    if no_coord:
        print('  Missing coords:', no_coord[:20], '…' if len(no_coord) > 20 else '')

if __name__ == '__main__':
    main()
