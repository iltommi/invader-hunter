#!/usr/bin/env python3
"""
Full POI update pipeline — run periodically to keep world_invaders.json fresh.

Steps:
  1. Scrape invader-spotter.art  →  spotter_pois.json   (IDs, status, points)
  2. Query Overpass / OSM        →  coordinates
  3. Merge everything            →  world_invaders.json

Usage:
  python3 update_pois.py              # full run (headless browser)
  python3 update_pois.py --show       # same but with visible browser
  python3 update_pois.py --skip-scrape  # skip step 1, reuse existing spotter_pois.json
"""

import json, math, re, sys, time, urllib.parse, urllib.request
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ── config ────────────────────────────────────────────────────────────────────

SPOTTER_BASE = 'https://www.invader-spotter.art/villes.php'
GITHUB_URL   = 'https://raw.githubusercontent.com/goguelnikov/SpaceInvaders/main/world_space_invaders_V05.json'
OVERPASS_URL = 'https://overpass-api.de/api/interpreter'
OVERPASS_Q   = '[out:json][timeout:120];\nnode["artist_name"="Invader"];\nout body;'

SPOTTER_FILE = Path('spotter_pois.json')
OUTPUT_FILE  = Path('world_invaders.json')

CITY_WAIT  = 4   # seconds after envoi()
PAGE_WAIT  = 3   # seconds after changepage()
PER_PAGE   = 50

STATUS_MAP = {
    'ok': 'ok', 'degraded': 'damaged', 'destroyed': 'destroyed',
    'unknown': 'unknown', 'neutre': 'ok',
}

# ── utilities ─────────────────────────────────────────────────────────────────

def fetch_json(url, post_data=None, retries=3):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=post_data,
                headers={'User-Agent': 'InvaderHunter/1.0',
                         'Content-Type': 'application/x-www-form-urlencoded'},
            )
            return json.loads(urllib.request.urlopen(req, timeout=130).read())
        except Exception as e:
            print(f'    attempt {attempt+1} failed: {e}')
            if attempt < retries - 1:
                time.sleep(5)
    raise RuntimeError(f'Failed to fetch {url}')

def parse_coord(v):
    try:
        return float(str(v).replace(',', '.'))
    except Exception:
        return None

def normalise_status(s):
    if not s:
        return 'ok'
    l = s.lower()
    if 'destroy' in l or 'gone' in l or 'missing' in l: return 'destroyed'
    if 'hidden'  in l or 'covered' in l:                 return 'hidden'
    if 'damage'  in l or 'little'  in l or 'partial' in l: return 'damaged'
    return 'ok'

ID_RE = re.compile(r'[A-Z]{1,6}_\d+', re.I)

def extract_osm_id(tags):
    for field in ('ref', 'name', 'description'):
        m = ID_RE.search(tags.get(field, ''))
        if m:
            return m.group(0).upper()
    return None

# ── step 1: scrape invader-spotter.art ───────────────────────────────────────

def make_driver(headless):
    opts = Options()
    if headless:
        opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--window-size=1280,900')
    return webdriver.Chrome(options=opts)

def get_city_codes(driver):
    driver.get(SPOTTER_BASE)
    time.sleep(2)
    links = driver.find_elements(By.XPATH, "//a[contains(@href,'envoi(')]")
    codes = []
    for l in links:
        m = re.search(r'envoi\("([^"]+)"\)', l.get_attribute('href') or '')
        if m:
            codes.append(m.group(1))
    return codes

def parse_spotter_page(html):
    entries = []
    id_re = re.compile(r'<b>([A-Z]{1,6}_\d+)\s+\[(\d+|\?\?)\s*pts?\]</b>', re.I)
    for m in id_re.finditer(html):
        inv_id = m.group(1)
        points = int(m.group(2)) if m.group(2).isdigit() else None
        window = html[m.start():m.start() + 800]
        st_m   = re.search(r'spot_invader_(\w+)\.png', window)
        status = STATUS_MAP.get(st_m.group(1) if st_m else 'unknown', 'unknown')
        entries.append({'id': inv_id, 'points': points, 'status': status})
    return entries

def total_from_page(html):
    m = re.search(r'résultats\s+\d+-\d+\s*/\s*(\d+)', html)
    return int(m.group(1)) if m else 0

def scrape_spotter(headless):
    print('\n── Step 1: scraping invader-spotter.art ──')
    driver = make_driver(headless)
    all_pois = []
    try:
        codes = get_city_codes(driver)
        print(f'  {len(codes)} cities found')
        for i, code in enumerate(codes, 1):
            print(f'  [{i:2d}/{len(codes)}] {code}…', end=' ', flush=True)
            try:
                driver.get(SPOTTER_BASE)
                time.sleep(1)
                driver.execute_script(f'envoi("{code}")')
                time.sleep(CITY_WAIT)
                html    = driver.page_source
                entries = parse_spotter_page(html)
                total   = total_from_page(html)
                pages   = math.ceil(total / PER_PAGE) if total > PER_PAGE else 1
                for page in range(2, pages + 1):
                    driver.execute_script(f'changepage({page})')
                    time.sleep(PAGE_WAIT)
                    entries += parse_spotter_page(driver.page_source)
                for e in entries:
                    e['city'] = code
                all_pois.extend(entries)
                print(f'{len(entries)} POIs')
            except Exception as ex:
                print(f'ERROR: {ex}')
    finally:
        driver.quit()

    SPOTTER_FILE.write_text(json.dumps(all_pois, indent=2, ensure_ascii=False))
    print(f'  → {len(all_pois)} total POIs saved to {SPOTTER_FILE}')
    return all_pois

# ── step 2: fetch OSM coordinates ────────────────────────────────────────────

def fetch_osm():
    print('\n── Step 2: querying OpenStreetMap (Overpass) ──')
    data = fetch_json(
        OVERPASS_URL,
        post_data=urllib.parse.urlencode({'data': OVERPASS_Q}).encode(),
    )
    osm = {}
    for node in data.get('elements', []):
        pid = extract_osm_id(node.get('tags', {}))
        if pid:
            osm[pid] = {'lat': node['lat'], 'lng': node['lon']}
    print(f'  → {len(osm)} geolocated POIs from OSM')
    return osm

# ── step 3: fetch GitHub fallback coords ─────────────────────────────────────

def fetch_github():
    print('\n── Step 3: fetching GitHub coordinate fallback ──')
    raw   = fetch_json(GITHUB_URL)
    items = raw if isinstance(raw, list) \
        else raw.get('invaders', list(raw.values())[0] if isinstance(raw, dict) else [])
    github = {}
    for p in items:
        pid = str(p.get('id') or p.get('name') or '').strip()
        lat = parse_coord(p.get('lat'))
        lng = parse_coord(p.get('lng'))
        if pid and lat and lng and lat != 0 and lng != 0:
            github[pid] = {'lat': lat, 'lng': lng,
                           'status': normalise_status(p.get('status')),
                           'points': int(p['points']) if str(p.get('points','')).isdigit() else None}
    print(f'  → {len(github)} geolocated POIs from GitHub')
    return github

# ── step 4: merge ─────────────────────────────────────────────────────────────

def merge(spotter, osm, github):
    print('\n── Step 4: merging ──')
    result   = []
    no_coord = []

    for p in spotter:
        pid   = p['id']
        entry = {'id': pid, 'city': p['city'], 'status': p['status'], 'points': p['points']}

        if pid in github:
            entry['lat'] = github[pid]['lat']
            entry['lng'] = github[pid]['lng']
        elif pid in osm:
            entry['lat'] = osm[pid]['lat']
            entry['lng'] = osm[pid]['lng']
        else:
            no_coord.append(pid)
            # still include in output — lat/lng simply absent

        result.append(entry)

    # Add any github POIs not in spotter (safety net)
    spotter_ids = {p['id'] for p in spotter}
    for pid, gh in github.items():
        if pid not in spotter_ids:
            result.append({'id': pid, 'city': '', 'status': gh['status'],
                           'points': gh['points'], 'lat': gh['lat'], 'lng': gh['lng']})

    OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Report
    from collections import Counter
    missing_by_city = Counter(pid.split('_')[0] for pid in no_coord)

    print(f'  → {len(result)} geolocated POIs written to {OUTPUT_FILE}')
    print(f'  → {len(no_coord)} POIs without coordinates:')
    for city, count in sorted(missing_by_city.items(), key=lambda x: -x[1]):
        print(f'     {city:<10} {count}')
    return result

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    headless     = '--show'         not in sys.argv
    skip_scrape  = '--skip-scrape'  in sys.argv

    if skip_scrape:
        if not SPOTTER_FILE.exists():
            sys.exit('--skip-scrape used but spotter_pois.json not found')
        print(f'Reusing existing {SPOTTER_FILE}')
        spotter = json.loads(SPOTTER_FILE.read_text())
    else:
        spotter = scrape_spotter(headless)

    osm    = fetch_osm()
    github = fetch_github()
    result = merge(spotter, osm, github)

    print(f'\nDone. {len(result)} POIs in {OUTPUT_FILE}')

if __name__ == '__main__':
    main()
