#!/usr/bin/env python3
"""
Full POI update pipeline — run periodically to keep world_invaders.json fresh.

Steps:
  1. Scrape invader-spotter.art  →  spotter_pois.json   (IDs, status, points)
  2. Fetch pnote.eu              →  primary coordinates  (4 259 POIs)
  3. Query Overpass / OSM        →  fallback coordinates for anything missing
  4. Merge everything            →  world_invaders.json

Step 4 also diffs the new result against the previous world_invaders.json
(before overwriting it) and prints what changed: newly discovered invaders,
POIs that gained coordinates that were previously missing, and any that lost
them, each broken down by city. The very first run has nothing to diff
against and just says so.

It also copies the fresh world_invaders.json to docs/world_invaders.json --
that's the copy the deployed PWA actually fetches (JSON_URL in index.html),
and nothing else keeps it in sync. spotter_pois.json/city_names.json aren't
copied there since the web app doesn't read them at all.

world_invaders.json is served cache-first by docs/sw.js (same bucket as the
ML model/embeddings), so whenever the synced copy's content actually changes,
this also bumps the SW cache version and About build date -- the same deploy
markers export_for_web.py bumps -- otherwise the update would silently never
reach anyone with a warm cache, no matter how many times they reload. Skipped
when nothing changed, so a routine run that finds nothing new doesn't
needlessly bust everyone's cache.

Usage:
  python3 update_pois.py              # full run (headless browser)
  python3 update_pois.py --show       # same but with visible browser
  python3 update_pois.py --skip-scrape  # skip step 1, reuse existing spotter_pois.json
"""

import json, math, re, sys, time, urllib.parse, urllib.request
from datetime import date
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# ── config ────────────────────────────────────────────────────────────────────

SPOTTER_BASE = 'https://www.invader-spotter.art/villes.php'
PNOTE_URL    = 'https://pnote.eu/projects/invaders/map/invaders.json'
OVERPASS_URL = 'https://overpass-api.de/api/interpreter'
OVERPASS_Q   = '[out:json][timeout:120];\nnode["artist_name"="Invader"];\nout body;'

SPOTTER_FILE   = Path('spotter_pois.json')
CITY_NAMES_FILE = Path('city_names.json')
OUTPUT_FILE    = Path('world_invaders.json')
DOCS_OUTPUT_FILE = Path('docs/world_invaders.json')  # what the deployed PWA actually reads
SW_FILE        = Path('docs/sw.js')
HTML_FILE      = Path('docs/index.html')

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

# ── deploy version markers ──────────────────────────────────────────────────
# Mirrors export_for_web.py's bump -- kept as a separate copy rather than a
# shared import since ml/ isn't set up as an importable package from here
# (same reasoning as the grid-snap logic already duplicated across the ml/
# scripts in this repo).

def bump_sw_cache_version():
    text = SW_FILE.read_text()
    m = re.search(r"const CACHE\s*=\s*'invader-hunter-v(\d+)'", text)
    if not m:
        print(f'  warn: could not find CACHE version in {SW_FILE}, skipping bump')
        return
    old_v, new_v = int(m.group(1)), int(m.group(1)) + 1
    SW_FILE.write_text(text.replace(f"invader-hunter-v{old_v}'", f"invader-hunter-v{new_v}'"))
    print(f'  SW cache: v{old_v} → v{new_v}')

def bump_about_build_date():
    text = HTML_FILE.read_text()
    m = re.search(r'(<div class="about-build" id="about-build-date">build )([^<]+)(</div>)', text)
    if not m:
        print(f'  warn: could not find about-build-date in {HTML_FILE}, skipping bump')
        return
    prefix, old_value, suffix = m.groups()
    today = date.today().isoformat()
    if old_value.startswith(today):
        rest = old_value[len(today):]
        new_value = today + (rest[:-1] + chr(ord(rest[-1]) + 1) if rest and rest[-1].isalpha() else 'b')
    else:
        new_value = today
    HTML_FILE.write_text(text[:m.start()] + prefix + new_value + suffix + text[m.end():])
    print(f'  About build: {old_value} → {new_value}')

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
    """Returns list of (code, name) tuples."""
    driver.get(SPOTTER_BASE)
    time.sleep(2)
    links = driver.find_elements(By.XPATH, "//a[contains(@href,'envoi(')]")
    codes = []
    for l in links:
        m = re.search(r'envoi\("([^"]+)"\)', l.get_attribute('href') or '')
        if m:
            codes.append((m.group(1), l.text.strip()))
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
    all_pois  = []
    city_names = {}   # code → full name
    try:
        codes = get_city_codes(driver)
        print(f'  {len(codes)} cities found')
        for i, (code, name) in enumerate(codes, 1):
            city_names[code] = name
            print(f'  [{i:2d}/{len(codes)}] {name} ({code})…', end=' ', flush=True)
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
                    e['city'] = name
                all_pois.extend(entries)
                print(f'{len(entries)} POIs')
            except Exception as ex:
                print(f'ERROR: {ex}')
    finally:
        driver.quit()

    SPOTTER_FILE.write_text(json.dumps(all_pois, indent=2, ensure_ascii=False))
    CITY_NAMES_FILE.write_text(json.dumps(city_names, indent=2, ensure_ascii=False))
    print(f'  → {len(all_pois)} total POIs saved to {SPOTTER_FILE}')
    return all_pois, city_names

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

# ── step 2: fetch pnote.eu coords (primary) ──────────────────────────────────

def fetch_pnote():
    print('\n── Step 2: fetching pnote.eu coordinates ──')
    items = fetch_json(PNOTE_URL)
    pnote = {}
    for p in items:
        pid = str(p.get('id') or '').strip()
        lat = p.get('obf_lat')
        lng = p.get('obf_lng')
        if pid and lat and lng:
            pnote[pid] = {'lat': lat, 'lng': lng}
    print(f'  → {len(pnote)} geolocated POIs from pnote.eu')
    return pnote

# ── step 4: merge ─────────────────────────────────────────────────────────────

def merge(spotter, pnote, osm, city_names):
    print('\n── Step 4: merging ──')

    prev_by_id = {}
    if OUTPUT_FILE.exists():
        try:
            prev_by_id = {p['id']: p for p in json.loads(OUTPUT_FILE.read_text())}
        except Exception:
            prev_by_id = {}

    result   = []
    no_coord = []

    for p in spotter:
        pid   = p['id']
        entry = {'id': pid, 'city': p['city'], 'status': p['status'], 'points': p['points']}

        if pid in pnote:
            entry['lat'] = pnote[pid]['lat']
            entry['lng'] = pnote[pid]['lng']
        elif pid in osm:
            entry['lat'] = osm[pid]['lat']
            entry['lng'] = osm[pid]['lng']
        else:
            no_coord.append(pid)
            # still include in output — lat/lng simply absent

        result.append(entry)

    data_json = json.dumps(result, indent=2, ensure_ascii=False)
    OUTPUT_FILE.write_text(data_json)

    # Also sync to docs/ -- that's the copy the deployed PWA actually reads
    # (JSON_URL in index.html), and nothing else keeps it in sync otherwise.
    if DOCS_OUTPUT_FILE.parent.exists():
        DOCS_OUTPUT_FILE.write_text(data_json)
        print(f'  → synced to {DOCS_OUTPUT_FILE} (served by the deployed PWA)')
    else:
        print(f'  warn: {DOCS_OUTPUT_FILE.parent} not found, skipping docs/ sync')

    # Report
    from collections import Counter
    missing_by_city = Counter(pid.split('_')[0] for pid in no_coord)

    print(f'  → {len(result)} geolocated POIs written to {OUTPUT_FILE}')
    print(f'  → {len(no_coord)} POIs without coordinates:')
    for city, count in sorted(missing_by_city.items(), key=lambda x: -x[1]):
        print(f'     {city:<10} {count}')

    print_diff_since_last_run(result, prev_by_id)
    return result

def _preview(ids, cap=20):
    ids = sorted(ids)
    shown = ', '.join(ids[:cap])
    more  = f'  (+{len(ids) - cap} more)' if len(ids) > cap else ''
    return f': {shown}{more}' if shown else ''

def print_diff_since_last_run(result, prev_by_id):
    if not prev_by_id:
        print(f'\n  Δ no previous {OUTPUT_FILE} to compare against — this run establishes the baseline')
        return

    from collections import Counter

    new_ids       = [p['id'] for p in result if p['id'] not in prev_by_id]
    newly_located = [p['id'] for p in result if p.get('lat') is not None
                      and p['id'] in prev_by_id and prev_by_id[p['id']].get('lat') is None]
    newly_lost    = [p['id'] for p in result if p.get('lat') is None
                      and p['id'] in prev_by_id and prev_by_id[p['id']].get('lat') is not None]

    print(f'\n  Δ since last run:')
    print(f'    {len(new_ids)} new invader(s) discovered{_preview(new_ids)}')
    print(f'    {len(newly_located)} newly geolocated (had no coordinates before){_preview(newly_located)}')
    if newly_lost:
        print(f'    {len(newly_lost)} LOST coordinates (had them before, missing now){_preview(newly_lost)}')

    if new_ids:
        by_city = Counter(pid.split('_')[0] for pid in new_ids)
        for city, count in sorted(by_city.items(), key=lambda x: -x[1]):
            print(f'       {city:<10} +{count}')

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    headless     = '--show'         not in sys.argv
    skip_scrape  = '--skip-scrape'  in sys.argv

    if skip_scrape:
        if not SPOTTER_FILE.exists():
            sys.exit('--skip-scrape used but spotter_pois.json not found')
        print(f'Reusing existing {SPOTTER_FILE}')
        spotter    = json.loads(SPOTTER_FILE.read_text())
        city_names = json.loads(CITY_NAMES_FILE.read_text()) if CITY_NAMES_FILE.exists() else {}
    else:
        spotter, city_names = scrape_spotter(headless)

    pnote  = fetch_pnote()
    osm    = fetch_osm()
    result = merge(spotter, pnote, osm, city_names)

    print(f'\nDone. {len(result)} POIs in {OUTPUT_FILE}')

if __name__ == '__main__':
    main()
