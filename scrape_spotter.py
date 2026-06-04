#!/usr/bin/env python3
"""
Scrapes invader IDs, status and points from invader-spotter.art
Output: spotter_pois.json  — [{id, city, status, points}]

Usage:
  python3 scrape_spotter.py              # scrape all cities
  python3 scrape_spotter.py --show       # same but with visible browser
  python3 scrape_spotter.py --city PA    # single city (for testing)
"""

import json, time, re, sys, math
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

BASE      = 'https://www.invader-spotter.art/villes.php'
PER_PAGE  = 50
CITY_WAIT = 4    # seconds after envoi()
PAGE_WAIT = 3    # seconds after changepage()

STATUS_MAP = {
    'ok':        'ok',
    'degraded':  'damaged',
    'destroyed': 'destroyed',
    'unknown':   'unknown',
    'neutre':    'ok',
}

def make_driver(headless=True):
    opts = Options()
    if headless:
        opts.add_argument('--headless=new')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--disable-dev-shm-usage')
    opts.add_argument('--window-size=1280,900')
    return webdriver.Chrome(options=opts)

def get_city_codes(driver):
    driver.get(BASE)
    time.sleep(2)
    links = driver.find_elements(By.XPATH, "//a[contains(@href, 'envoi(')]")
    codes = []
    for l in links:
        href = l.get_attribute('href') or ''
        m = re.search(r'envoi\("([^"]+)"\)', href)
        if m:
            codes.append(m.group(1))
    return codes

def parse_page(html):
    """Extract list of (id, points, status) from a city page HTML."""
    entries = []

    # Find all ID+points occurrences: <b>NY_01 [10 pts]</b> or <b>NY_03 [?? pts]</b>
    id_re = re.compile(r'<b>([A-Z]{1,6}_\d+)\s+\[(\d+|\?\?)\s*pts?\]</b>', re.I)

    for m in id_re.finditer(html):
        inv_id = m.group(1)
        points = int(m.group(2)) if m.group(2).isdigit() else None

        # Look for the nearest status image within the next 800 chars
        window = html[m.start():m.start() + 800]
        st_m   = re.search(r'spot_invader_(\w+)\.png', window)
        raw_st = st_m.group(1) if st_m else 'unknown'
        status = STATUS_MAP.get(raw_st, raw_st)

        entries.append({'id': inv_id, 'points': points, 'status': status})

    return entries

def total_from_page(html):
    """Extract total count from 'résultats 1-50 / 219' string."""
    m = re.search(r'résultats\s+\d+-\d+\s*/\s*(\d+)', html)
    return int(m.group(1)) if m else 0

def scrape_city(driver, code):
    driver.get(BASE)
    time.sleep(1)
    driver.execute_script(f'envoi("{code}")')
    time.sleep(CITY_WAIT)

    html    = driver.page_source
    entries = parse_page(html)
    total   = total_from_page(html)
    pages   = math.ceil(total / PER_PAGE) if total else 1

    print(f'  {code}: {total} total, {pages} page(s)', flush=True)

    for page in range(2, pages + 1):
        driver.execute_script(f'changepage({page})')
        time.sleep(PAGE_WAIT)
        entries += parse_page(driver.page_source)

    # Tag each entry with its city code
    for e in entries:
        e['city'] = code

    return entries

def main():
    headless = '--show' not in sys.argv
    single   = None
    if '--city' in sys.argv:
        idx = sys.argv.index('--city')
        single = sys.argv[idx + 1]

    driver = make_driver(headless=headless)
    all_pois = []

    try:
        if single:
            codes = [single]
        else:
            print('Fetching city list…')
            codes = get_city_codes(driver)
            print(f'Found {len(codes)} cities\n')

        for i, code in enumerate(codes, 1):
            print(f'[{i}/{len(codes)}] Scraping {code}…', end=' ', flush=True)
            try:
                pois = scrape_city(driver, code)
                all_pois.extend(pois)
                print(f'  → {len(pois)} entries', flush=True)
            except Exception as ex:
                print(f'  ERROR: {ex}', flush=True)

    finally:
        driver.quit()

    out = 'spotter_pois.json'
    with open(out, 'w') as f:
        json.dump(all_pois, f, indent=2, ensure_ascii=False)
    print(f'\nDone. {len(all_pois)} POIs saved to {out}')

if __name__ == '__main__':
    main()
