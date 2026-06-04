# Invader Hunter

A PWA for tracking [Space Invader](https://space-invaders.com) street art mosaics on an interactive map. Hosted on GitHub Pages.

## App features

- Map of all known mosaics worldwide, clustered by zoom level
- Filter by status: Unflashed / Flashed / Destroyed / All
- Tap a marker to flash it (mark as found), take an annotated photo, or search Google Images
- Stats popup: remaining, total, points, completion %
- Export / import your flashed list as JSON
- Works offline (service worker caches the app shell and POI data)
- Installable as a PWA on Android and iOS

---

## POI data pipeline

The map data comes from two sources merged together:

| Source | Provides |
|---|---|
| [invader-spotter.art](https://www.invader-spotter.art) | IDs, status, points (scraped via Selenium) |
| [OpenStreetMap](https://www.openstreetmap.org) (Overpass API) | GPS coordinates |

The merge is handled by `update_pois.py`, which writes `world_invaders.json` — the file the app loads at runtime.

### Prerequisites

**Python 3.8+** and **Google Chrome** must be installed.

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Selenium 4 includes automatic ChromeDriver management — no manual driver installation needed.

### Update the POI data

Run the full pipeline (scrape + OSM fetch + merge):

```bash
python3 update_pois.py
```

This takes roughly 10–15 minutes (the spotter scrape is the slow part — 88 cities, paginated).

**Options:**

```bash
python3 update_pois.py --show         # open a visible browser window (useful to debug)
python3 update_pois.py --skip-scrape  # skip scraping, just refresh OSM coords and re-merge
```

Use `--skip-scrape` for a quick coord refresh when you don't need to pick up new IDs or status changes from spotter.

### Individual scripts

| Script | What it does |
|---|---|
| `update_pois.py` | Full pipeline — the one to run |
| `scrape_spotter.py` | Scrape invader-spotter.art only → `spotter_pois.json` |
| `fetch_osm.py` | Fetch OSM coords + merge → `world_invaders.json` |

### Commit the result

After running, commit `world_invaders.json` and optionally `spotter_pois.json`:

```bash
git add world_invaders.json spotter_pois.json
git commit -m "update POI data"
git push
```

GitHub Pages will serve the updated data within a minute.
