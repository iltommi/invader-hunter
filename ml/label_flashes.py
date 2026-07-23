#!/usr/bin/env python3
"""
Flash identity labeler — manually confirm which known invader each freshly
scanned FlashInvaders photo actually is.

For each photo in flashes/images/, embeds a multi-scale grid of crops (many
flash photos are wide, uncropped street shots where the invader is a small
part of the frame) with the (fine-tuned, if present) CLIP encoder, and shows
the top-K nearest known invaders from embeddings_clip.npz. Click the correct
match, type an ID if the right one isn't shown, or mark unknown/skip. Labels
are saved incrementally; restart resumes where you left off.

If the flash's reported city matches a known city code, candidates are
restricted to that city (falls back to the unrestricted global top-K if the
city can't be resolved). This is a labeling-time convenience only. The real
recognition pipeline (and train_clip.py) never sees city info, since a real
scan in the wild won't have it either, so it must not become a crutch the
model learns to rely on.

Confirmed labels are picked up by train_clip.py: a POI with one or more
confirmed real photos gets genuine multi-photo contrastive pairs instead of
only synthetic augmentations of its single reference image. Since coverage
across invaders is uneven (some have a dozen confirmed photos, most have
none), each candidate shows how many confirmed photos it already has, and
--breadth reorders the queue to surface likely-uncovered invaders first.

Skip has three reasons (tracked separately, shown live and at exit) so you
can tell photo-quality losses (dark/blurry -- not a fair test of the model)
apart from genuinely-hard-to-place photos. All skip reasons are excluded
from training/eval either way; the reason is just for your own bookkeeping.

Controls:
  skew / v-skew sliders — shear the photo to correct for it being taken at an
                angle to the wall (skew = left/right camera angle, v-skew =
                up/down camera angle, e.g. shooting a mural mounted high on a
                wall from below). Re-searches the whole (now-sheared) photo
                automatically ~300ms after you stop moving the slider (not on
                every tick, to avoid re-running multi-crop search dozens of
                times a second while dragging). Reset clears both back to 0.
  drag on photo — search only the selected region, embedded directly with no
                  further sub-cropping (you've already isolated the invader,
                  unlike the automatic whole-photo multi-crop search)
  double-click / r on photo — reset to the full-photo multi-crop search (also
                  resets both skew sliders to 0)
  left/right  — nudge horizontal skew by 1°
  up/down     — nudge vertical skew by 1°
  1-9         — pick that candidate (ranks 10+ are click-only)
  u           — Unknown / not in database
  s           — Skip (other/unclear reason)
  d           — Skip (photo too dark)
  f           — Skip (photo blurry)
  b           — Back (undo last label)
  Return      — confirm typed ID
  q / Escape  — Quit

Usage (from ml/ directory):
  python3 label_flashes.py                    # 4x4 grid (16 candidates)
  python3 label_flashes.py --top 24 --cols 6  # 4x6 grid (24 candidates)
  python3 label_flashes.py --top 9
  python3 label_flashes.py --top 240 --cols 6 # 40 rows -- only 4 visible at
                                               # once, scroll for the rest
  python3 label_flashes.py --breadth          # prioritize under-covered invaders
"""

import argparse
import json
import math
import tkinter as tk
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageTk
from transformers import CLIPModel, CLIPProcessor

# ── Config ────────────────────────────────────────────────────────────────────

FLASHES_DIR    = Path('flashes/images')
METADATA_FILE  = Path('flashes/metadata.jsonl')
LABELS_FILE    = Path('flashes/identity_labels.json')
IMAGES_DIR     = Path('images')
EMB_FILE       = Path('embeddings_clip.npz')
FINETUNED_FILE = Path('clip_finetuned.pt')
CITY_NAMES_FILE = Path('../city_names.json')

QUERY_SIZE = 320
CAND_SIZE  = 135
MAX_VISIBLE_ROWS = 4  # candidate grid viewport height cap; rest scrolls
BG      = '#1a1a2e'
FG      = '#00f5ff'
GOOD_COL = '#39ff14'
BAD_COL  = '#ff2d55'
SKIP_COL = '#888888'
CITY_COL = '#ffab00'

# Multi-crop search: many flash photos are wide street shots where the
# invader is a small part of the frame, so we embed the full image plus a
# multi-scale grid of sub-crops and take, per candidate, the best (max)
# similarity across all of them.
CROP_SCALES  = (0.5, 0.7)
CROP_GRID    = 3

# ── Model + index ─────────────────────────────────────────────────────────────

def load_model():
    device = (
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available()         else
        torch.device('cpu')
    )
    model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
    if FINETUNED_FILE.exists():
        ckpt = torch.load(FINETUNED_FILE, map_location=device)
        model.vision_model.load_state_dict(ckpt['vision_model'])
        model.visual_projection.load_state_dict(ckpt['visual_projection'])
        print('Using fine-tuned weights')
    return model, processor, device

def skew_image(img, h_degrees, v_degrees=0):
    """Shear correction for photos taken at an angle to the wall.
    h_degrees: horizontal shear, tilts top vs bottom (camera angled left/right).
    v_degrees: vertical shear, tilts one side vs the other (camera angled
    up/down -- common when a mural is mounted high and shot from below).
    Both are applied as a single affine transform (not two sequential warps,
    which would resample/blur twice). Output is the same size as the input;
    newly-exposed corners are filled gray."""
    if not h_degrees and not v_degrees:
        return img
    w, h = img.size
    kh = math.tan(math.radians(h_degrees))
    kv = math.tan(math.radians(v_degrees))
    return img.transform(
        (w, h), Image.AFFINE,
        (1, kh, -kh * h / 2, kv, 1, -kv * w / 2),
        resample=Image.BICUBIC, fillcolor=(128, 128, 128),
    )

def generate_crops(img, scales=CROP_SCALES, grid=CROP_GRID):
    """Full image plus a multi-scale grid of sub-crops, so a small invader in
    a wide street photo still gets a tight, well-matched view."""
    w, h = img.size
    crops = [img]
    for s in scales:
        cw, ch = int(w * s), int(h * s)
        if cw < 20 or ch < 20:
            continue
        xs = np.linspace(0, w - cw, grid) if w > cw else [0]
        ys = np.linspace(0, h - ch, grid) if h > ch else [0]
        for y in ys:
            for x in xs:
                x, y = int(x), int(y)
                crops.append(img.crop((x, y, x + cw, y + ch)))
    return crops

def embed_batch(model, processor, device, imgs):
    inputs = processor(images=imgs, return_tensors='pt').to(device)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs['pixel_values'])
        feats = model.visual_projection(vision_out.pooler_output).cpu().numpy()
    feats /= np.linalg.norm(feats, axis=1, keepdims=True)
    return feats

def search_multi_crop(model, processor, device, img, all_emb):
    """Per-candidate max similarity across the full image and all sub-crops."""
    crops     = generate_crops(img)
    crop_embs = embed_batch(model, processor, device, crops)
    return (crop_embs @ all_emb.T).max(axis=0)

def search_single(model, processor, device, img, all_emb):
    """Direct embedding similarity, no sub-cropping — for manually drawn
    selections, which are already a tight crop of the invader."""
    emb = embed_batch(model, processor, device, [img])[0]
    return all_emb @ emb

def precompute_top1(model, processor, device, paths, all_ids, all_emb, batch_size=64):
    """Fast whole-image top-1 guess per path (no multi-crop -- this is only
    used to roughly rank the queue, not to label), for --breadth ordering."""
    guesses = {}
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        imgs = []
        for p in batch:
            try:
                imgs.append(Image.open(p).convert('RGB'))
            except Exception:
                imgs.append(Image.new('RGB', (224, 224)))
        sims = embed_batch(model, processor, device, imgs) @ all_emb.T
        for p, row in zip(batch, sims):
            guesses[p.stem] = str(all_ids[int(row.argmax())])
        print(f'  breadth precompute {min(i + batch_size, len(paths))}/{len(paths)}', end='\r', flush=True)
    print()
    return guesses

def load_metadata():
    records = {}
    if METADATA_FILE.exists():
        for line in METADATA_FILE.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            records[rec['flash_id']] = rec
    return records

# FlashInvaders' own "city" field doesn't always match invader-spotter's
# naming in city_names.json (scraped upstream, used across the rest of this
# repo) -- different language/spelling conventions or abbreviations. Small
# supplementary table for known mismatches; extend it if you spot more.
CITY_ALIASES = {
    'cologne':      'KLN',   # city_names.json has the native "Köln"
    'rome':         'ROM',   # city_names.json has the native "Roma"
    'barcelone':    'BRC',   # city_names.json has "Barcelona"
    "cote d'azur":  'CAZ',   # city_names.json has "Côte d'Azur"
    'clermont-frd': 'CLR',   # city_names.json has "Clermont-Ferrand"
    'hong-kong':    'HK',    # city_names.json has "Hong Kong"
}

def load_city_lookup():
    try:
        codes = json.loads(CITY_NAMES_FILE.read_text())
    except Exception:
        codes = {}
    lookup = {name.lower(): code for code, name in codes.items()}
    lookup.update({code.lower(): code for code in codes})  # flash sometimes reports the code itself, e.g. "bab"
    lookup.update(CITY_ALIASES)
    return lookup

# ── App ───────────────────────────────────────────────────────────────────────

class FlashLabeler:
    def __init__(self, root, top_k, cols=4, breadth=False):
        self.root  = root
        self.top_k = top_k
        self.cols  = cols
        self.root.title('Flash Identity Labeler')
        self.root.configure(bg=BG)
        # Not resizable(False, False) yet -- on this platform Tk locks that
        # as a hard ceiling on the window's size at the moment it's called,
        # and _presize_window() below needs to grow the window past its
        # empty-candidate-grid starting size first. Locked at the end of
        # _presize_window() once the real (worst-case) size is set.

        print('Loading CLIP...')
        self.model, self.processor, self.device = load_model()

        data = np.load(EMB_FILE)
        self.all_ids = data['ids']
        self.all_emb = data['embeddings']

        self.metadata   = load_metadata()
        self.city_lookup = load_city_lookup()

        self.labels  = {}
        self.history = []
        self._load_labels()
        self._confirmed_counts = Counter(
            l['poi_id'] for l in self.labels.values() if l.get('verdict') == 'confirmed'
        )
        self.session_confirmed = 0  # confirms made in this run
        self.session_new       = 0  # of those, first-ever confirmed photo for that invader
        self.session_skips     = Counter()  # skip reason -> count this run ('dark', 'blurry', 'other')

        all_paths  = sorted(FLASHES_DIR.glob('*.jpg'), key=lambda p: int(p.stem))
        self.paths = [p for p in all_paths if p.stem not in self.labels]
        self.total   = len(all_paths)
        self.labeled = len(self.labels)
        self.idx     = 0

        if breadth and self.paths:
            print(f'Precomputing guesses for {len(self.paths)} remaining photos '
                  f'to prioritize under-covered invaders...')
            guesses = precompute_top1(self.model, self.processor, self.device,
                                       self.paths, self.all_ids, self.all_emb)
            self.paths.sort(key=lambda p: (self._confirmed_counts.get(guesses.get(p.stem), 0), int(p.stem)))

        self._current_candidates = []  # [(poi_id, similarity), ...] for current image
        self._cand_widgets = []

        self._current_orig      = None  # full-resolution PIL image of the current flash photo
        self._current_skewed    = None  # _current_orig with the skew slider's shear applied
        self._current_city_code = None
        self._query_scale       = 1.0
        self._query_offset      = (0, 0)
        self._query_thumb_size  = (QUERY_SIZE, QUERY_SIZE)
        self._drag_start        = None
        self._rect_id           = None
        self._skew_search_job   = None  # pending after() id for the debounced skew re-search

        self._build_ui()
        self._presize_window()
        self._load_current()

    def _load_labels(self):
        if LABELS_FILE.exists():
            try:
                self.labels = json.loads(LABELS_FILE.read_text())
            except Exception:
                self.labels = {}

    def _save_labels(self):
        LABELS_FILE.write_text(json.dumps(self.labels, indent=2))

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=10, pady=6)

        tk.Label(self.root, text='FLASH IDENTITY LABELER', font=('Courier', 13, 'bold'),
                 bg=BG, fg=FG).pack(**pad)

        self.prog_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.prog_var,
                 font=('Courier', 10), bg=BG, fg='#888').pack()
        self.bar_canvas = tk.Canvas(self.root, width=900, height=6, bg='#2a2a3e', highlightthickness=0)
        self.bar_canvas.pack(pady=(0, 8))
        self.bar_fill = self.bar_canvas.create_rectangle(0, 0, 0, 6, fill=FG, width=0)

        main = tk.Frame(self.root, bg=BG)
        main.pack(padx=10)

        # Query panel
        query_frame = tk.Frame(main, bg=BG)
        query_frame.grid(row=0, column=0, sticky='n', padx=(0, 16))
        self.query_canvas = tk.Canvas(query_frame, width=QUERY_SIZE, height=QUERY_SIZE,
                                       bg='#0a0a0f', highlightthickness=0, cursor='crosshair')
        self.query_canvas.pack()
        self.query_canvas.bind('<ButtonPress-1>', self._on_drag_start)
        self.query_canvas.bind('<B1-Motion>', self._on_drag_move)
        self.query_canvas.bind('<ButtonRelease-1>', self._on_drag_end)
        self.query_canvas.bind('<Double-Button-1>', lambda _: self._reset_crop())
        tk.Label(query_frame, text='drag to search a region  ·  double-click / r to reset',
                 font=('Courier', 8), bg=BG, fg='#444').pack(pady=(4, 0))

        skew_frame = tk.Frame(query_frame, bg=BG)
        skew_frame.pack(pady=(4, 0), fill='x')
        tk.Label(skew_frame, text='skew:', font=('Courier', 8), bg=BG, fg='#888').pack(side='left')
        self.skew_var = tk.DoubleVar(value=0.0)
        tk.Scale(skew_frame, from_=-30, to=30, resolution=1, orient=tk.HORIZONTAL,
                 variable=self.skew_var, command=self._on_skew_change,
                 bg=BG, fg=FG, troughcolor='#2a2a3e', highlightthickness=0,
                 font=('Courier', 8), showvalue=True, relief='flat').pack(side='left', fill='x', expand=True)

        vskew_frame = tk.Frame(query_frame, bg=BG)
        vskew_frame.pack(pady=(2, 0), fill='x')
        tk.Label(vskew_frame, text='v-skew:', font=('Courier', 8), bg=BG, fg='#888').pack(side='left')
        self.vskew_var = tk.DoubleVar(value=0.0)
        tk.Scale(vskew_frame, from_=-30, to=30, resolution=1, orient=tk.HORIZONTAL,
                 variable=self.vskew_var, command=self._on_skew_change,
                 bg=BG, fg=FG, troughcolor='#2a2a3e', highlightthickness=0,
                 font=('Courier', 8), showvalue=True, relief='flat').pack(side='left', fill='x', expand=True)

        self.query_caption = tk.Label(query_frame, text='', font=('Courier', 10),
                                       bg=BG, fg='#aaa', justify='left')
        self.query_caption.pack(pady=(6, 0))

        # Manual ID entry
        entry_frame = tk.Frame(query_frame, bg=BG)
        entry_frame.pack(pady=(10, 0))
        tk.Label(entry_frame, text='ID:', font=('Courier', 10), bg=BG, fg='#888').grid(row=0, column=0)
        self.id_entry = tk.Entry(entry_frame, font=('Courier', 10), width=14,
                                  bg='#0a0a0f', fg=FG, insertbackground=FG, relief='flat')
        self.id_entry.grid(row=0, column=1, padx=4)
        self.id_entry.bind('<Return>', lambda _: self._confirm_typed())
        tk.Button(entry_frame, text='Confirm', font=('Courier', 9), command=self._confirm_typed,
                  bg='#2a2a3e', fg=FG, relief='flat', cursor='hand2').grid(row=0, column=2)
        self.entry_error = tk.Label(query_frame, text='', font=('Courier', 9), bg=BG, fg=BAD_COL)
        self.entry_error.pack()

        # Candidates panel -- a fixed-height scrollable viewport (a Canvas,
        # which unlike a Frame never auto-resizes to its embedded content) so
        # a large --top doesn't demand an absurdly tall window; extra rows
        # scroll instead.
        cand_panel = tk.Frame(main, bg=BG)
        cand_panel.grid(row=0, column=1, sticky='n')
        self.filter_var = tk.StringVar()
        tk.Label(cand_panel, textvariable=self.filter_var, font=('Courier', 9),
                 bg=BG, fg=CITY_COL).pack(anchor='w', pady=(0, 4))

        cand_scroll_area = tk.Frame(cand_panel, bg=BG)
        cand_scroll_area.pack()
        self.cand_canvas = tk.Canvas(cand_scroll_area, bg=BG, highlightthickness=0)
        cand_scrollbar = tk.Scrollbar(cand_scroll_area, orient='vertical', command=self.cand_canvas.yview)
        self.cand_canvas.configure(yscrollcommand=cand_scrollbar.set)
        self.cand_canvas.pack(side='left')
        cand_scrollbar.pack(side='left', fill='y')

        self.cand_frame = tk.Frame(self.cand_canvas, bg=BG)
        self.cand_canvas.create_window((0, 0), window=self.cand_frame, anchor='nw')
        self.cand_frame.bind('<Configure>',
                              lambda _: self.cand_canvas.configure(scrollregion=self.cand_canvas.bbox('all')))
        self.cand_canvas.bind('<MouseWheel>',
                              lambda e: self.cand_canvas.yview_scroll(-1 if e.delta > 0 else 1, 'units'))

        # Buttons
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=10)
        btn_cfg = dict(font=('Courier', 10, 'bold'), width=12, relief='flat', cursor='hand2', pady=6)
        tk.Button(btn_frame, text='? UNKNOWN [u]', bg=BAD_COL, fg='#fff',
                  command=self._unknown, **btn_cfg).grid(row=0, column=0, padx=6)
        tk.Button(btn_frame, text='→ SKIP [s]', bg='#2a2a3e', fg='#aaa',
                  command=self._skip, **btn_cfg).grid(row=0, column=1, padx=6)
        tk.Button(btn_frame, text='DARK [d]', bg='#2a2a3e', fg='#aaa',
                  command=lambda: self._skip('dark'), **btn_cfg).grid(row=0, column=2, padx=6)
        tk.Button(btn_frame, text='BLURRY [f]', bg='#2a2a3e', fg='#aaa',
                  command=lambda: self._skip('blurry'), **btn_cfg).grid(row=0, column=3, padx=6)
        tk.Button(btn_frame, text='← BACK [b]', bg='#2a2a3e', fg='#aaa',
                  command=self._back, **btn_cfg).grid(row=0, column=4, padx=6)

        tk.Label(self.root,
                 text='1-9 pick candidate (click for 10+)  ·  u unknown  ·  s skip (other)  ·  d skip (dark)  ·  '
                      'f skip (blurry)  ·  b back  ·  r reset crop  ·  q quit',
                 font=('Courier', 8), bg=BG, fg='#444').pack(pady=(0, 10))

        for i in range(1, 10):
            self.root.bind(str(i), lambda _, n=i: self._pick(n - 1))
        self.root.bind('u', lambda _: self._unknown())
        self.root.bind('s', lambda _: self._skip())
        self.root.bind('d', lambda _: self._skip('dark'))
        self.root.bind('f', lambda _: self._skip('blurry'))
        self.root.bind('b', lambda _: self._back())
        self.root.bind('r', lambda _: self._reset_crop())
        self.root.bind('q', lambda _: self.root.destroy())
        self.root.bind('<Escape>', lambda _: self.root.destroy())
        self.root.bind('<Left>',  lambda _: self._nudge_skew(-1, 0))
        self.root.bind('<Right>', lambda _: self._nudge_skew(+1, 0))
        self.root.bind('<Up>',    lambda _: self._nudge_skew(0, -1))
        self.root.bind('<Down>',  lambda _: self._nudge_skew(0, +1))

    def _update_progress(self):
        pct = self.labeled / self.total if self.total else 0
        dupes = self.session_confirmed - self.session_new
        skip_total = sum(self.session_skips.values())
        skip_bits = ', '.join(f'{n} {reason}' for reason, n in self.session_skips.items() if n)
        skip_str = f'  •  skipped: {skip_total} ({skip_bits})' if skip_total else ''
        self.prog_var.set(
            f'{self.labeled} / {self.total} labeled  •  '
            f'{self.total - self.labeled} remaining  •  {round(pct*100)}%  •  '
            f'session: {self.session_confirmed} confirmed ({self.session_new} new invaders, {dupes} dupes)'
            f'{skip_str}'
        )
        self.bar_canvas.coords(self.bar_fill, 0, 0, int(900 * pct), 6)

    # ── Image helpers ─────────────────────────────────────────────────────────

    def _fit(self, img, size):
        img = img.copy()
        img.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new('RGB', (size, size), (10, 10, 15))
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        return ImageTk.PhotoImage(canvas)

    def _placeholder(self, size, text=''):
        canvas = Image.new('RGB', (size, size), (10, 10, 15))
        if text:
            ImageDraw.Draw(canvas).multiline_text((size//2, size//2), text,
                                                    fill=(80, 80, 100), anchor='mm', align='center')
        return ImageTk.PhotoImage(canvas)

    def _show_query(self, img):
        """Render img into query_canvas and remember the scale/offset so
        drag-selected canvas coordinates can be mapped back to img pixels."""
        self.query_canvas.delete('all')
        size = QUERY_SIZE
        w, h = img.size
        scale = min(size / w, size / h, 1.0)
        tw, th = max(1, round(w * scale)), max(1, round(h * scale))
        thumb = img.resize((tw, th), Image.LANCZOS) if scale != 1.0 else img

        canvas_img = Image.new('RGB', (size, size), (10, 10, 15))
        canvas_img.paste(thumb, ((size - tw) // 2, (size - th) // 2))
        self._tk_query = ImageTk.PhotoImage(canvas_img)
        self.query_canvas.create_image(0, 0, anchor='nw', image=self._tk_query)

        self._query_scale      = scale
        self._query_offset     = ((size - tw) // 2, (size - th) // 2)
        self._query_thumb_size = (tw, th)

    def _candidate_thumb(self, poi_id):
        code = poi_id.split('_')[0]
        path = IMAGES_DIR / code / f'{poi_id}.png'
        try:
            return Image.open(path).convert('RGB')
        except Exception:
            return None

    # ── Load current ─────────────────────────────────────────────────────────

    def _load_current(self):
        self.entry_error.configure(text='')
        self.id_entry.delete(0, tk.END)
        self._cancel_skew_search()  # a stale job must not fire later against a different photo

        if self.idx >= len(self.paths):
            self._show_done()
            return

        path = self.paths[self.idx]
        flash_id = int(path.stem)
        self._update_progress()

        try:
            orig = Image.open(path).convert('RGB')
        except Exception:
            orig = Image.new('RGB', (224, 224), (30, 30, 40))

        self._current_orig = orig
        self._current_skewed = orig
        self.skew_var.set(0.0)
        self.vskew_var.set(0.0)
        self._show_query(orig)

        rec = self.metadata.get(flash_id, {})
        city = rec.get('city', '?')
        player = rec.get('player', '?')
        self._current_city_code = self.city_lookup.get(city.lower())
        hint = f' ({self._current_city_code})' if self._current_city_code else ''
        self.query_caption.configure(
            text=f'flash_id {flash_id}\ncity: {city}{hint}\nplayer: {player}'
        )

        self._run_search(orig)

    def _run_search(self, img, multi_crop=True):
        if multi_crop:
            sims = search_multi_crop(self.model, self.processor, self.device, img, self.all_emb)
        else:
            sims = search_single(self.model, self.processor, self.device, img, self.all_emb)
        self._current_candidates, filtered = self._rank_candidates(sims, self._current_city_code)
        self.filter_var.set(f'showing top {len(self._current_candidates)} matches in {self._current_city_code}'
                             if filtered else 'showing global top matches (no city filter)')
        self._render_candidates()

    def _rank_candidates(self, sims, city_code):
        """Top-K by similarity, restricted to the reported city when it's
        resolvable to known invaders (falls back to the unrestricted global
        top-K otherwise). Display-only — see module docstring on why city
        never touches training."""
        mask = None
        if city_code:
            candidate_mask = np.array([str(cid).split('_')[0] == city_code for cid in self.all_ids])
            if candidate_mask.any():
                mask = candidate_mask

        scoped = np.where(mask, sims, -np.inf) if mask is not None else sims
        order  = np.argsort(scoped)[::-1]
        top    = [i for i in order if np.isfinite(scoped[i])][:self.top_k]
        return [(str(self.all_ids[i]), float(sims[i])) for i in top], mask is not None

    def _render_candidates(self):
        for w in self._cand_widgets:
            w.destroy()
        self._cand_widgets = []

        cols = min(self.cols, self.top_k)
        for rank, (poi_id, sim) in enumerate(self._current_candidates):
            r, c = divmod(rank, cols)
            cell = tk.Frame(self.cand_frame, bg=BG)
            cell.grid(row=r, column=c, padx=3, pady=3)

            thumb_img = self._candidate_thumb(poi_id)
            tkimg = self._fit(thumb_img, CAND_SIZE) if thumb_img else self._placeholder(CAND_SIZE, 'no image')

            btn = tk.Button(cell, image=tkimg, relief='flat', bd=0, highlightthickness=0,
                             cursor='hand2', command=lambda n=rank: self._pick(n))
            btn.image = tkimg
            btn.pack()
            fg = GOOD_COL if sim >= 0.8 else (FG if sim >= 0.6 else '#888')
            tk.Label(cell, text=f'[{rank+1}] {poi_id}  {sim:.2f}',
                     font=('Courier', 9), bg=BG, fg=fg).pack(pady=(2, 0))

            count = self._confirmed_counts.get(poi_id, 0)
            cov_text, cov_col = ('new!', GOOD_COL) if count == 0 else (f'have {count}', '#666')
            tk.Label(cell, text=cov_text, font=('Courier', 8), bg=BG, fg=cov_col).pack()

            self._cand_widgets.append(cell)

        self.cand_canvas.yview_moveto(0)  # start each new render scrolled to the top

    def _presize_window(self):
        """Render one full top_k-candidate placeholder grid before the first
        real photo loads, to measure the true cell size. The candidate
        viewport (cand_canvas) is capped at MAX_VISIBLE_ROWS tall regardless
        of top_k -- a huge --top (e.g. 240) would otherwise demand an
        absurdly tall window, so extra rows scroll instead. cand_frame
        itself (inside the canvas) is left free to grow to whatever height
        the actual candidate count needs; unlike a Frame, a Canvas never
        auto-resizes to its embedded content, so no propagate/freeze dance
        is needed here for it to stay put at that capped size."""
        cols = min(self.cols, self.top_k)
        rows = math.ceil(self.top_k / cols)
        placeholders = []
        for rank in range(self.top_k):
            r, c = divmod(rank, cols)
            cell = tk.Frame(self.cand_frame, bg=BG)
            cell.grid(row=r, column=c, padx=3, pady=3)
            tkimg = self._placeholder(CAND_SIZE)
            lbl = tk.Label(cell, image=tkimg, bg=BG)
            lbl.image = tkimg
            lbl.pack()
            tk.Label(cell, text=f'[{rank+1}] XX_0000  0.00', font=('Courier', 9), bg=BG, fg='#888').pack(pady=(2, 0))
            tk.Label(cell, text='have 0', font=('Courier', 8), bg=BG, fg='#666').pack()
            placeholders.append(cell)

        self.root.update()
        full_w = self.cand_frame.winfo_reqwidth()
        row_h  = self.cand_frame.winfo_reqheight() / rows

        for widget in placeholders:
            widget.destroy()

        visible_rows = min(rows, MAX_VISIBLE_ROWS)
        self.cand_canvas.configure(width=full_w, height=round(row_h * visible_rows))

        self.root.update()
        self.root.update()
        self.root.geometry(f'{self.root.winfo_reqwidth()}x{self.root.winfo_reqheight()}')
        self.root.resizable(False, False)

    def _show_done(self):
        self.query_caption.configure(text='All done!')
        self.query_canvas.delete('all')
        self._tk_query = self._placeholder(QUERY_SIZE, '✓ all flashes labeled')
        self.query_canvas.create_image(0, 0, anchor='nw', image=self._tk_query)
        self._current_orig = None
        for w in self._cand_widgets:
            w.destroy()
        self._cand_widgets = []
        self._update_progress()

    # ── Region selection ─────────────────────────────────────────────────────

    def _on_drag_start(self, event):
        if self._current_orig is None:
            return
        self._cancel_skew_search()  # a manual region selection supersedes the whole-image search
        self._drag_start = (event.x, event.y)
        self.query_canvas.delete('rect')
        self._rect_id = self.query_canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline=GOOD_COL, width=2, tags='rect')

    def _on_drag_move(self, event):
        if self._rect_id is None or self._drag_start is None:
            return
        x0, y0 = self._drag_start
        self.query_canvas.coords(self._rect_id, x0, y0, event.x, event.y)

    def _on_drag_end(self, event):
        if self._drag_start is None or self._current_orig is None:
            return
        x0, y0 = self._drag_start
        x1, y1 = event.x, event.y
        self._drag_start = None

        cx0, cx1 = sorted((x0, x1))
        cy0, cy1 = sorted((y0, y1))

        # too small to be a deliberate selection -> ignore, keep current results
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            self.query_canvas.delete('rect')
            self._rect_id = None
            return

        ox, oy = self._query_offset
        tw, th = self._query_thumb_size
        ix0, iy0 = max(cx0, ox), max(cy0, oy)
        ix1, iy1 = min(cx1, ox + tw), min(cy1, oy + th)
        if ix1 - ix0 < 8 or iy1 - iy0 < 8:
            return

        scale = self._query_scale
        w, h  = self._current_skewed.size
        px0 = max(0, (ix0 - ox) / scale)
        py0 = max(0, (iy0 - oy) / scale)
        px1 = min(w, (ix1 - ox) / scale)
        py1 = min(h, (iy1 - oy) / scale)

        crop = self._current_skewed.crop((int(px0), int(py0), int(px1), int(py1)))
        self._run_search(crop, multi_crop=False)

    def _on_skew_change(self, _=None):
        if self._current_orig is None:
            return
        self._current_skewed = skew_image(self._current_orig, self.skew_var.get(), self.vskew_var.get())
        self._show_query(self._current_skewed)
        # any in-progress/finished selection was drawn against the old skew, no longer valid
        self.query_canvas.delete('rect')
        self._rect_id = None
        # Debounced: a slider drag fires this many times a second, and a full
        # multi-crop search (~19 CLIP passes) on every tick would be laggy --
        # only actually re-search once the slider settles for a moment.
        self._cancel_skew_search()
        self._skew_search_job = self.root.after(300, self._skew_search_now)

    def _cancel_skew_search(self):
        if self._skew_search_job:
            self.root.after_cancel(self._skew_search_job)
            self._skew_search_job = None

    def _skew_search_now(self):
        self._skew_search_job = None
        if self._current_skewed is not None:
            self._run_search(self._current_skewed, multi_crop=True)

    def _nudge_skew(self, dh, dv):
        if self._current_orig is None:
            return
        self.skew_var.set(max(-30, min(30, self.skew_var.get() + dh)))
        self.vskew_var.set(max(-30, min(30, self.vskew_var.get() + dv)))
        self._on_skew_change()

    def _reset_crop(self):
        if self._current_orig is None:
            return
        self._cancel_skew_search()
        self.query_canvas.delete('rect')
        self._rect_id = None
        self.skew_var.set(0.0)
        self.vskew_var.set(0.0)
        self._current_skewed = self._current_orig
        self._show_query(self._current_orig)
        self._run_search(self._current_orig)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _label(self, poi_id, verdict, similarity=None, rank=None, reason=None):
        if self.idx >= len(self.paths):
            return
        path = self.paths[self.idx]
        flash_id = int(path.stem)
        rec = self.metadata.get(flash_id, {})
        label = {
            'poi_id': poi_id,
            'verdict': verdict,
            'similarity': similarity,
            'rank': rank,
            'reason': reason,
            'city': rec.get('city'),
            'player': rec.get('player'),
            'timestamp': rec.get('timestamp'),
        }
        self.history.append((path.stem, label))
        self.labels[path.stem] = label
        if verdict == 'confirmed' and poi_id:
            self.session_confirmed += 1
            if self._confirmed_counts.get(poi_id, 0) == 0:
                self.session_new += 1
            self._confirmed_counts[poi_id] += 1
        if verdict == 'skip':
            self.session_skips[reason or 'other'] += 1
        self._save_labels()
        self.labeled += 1
        self.idx += 1
        self._load_current()

    def _pick(self, rank):
        if rank >= len(self._current_candidates):
            return
        poi_id, sim = self._current_candidates[rank]
        self._label(poi_id, 'confirmed', similarity=sim, rank=rank + 1)

    def _confirm_typed(self):
        typed = self.id_entry.get().strip().upper()
        if not typed:
            return
        if typed not in set(self.all_ids.tolist()):
            self.entry_error.configure(text=f'"{typed}" not a known invader ID')
            return
        self._label(typed, 'confirmed', similarity=None, rank=None)

    def _unknown(self): self._label(None, 'unknown')
    def _skip(self, reason=None): self._label(None, 'skip', reason=reason)

    def _back(self):
        if not self.history:
            return
        stem, label = self.history.pop()
        if stem in self.labels:
            del self.labels[stem]
            self._save_labels()
            self.labeled -= 1
        if label.get('verdict') == 'confirmed' and label.get('poi_id'):
            poi_id = label['poi_id']
            self._confirmed_counts[poi_id] -= 1
            self.session_confirmed -= 1
            if self._confirmed_counts[poi_id] <= 0:
                if self._confirmed_counts[poi_id] == 0:
                    self.session_new -= 1
                del self._confirmed_counts[poi_id]
        elif label.get('verdict') == 'skip':
            reason = label.get('reason') or 'other'
            self.session_skips[reason] -= 1
            if self.session_skips[reason] <= 0:
                del self.session_skips[reason]
        path = next((p for p in sorted(FLASHES_DIR.glob('*.jpg'), key=lambda p: int(p.stem))
                     if p.stem == stem), None)
        if path:
            self.paths.insert(self.idx, path)
        self.idx = max(0, self.idx - 1)
        self._load_current()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--top', type=int, default=16, help='number of candidate matches to show (default: 16, a 4x4 grid)')
    parser.add_argument('--cols', type=int, default=4, help='columns in the candidate grid (default: 4; e.g. --top 24 --cols 6 for 4x6)')
    parser.add_argument('--breadth', action='store_true',
                         help='reorder the queue to surface likely-uncovered invaders first (slower startup)')
    args = parser.parse_args()

    root = tk.Tk()
    app  = FlashLabeler(root, top_k=args.top, cols=args.cols, breadth=args.breadth)
    root.mainloop()
    confirmed = sum(1 for v in app.labels.values() if v.get('verdict') == 'confirmed')
    print(f'\nSaved {len(app.labels)} labels ({confirmed} confirmed) → {LABELS_FILE}')
    dupes = app.session_confirmed - app.session_new
    print(f'This session: {app.session_confirmed} confirmed '
          f'({app.session_new} new invaders, {dupes} dupes of already-covered invaders)')
    skip_total = sum(app.session_skips.values())
    if skip_total:
        skip_bits = ', '.join(f'{n} {reason}' for reason, n in app.session_skips.items())
        print(f'Skipped: {skip_total} ({skip_bits})')
