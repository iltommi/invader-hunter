#!/usr/bin/env python3
"""
Snap labeling tool — evaluate grid snap quality on reference images.

Shows original photo alongside grid-snapped reconstruction.
Labels are saved incrementally; restart resumes where you left off.

Controls:
  y / Return  — Good snap
  n           — Bad snap
  s           — Skip (unlabeled)
  b           — Back (undo last label)
  q / Escape  — Quit

Usage (from ml/ directory):
  python3 label_snaps.py
"""

import json
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageTk

# ── Grid snap (self-contained, mirrors train_clip.py) ─────────────────────────

SNAP_CONF = 0.10

def _gs_fft_period(profile, min_p=5):
    n = len(profile)
    k_min, k_max = 3, n // min_p
    if k_max < k_min:
        return 0, 0.0
    ps = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
    k_max = min(k_max, len(ps) - 1)
    sub = ps[k_min:k_max + 1]
    total = sub.sum()
    if total == 0:
        return 0, 0.0
    best_idx = int(sub.argmax())
    best_k = k_min + best_idx
    return round(n / best_k), float(sub[best_idx] / total)

def _gs_phase(profile, period):
    T = max(1, round(period))
    n = len(profile)
    best_phase, best_score = 0, np.inf
    for p in range(T):
        score, s = 0.0, p
        while s < n:
            block = profile[s:min(s + T, n)]
            if len(block) > 1:
                score += float(block.var()) * len(block)
            s += T
        if score < best_score:
            best_score = score
            best_phase = p
    return best_phase

def _gs_kmeans(colors, k=4, iters=25):
    n = len(colors)
    idx = np.linspace(0, n - 1, k, dtype=int)
    centroids = colors[idx].astype(np.float64).copy()
    assign = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        dists = np.stack([((colors - c) ** 2).sum(axis=1) for c in centroids], axis=1)
        new_assign = dists.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_assign, assign):
            break
        assign = new_assign
        for c in range(k):
            mask = assign == c
            if mask.any():
                centroids[c] = colors[mask].mean(axis=0)
    return centroids, assign

def _gs_kmeans_adaptive(colors, k_min=2, k_max=8, threshold=0.10, iters=25):
    k_max = min(k_max, len(colors))
    best_centroids, best_assign = None, None
    prev_wcss = None
    first_improvement = None
    for k in range(k_min, k_max + 1):
        centroids, assign = _gs_kmeans(colors, k=k, iters=iters)
        wcss = sum(float(((colors[assign == j] - centroids[j]) ** 2).sum())
                   for j in range(k) if (assign == j).any())
        if prev_wcss is not None:
            imp = prev_wcss - wcss
            if first_improvement is None:
                first_improvement = imp
            elif first_improvement > 0 and imp < threshold * first_improvement:
                break
        best_centroids, best_assign = centroids, assign
        prev_wcss = wcss
    return best_centroids, best_assign

def _gs_morph_clean(cells, assign):
    gx_vals = sorted(set(c[0] for c in cells))
    gy_vals = sorted(set(c[1] for c in cells))
    to_col = {gx: i for i, gx in enumerate(gx_vals)}
    to_row = {gy: i for i, gy in enumerate(gy_vals)}
    grid   = {(to_col[c[0]], to_row[c[1]]): i for i, c in enumerate(cells)}
    out    = assign.copy()
    for i, c in enumerate(cells):
        ci, ri = to_col[c[0]], to_row[c[1]]
        nbrs = [grid.get((ci-1, ri)), grid.get((ci+1, ri)),
                grid.get((ci, ri-1)), grid.get((ci, ri+1))]
        nbrs = [j for j in nbrs if j is not None]
        if len(nbrs) < 3:
            continue
        votes = {}
        for j in nbrs:
            v = assign[j]; votes[v] = votes.get(v, 0) + 1
        top, cnt = max(votes.items(), key=lambda x: x[1])
        if cnt >= 3 and top != assign[i]:
            out[i] = top
    return out

def grid_snap_pil(img):
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    h, w = arr.shape[:2]
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    col_profile = np.abs(np.diff(gray.mean(axis=0)))
    row_profile = np.abs(np.diff(gray.mean(axis=1)))
    Tx, cx = _gs_fft_period(col_profile)
    Ty, cy = _gs_fft_period(row_profile)
    if cx < SNAP_CONF and cy < SNAP_CONF:
        return None, 0.0
    T = Tx if cx >= cy else Ty
    confidence = max(cx, cy)
    px = _gs_phase(col_profile, T)
    py = _gs_phase(row_profile, T)
    margin = max(1, round(T * 0.15))
    cells = []
    gy = py
    while gy < h:
        y1 = min(gy + T, h)
        gx = px
        while gx < w:
            x1 = min(gx + T, w)
            patch = arr[gy + margin:y1 - margin, gx + margin:x1 - margin]
            if patch.size > 0:
                cells.append((gx, gy, x1, y1, patch.reshape(-1, 3).mean(axis=0)))
            gx += T
        gy += T
    if len(cells) < 9:
        return None, 0.0
    colors = np.array([c[4] for c in cells], dtype=np.float32)
    centroids, raw_assign = _gs_kmeans_adaptive(colors)
    assign = _gs_morph_clean(cells, raw_assign)
    out = np.zeros_like(arr)
    for i, (gx, gy, x1, y1, _) in enumerate(cells):
        out[gy:y1, gx:x1] = centroids[assign[i]]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)), confidence

# ── Config ────────────────────────────────────────────────────────────────────

IMAGES_DIR  = Path('images')
LABELS_FILE = Path('snap_labels.json')
PANEL_SIZE  = 380   # px per image panel
BG          = '#1a1a2e'
FG          = '#00f5ff'
GOOD_COL    = '#39ff14'
BAD_COL     = '#ff2d55'
SKIP_COL    = '#888888'

# ── App ───────────────────────────────────────────────────────────────────────

class SnapLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title('Snap Labeler')
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.labels  = {}
        self.history = []   # list of (stem, old_verdict) for undo
        self._load_labels()

        all_paths = sorted(IMAGES_DIR.rglob('*.png'))
        self.paths   = [p for p in all_paths if p.stem not in self.labels]
        self.total   = len(all_paths)
        self.labeled = len(self.labels)
        self.idx     = 0

        self._build_ui()
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

        title = tk.Label(self.root, text='SNAP LABELER', font=('Courier', 13, 'bold'),
                         bg=BG, fg=FG)
        title.pack(**pad)

        self.prog_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.prog_var,
                 font=('Courier', 10), bg=BG, fg='#888').pack()
        self.bar_canvas = tk.Canvas(self.root, width=PANEL_SIZE*2+20, height=6,
                                    bg='#2a2a3e', highlightthickness=0)
        self.bar_canvas.pack(pady=(0, 8))
        self.bar_fill = self.bar_canvas.create_rectangle(0, 0, 0, 6, fill=FG, width=0)

        img_frame = tk.Frame(self.root, bg=BG)
        img_frame.pack(padx=10)

        self.lbl_orig = tk.Label(img_frame, bg='#0a0a0f', relief='flat')
        self.lbl_orig.grid(row=0, column=0, padx=5)

        self.lbl_snap = tk.Label(img_frame, bg='#0a0a0f', relief='flat')
        self.lbl_snap.grid(row=0, column=1, padx=5)

        self.cap_orig = tk.Label(img_frame, text='Original',
                                 font=('Courier', 9), bg=BG, fg='#666')
        self.cap_orig.grid(row=1, column=0)

        self.cap_snap = tk.Label(img_frame, text='Snapped',
                                 font=('Courier', 9), bg=BG, fg='#666')
        self.cap_snap.grid(row=1, column=1)

        self.id_var = tk.StringVar()
        tk.Label(self.root, textvariable=self.id_var,
                 font=('Courier', 11, 'bold'), bg=BG, fg=FG).pack(pady=(10, 4))

        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=6)
        btn_cfg = dict(font=('Courier', 10, 'bold'), width=10,
                       relief='flat', cursor='hand2', pady=6)

        tk.Button(btn_frame, text='✓ GOOD  [y]', bg=GOOD_COL, fg='#000',
                  command=self._good, **btn_cfg).grid(row=0, column=0, padx=6)
        tk.Button(btn_frame, text='✗ BAD   [n]', bg=BAD_COL, fg='#fff',
                  command=self._bad, **btn_cfg).grid(row=0, column=1, padx=6)
        tk.Button(btn_frame, text='→ SKIP  [s]', bg='#2a2a3e', fg='#aaa',
                  command=self._skip, **btn_cfg).grid(row=0, column=2, padx=6)
        tk.Button(btn_frame, text='← BACK  [b]', bg='#2a2a3e', fg='#aaa',
                  command=self._back, **btn_cfg).grid(row=0, column=3, padx=6)

        tk.Label(self.root, text='q / Escape to quit',
                 font=('Courier', 8), bg=BG, fg='#444').pack(pady=(4, 10))

        self.root.bind('y', lambda _: self._good())
        self.root.bind('<Return>', lambda _: self._good())
        self.root.bind('n', lambda _: self._bad())
        self.root.bind('s', lambda _: self._skip())
        self.root.bind('b', lambda _: self._back())
        self.root.bind('q', lambda _: self.root.destroy())
        self.root.bind('<Escape>', lambda _: self.root.destroy())

    def _update_progress(self):
        total_labeled = self.labeled
        pct = total_labeled / self.total if self.total else 0
        remaining = self.total - total_labeled
        self.prog_var.set(
            f'{total_labeled} / {self.total} labeled  •  {remaining} remaining  •  '
            f'{round(pct*100)}%'
        )
        w = int((PANEL_SIZE * 2 + 20) * pct)
        self.bar_canvas.coords(self.bar_fill, 0, 0, w, 6)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _fit(self, img, size=PANEL_SIZE):
        img = img.copy()
        img.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new('RGB', (size, size), (10, 10, 15))
        x = (size - img.width) // 2
        y = (size - img.height) // 2
        canvas.paste(img, (x, y))
        return ImageTk.PhotoImage(canvas)

    def _placeholder(self, text):
        canvas = Image.new('RGB', (PANEL_SIZE, PANEL_SIZE), (10, 10, 15))
        d = ImageDraw.Draw(canvas)
        d.multiline_text((PANEL_SIZE//2, PANEL_SIZE//2), text,
                         fill=(80, 80, 100), anchor='mm', align='center')
        return ImageTk.PhotoImage(canvas)

    def _load_current(self):
        if self.idx >= len(self.paths):
            self._show_done()
            return

        path = self.paths[self.idx]
        self.id_var.set(path.stem)
        self._update_progress()

        try:
            orig = Image.open(path).convert('RGB')
        except Exception:
            orig = Image.new('RGB', (224, 224), (30, 30, 40))

        snapped, conf = grid_snap_pil(orig)

        self._tk_orig = self._fit(orig)
        self.lbl_orig.configure(image=self._tk_orig)
        self.cap_orig.configure(text=f'Original  {orig.width}×{orig.height}px')

        if snapped is not None:
            self._tk_snap = self._fit(snapped)
            self.lbl_snap.configure(image=self._tk_snap)
            self.cap_snap.configure(
                text=f'conf={conf:.2f}',
                fg=GOOD_COL if conf >= 0.6 else '#ffd60a'
            )
        else:
            self._tk_snap = self._placeholder('snap failed')
            self.lbl_snap.configure(image=self._tk_snap)
            self.cap_snap.configure(text='snap failed', fg=BAD_COL)

        self._current_conf = conf

    def _show_done(self):
        self.id_var.set('All done!')
        self._update_progress()
        blank = self._placeholder('✓ all images labeled')
        self._tk_orig = self._tk_snap = blank
        self.lbl_orig.configure(image=blank)
        self.lbl_snap.configure(image=blank)
        self.cap_orig.configure(text='')
        self.cap_snap.configure(text='')

    # ── Actions ───────────────────────────────────────────────────────────────

    def _label(self, verdict):
        if self.idx >= len(self.paths):
            return
        stem = self.paths[self.idx].stem
        self.history.append((stem, None))
        self.labels[stem] = {'verdict': verdict, 'confidence': self._current_conf}
        self._save_labels()
        self.labeled += 1
        self.idx += 1
        self._load_current()

    def _good(self): self._label('good')
    def _bad(self):  self._label('bad')
    def _skip(self): self._label('skip')

    def _back(self):
        if not self.history:
            return
        stem, _ = self.history.pop()
        if stem in self.labels:
            del self.labels[stem]
            self._save_labels()
            self.labeled -= 1
        path = next((p for p in sorted(IMAGES_DIR.rglob('*.png')) if p.stem == stem), None)
        if path:
            self.paths.insert(self.idx, path)
        self.idx = max(0, self.idx - 1)
        self._load_current()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = tk.Tk()
    app  = SnapLabeler(root)
    root.mainloop()
    print(f'\nSaved {len(app.labels)} labels → {LABELS_FILE}')
