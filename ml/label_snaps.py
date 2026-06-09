#!/usr/bin/env python3
"""
Snap labeling tool — evaluate grid snap quality on reference images.

Shows original photo alongside grid-snapped reconstruction.
Labels are saved incrementally; restart resumes where you left off.
Use the T and k sliders + Force checkboxes to manually tune the snap
when auto-detection fails.  Forced parameters are saved in the label
so train_clip.py can replay the exact same snap.

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

def _gs_wavelet_period(profile, min_p=2.0, max_p=None):
    """Period detection via Morlet CWT power spectrum (pure numpy).

    Evaluates CWT power at a dense grid of periods using Parseval's theorem:
    P(T) = Σ_f |X(f)|² · exp(−(ω₀·T·f − ω₀)²)
    i.e. the signal's spectral power weighted by a Gaussian centred at f=1/T.
    This gives sub-pixel accuracy and handles small images with few tiles
    better than FFT because the scale axis is continuous.
    """
    n = len(profile)
    if max_p is None:
        max_p = n / 3.0
    if max_p < min_p or n < min_p * 3:
        return 0, 0.0

    w0   = 6.0   # Morlet centre frequency (higher = sharper in freq, wider in time)
    step = 0.1
    periods = np.arange(min_p, max_p + step, step)
    if len(periods) < 3:
        return 0, 0.0

    p    = profile - profile.mean()
    X    = np.fft.rfft(p)
    Xpow = np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n)   # cycles per sample

    # For each candidate period T, the Morlet at scale s = w0·T/(2π)
    # has |Ψ̂_s(f)|² ∝ exp(−(2π·f·s − w0)²) = exp(−w0²·(T·f − 1)²)
    power = np.array([
        float((Xpow * np.exp(-(w0 * (freqs * T - 1.0)) ** 2)).sum())
        for T in periods
    ])

    best_idx = int(power.argmax())

    # Sub-pixel refinement via parabolic interpolation on the power peak
    if 0 < best_idx < len(periods) - 1:
        p0, p1, p2 = power[best_idx - 1], power[best_idx], power[best_idx + 1]
        denom = 2 * p1 - p2 - p0
        delta = (0.5 * (p2 - p0) / denom) if abs(denom) > 1e-12 else 0.0
        best_period = float(periods[best_idx] + delta * step)
    else:
        best_period = float(periods[best_idx])

    # Confidence: how much the peak stands above the median background
    bg   = float(np.median(power))
    peak = float(power[best_idx])
    conf = max(0.0, min(1.0, (peak / (bg + 1e-9) - 1.0) / 2.0))

    return best_period, conf

def _gs_phase(profile, period):
    T = max(1.0, float(period))
    T_int = max(1, round(T))
    n = len(profile)
    best_phase, best_score = 0, np.inf
    for p in range(T_int):
        score, i = 0.0, 0
        while True:
            s = p + round(i * T)
            if s >= n:
                break
            e = p + round((i + 1) * T)
            block = profile[s:min(e, n)]
            if len(block) > 1:
                score += float(block.var()) * len(block)
            i += 1
        if score < best_score:
            best_score = score
            best_phase = p
    return best_phase

def grid_snap_pil(img, force_T=None):
    """Returns (snapped_img_or_None, confidence, T_used)."""
    arr = np.array(img.convert('RGB'), dtype=np.float32)
    h, w = arr.shape[:2]
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    col_profile = np.abs(np.diff(gray.mean(axis=0)))
    row_profile = np.abs(np.diff(gray.mean(axis=1)))

    if force_T is not None:
        T = force_T
        confidence = 1.0
    else:
        Tx, cx = _gs_wavelet_period(col_profile)
        Ty, cy = _gs_wavelet_period(row_profile)
        if cx < SNAP_CONF and cy < SNAP_CONF:
            return None, 0.0, 0
        T = Tx if cx >= cy else Ty
        confidence = max(cx, cy)

    if T < 4 and force_T is None:
        return None, 0.0, T

    px = _gs_phase(col_profile, T)
    py = _gs_phase(row_profile, T)
    margin = min(max(0, round(T * 0.15)), max(0, int(T) // 2 - 1))

    out = np.zeros_like(arr)
    n_cells = 0
    yi = 0
    while True:
        gy = py + round(yi * T)
        if gy >= h:
            break
        y1 = min(py + round((yi + 1) * T), h)
        xi = 0
        while True:
            gx = px + round(xi * T)
            if gx >= w:
                break
            x1 = min(px + round((xi + 1) * T), w)
            patch = arr[gy + margin:y1 - margin, gx + margin:x1 - margin]
            if patch.size > 0:
                out[gy:y1, gx:x1] = patch.reshape(-1, 3).mean(axis=0)
                n_cells += 1
            xi += 1
        yi += 1

    if n_cells < 9:
        return None, 0.0, T
    return Image.fromarray(out.clip(0, 255).astype(np.uint8)), confidence, T

# ── Config ────────────────────────────────────────────────────────────────────

IMAGES_DIR  = Path('images')
LABELS_FILE = Path('snap_labels.json')
PANEL_SIZE  = 380
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
        self.history = []
        self._load_labels()

        all_paths = sorted(IMAGES_DIR.rglob('*.png'))
        self.paths   = [p for p in all_paths if p.stem not in self.labels]
        self.total   = len(all_paths)
        self.labeled = len(self.labels)
        self.idx     = 0

        self._loading    = False   # suppress resnap during image load
        self._resnap_job = None    # pending after() id
        self._current_orig = None  # PIL image currently shown

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

        tk.Label(self.root, text='SNAP LABELER', font=('Courier', 13, 'bold'),
                 bg=BG, fg=FG).pack(**pad)

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

        # ── Parameter sliders ─────────────────────────────────────────────────
        param_frame = tk.Frame(self.root, bg=BG)
        param_frame.pack(padx=10, pady=4)

        slider_cfg = dict(orient=tk.HORIZONTAL, length=220, bg=BG, fg=FG,
                          troughcolor='#2a2a3e', highlightthickness=0,
                          font=('Courier', 9), relief='flat', bd=0,
                          activebackground=FG, showvalue=False)
        lbl_cfg  = dict(font=('Courier', 9), bg=BG, fg='#aaa', width=3, anchor='e')
        val_cfg  = dict(font=('Courier', 9, 'bold'), bg=BG, fg=FG, width=5, anchor='w')
        chk_cfg  = dict(font=('Courier', 9), bg=BG, fg='#888', activebackground=BG,
                        activeforeground=FG, selectcolor='#0a0a0f',
                        relief='flat', bd=0, cursor='hand2')

        # T row
        tk.Label(param_frame, text='T:', **lbl_cfg).grid(row=0, column=0, padx=(0,4))
        self.t_var = tk.DoubleVar(value=8.0)
        tk.Scale(param_frame, from_=2.0, to=40.0, resolution=0.2, variable=self.t_var,
                 command=self._on_slider, **slider_cfg).grid(row=0, column=1)
        self.t_val_lbl = tk.Label(param_frame, text='8.0 px', **val_cfg)
        self.t_val_lbl.grid(row=0, column=2, padx=(6, 0))
        self.force_t = tk.BooleanVar(value=False)
        tk.Checkbutton(param_frame, text='force', variable=self.force_t,
                       command=self._on_force_toggle, **chk_cfg).grid(row=0, column=3, padx=(8,0))

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=8)
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

        tk.Label(self.root,
                 text='y/n/s/b/q  ·  ↑↓ tile size',
                 font=('Courier', 8), bg=BG, fg='#444').pack(pady=(2, 10))

        self.root.bind('y', lambda _: self._good())
        self.root.bind('<Return>', lambda _: self._good())
        self.root.bind('n', lambda _: self._bad())
        self.root.bind('s', lambda _: self._skip())
        self.root.bind('b', lambda _: self._back())
        self.root.bind('q', lambda _: self.root.destroy())
        self.root.bind('<Escape>', lambda _: self.root.destroy())
        self.root.bind('<Up>',    lambda _: self._nudge_T(+0.2))
        self.root.bind('<Down>',  lambda _: self._nudge_T(-0.2))

    def _update_progress(self):
        pct = self.labeled / self.total if self.total else 0
        self.prog_var.set(
            f'{self.labeled} / {self.total} labeled  •  '
            f'{self.total - self.labeled} remaining  •  {round(pct*100)}%'
        )
        w = int((PANEL_SIZE * 2 + 20) * pct)
        self.bar_canvas.coords(self.bar_fill, 0, 0, w, 6)

    # ── Slider / force callbacks ──────────────────────────────────────────────

    def _on_slider(self, _=None):
        self.t_val_lbl.configure(text=f'{self.t_var.get():.1f} px')
        if self._loading:
            return
        if self._resnap_job:
            self.root.after_cancel(self._resnap_job)
        self._resnap_job = self.root.after(250, self._resnap)

    def _on_force_toggle(self):
        if self._loading:
            return
        self._resnap()

    def _nudge_T(self, delta):
        self.force_t.set(True)
        self.t_var.set(round(max(2.0, min(40.0, self.t_var.get() + delta)) * 5) / 5)
        self._on_slider()

    def _resnap(self):
        self._resnap_job = None
        if self._current_orig is None or self.idx >= len(self.paths):
            return
        force_T = self.t_var.get() if self.force_t.get() else None
        snapped, conf, T = grid_snap_pil(self._current_orig, force_T=force_T)
        self._update_snap_panel(snapped, conf, T)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _fit(self, img, size=PANEL_SIZE):
        img = img.copy()
        img.thumbnail((size, size), Image.LANCZOS)
        canvas = Image.new('RGB', (size, size), (10, 10, 15))
        canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
        return ImageTk.PhotoImage(canvas)

    def _placeholder(self, text):
        canvas = Image.new('RGB', (PANEL_SIZE, PANEL_SIZE), (10, 10, 15))
        d = ImageDraw.Draw(canvas)
        d.multiline_text((PANEL_SIZE//2, PANEL_SIZE//2), text,
                         fill=(80, 80, 100), anchor='mm', align='center')
        return ImageTk.PhotoImage(canvas)

    def _update_snap_panel(self, snapped, conf, T):
        if snapped is not None:
            self._tk_snap = self._fit(snapped)
            self.lbl_snap.configure(image=self._tk_snap)
            tag = ' [F]' if self.force_t.get() else ''
            self.cap_snap.configure(
                text=f'T={T:.1f}px{tag}  conf={conf:.2f}',
                fg=GOOD_COL if conf >= 0.6 else '#ffd60a'
            )
        else:
            self._tk_snap = self._placeholder('snap failed')
            self.lbl_snap.configure(image=self._tk_snap)
            self.cap_snap.configure(text='snap failed', fg=BAD_COL)
        self._current_conf = conf
        self._current_T    = T

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

        self._current_orig = orig

        # Run auto-snap first to seed slider defaults
        snapped, conf, T = grid_snap_pil(orig)

        self._loading = True
        self.force_t.set(False)
        self.t_var.set(float(T) if T > 0 else 8.0)
        self.t_val_lbl.configure(text=f'{self.t_var.get():.1f} px')
        self._loading = False

        self._tk_orig = self._fit(orig)
        self.lbl_orig.configure(image=self._tk_orig)
        self.cap_orig.configure(text=f'Original  {orig.width}×{orig.height}px')

        self._update_snap_panel(snapped, conf, T)

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
        label = {
            'verdict':    verdict,
            'confidence': self._current_conf,
            'T':          self._current_T,
            'forced_T':   bool(self.force_t.get()),
        }
        self.history.append((stem, label))
        self.labels[stem] = label
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
        stem, saved = self.history.pop()
        if stem in self.labels:
            del self.labels[stem]
            self._save_labels()
            self.labeled -= 1
        path = next((p for p in sorted(IMAGES_DIR.rglob('*.png')) if p.stem == stem), None)
        if path:
            self.paths.insert(self.idx, path)
        self.idx = max(0, self.idx - 1)
        self._load_current()
        if saved and saved.get('forced_T'):
            self._loading = True
            self.force_t.set(True)
            self.t_var.set(float(saved['T']))
            self.t_val_lbl.configure(text=f"{saved['T']:.1f} px")
            self._loading = False
            self._resnap()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    root = tk.Tk()
    app  = SnapLabeler(root)
    root.mainloop()
    print(f'\nSaved {len(app.labels)} labels → {LABELS_FILE}')
