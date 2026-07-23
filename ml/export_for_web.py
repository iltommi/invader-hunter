#!/usr/bin/env python3
"""
Export fine-tuned CLIP visual encoder to ONNX for in-browser inference,
and convert the embeddings index to browser-friendly binary format.

Outputs (in project root, next to index.html):
  clip_visual.onnx      — fp32 ONNX model (~350 MB, intermediate)
  clip_visual_q.onnx    — uint8 quantized (~87 MB, intermediate)
  clip_visual_int8.onnx — int8 quantized (~22 MB, intermediate, preferred)
  model.part0          — first chunk (~47 MB, commit to git)
  model.part1          — second chunk (~42 MB, commit to git)
  embeddings.bin       — Float32Array [N × 512], row-major
  ids.json             — ["PA_1251", ...]

Also bumps docs/sw.js's cache version and docs/index.html's About build date,
so the deploy actually reaches users (docs/sw.js caches model/embeddings
cache-first for offline use, so without this a new export silently never
shows up for anyone with a warm cache, no matter how many times they reload).
Skipped if the working tree already has an uncommitted bump (e.g. you also
ran update_pois.py first in this same session) -- one bump covers everything
not yet pushed, so bumping again here would double-count.

Run after training:
  python3 export_for_web.py
"""

import json
import re
import subprocess
import numpy as np
from pathlib import Path
import torch
from transformers import CLIPModel

FINETUNED_FILE = Path('clip_finetuned.pt')
EMB_FILE       = Path('embeddings_clip.npz')
ROOT_DIR       = Path('..')         # project root (intermediary ONNX files)
DOCS_DIR       = Path('../docs')    # served by GitHub Pages

ONNX_FP32  = ROOT_DIR / 'clip_visual.onnx'
ONNX_Q     = ROOT_DIR / 'clip_visual_q.onnx'
ONNX_INT8  = ROOT_DIR / 'clip_visual_int8.onnx'

# ── Wrapper: visual encoder + L2 normalisation ────────────────────────────────

class CLIPVisual(torch.nn.Module):
    def __init__(self, clip):
        super().__init__()
        self.vision_model      = clip.vision_model
        self.visual_projection = clip.visual_projection

    def forward(self, pixel_values):
        out  = self.vision_model(pixel_values=pixel_values)
        feat = self.visual_projection(out.pooler_output)
        return feat / feat.norm(dim=-1, keepdim=True)

# ── Load model ────────────────────────────────────────────────────────────────

print('Loading CLIP...')
clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').eval()
if FINETUNED_FILE.exists():
    ckpt = torch.load(FINETUNED_FILE, map_location='cpu')
    clip.vision_model.load_state_dict(ckpt['vision_model'])
    clip.visual_projection.load_state_dict(ckpt['visual_projection'])
    print('  loaded fine-tuned weights')
else:
    print('  no fine-tuned weights found, using base CLIP')

visual = CLIPVisual(clip).eval()
dummy  = torch.zeros(1, 3, 224, 224)

# ── Export ONNX fp32 ──────────────────────────────────────────────────────────

print(f'Exporting {ONNX_FP32} ...')
torch.onnx.export(
    visual, dummy, str(ONNX_FP32),
    input_names=['pixel_values'],
    output_names=['image_embeds'],
    dynamic_axes={'pixel_values': {0: 'batch'}, 'image_embeds': {0: 'batch'}},
    opset_version=14,
    dynamo=False,
)
print(f'  {ONNX_FP32.stat().st_size/1e6:.0f} MB')

# ── Quantize: uint8 (legacy) then int8 (smaller, faster on mobile) ───────────

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f'Quantizing uint8 → {ONNX_Q} ...')
    quantize_dynamic(str(ONNX_FP32), str(ONNX_Q), weight_type=QuantType.QUInt8)
    print(f'  {ONNX_Q.stat().st_size/1e6:.0f} MB')

    print(f'Quantizing int8  → {ONNX_INT8} ...')
    quantize_dynamic(str(ONNX_FP32), str(ONNX_INT8), weight_type=QuantType.QInt8)
    print(f'  {ONNX_INT8.stat().st_size/1e6:.0f} MB')

except ImportError:
    print('onnxruntime not installed — skipping quantization')
    print('  pip install onnxruntime   then re-run to get the smaller quantized model')

# ── Split quantized model into chunks for GitHub Pages ────────────────────────

CHUNK = 47 * 1024 * 1024   # 47 MB — safely under GitHub's 50 MB soft limit

if ONNX_INT8.exists():
    src = ONNX_INT8
elif ONNX_Q.exists():
    src = ONNX_Q
elif ONNX_FP32.exists():
    src = ONNX_FP32
    print('Warning: using fp32 model for splitting (no quantized model available)')
else:
    src = None

if src:
    print(f'Splitting {src.name} into chunks → docs/ ...')
    DOCS_DIR.mkdir(exist_ok=True)
    data_bytes = src.read_bytes()
    for i, offset in enumerate(range(0, len(data_bytes), CHUNK)):
        out = DOCS_DIR / f'model.part{i}'
        out.write_bytes(data_bytes[offset:offset + CHUNK])
        print(f'  → {out}  ({out.stat().st_size/1e6:.1f} MB)')

# ── Export embeddings ─────────────────────────────────────────────────────────

print('Exporting embeddings → docs/ ...')
DOCS_DIR.mkdir(exist_ok=True)
data = np.load(EMB_FILE)
ids  = data['ids'].tolist()
emb  = data['embeddings'].astype(np.float32)

emb.tofile(str(DOCS_DIR / 'embeddings.bin'))
(DOCS_DIR / 'ids.json').write_text(json.dumps(ids))

# ── Export city padding ────────────────────────────────────────────────────────
# For linking to the original invader-spotter.art close-up photo:
# https://www.invader-spotter.art/grosplan/{code}/{code}_{padded_num}-grosplan.png
# The zero-padding width is per-city (the digit count of that city's highest
# ID number) and isn't derivable from a single ID string alone -- e.g. PA_5
# needs "PA_0005" since Paris goes up into 4 digits. Computed from the full
# spotter_pois.json catalog (not just `ids`, the embeddings index subset --
# that could under-count a city's max if it doesn't yet include that city's
# highest-numbered invader, producing too little padding and a 404).

print('Computing city padding lookup...')
SPOTTER_FILE = ROOT_DIR / 'spotter_pois.json'
city_padding = {}
if SPOTTER_FILE.exists():
    for p in json.loads(SPOTTER_FILE.read_text()):
        code, num = p['id'].split('_', 1)
        if num.isdigit():
            city_padding[code] = max(city_padding.get(code, 0), len(num))
    (DOCS_DIR / 'city_padding.json').write_text(json.dumps(city_padding))
    print(f'  {len(city_padding)} cities → city_padding.json')
else:
    print(f'  warn: {SPOTTER_FILE} not found, skipping (view-original links will use unpadded IDs)')
print(f'  {len(ids)} vectors → embeddings.bin ({emb.nbytes/1e6:.1f} MB) + ids.json')

# ── Export thumbnails ─────────────────────────────────────────────────────────
# One 80×80 JPEG per embedding, packed into thumbs.bin.
# thumbs_idx.bin holds Uint32 pairs [offset, length] — one per entry in ids.json.
# A zero-length entry means no image was available for that POI.

from PIL import Image
import io
import array as arr_mod

THUMB_PX  = 80
IMAGE_DIR = Path('images')

print('Exporting thumbnails → docs/ ...')
offsets_flat = []
blobs = []
total_bytes = 0

for poi_id in ids:
    code     = poi_id.split('_')[0]
    img_path = IMAGE_DIR / code / f'{poi_id}.png'
    blob = b''
    if img_path.exists():
        try:
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            s = min(w, h)
            img = img.crop(((w - s) // 2, (h - s) // 2,
                            (w - s) // 2 + s, (h - s) // 2 + s))
            img = img.resize((THUMB_PX, THUMB_PX), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, 'JPEG', quality=65)
            blob = buf.getvalue()
        except Exception as exc:
            print(f'  warn: {poi_id}: {exc}')
    offsets_flat.extend([total_bytes, len(blob)])
    blobs.append(blob)
    total_bytes += len(blob)

thumb_bin = b''.join(blobs)
(DOCS_DIR / 'thumbs.bin').write_bytes(thumb_bin)
thumb_idx = arr_mod.array('I', offsets_flat)
(DOCS_DIR / 'thumbs_idx.bin').write_bytes(thumb_idx.tobytes())
print(f'  {len(ids)} thumbnails → thumbs.bin ({len(thumb_bin)/1e6:.1f} MB) + thumbs_idx.bin')

# ── Bump deploy version markers ────────────────────────────────────────────────
# docs/sw.js caches model/embeddings/thumbnails cache-first (so the PWA works
# offline), so without bumping this, users with a warm cache never see a new
# export no matter how many times they reload. The About build date is a
# separate, purely cosmetic marker with its own long-standing convention
# (bare date for the day's first build, then a suffix letter b, c, ... for
# same-day re-runs) -- bumped here too so it stays a reliable "did this
# actually deploy" signal instead of silently going stale.
#
# Both are skipped if the working tree is already ahead of the last commit
# (update_pois.py may have already bumped them in this same session) -- one
# bump covers everything not yet pushed, so bumping again would double-count.

from datetime import date

SW_FILE   = ROOT_DIR / 'docs' / 'sw.js'
HTML_FILE = ROOT_DIR / 'docs' / 'index.html'

def _committed_sw_version():
    """CACHE version number from the last commit, or None if it can't be
    determined (no git, no prior commit, etc.)."""
    try:
        text = subprocess.run(
            ['git', 'show', 'HEAD:docs/sw.js'],
            capture_output=True, text=True, check=True, cwd=str(ROOT_DIR),
        ).stdout
    except Exception:
        return None
    m = re.search(r'invader-hunter-v(\d+)', text)
    return int(m.group(1)) if m else None

def _sw_already_bumped():
    committed = _committed_sw_version()
    if committed is None:
        return False  # can't tell -- default to allowing the bump
    m = re.search(r'invader-hunter-v(\d+)', SW_FILE.read_text())
    current = int(m.group(1)) if m else None
    return current is not None and current != committed

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

print('Bumping deploy version markers...')
if _sw_already_bumped():
    print('  (SW cache already bumped since last commit -- one bump covers this too, skipping)')
else:
    bump_sw_cache_version()
    bump_about_build_date()

print('Done.')
