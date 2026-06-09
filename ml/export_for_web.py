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

Run after training:
  python3 export_for_web.py
"""

import json
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

print('Done.')
