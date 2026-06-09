# Invader Hunter — ML Pipeline

CLIP-based image similarity search for identifying Space Invader artworks from photos.

## Large files (not tracked in git)

The following are gitignored and must be regenerated locally:

| File(s) | How to regenerate |
|---|---|
| `images/`, `clip_finetuned.pt` | Steps 1–2 below (`download_images.py` → `train_clip.py`) |
| `embeddings.npz`, `embeddings_clip.npz` | Step 3 below (`extract_features_clip.py`) |
| `../docs/embeddings.bin`, `../docs/ids.json`, `../docs/thumbs.bin`, `../docs/thumbs_idx.bin`, `../docs/model.part*` | Step 5 below (`export_for_web.py`) — commit the outputs afterwards |

## Setup

```bash
pip install torch torchvision transformers pillow numpy onnxruntime
```

## Pipeline

### 1. Download reference images

```bash
python3 download_images.py
```

Populates `images/<CITY>/<CITY_ID>.png` from the Invader Spotter database.

---

### 2. Fine-tune CLIP

```bash
python3 train_clip.py
```

Fine-tunes the CLIP ViT-B/32 visual encoder using multi-view contrastive learning. Each image produces 3 views per step: two independently augmented raw photos and one grid-snapped pixel-art reconstruction. The model learns to pull all three views of the same invader together and push different invaders apart, bridging raw photos and clean pixel art.

- Trains for 10 epochs, ~48 images per batch
- Freezes early layers; trains last 4 transformer blocks + projection head
- Output: `clip_finetuned.pt`

---

### 3. Build the embeddings index

```bash
python3 extract_features_clip.py
```

Embeds all reference images using the fine-tuned CLIP encoder with mixed TTA: 8 raw augmented views (rotation, brightness/contrast jitter, blur, crop, aspect ratio distortion) plus 4 grid-snapped views for images where the tile grid is detectable. Views are averaged per image to produce the final embedding.

- Output: `embeddings_clip.npz` — arrays `ids` and `embeddings` [N × 512]
- Automatically loads `clip_finetuned.pt` if present

---

### 4. Search from the command line

```bash
python3 search_clip.py path/to/photo.jpg
python3 search_clip.py path/to/photo.jpg --top 10
python3 search_clip.py path/to/photo.jpg --city PA
```

Embeds the query image and returns the top matches by cosine similarity. Opens the top result image with `open`. Automatically loads `clip_finetuned.pt` if present.

---

### 5. Export for browser (ONNX)

```bash
python3 export_for_web.py
```

Exports the fine-tuned visual encoder to ONNX and the embeddings index to browser-friendly binary files.

Outputs (written to the project root, next to `index.html`):

| File | Size | Notes |
|------|------|-------|
| `clip_visual.onnx` | ~350 MB | fp32 intermediate, not committed |
| `clip_visual_q.onnx` | ~87 MB | uint8 quantized intermediate, not committed |
| `clip_visual_int8.onnx` | ~22 MB | int8 quantized intermediate, not committed |
| `model.part0` | ~47 MB | commit to git |
| `model.part1` | ~42 MB | commit to git |
| `embeddings.bin` | ~9 MB | commit to git |
| `ids.json` | ~60 KB | commit to git |

The quantized model requires `onnxruntime`:
```bash
pip install onnxruntime
```

After export, commit the outputs and `index.html` will load the model automatically on page open.

---

## Files

| File | Description |
|------|-------------|
| `train_clip.py` | Fine-tune CLIP with SimCLR |
| `extract_features_clip.py` | Build CLIP embeddings index |
| `search_clip.py` | CLI search against the index |
| `export_for_web.py` | Export ONNX model + embeddings for browser |
| `extract_features.py` | EfficientNet-based index (baseline) |
| `search.py` | CLI search using EfficientNet index |
| `download_images.py` | Download reference images |
| `clip_finetuned.pt` | Fine-tuned model weights |
| `embeddings_clip.npz` | CLIP embeddings index |
| `embeddings.npz` | EfficientNet embeddings index |
| `images/` | Reference images, one per invader |
| `test.jpg` | Test query (PA_1251) |
