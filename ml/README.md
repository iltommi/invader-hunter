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

### 6. Live flash feed (optional, extra training signal)

```bash
python3 download_flashes.py            # poll the FlashInvaders live feed → flashes/images/, flashes/metadata.jsonl
python3 label_flashes.py               # manually confirm each photo's real invader ID → flashes/identity_labels.json
python3 train_clip.py                  # picks up confirmed photos automatically
```

`download_flashes.py` polls the site's own public JSON feed and saves new scans. `label_flashes.py` is a Tkinter GUI: for each photo it shows the top-K nearest known invaders (via the current embeddings index) so you can click the correct match, type an ID manually, or mark unknown/skip — resumable, same pattern as `label_snaps.py`. Invaders with one or more confirmed real photos get genuine multi-photo contrastive pairs in `train_clip.py` instead of only synthetic augmentations of the single reference image.

Check accuracy on a held-out split of confirmed photos (never trained on — see `flash_split.py`) at any time:

```bash
python3 eval_recognition.py
```

Reports top-1/5/16 accuracy and MRR, appends to `eval_history.log`, and writes `eval_latest.json` (machine-readable, used by `auto_retrain.py` below).

---

### 7. Unattended retrain-if-improved (optional)

```bash
python3 auto_retrain.py --dry-run   # check whether it would trigger, no changes
python3 auto_retrain.py             # run for real
```

Safe to run on a schedule (cron/launchd): downloads new scans, and if at least `--min-new` (default 300) new confirmed labels have accumulated since the last retrain, runs `train_clip.py` → `extract_features_clip.py` → `eval_recognition.py` and **keeps the new checkpoint only if multi-crop top-1 actually improved** — otherwise it reverts `clip_finetuned.pt` / `embeddings_clip.npz` / `eval_latest.json` back to the pre-retrain versions. State lives in `retrain_state.json`.

This automates the mechanical pipeline only — identity confirmation in `label_flashes.py` stays manual. At current accuracy, auto-accepting the model's own guesses as ground truth would inject wrong labels often enough to actively hurt the model, since mislabeled pairs teach contrastive training to merge genuinely different invaders.

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
| `download_flashes.py` | Poll the live FlashInvaders feed for new scan photos |
| `label_flashes.py` | GUI to confirm each flash photo's real invader ID |
| `flash_split.py` | Deterministic train/eval split for confirmed flash photos |
| `eval_recognition.py` | Accuracy on the held-out eval split |
| `auto_retrain.py` | Unattended retrain-if-improved orchestration |
| `clip_finetuned.pt` | Fine-tuned model weights |
| `embeddings_clip.npz` | CLIP embeddings index |
| `embeddings.npz` | EfficientNet embeddings index |
| `images/` | Reference images, one per invader |
| `test.jpg` | Test query (PA_1251) |




cd ml

# 1. (optional) pull in more freshly-scanned photos
python3 download_flashes.py

# 2. label them, prioritizing invaders with zero confirmed photos first
python3 label_flashes.py --breadth

# 3. once you've built up a meaningful new batch of confirmed photos, retrain
python3 train_clip.py

# 4. rebuild the search index from the new checkpoint (always do this right after training)
python3 extract_features_clip.py

# 5. check whether it actually helped, on the held-out set
python3 eval_recognition.py

Then repeat from step 1/2 — download more, label more (breadth first), retrain, rebuild, eval. Each eval_recognition.py run appends to eval_history.log, so you can watch top-1/top-5/MRR move over time and confirm each retrain is actually helping before committing to it.

One rule of thumb from what we found: don't retrain after every single labeling session — batch it up. Going from ~200 to ~2,489 confirmed photos is what took multi-crop top-1 from 66.7% to 75.6%; a retrain after only a few dozen new labels wouldn't move the needle much and isn't worth the ~2 hour training + rebuild time.
