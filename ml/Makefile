PYTHON   = python3
IMAGES   = images
LABELS   = snap_labels.json
WEIGHTS  = clip_finetuned.pt
EMB_NPZ  = embeddings_clip.npz
WEB_BIN  = ../docs/embeddings.bin

.PHONY: all train extract export download label clean

# Full pipeline: train → extract → export
all: export

download:
	$(PYTHON) download_images.py

label:
	$(PYTHON) label_snaps.py

$(WEIGHTS): $(IMAGES) $(LABELS)
	$(PYTHON) train_clip.py

train: $(WEIGHTS)

$(EMB_NPZ): $(WEIGHTS) $(IMAGES) $(LABELS)
	$(PYTHON) extract_features_clip.py

extract: $(EMB_NPZ)

$(WEB_BIN): $(EMB_NPZ)
	$(PYTHON) export_for_web.py

export: $(WEB_BIN)

clean:
	rm -f $(WEIGHTS) $(EMB_NPZ) \
	      ../docs/embeddings.bin ../docs/ids.json \
	      ../docs/model.part0 ../docs/model.part1 \
	      ../docs/thumbs.bin ../docs/thumbs_idx.bin
