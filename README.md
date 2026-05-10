# RxRead — Doctor Handwriting Recognition

A deep-learning pipeline that reads messy handwritten text from prescription images. Built with a **ResNet-CRNN + CTC** model trained on the [GNHK](https://doi.org/10.1109/ICDAR.2021.00060) wild-handwriting dataset (with optional IAM augmentation), with beam search decoding, test-time augmentation, and character language model rescoring. Served through a Flask web interface that highlights low-confidence words.

---

## Change History Since Last Push

Current branch is aligned with `origin/master` (no new local commits ahead of remote), but there are local working-tree updates not yet pushed.

### Local architecture and code changes

- Migrated from flat files to a layered package structure:
        - Added `config.py`
        - Added `core/`, `pipeline/`, `services/`, `web/`
        - Replaced old top-level modules (`model.py`, `dataset.py`, `train.py`, `predict.py`, `app.py`, `generate_synthetic.py`)
- Updated CLI orchestration in `main.py` for the new package layout.
- Updated dependencies in `requirements.txt`.
- Updated the analysis notebook in `notebooks/training_curves.ipynb` for new imports/paths.

### Local assets and UI changes

- Updated generated training plots in `static/plots/`.
- Moved web assets into `web/` and removed legacy top-level `templates/` and `static/` usage in app code.

### Notes

- These changes are in the working tree and need to be committed and pushed to appear in remote history.

---

## Method Evolution (Design Iterations)

This section summarizes the major architectural and training decisions made during development.

| Phase | Previous approach | Current approach | Rationale |
|------|------------------|----------------|----------|
| 1. Visual encoder | Custom CNN | ResNet-18 backbone | Better features + ImageNet transfer learning |
| 2. Column pooling | AdaptiveAvgPool over height | Mean+max concat over height | Preserves peak-stroke information that average smears out |
| 3. Sequence head | 3-layer BiLSTM @ 256 hidden | 2-layer BiLSTM @ 320 hidden + LayerNorm | Faster, less overfitting, better stabilised input distribution |
| 4. Decoding | Greedy CTC decoding | Beam search + char bigram LM | Reduces implausible sequences; LM weight 0.4 to avoid over-correction |
| 5. Data sources | GNHK only | GNHK + IAM (5:4 weighted) | IAM regularises; previous 7:3 underused its diversity |
| 6. Optimization | Standard Adam | AdamW + differential LR + OneCycleLR | Stable convergence with backbone fine-tuning |
| 7. Training efficiency | CPU-heavy pipeline | GPU augmentation + AMP + tensor caching | ~4× speedup |
| 8. Code structure | Monolithic scripts | Layered packages (core / pipeline / services / web) | Clear separation of concerns |
| 9. Output UX | Plain text string | Per-word `{text, confidence, bbox}` with colour-coded UI | Surfaces uncertainty so users know when to double-check |

Stack: **ResNet backbone + gated conv + mean+max pool → BiLSTM → CTC**.

---

## Architecture

### ResNet-CRNN + CTC Pipeline

```
Input Image (32×320, grayscale)
        │
        ▼
┌──────────────────────────┐
│   ResNet-18 Backbone     │   1×1 input adapter (1ch → 3ch),
│   (relaxed strides)      │   stride tweaks in layer3/4 preserve width
└──────────┬───────────────┘
           │  (B, 512, H, W)
           ▼
┌──────────────────────────┐
│   Gated Convolution      │   conv ⊙ sigmoid(gate) — sharper feature select
└──────────┬───────────────┘
           │  (B, 512, H, W)
           ▼
┌──────────────────────────┐
│   Mean+Max Column Pool   │   concat over height axis → 1024 chans
└──────────┬───────────────┘
           │  (B, W, 1024)
           ▼
┌──────────────────────────┐
│   Linear 1024 → 384      │
│   + LayerNorm            │   stabilises BiLSTM input distribution
└──────────┬───────────────┘
           │  (B, W, 384)
           ▼
┌──────────────────────────┐
│   Bidirectional LSTM     │   2 layers, 320 hidden × 2 directions
│   + dropout              │   captures left/right context
└──────────┬───────────────┘
           │  (B, W, 640)
           ▼
┌──────────────────────────┐
│   Linear Classifier      │   640 → num_classes (len(CHARS)+1)
└──────────┬───────────────┘
           │  (T, B, C)
           ▼
  CTC Beam Search Decode
   + Char-LM Rescoring + 3-view TTA
           │
           ▼
      "Aspirin 500mg"
```

### How Each Stage Works

| Stage | What it does | Details |
|-------|-------------|---------|
| **ResNet backbone** | Extracts visual features from a grayscale word image | ResNet-18 with a 1×1 input adapter for grayscale and stride relaxations in layer3/4 so horizontal resolution stays high enough for CTC alignment. |
| **Gated conv** | Learns a soft feature mask on top of the backbone | `conv ⊙ sigmoid(gate)` lets the network suppress backbone activations that aren't text-relevant. |
| **Mean+max column pooling** | Collapses the height axis into a sequence | Concatenates per-column mean and max → 1024-d feature per timestep, preserving stroke peaks that average pooling smears out. |
| **Bidirectional LSTM** | Reads the feature sequence in both directions | 2-layer BiLSTM with 320 hidden per direction (640 output per timestep). LayerNorm before the LSTM stabilises the recurrent input. |
| **CTC decoder** | Aligns predictions to variable-length text | CTC loss for training (alignment-free). Inference uses **beam search** (width 15) with **char-LM rescoring** (weight 0.4) and **3-view TTA** (base + ±3° rotation). |

---

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Input size | 32 × 320 × 1 (H × W × C, grayscale) |
| Character set | Printable ASCII from `config.CHARS` |
| Vocabulary size | `len(CHARS) + 1` (CTC blank at index 0) |
| Visual backbone | ResNet-18 (ImageNet init) + 1×1 grayscale adapter + relaxed strides |
| Sequence head | Gated conv → mean+max column pool → Linear 1024→384 + LayerNorm |
| RNN | BiLSTM, 2 layers, 320 hidden per direction (640 output) |
| Optimizer | AdamW (differential LR: backbone × 0.2, head × 1.0) |
| Base LR | `2e-4` (`config.LR`) |
| LR schedule | OneCycleLR (cosine, `pct_start=0.1`) |
| Batch size | `256` |
| Epochs | `100` max with early stopping (`patience=6` full-CER checks) |
| Gradient accumulation | `1` step |
| Loss function | Smoothed CTC (`blank=0`, smoothing=0.1 over non-blank classes) |
| Gradient clipping | `max_norm=5` |
| Regularisation | Dropout 0.5, weight decay `3e-4` |
| Data augmentation | GPU batch affine (rot/shear/translate/horiz-stretch), elastic, photometric, morphological |
| TTA at inference | 3 views: base + ±3° rotation (rotate after resize for stable scale) |
| Validation cadence | Sampled CER each epoch, full CER every 10 epochs |

---

## Project Structure

```
├── main.py                # Single entry point — train / predict / serve
├── config.py              # Global paths, charset, and training hyperparameters
├── core/
│   ├── model.py           # ResNetCRNN architecture
│   ├── decoding.py        # CTC greedy/beam decoding + char LM
│   └── metrics.py         # CER and alignment metrics
├── pipeline/
│   ├── dataset.py         # GNHK / IAM dataset loaders + collate
│   └── preprocessing.py   # Base transforms, TTA, GPU augmentation
├── services/
│   ├── training.py        # Training loop, checkpoints, history, early stopping
│   ├── inference.py       # Segmentation + TTA + beam decode + per-word output
│   ├── evaluation.py      # Curves and confusion matrix generation
│   └── lm_builder.py      # Build the character LM from training transcripts
├── web/
│   ├── app.py             # Flask app
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── plots/
├── notebooks/
│   └── training_curves.ipynb  # Jupyter notebook — individual training plots
├── checkpoints/      # Model weights + char LM (gitignored)
├── outputs/          # Training curves & history (gitignored)
├── requirements.txt
└── data/             # Datasets (gitignored)
    ├── gnhk/
    │   ├── train_data/
    │   └── test_data/
    └── iam/
```

### File Details

- **`main.py`** — Entry point. `python main.py` runs train (if no weights) → build LM → serve in one go.
- **`core/model.py`** — `ResNetCRNN`: ResNet18 backbone → gated conv → mean+max column pool → 2-layer BiLSTM (hidden 320, bidir).
- **`pipeline/dataset.py`** — Dataset loaders for GNHK and IAM sources.
- **`services/training.py`** — Training orchestration, scheduling, checkpoints, and history export.
- **`services/inference.py`** — Segmentation + recognition; returns per-word `{text, confidence, bbox}`.
- **`services/lm_builder.py`** — Builds the character bigram LM from training transcripts.
- **`web/app.py`** — Flask routes and server startup.

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/MessyWriting.git
cd MessyWriting

# 2. Install dependencies
pip install -r requirements.txt

# 3. For GPU support (NVIDIA), install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Download the GNHK dataset into data/gnhk/
#    (see https://doi.org/10.1109/ICDAR.2021.00060)

# 5. (Optional) Add IAM into data/iam/ for extra character coverage

# 6. Full pipeline — trains (if needed) → builds LM → launches the web app
python main.py

# Or run individual steps:
python main.py train              # Force a fresh train run, then build LM + serve
python main.py lm                 # Rebuild the character LM only
python main.py predict image.jpg  # CLI inference (prints JSON with per-word output)
python main.py serve              # Web UI only (port 5000)
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **PyTorch** | Model definition, training, GPU-accelerated inference |
| **torchvision** | Image transforms (resize, normalize, grayscale) |
| **OpenCV** | Image I/O and bounding box cropping from polygon annotations |
| **Pillow** | Image loading for inference pipeline |
| **Flask** | Lightweight REST API and web server |
| **matplotlib** | Training curve visualization (loss, accuracy, CER) |
| **trdg** | Synthetic handwriting data generation |
| **Jupyter** | Notebook for generating individual training plots |

---

## Dataset

**GNHK (Handwriting in the Wild)** — introduced at ICDAR 2021 ([paper](https://doi.org/10.1109/ICDAR.2021.00060)). Each sample is a photograph of handwritten English text with word-level polygon annotations. The dataset contains diverse handwriting styles captured in natural settings.

| Split | Samples |
|-------|---------|
| Train (GNHK) | ~32,500 word crops |
| Val/Test (GNHK) | ~10,000 word crops |
| Synthetic (optional) | up to 50,000 generated word images |

**Annotation format:** Each JSON file contains a flat list of objects with `"text"` and `"polygon"` (8-point x0,y0…x3,y3) keys. The dataset loader derives axis-aligned bounding boxes from the polygon coordinates.

---

## Training Metrics

| Metric | Description |
|--------|-------------|
| **CTC Loss** | Connectionist Temporal Classification — alignment-free sequence loss |
| **Word Accuracy** | Exact match: predicted text == ground truth text |
| **CER** | Character Error Rate via Levenshtein edit distance |

Training saves:
- `checkpoints/crnn_gnhk_best.pth` — best model by validation loss
- `checkpoints/crnn_gnhk.pth` — final epoch model
- `outputs/training_curves.png` — combined 3-panel plot

> Open `notebooks/training_curves.ipynb` to generate individual plots you can embed on a website.

---

## Results

Trained for **35 epochs** on an NVIDIA RTX 4060 (8 GB) with CUDA + AMP.

| Metric | Best | Final (Epoch 35) |
|--------|------|-------------------|
| **Train Loss** | — | 0.3952 |
| **Val Loss** | 0.9458 (epoch 28) | 0.9904 |
| **Word Accuracy** | 55.5% (epoch 35) | 55.5% |
| **CER** | 22.0% (epoch 35) | 22.0% |
| **Epoch Time** | 25.7s | ~26s avg |

### Training Speed

| Metric | Before Optimisation | After Optimisation | Speedup |
|--------|--------------------|--------------------|---------|
| **Time per epoch** | ~115s | ~26s | **4.4×** |
| **Total training (35 epochs)** | ~67 min | ~16 min | **4.2×** |

Key contributors: GPU batch augmentation, AMP (FP16), full tensor pre-caching, `cudnn.benchmark`.

### Training Curves

![Training Curves](outputs/training_curves.png)

The model learns rapidly in the first 10 epochs, reaching ~40% word accuracy and 31% CER. With GPU batch augmentation (elastic distortion, affine, morphological ops), the train-val loss gap stays narrow throughout training (0.40 vs 0.99 at epoch 35), showing effective regularisation. Early stopping triggers at epoch 35 after 7 epochs without val loss improvement.

---

## Challenges & Improvements

### Overfitting Mitigations

The GNHK dataset (32K training samples) with a 2.4M parameter model is prone to overfitting. The following techniques keep the train-val loss gap manageable (0.40 vs 0.99 at epoch 35 — compared to 0.009 vs 1.91 in the baseline):

| Technique | What changed | File |
|-----------|-------------|------|
| **Dropout (RNN)** | Dropout 0.5 between BiLSTM layers and before the classifier | `core/model.py` |
| **GPU batch augmentation** | Affine (incl. horizontal stretch), elastic distortion, brightness/contrast, blur, morphology — all on GPU per batch | `pipeline/preprocessing.py` |
| **Weight decay** | L2 regularisation via `AdamW(weight_decay=3e-4)` | `services/training.py` |
| **CTC label smoothing** | Smooths only non-blank classes so it doesn't fight CTC's natural blank sparsity | `services/training.py` |
| **Early stopping** | CER-based early stopping on full-validation checkpoints | `services/training.py` |
| **IAM regularisation** | IAM samples (5:4 weighted) supply character diversity GNHK alone misses | `pipeline/dataset.py` |

### Accuracy Improvements

| Technique | What changed | Expected Impact |
|-----------|-------------|----------------|
| **Mean+max column pool** | Replaces AdaptiveAvgPool over height; concat preserves stroke peaks | ~0.5-1% CER improvement, free at runtime |
| **Beam search decoding** | Width-15 beam search replaces greedy argmax at inference | ~2-5% accuracy gain (no retraining needed) |
| **Character LM rescoring** | Bigram LM built from training transcripts rescores beam hypotheses (weight 0.4) | Catches implausible sequences without over-correcting rare tokens |
| **Test-time augmentation** | 3 averaged views (base + ±3° rotation, applied after resize) at inference | Reduces prediction variance, ~1-2% accuracy gain |
| **Elastic distortion** | Smooth random displacement fields simulate natural handwriting deformation | Better training generalisation |
| **Horizontal stretch aug** | 0.85-1.15× random width scale folded into the affine grid | Handwriting has high width variance; consistently helps OCR |
| **Morphological ops** | Erosion/dilation via max/min pooling simulates pen stroke thickness variation | Robustness to varied pen strokes |
| **CLAHE + relative segmentation thresholds** | Word-spacing thresholds scale with line height; CLAHE before Otsu | Robust word segmentation across resolutions and lighting |
| **Per-word confidence + colour-coded UI** | Each word's mean non-blank softmax surfaced in the web UI | Users see which words to double-check |

### Training Speed Optimizations (115s → 26s per epoch, **4.4× speedup**)

| Technique | What changed | Impact |
|-----------|-------------|--------|
| **Mixed precision (AMP)** | Forward pass runs in FP16 via `torch.amp.autocast` + `GradScaler` | ~2× speedup on tensor cores, halves VRAM usage |
| **Batch size scaled to 256** | AMP + tensor caching free enough memory; large batches improve GPU utilisation | Better step throughput |
| **`cudnn.benchmark`** | cuDNN autotuner selects fastest convolution kernels for fixed 32×320 input | ~10–20% faster convolutions |
| **`pin_memory` + `non_blocking`** | Pinned CPU memory with async GPU transfers | Overlaps data transfer with compute |
| **OneCycleLR** | Cosine annealing per batch (replaces StepLR) | Faster convergence, often fewer epochs needed |
| **`zero_grad(set_to_none=True)`** | Avoids memset on gradient buffers | Minor per-step speedup |
| **Full tensor pre-caching** | All images converted to tensors once at init — GPU batch augmentation handles the rest | Eliminates CPU data-loading bottleneck |

> **Note:** The deployed model uses `crnn_gnhk_best.pth` (best val loss checkpoint). At inference time, beam search + TTA + LM rescoring are applied automatically for maximum accuracy.

---

## Web Interface

The Flask app serves a prescription-pad styled UI at `http://localhost:5000`:

1. **Drag & drop** (or browse) an image of handwritten text
2. Click **Transcribe Handwriting**
3. The CRNN decodes the image and returns the predicted text

The `/predict` endpoint accepts `POST` with a `multipart/form-data` image and returns:
```json
{"text": "Aspirin 500mg"}
```

---

## License

This project is for educational and portfolio purposes.
