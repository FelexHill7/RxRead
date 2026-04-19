# RxRead — Doctor Handwriting Recognition

A deep-learning pipeline that reads messy handwritten text from prescription images. Built with a **ResNet-CRNN + Attention + CTC** model trained on the [GNHK](https://doi.org/10.1109/ICDAR.2021.00060) wild-handwriting dataset (+ optional IAM and synthetic data), with beam search decoding, test-time augmentation, and character language model rescoring. Served through a clean Flask web interface.

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

## Method Evolution (What Changed and Why)

This section summarizes the major method decisions made over time and the rationale behind each change.

## Method Evolution (Design Iterations)

This section summarizes the major architectural and training decisions made during development.

| Phase | Previous approach | Current approach | Rationale |
|------|------------------|----------------|----------|
| 1. Visual encoder | Custom CNN | ResNet-18 backbone | Improved feature extraction and transfer learning from ImageNet weights |
| 2. Sequence modeling | CNN → BiLSTM | CNN → BiLSTM → Attention | Attention provides contextual refinement over sequential features |
| 3. Decoding | Greedy CTC decoding | Beam search + character bigram LM | Improves stability and reduces implausible character sequences |
| 4. Data sources | GNHK only | GNHK + IAM + synthetic | Improves robustness across handwriting styles |
| 5. Optimization | Standard Adam | AdamW + differential LR + OneCycleLR | Better convergence and stability |
| 6. Training efficiency | CPU-heavy pipeline | GPU augmentation + AMP + caching | Significant reduction in training time |
| 7. Code structure | Monolithic scripts | Modular architecture (core/pipeline/services/web) | Separation of concerns and maintainability |

### Notes on CTC vs ResNet

- The project did **not** replace CTC with ResNet; these are different parts of the system.
- **ResNet** is the visual backbone (feature extractor).
- **CTC** is still the sequence training objective/decoder interface.
- Current stack is: **ResNet backbone + BiLSTM + Attention + CTC**.

---

## Architecture

### ResNet-CRNN + Attention + CTC Pipeline

```
Input Image (32×128, grayscale)
        │
        ▼
┌──────────────────────────┐
│   ResNet-18 Backbone     │   conv1(1ch), stride tweaks in layer3/4
│   (+ AdaptiveAvgPool2d ) │   preserves sequence width for CTC
└──────────┬───────────────┘
           │  (B, T, 512)
           ▼
┌──────────────────────────┐
│    Bidirectional LSTM     │   3 layers, 256 hidden units × 2 directions
│    + dropout              │   Captures left-to-right & right-to-left context
└──────────┬───────────────┘
           │  (B, T, 512)
           ▼
┌──────────────────────────┐
│    Additive Attention     │   Focuses on relevant spatial positions
│    (residual connection)  │   Query-based soft attention over timesteps
└──────────┬───────────────┘
           │  (B, T, 512)
           ▼
┌──────────────────────────┐
│      Linear Classifier    │   512 → num_classes (len(CHARS)+1)
└──────────┬───────────────┘
           │  (T, B, C)
           ▼
  CTC Beam Search Decode
   + LM Rescoring + TTA
           │
           ▼
      "Aspirin 500mg"
```

### How Each Stage Works

| Stage | What it does | Details |
|-------|-------------|---------|
| **ResNet backbone** | Extracts visual features from a grayscale word image | ResNet-18 backbone adapted for 1-channel input with stride modifications to preserve temporal width. |
| **Bidirectional LSTM** | Reads the feature sequence in both directions | 3-layer BiLSTM with 256 hidden units per direction (512 output features per timestep). |
| **Attention** | Focuses on relevant spatial positions | Additive attention computes weighted context over all timesteps with a residual connection. Helps the classifier attend to the most informative positions for each character. |
| **CTC decoder** | Aligns predictions to variable-length text | CTC loss for training (alignment-free). Inference uses **beam search** (width 10) with **character LM rescoring** and **test-time augmentation** (5 views). Greedy decode available as a faster fallback. |

---

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Input size | 32 × 128 × 1 (H × W × C, grayscale) |
| Character set | Printable ASCII from `config.CHARS` |
| Vocabulary size | `len(CHARS) + 1` (CTC blank at index 0) |
| Visual backbone | ResNet-18 (ImageNet init) adapted for grayscale + width-preserving strides |
| RNN type | BiLSTM, 3 layers, 256 hidden per direction |
| Attention | Additive attention with residual fusion |
| Optimizer | AdamW (differential LR: backbone + head) |
| Base LR | `3e-4` (`config.LR`) |
| LR schedule | OneCycleLR (cosine, `pct_start=0.1`) |
| Batch size | `64` |
| Epochs | `80` max with early stopping (`patience=10`) |
| Gradient accumulation | `2` steps |
| Loss function | CTC Loss (`blank=0`, `zero_infinity=True`) |
| Gradient clipping | `max_norm=5` |
| Regularisation | Dropout (0.3), weight decay `1e-4` |
| Data augmentation | GPU batch affine/elastic/photometric/morphological transforms |
| Validation cadence | Sampled CER on most epochs, full CER every 5 epochs |

---

## Project Structure

```
├── main.py                # Single entry point — train / predict / serve
├── config.py              # Global paths, charset, and training hyperparameters
├── core/
│   ├── model.py           # ResNetCRNN + Attention architecture
│   ├── decoding.py        # CTC greedy/beam decoding + char LM
│   └── metrics.py         # CER and alignment metrics
├── pipeline/
│   ├── dataset.py         # GNHK/IAM/synthetic dataset loaders + collate
│   ├── preprocessing.py   # Base transforms, TTA, GPU augmentation
│   └── generate_synthetic.py
├── services/
│   ├── training.py        # Training loop, checkpoints, history, early stopping
│   ├── inference.py       # Inference + segmentation + TTA + beam decode
│   └── evaluation.py      # Curves and confusion matrix generation
├── web/
│   ├── app.py             # Flask app
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── plots/
├── notebooks/
│   └── training_curves.ipynb  # Jupyter notebook — individual training plots
├── checkpoints/      # Model weights (gitignored)
├── outputs/          # Training curves & history (gitignored)
├── requirements.txt
└── data/             # Datasets (gitignored)
    ├── gnhk/
    │   ├── train_data/
    │   └── test_data/
    └── synthetic/    # Generated by generate_synthetic.py
```

### File Details

- **`main.py`** — Entry point for train/serve/predict commands.
- **`core/model.py`** — `ResNetCRNN` model definition.
- **`pipeline/dataset.py`** — Dataset loaders for GNHK, IAM, and synthetic sources.
- **`services/training.py`** — Training orchestration, scheduling, checkpoints, and history export.
- **`services/inference.py`** — Runtime prediction pipeline for CLI and web app.
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

# 5. (Optional) Generate synthetic training data
python -m pipeline.generate_synthetic --count 50000

# 6. Full pipeline — trains (if needed) then launches the web app
python main.py

# Or run individual steps:
python main.py train              # Train only
python main.py predict image.jpg  # CLI inference
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
| **Dropout (CNN/RNN)** | Dropout in CNN feature stack and recurrent stack | `core/model.py` |
| **GPU batch augmentation** | Affine transforms, elastic distortion, brightness/contrast, blur, morphological ops — all on GPU at batch level | `pipeline/preprocessing.py` |
| **Weight decay** | L2 regularisation via `AdamW(weight_decay=1e-4)` | `services/training.py` |
| **Early stopping** | CER-based early stopping on full-validation checkpoints | `services/training.py` |
| **Synthetic data** | Optional generated word images to supplement training | `pipeline/generate_synthetic.py` |

### Accuracy Improvements

| Technique | What changed | Expected Impact |
|-----------|-------------|----------------|
| **Attention layer** | Additive attention with residual connection between BiLSTM and classifier | Better character-level focus; ~2-3% accuracy gain |
| **Beam search decoding** | Width-10 beam search replaces greedy argmax at inference | ~2-5% accuracy gain (no retraining needed) |
| **Character LM rescoring** | Bigram language model built from training data rescores beam hypotheses | Catches implausible character sequences |
| **Test-time augmentation** | 5 augmented views (rotation, scale) averaged at inference | Reduces prediction variance, ~1-2% accuracy gain |
| **Elastic distortion** | Smooth random displacement fields simulate natural handwriting deformation | Better training generalisation |
| **Morphological ops** | Erosion/dilation via max/min pooling simulates pen stroke thickness variation | Robustness to varied pen strokes |

### Training Speed Optimizations (115s → 26s per epoch, **4.4× speedup**)

| Technique | What changed | Impact |
|-----------|-------------|--------|
| **Mixed precision (AMP)** | Forward pass runs in FP16 via `torch.amp.autocast` + `GradScaler` | ~2× speedup on tensor cores, halves VRAM usage |
| **Batch size 32 → 64** | Doubled batch size (AMP frees enough memory) | Better GPU utilisation per step |
| **`cudnn.benchmark`** | cuDNN autotuner selects fastest convolution kernels for fixed 32×128 input | ~10–20% faster convolutions |
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