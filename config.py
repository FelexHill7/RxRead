"""
config.py — Shared constants, paths, and hyperparameters.

Bottom layer of the architecture: no project imports.
Every other module imports from here instead of defining its own constants.
"""

import os

# ── Project root (all relative paths resolve from here) ──────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Character set ─────────────────────────────────────────────────────────────
CHARS = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}   # 0 reserved for CTC blank
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

# ── Image dimensions ──────────────────────────────────────────────────────────
IMG_HEIGHT = 32
IMG_WIDTH = 128

# ── Data paths ────────────────────────────────────────────────────────────────
TRAIN_DIR = "data/gnhk/train_data"
TEST_DIR = "data/gnhk/test_data"
SYNTHETIC_DIR = "data/synthetic"
IAM_DIR = "data/iam"

# ── Checkpoint paths ──────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
BEST_WEIGHTS = os.path.join(CHECKPOINT_DIR, "crnn_gnhk_best.pth")
FINAL_WEIGHTS = os.path.join(CHECKPOINT_DIR, "crnn_gnhk.pth")
CHAR_LM_PATH = os.path.join(CHECKPOINT_DIR, "char_lm.json")

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join("web", "static", "plots")

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE = 198
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10
ACCUMULATION_STEPS = 1
DROPOUT = 0.3
GRAD_CLIP_NORM = 5
BACKBONE_LR_MULT = 0.2
ONECYCLE_PCT_START = 0.1
AUGMENT_START_EPOCH = 3

# ── Performance tuning knobs ────────────────────────────────────────────────
NUM_WORKERS = 0
PREFETCH_FACTOR = None

# Evaluate CER on a subset for most epochs (faster), full set periodically.
VAL_CER_SAMPLE_LIMIT = 2048
FULL_VAL_INTERVAL = 5

# Plot less frequently to avoid per-epoch matplotlib overhead.
PLOT_EVERY_N_EPOCHS = 5


def encode_text(text):
    """Convert a string to a list of integer indices, skipping unknown chars."""
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]
