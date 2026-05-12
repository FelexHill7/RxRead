"""
config.py — Shared constants, paths, and hyperparameters.

Bottom layer of the architecture: no project imports.
Every other module imports from here instead of defining its own constants.
"""

import os

# ── Character set ─────────────────────────────────────────────────────────────
CHARS = " !\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARS)}   # 0 reserved for CTC blank
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

# ── Image dimensions ──────────────────────────────────────────────────────────
IMG_HEIGHT = 32
IMG_WIDTH = 320

# ── Data paths ────────────────────────────────────────────────────────────────
TRAIN_DIR = "data/gnhk/train_data"
TEST_DIR = "data/gnhk/test_data"
IAM_DIR = "data/iam"
# Imgur5K is optional. Set up via Meta's downloader; loader skips silently
# if the directory is missing or empty.
IMGUR5K_DIR = "data/imgur5k"

# ── Checkpoint paths ──────────────────────────────────────────────────────────
CHECKPOINT_DIR = "checkpoints"
CHAR_LM_PATH = os.path.join(CHECKPOINT_DIR, "char_lm.json")

# Seeds to train. Multi-seed runs save to crnn_gnhk_seed{N}_best.pth and are
# auto-ensembled at inference. Default to a single seed for fast iteration —
# bump to e.g. [42, 1337] when you want the ~1 % CER ensemble lift.
SEEDS = [42]


def seed_weights_path(seed):
    """Path for a per-seed best-CER checkpoint."""
    return os.path.join(CHECKPOINT_DIR, f"crnn_gnhk_seed{seed}_best.pth")

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join("web", "static", "plots")

# ── Synthetic data ───────────────────────────────────────────────────────────
# Font-rendered word images mixed into training. Set to 0 to disable.
# Pre-cached in RAM at startup (~40 KB/sample → 30000 ≈ 1.2 GB).
# Bump to 60000+ if you have RAM headroom — vocabulary coverage matters more
# than realism for synthetic, so more is generally better for diversity.
SYNTHETIC_SAMPLES = 30000

# ── Training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE = 256
# Reduced from 100 → 60 for the medium-time training budget. Early stopping
# (PATIENCE) will cut it shorter if validation CER plateaus.
EPOCHS = 60
LR = 2e-4
WEIGHT_DECAY = 3e-4
PATIENCE = 6
ACCUMULATION_STEPS = 1
GRAD_CLIP_NORM = 5
BACKBONE_LR_MULT = 0.2
ONECYCLE_PCT_START = 0.1

# ── Performance tuning knobs ────────────────────────────────────────────────
NUM_WORKERS = 0  # Set to 0 for Windows compatibility; can be >0 on Linux/Mac for faster data loading.

# Evaluate CER on a subset for most epochs (faster), full set periodically.
VAL_CER_SAMPLE_LIMIT = 8192
FULL_VAL_INTERVAL = 10

# Plot less frequently to avoid per-epoch matplotlib overhead.
PLOT_EVERY_N_EPOCHS = 5

# ── Hard-example mining ──────────────────────────────────────────────────────
# After this fraction of training, switch to a sampler that oversamples
# high-loss examples (computed in a one-shot eval pass). Set to None to disable.
HARD_MINE_START_FRAC = 0.8


def encode_text(text):
    """Convert a string to a list of integer indices, skipping unknown chars."""
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]
