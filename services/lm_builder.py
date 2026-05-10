"""
lm_builder.py — Build the character bigram LM from training transcriptions.

Run once after the dataset is in place. The LM is read by inference.py at
prediction time to gently rescore CTC beam-search outputs.

Usage:
    python -c "from services.lm_builder import build_lm; build_lm()"
"""

from config import TRAIN_DIR, IAM_DIR
from pipeline.dataset import GNHKDataset, IAMDataset
from core.decoding import CharLM
import os


def build_lm():
    texts = []

    gnhk = GNHKDataset(TRAIN_DIR)
    texts.extend(t for _, _, t in gnhk.cached)
    print(f"  collected {len(gnhk)} GNHK texts")

    if os.path.isdir(IAM_DIR):
        iam = IAMDataset(IAM_DIR)
        texts.extend(t for _, _, t in iam.cached)
        print(f"  collected {len(iam)} IAM texts")

    lm = CharLM()
    lm.build_from_texts(texts)
    lm.save()
    print(f"  CharLM built from {len(texts)} sequences → {lm.path}")
    return lm


if __name__ == "__main__":
    build_lm()
