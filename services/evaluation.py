"""
evaluation.py — Visualization and evaluation utilities.

Generates training curve plots and character confusion matrices.
Depends on: config, metrics, decoding (no model/dataset coupling).
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

import torch

from config import OUTPUT_DIR, PLOTS_DIR
from core.metrics import align_chars
from core.decoding import ctc_greedy_decode_batch


def plot_training_curves(history):
    """Save loss / accuracy / CER plots from training history dict."""
    epochs_so_far = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Drop epoch 1 from the loss plot — the random-init train loss is ~6x
    # larger than every subsequent epoch and crushes the y-axis. Accuracy
    # and CER plots keep epoch 1 since their natural range (0-100%) handles
    # the starting point fine.
    loss_epochs = list(epochs_so_far)[1:]
    if loss_epochs:
        axes[0].plot(loss_epochs, history["train_loss"][1:], "o-", label="Train Loss")
        axes[0].plot(loss_epochs, history["val_loss"][1:],   "o-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CTC Loss")
    axes[0].set_title("Training & Validation Loss (epoch 1 omitted)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_so_far, history["val_accuracy"], "o-", color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Word-Level Accuracy")
    axes[1].set_ylim(0, 100)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_so_far, history["val_cer"], "o-", color="red")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("CER (%)")
    axes[2].set_title("Character Error Rate")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
    plt.close(fig)


def generate_confusion_matrix(model, val_loader, device, use_amp):
    """Run the model on the validation set and save a per-character TP/FP/FN chart."""
    model.eval()
    true_chars_all = []
    pred_chars_all = []

    with torch.no_grad():
        for images, labels, label_lengths, ground_truths in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
            decoded = ctc_greedy_decode_batch(outputs)
            for pred_text, gt_text in zip(decoded, ground_truths):
                for true_c, pred_c in align_chars(pred_text, gt_text):
                    if true_c != "\u2205" and pred_c != "\u2205":
                        true_chars_all.append(true_c)
                        pred_chars_all.append(pred_c)

    print(f"Aligned {len(true_chars_all)} character pairs for confusion matrix")

    char_freq = Counter(true_chars_all)
    top_30 = [c for c, _ in char_freq.most_common(30)]

    mask = [(t in top_30 and p in top_30) for t, p in zip(true_chars_all, pred_chars_all)]
    filtered_true = [t for t, m in zip(true_chars_all, mask) if m]
    filtered_pred = [p for p, m in zip(pred_chars_all, mask) if m]

    cm = sklearn_confusion_matrix(filtered_true, filtered_pred, labels=top_30)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    display_labels = [f"'{c}'" if c != " " else "sp" for c in top_30]

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(top_30))
    bar_w = 0.25

    ax.bar(x - bar_w, tp, bar_w, label="True Positives (TP)", color="#2ecc71")
    ax.bar(x,         fp, bar_w, label="False Positives (FP)", color="#e74c3c")
    ax.bar(x + bar_w, fn, bar_w, label="False Negatives (FN)", color="#f39c12")

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=9, rotation=45, ha="right")
    ax.set_xlabel("Character", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Per-Character TP / FP / FN", fontweight="bold", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {PLOTS_DIR}/ and {OUTPUT_DIR}/")
