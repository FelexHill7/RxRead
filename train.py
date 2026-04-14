"""
train.py — Training loop for the CRNN handwriting recogniser.

Trains the CRNN on the GNHK dataset using CTC loss.  Saves the best
checkpoint (by validation loss) and a final checkpoint after all epochs.
Generates training curves (loss, accuracy, CER) saved to training_curves.png.

Usage:
    python train.py
"""

import os
import json
import math
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")                              # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from dataset import GNHKDataset, build_dataset, CHARS, idx2char
from model import CRNN


# ── CTC greedy decoder ───────────────────────────────────────────────────────
def ctc_decode(output):
    """Greedy CTC decode: argmax at each timestep, collapse repeats, drop blanks."""
    preds = output.argmax(2)                         # (T, B)
    batch_texts = []
    for b in range(preds.size(1)):
        seq = preds[:, b].tolist()
        chars, prev = [], None
        for idx in seq:
            if idx != prev and idx != 0:
                chars.append(idx2char.get(idx, ""))
            prev = idx
        batch_texts.append("".join(chars))
    return batch_texts


def char_error_rate(pred, target):
    """Character Error Rate (CER) via Levenshtein edit distance."""
    n = len(target)
    m = len(pred)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            temp = dp[j]
            if target[i - 1] == pred[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m] / max(n, 1)


def align_chars(pred, target):
    """Align predicted and target strings using Levenshtein DP to get
    character-level (true_char, pred_char) pairs including substitutions."""
    n, m = len(target), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if target[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and (target[i - 1] == pred[j - 1] or
                                  dp[i][j] == dp[i - 1][j - 1] + 1):
            pairs.append((target[i - 1], pred[j - 1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            pairs.append((target[i - 1], "\u2205"))
            i -= 1
        else:
            pairs.append(("\u2205", pred[j - 1]))
            j -= 1
    return list(reversed(pairs))


def generate_confusion_matrix(model, val_loader, device, use_amp):
    """Run the model on test samples and generate a character confusion matrix."""
    model.eval()
    true_chars_all = []
    pred_chars_all = []

    with torch.no_grad():
        for images, labels, label_lengths, ground_truths in val_loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
            decoded = ctc_decode(outputs)
            for pred_text, gt_text in zip(decoded, ground_truths):
                for true_c, pred_c in align_chars(pred_text, gt_text):
                    if true_c != "\u2205" and pred_c != "\u2205":
                        true_chars_all.append(true_c)
                        pred_chars_all.append(pred_c)

    print(f"Aligned {len(true_chars_all)} character pairs for confusion matrix")

    # Top 30 most frequent characters for readability
    char_freq = Counter(true_chars_all)
    top_30 = [c for c, _ in char_freq.most_common(30)]

    mask = [(t in top_30 and p in top_30) for t, p in zip(true_chars_all, pred_chars_all)]
    filtered_true = [t for t, m in zip(true_chars_all, mask) if m]
    filtered_pred = [p for p, m in zip(pred_chars_all, mask) if m]

    cm = sklearn_confusion_matrix(filtered_true, filtered_pred, labels=top_30)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)

    display_labels = [f"'{c}'" if c != " " else "sp" for c in top_30]
    ax.set_xticks(range(len(top_30)))
    ax.set_yticks(range(len(top_30)))
    ax.set_xticklabels(display_labels, fontsize=8, rotation=45, ha="right")
    ax.set_yticklabels(display_labels, fontsize=8)
    ax.set_xlabel("Predicted Character", fontsize=12)
    ax.set_ylabel("True Character", fontsize=12)
    ax.set_title("Character Confusion Matrix (normalized by row)", fontweight="bold", fontsize=14)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Rate", fontsize=10)

    for i in range(len(top_30)):
        for j in range(len(top_30)):
            val = cm_norm[i, j]
            if val > 0.05:
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    plt.tight_layout()
    os.makedirs("static/plots", exist_ok=True)
    fig.savefig("static/plots/confusion_matrix.png", dpi=150, bbox_inches="tight")
    fig.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Confusion matrix saved to static/plots/confusion_matrix.png and outputs/confusion_matrix.png")


def collate_fn(batch):
    """Custom collate: stack images and concatenate variable-length labels
    into the flat format required by CTCLoss."""
    images, labels, texts = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels])
    labels_concat = torch.cat(labels)
    return images, labels_concat, label_lengths, texts


# ── GPU batch augmentation ────────────────────────────────────────────────────
@torch.no_grad()
def gpu_augment(images):
    """Apply random augmentation to an entire batch on GPU in one shot.

    Replaces per-sample CPU transforms with batch-parallel GPU operations:
    - Random affine (rotation, translation, shear)
    - Elastic distortion (simulates handwriting deformation)
    - Random brightness / contrast
    - Gaussian blur
    - Morphological erosion / dilation (pen stroke thickness variation)

    Input/output: (B, 1, 32, 128) tensors in [-1, 1] range.
    """
    B, C, H, W = images.shape
    device = images.device

    # ── Random affine: rotation (±5°) + translation (±5%) + shear (±5°) ───
    angles = (torch.rand(B, device=device) * 2 - 1) * 5 * (math.pi / 180)
    shear  = (torch.rand(B, device=device) * 2 - 1) * 5 * (math.pi / 180)
    tx     = (torch.rand(B, device=device) * 2 - 1) * 0.05
    ty     = (torch.rand(B, device=device) * 2 - 1) * 0.05

    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = cos_a + shear * sin_a
    theta[:, 0, 1] = -sin_a + shear * cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty

    grid = torch.nn.functional.affine_grid(theta, images.shape, align_corners=False)
    images = torch.nn.functional.grid_sample(images, grid, align_corners=False, padding_mode="border")

    # ── Elastic distortion (50% chance per batch) ─────────────────────────
    # Generates smooth random displacement fields to simulate natural
    # handwriting deformation (warped strokes, uneven spacing)
    if torch.rand(1).item() > 0.5:
        alpha = 3.0   # displacement magnitude
        sigma = 4.0   # smoothing sigma
        # Random displacement field
        dx = torch.rand(B, 1, H, W, device=device) * 2 - 1
        dy = torch.rand(B, 1, H, W, device=device) * 2 - 1
        # Smooth with Gaussian
        ek = 7
        eax = torch.arange(ek, dtype=torch.float32, device=device) - ek // 2
        ekernel = torch.exp(-0.5 * (eax / sigma) ** 2)
        ekernel = ekernel / ekernel.sum()
        ek2d = (ekernel[:, None] * ekernel[None, :]).view(1, 1, ek, ek)
        dx = torch.nn.functional.conv2d(dx, ek2d, padding=ek // 2) * alpha
        dy = torch.nn.functional.conv2d(dy, ek2d, padding=ek // 2) * alpha
        # Normalize to [-1, 1] grid space
        dx = dx.squeeze(1) / (W / 2)
        dy = dy.squeeze(1) / (H / 2)
        # Base grid + displacement
        base_grid = torch.nn.functional.affine_grid(
            torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1),
            images.shape, align_corners=False
        )
        base_grid[..., 0] += dx
        base_grid[..., 1] += dy
        images = torch.nn.functional.grid_sample(images, base_grid, align_corners=False, padding_mode="border")

    # ── Random brightness (±30%) and contrast (±30%) ─────────────────────
    brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
    contrast   = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
    mean = images.mean(dim=(2, 3), keepdim=True)
    images = contrast * (images - mean) + mean + (brightness - 1.0)

    # ── Gaussian blur (kernel_size=3, random sigma 0.1–1.0) ──────────────
    sigma = 0.1 + torch.rand(1, device=device).item() * 0.9
    k = 3
    ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel_2d = (kernel[:, None] * kernel[None, :]).view(1, 1, k, k)
    images = torch.nn.functional.conv2d(images, kernel_2d, padding=k // 2, groups=1)

    # ── Morphological erosion / dilation (30% chance each) ────────────────
    # Simulates pen stroke thickness variation: dilation = thicker strokes,
    # erosion = thinner strokes. Uses max/min pooling as GPU-friendly morph ops.
    morph_roll = torch.rand(1).item()
    if morph_roll < 0.3:
        # Dilation (thicken dark strokes): min-pool on inverted → invert back
        # For [-1,1] range where -1 is dark ink: dilation = take the min in neighbourhood
        images = -torch.nn.functional.max_pool2d(-images, kernel_size=3, stride=1, padding=1)
    elif morph_roll < 0.6:
        # Erosion (thin strokes): max-pool (takes brightest/background in neighbourhood)
        images = torch.nn.functional.max_pool2d(images, kernel_size=3, stride=1, padding=1)

    return images.clamp(-1, 1)


def train():
    """Run the full training pipeline."""
    # ── Hyper-parameters & paths ─────────────────────────────────────────────
    TRAIN_DIR     = "data/gnhk/train_data"
    TEST_DIR      = "data/gnhk/test_data"
    SYNTHETIC_DIR = "data/synthetic"
    BATCH_SIZE = 64         # doubled (AMP halves memory usage)
    EPOCHS = 50
    LR = 0.001
    WEIGHT_DECAY = 1e-4
    PATIENCE = 7           # early stopping patience (epochs without val loss improvement)
    NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # mixed precision only on GPU
    print(f"Using device: {device} | AMP: {use_amp}")

    # Let cuDNN autotuner find the fastest kernels for fixed input size (32×128)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # ── Datasets & DataLoaders ───────────────────────────────────────────────
    # Training: GNHK + optional synthetic data (if generated)
    train_set = build_dataset(TRAIN_DIR, synthetic_dir=SYNTHETIC_DIR)
    # Validation: GNHK only (always evaluate on real handwriting)
    val_set   = GNHKDataset(TEST_DIR)

    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    loader_kwargs = dict(
        collate_fn=collate_fn,
        num_workers=0,                       # 0 avoids Windows multiprocessing deadlocks
        pin_memory=(device.type == "cuda"),   # faster CPU→GPU copies
    )
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs
    )

    # ── Model, optimiser, scheduler, loss ────────────────────────────────────
    model     = CRNN(NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_loader),   # cosine anneal per step — faster convergence
    )
    criterion = CTCLoss(blank=0, zero_infinity=True)  # blank=0 matches our char2idx offset
    scaler = torch.amp.GradScaler(enabled=use_amp)    # mixed-precision grad scaling

    best_val_loss = float("inf")
    patience_counter = 0  # early stopping counter

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # ── Metric history for plotting ──────────────────────────────────────────
    history = {
        "train_loss": [], "val_loss": [],
        "val_accuracy": [], "val_cer": [],
    }

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        # ── Training ──────────────────────────────────────────────
        model.train()
        total_train_loss = 0

        for images, labels, label_lengths, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Batch augmentation on GPU — processes all 64 images at once
            images = gpu_augment(images)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)                  # (T, B, C) — CTC time-first
                input_lengths = torch.full(              # every sample has the same seq length
                    (images.size(0),), outputs.size(0), dtype=torch.long
                )
                loss = criterion(
                    outputs.log_softmax(2), labels, input_lengths, label_lengths
                )

            optimizer.zero_grad(set_to_none=True)        # slightly faster than zeroing
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:          # optimizer actually stepped
                scheduler.step()                         # OneCycleLR steps per batch
            total_train_loss += loss.item()

        # ── Validation (loss + accuracy + CER) ────────────────────
        model.eval()
        total_val_loss = 0
        correct, total_words = 0, 0
        total_cer = 0.0

        with torch.no_grad():
            for images, labels, label_lengths, ground_truths in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    input_lengths = torch.full(
                        (images.size(0),), outputs.size(0), dtype=torch.long
                    )
                    loss = criterion(
                        outputs.log_softmax(2), labels, input_lengths, label_lengths
                    )
                total_val_loss += loss.item()

                # Decode predictions and compare to ground truth
                decoded = ctc_decode(outputs)
                for pred_text, gt_text in zip(decoded, ground_truths):
                    total_words += 1
                    if pred_text == gt_text:
                        correct += 1
                    total_cer += char_error_rate(pred_text, gt_text)

        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)
        accuracy  = correct / max(total_words, 1) * 100      # word-level accuracy %
        avg_cer   = total_cer / max(total_words, 1) * 100    # character error rate %
        epoch_time = time.time() - epoch_start

        # Store metrics
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_accuracy"].append(accuracy)
        history["val_cer"].append(avg_cer)

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Accuracy: {accuracy:.1f}% | "
            f"CER: {avg_cer:.1f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model + early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), "checkpoints/crnn_gnhk_best.pth")
            print(f"  ✓ Best model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  ✗ Early stopping triggered (no improvement for {PATIENCE} epochs)")
                break

        # ── Update plots after every epoch ────────────────────────
        epochs_so_far = range(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Plot 1 — Loss curves
        axes[0].plot(epochs_so_far, history["train_loss"], "o-", label="Train Loss")
        axes[0].plot(epochs_so_far, history["val_loss"],   "o-", label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("CTC Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2 — Word-level accuracy
        axes[1].plot(epochs_so_far, history["val_accuracy"], "o-", color="green")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Word-Level Accuracy")
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, alpha=0.3)

        # Plot 3 — Character Error Rate
        axes[2].plot(epochs_so_far, history["val_cer"], "o-", color="red")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("CER (%)")
        axes[2].set_title("Character Error Rate")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("outputs/training_curves.png", dpi=150)
        plt.close(fig)

    # Always save final model too
    torch.save(model.state_dict(), "checkpoints/crnn_gnhk.pth")

    # Export training history as JSON for the notebook
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Build character language model from training data labels
    from predict import CharLM
    lm = CharLM()
    train_texts = []
    if hasattr(train_set, 'cached'):
        train_texts = [text for _, _, text in train_set.cached]
    elif hasattr(train_set, 'datasets'):
        for ds in train_set.datasets:
            if hasattr(ds, 'cached'):
                train_texts.extend(text for _, _, text in ds.cached)
    if train_texts:
        lm.build_from_texts(train_texts)
        lm.save()
        print(f"Character LM built from {len(train_texts)} training texts")

    # ── Generate confusion matrix on test set using best checkpoint ───────
    best_path = "checkpoints/crnn_gnhk_best.pth"
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        print("Loaded best checkpoint for confusion matrix generation")
    generate_confusion_matrix(model, val_loader, device, use_amp)

    print("Training complete. Final model saved to checkpoints/crnn_gnhk.pth")
    print("Training curves saved to outputs/training_curves.png")
    print("Training history saved to outputs/training_history.json")


if __name__ == "__main__":
    train()