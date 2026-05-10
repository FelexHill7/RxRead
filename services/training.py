"""
training.py — Training loop for the handwriting recogniser.

Orchestrates data loading, model creation, training/validation epochs,
checkpointing, and post-training evaluation. Validation uses greedy CTC
decode (beam search is reserved for inference) so epochs stay fast.

Usage:
    python main.py train
"""

import os
import json
import math
import time

import torch
from torch.nn import CTCLoss

from config import (
    TRAIN_DIR, TEST_DIR, IAM_DIR,
    BEST_WEIGHTS, FINAL_WEIGHTS, CHECKPOINT_DIR, OUTPUT_DIR,
    NUM_CLASSES, EPOCHS, LR, WEIGHT_DECAY,
    PATIENCE, ACCUMULATION_STEPS, GRAD_CLIP_NORM,
    BACKBONE_LR_MULT, ONECYCLE_PCT_START, VAL_CER_SAMPLE_LIMIT,
    FULL_VAL_INTERVAL, PLOT_EVERY_N_EPOCHS,
)
from core.model import ResNetCRNN
from pipeline.dataset import GNHKDataset, build_weighted_train_set, build_dataloader
from pipeline.preprocessing import gpu_augment
from core.decoding import ctc_greedy_decode_batch          # ← greedy only for training
from core.metrics import char_error_rate
from services.evaluation import plot_training_curves, generate_confusion_matrix


def train():
    """Run the full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device} | AMP: {use_amp}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────────
    train_set, sampler = build_weighted_train_set(
        gnhk_dir=TRAIN_DIR,
        iam_dir=IAM_DIR,
    )
    val_set = GNHKDataset(TEST_DIR)
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    train_loader = build_dataloader(train_set, sampler=sampler)
    val_loader = build_dataloader(val_set)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model = ResNetCRNN(NUM_CLASSES).to(device)

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR * BACKBONE_LR_MULT},
        {'params': head_params, 'lr': LR},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[LR * BACKBONE_LR_MULT, LR], epochs=EPOCHS,
        steps_per_epoch=math.ceil(len(train_loader) / ACCUMULATION_STEPS),
        pct_start=ONECYCLE_PCT_START,
        anneal_strategy="cos",
    )
    class SmoothedCTCLoss(torch.nn.Module):
        def __init__(self, blank=0, smoothing=0.1):
            super().__init__()
            self.ctc = CTCLoss(blank=blank, zero_infinity=True)
            self.smoothing = smoothing
            self.blank = blank

        def forward(self, log_probs, targets, input_lengths, target_lengths):
            ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
            # Smooth only the non-blank classes — including blank fights CTC's
            # natural sparsity and silently inflates blank-frame entropy.
            non_blank = torch.cat(
                [log_probs[..., :self.blank], log_probs[..., self.blank + 1:]],
                dim=-1,
            )
            smooth_loss = -non_blank.mean()
            return (1 - self.smoothing) * ctc_loss + self.smoothing * smooth_loss

    criterion = SmoothedCTCLoss(blank=0, smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_cer = float("inf")
    patience_counter = 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_cer": []}

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (images, labels, label_lengths, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = gpu_augment(images)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                input_lengths = torch.full(
                    (images.size(0),), outputs.size(0), dtype=torch.long
                )
                loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_train_loss += loss.item() * ACCUMULATION_STEPS

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0
        correct, total_words = 0, 0
        total_cer = 0.0

        evaluate_full_cer = ((epoch + 1) % FULL_VAL_INTERVAL == 0) or (epoch == 0)
        cer_budget = None if evaluate_full_cer else VAL_CER_SAMPLE_LIMIT

        with torch.no_grad():
            for images, labels, label_lengths, ground_truths in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(images)
                    input_lengths = torch.full(
                        (images.size(0),), outputs.size(0), dtype=torch.long
                    )
                    loss = criterion(outputs.log_softmax(2), labels, input_lengths, label_lengths)
                total_val_loss += loss.item()

                if cer_budget is not None and total_words >= cer_budget:
                    continue

                # Greedy decode — fast, GPU-friendly, sufficient for training signal.
                # Beam search + LM is applied only at inference time (inference.py).
                decoded = ctc_greedy_decode_batch(outputs)

                for pred_text, gt_text in zip(decoded, ground_truths):
                    if cer_budget is not None and total_words >= cer_budget:
                        break
                    total_words += 1
                    if pred_text == gt_text:
                        correct += 1
                    total_cer += char_error_rate(pred_text, gt_text)

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        accuracy = correct / max(total_words, 1) * 100
        avg_cer = total_cer / max(total_words, 1) * 100
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["val_accuracy"].append(accuracy)
        history["val_cer"].append(avg_cer)

        cer_tag = "CER(full)" if evaluate_full_cer else f"CER(sample:{total_words})"
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Accuracy: {accuracy:.1f}% | "
            f"{cer_tag}: {avg_cer:.1f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        if evaluate_full_cer:
            if avg_cer < best_cer:
                best_cer = avg_cer
                patience_counter = 0
                torch.save(model.state_dict(), BEST_WEIGHTS)
                print(f"  ✓ Best model saved (CER: {best_cer:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  ✗ Early stopping triggered (no improvement for {PATIENCE} full CER checks)")
                    break

        if (epoch + 1) % PLOT_EVERY_N_EPOCHS == 0 or (epoch + 1) == EPOCHS:
            plot_training_curves(history)

    # ── Post-training ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), FINAL_WEIGHTS)

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Confusion matrix on best checkpoint
    if os.path.exists(BEST_WEIGHTS):
        model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device, weights_only=True))
        print("Loaded best checkpoint for confusion matrix generation")
    generate_confusion_matrix(model, val_loader, device, use_amp)

    print(f"Training complete. Final model saved to {FINAL_WEIGHTS}")
    print(f"Training history saved to {OUTPUT_DIR}/training_history.json")


if __name__ == "__main__":
    train()