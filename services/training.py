"""
training.py — Training loop for the handwriting recogniser.

Orchestrates: data loading, model creation, training/validation epochs,
checkpointing, and post-training evaluation. All domain logic is delegated
to focused modules.

Usage:
    python train.py
"""

import os
import json
import math
import time

import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss

from config import (
    TRAIN_DIR, TEST_DIR, SYNTHETIC_DIR, IAM_DIR,
    BEST_WEIGHTS, FINAL_WEIGHTS, CHECKPOINT_DIR, OUTPUT_DIR,
    NUM_CLASSES, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    PATIENCE, ACCUMULATION_STEPS, GRAD_CLIP_NORM,
    BACKBONE_LR_MULT, ONECYCLE_PCT_START, AUGMENT_START_EPOCH,
    NUM_WORKERS, PREFETCH_FACTOR,
    VAL_CER_SAMPLE_LIMIT, FULL_VAL_INTERVAL, PLOT_EVERY_N_EPOCHS,
)
from core.model import ResNetCRNN
from pipeline.dataset import GNHKDataset, IAMDataset, SyntheticDataset, collate_fn
from pipeline.preprocessing import gpu_augment
from core.decoding import ctc_greedy_decode_batch, ctc_beam_decode_batch, CharLM
from core.metrics import char_error_rate
from services.evaluation import plot_training_curves, generate_confusion_matrix
from torch.utils.data import WeightedRandomSampler, ConcatDataset


def _collect_training_texts(dataset):
    """Extract raw text labels from a (possibly concatenated) dataset."""
    texts = []
    if hasattr(dataset, 'cached'):
        texts = [text for _, _, text in dataset.cached]
    elif hasattr(dataset, 'datasets'):
        for ds in dataset.datasets:
            if hasattr(ds, 'cached'):
                texts.extend(text for _, _, text in ds.cached)
    return texts


def train():
    """Run the full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device} | AMP: {use_amp}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────────
    val_set = GNHKDataset(TEST_DIR)
    gnhk = GNHKDataset(TRAIN_DIR)
    iam = IAMDataset(IAM_DIR) if IAM_DIR and os.path.isdir(IAM_DIR) else None
    syn = SyntheticDataset(SYNTHETIC_DIR) if SYNTHETIC_DIR and os.path.isdir(SYNTHETIC_DIR) else None

    datasets, weights = [gnhk], [3.0] * len(gnhk)
    if iam and len(iam) > 0:
        datasets.append(iam)
        weights += [1.0] * len(iam)
    if syn and len(syn) > 0:
        datasets.append(syn)
        weights += [2.0] * len(syn)  # synthetic is cleaner than IAM, weight middle

    train_set = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)

    loader_kwargs = dict(
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=(device.type == "cuda"),
    )

    # shuffle must be False when using a sampler
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=sampler, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

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
    criterion = CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_cer = float("inf")
    patience_counter = 0

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_cer": []}

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # ── Training phase ────────────────────────────────────────
        model.train()
        total_train_loss = 0

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, (images, labels, label_lengths, _) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if (epoch + 1) >= AUGMENT_START_EPOCH:
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

        # ── Validation phase ──────────────────────────────────────
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

                decoded = ctc_beam_decode_batch(outputs, beam_width=5)
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
                print(f"  \u2713 Best model saved (CER: {best_cer:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  \u2717 Early stopping triggered (no improvement for {PATIENCE} full CER checks)")
                    break

        if (epoch + 1) % PLOT_EVERY_N_EPOCHS == 0 or (epoch + 1) == EPOCHS:
            plot_training_curves(history)

    # ── Post-training ─────────────────────────────────────────────────────────
    torch.save(model.state_dict(), FINAL_WEIGHTS)

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Build character language model from training labels
    train_texts = _collect_training_texts(train_set)
    if train_texts:
        lm = CharLM()
        lm.build_from_texts(train_texts)
        lm.save()
        print(f"Character LM built from {len(train_texts)} training texts")

    # Confusion matrix on best checkpoint
    if os.path.exists(BEST_WEIGHTS):
        model.load_state_dict(torch.load(BEST_WEIGHTS, map_location=device, weights_only=True))
        print("Loaded best checkpoint for confusion matrix generation")
    generate_confusion_matrix(model, val_loader, device, use_amp)

    print(f"Training complete. Final model saved to {FINAL_WEIGHTS}")
    print(f"Training history saved to {OUTPUT_DIR}/training_history.json")


if __name__ == "__main__":
    train()