"""
training.py — Training loop for the handwriting recogniser.

Trains one model per seed in ``config.SEEDS`` (saved to per-seed checkpoint
paths so inference can ensemble them). Late-stage epochs switch to a sampler
that oversamples high-loss training examples (hard-example mining).
"""

import os
import json
import math
import time
import random

import numpy as np
import torch
from torch.nn import CTCLoss
from torch.utils.data import WeightedRandomSampler

from config import (
    TRAIN_DIR, TEST_DIR, IAM_DIR, IMGUR5K_DIR,
    SYNTHETIC_SAMPLES,
    CHECKPOINT_DIR, OUTPUT_DIR,
    NUM_CLASSES, EPOCHS, LR, WEIGHT_DECAY,
    PATIENCE, ACCUMULATION_STEPS, GRAD_CLIP_NORM,
    BACKBONE_LR_MULT, ONECYCLE_PCT_START, VAL_CER_SAMPLE_LIMIT,
    FULL_VAL_INTERVAL, PLOT_EVERY_N_EPOCHS,
    SEEDS, HARD_MINE_START_FRAC,
    seed_weights_path,
)
from torch.utils.data import ConcatDataset
from core.model import ResNetCRNN
from pipeline.dataset import (
    GNHKDataset, IndexedDataset,
    build_weighted_train_set, build_dataloader,
)
from pipeline.preprocessing import gpu_augment
from core.decoding import ctc_greedy_decode_batch
from core.metrics import char_error_rate
from services.evaluation import plot_training_curves, generate_confusion_matrix


# ── Loss ─────────────────────────────────────────────────────────────────────

class SmoothedCTCLoss(torch.nn.Module):
    """CTC loss + label-smoothing on non-blank classes only."""

    def __init__(self, blank=0, smoothing=0.1):
        super().__init__()
        self.ctc = CTCLoss(blank=blank, zero_infinity=True)
        self.smoothing = smoothing
        self.blank = blank

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)
        non_blank = torch.cat(
            [log_probs[..., :self.blank], log_probs[..., self.blank + 1:]],
            dim=-1,
        )
        smooth_loss = -non_blank.mean()
        return (1 - self.smoothing) * ctc_loss + self.smoothing * smooth_loss


# ── Hard-example mining ──────────────────────────────────────────────────────

@torch.no_grad()
def _compute_per_sample_losses(model, train_set, device, use_amp):
    """One-shot per-sample CTC loss pass over the (indexed) training set.

    Used to weight a fresh sampler for the hard-mining phase. Runs at
    eval-mode without augmentation so the loss reflects "what the model
    still gets wrong" rather than augmentation-induced noise.
    """
    model.eval()
    raw_ctc = CTCLoss(blank=0, zero_infinity=True, reduction="none")

    # Sequential loader without sampler so every sample is touched exactly once.
    loader = build_dataloader(train_set, sampler=None, shuffle=False)

    losses = torch.zeros(len(train_set), dtype=torch.float32)
    seen = torch.zeros(len(train_set), dtype=torch.bool)

    for batch in loader:
        images, labels, label_lengths, _, indices = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            input_lengths = torch.full(
                (images.size(0),), outputs.size(0), dtype=torch.long
            )
            per_sample = raw_ctc(
                outputs.log_softmax(2), labels, input_lengths, label_lengths
            )

        losses[indices] = per_sample.detach().float().cpu()
        seen[indices] = True

    if not seen.all():
        # Defensive: any sample we didn't see gets the median, so it isn't
        # silently zero-weighted out of the next sampler.
        losses[~seen] = float(losses[seen].median())
    return losses.numpy()


def _build_hard_sampler(base_weights, sample_losses):
    """Combine source-bias weights with per-sample loss.

    Result: a sample's probability ∝ source_weight × max(loss, eps). Hard
    examples in the target domain (GNHK) get the largest amplification
    because their source weight is already highest.
    """
    eps = float(np.percentile(sample_losses, 10))  # don't zero out easy ones
    eps = max(eps, 0.01)
    weights = base_weights * np.maximum(sample_losses, eps)
    return WeightedRandomSampler(
        weights.tolist(), num_samples=len(weights), replacement=True
    )


# ── Per-seed training ────────────────────────────────────────────────────────

def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_one_seed(seed):
    print(f"\n{'═' * 64}\n  Training seed {seed}\n{'═' * 64}")
    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device} | AMP: {use_amp}")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # ── Data ─────────────────────────────────────────────────────────────
    raw_train_set, sampler, base_weights, iam_val = build_weighted_train_set(
        gnhk_dir=TRAIN_DIR, iam_dir=IAM_DIR, imgur5k_dir=IMGUR5K_DIR,
        synthetic_samples=SYNTHETIC_SAMPLES,
    )
    train_set = IndexedDataset(raw_train_set)

    gnhk_val = GNHKDataset(TEST_DIR)
    if iam_val is not None and len(iam_val) > 0:
        val_set = ConcatDataset([gnhk_val, iam_val])
        print(
            f"Train samples: {len(train_set)} | "
            f"Val samples: {len(val_set)} ({len(gnhk_val)} GNHK + {len(iam_val)} IAM)"
        )
    else:
        val_set = gnhk_val
        print(f"Train samples: {len(train_set)} | Val samples: {len(val_set)}")

    train_loader = build_dataloader(train_set, sampler=sampler)
    val_loader = build_dataloader(val_set)

    # ── Model & optimiser ────────────────────────────────────────────────
    model = ResNetCRNN(NUM_CLASSES).to(device)

    backbone_prefixes = ('conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4')
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in backbone_prefixes):
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': LR * BACKBONE_LR_MULT},
            {'params': head_params, 'lr': LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR * BACKBONE_LR_MULT, LR],
        epochs=EPOCHS,
        steps_per_epoch=math.ceil(len(train_loader) / ACCUMULATION_STEPS),
        pct_start=ONECYCLE_PCT_START,
        anneal_strategy="cos",
    )

    criterion = SmoothedCTCLoss(blank=0, smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_cer = float("inf")
    patience_counter = 0
    hard_mining_active = False
    hard_mine_epoch = (
        int(EPOCHS * HARD_MINE_START_FRAC) if HARD_MINE_START_FRAC else None
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    best_path = seed_weights_path(seed)

    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_cer": []}

    # ── Epoch loop ───────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        epoch_start = time.time()

        # Hard-example mining trigger: rebuild the sampler once.
        if (
            hard_mine_epoch is not None
            and not hard_mining_active
            and epoch == hard_mine_epoch
        ):
            print("  ⚙ Computing per-sample losses for hard-example mining...")
            sample_losses = _compute_per_sample_losses(
                model, train_set, device, use_amp
            )
            new_sampler = _build_hard_sampler(base_weights, sample_losses)
            train_loader = build_dataloader(train_set, sampler=new_sampler)
            hard_mining_active = True
            print(
                f"  ⚙ Hard mining active "
                f"(loss range: {sample_losses.min():.2f}–{sample_losses.max():.2f}, "
                f"mean {sample_losses.mean():.2f})"
            )

        # ── Training phase ──────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0

        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader):
            images, labels, label_lengths, _, _ = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = gpu_augment(images)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                input_lengths = torch.full(
                    (images.size(0),), outputs.size(0), dtype=torch.long
                )
                loss = criterion(
                    outputs.log_softmax(2), labels, input_lengths, label_lengths
                )
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            step_now = (
                (batch_idx + 1) % ACCUMULATION_STEPS == 0
                or (batch_idx + 1) == len(train_loader)
            )
            if step_now:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_train_loss += loss.item() * ACCUMULATION_STEPS

        # ── Validation phase ────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
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
                    loss = criterion(
                        outputs.log_softmax(2), labels, input_lengths, label_lengths
                    )
                total_val_loss += loss.item()

                if cer_budget is not None and total_words >= cer_budget:
                    continue

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
        hm_tag = " | hard-mine" if hard_mining_active else ""
        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Accuracy: {accuracy:.1f}% | "
            f"{cer_tag}: {avg_cer:.1f}% | "
            f"Time: {epoch_time:.1f}s{hm_tag}"
        )

        if evaluate_full_cer:
            if avg_cer < best_cer:
                best_cer = avg_cer
                patience_counter = 0
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best model saved (CER: {best_cer:.2f}%) → {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(
                        f"  ✗ Early stopping (no improvement for {PATIENCE} "
                        f"full CER checks)"
                    )
                    break

        if (epoch + 1) % PLOT_EVERY_N_EPOCHS == 0 or (epoch + 1) == EPOCHS:
            plot_training_curves(history)

    # ── Post-training ────────────────────────────────────────────────────
    history_path = os.path.join(OUTPUT_DIR, f"training_history_seed{seed}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if os.path.exists(best_path):
        model.load_state_dict(
            torch.load(best_path, map_location=device, weights_only=True)
        )
    generate_confusion_matrix(model, val_loader, device, use_amp)

    print(f"\nSeed {seed} complete | best CER {best_cer:.2f}% | weights → {best_path}")


# ── Public entry point ───────────────────────────────────────────────────────

def train():
    """Train every seed in ``config.SEEDS``. Per-seed checkpoints are
    auto-ensembled by the inference module."""
    print(f"Training {len(SEEDS)} seed(s): {SEEDS}")
    for seed in SEEDS:
        _train_one_seed(seed)
    print(f"\nAll seeds complete. Run `python main.py lm` next, then `serve`.")




if __name__ == "__main__":
    train()
