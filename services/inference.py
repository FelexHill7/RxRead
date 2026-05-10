"""
inference.py — Inference pipeline for handwriting recognition.

Returns structured per-word output: each word carries text, bounding box, and
confidence so the front-end can colour-code low-confidence predictions.

Key fixes vs prior version:
- CLAHE before Otsu so shadows / uneven lighting don't break thresholding
- Word-segmentation thresholds are relative to detected line height instead
  of absolute pixels (works across different image resolutions)
- TTA averaging rotates AFTER resize (preprocessing.py) so all branches
  see the same effective scale
- LM weight default lowered from 0.7 → 0.4 (a bigram LM at 0.7 overcorrects
  rare-but-correct tokens like drug names)
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image

from config import NUM_CLASSES, BEST_WEIGHTS, FINAL_WEIGHTS
from core.model import ResNetCRNN
from pipeline.preprocessing import base_transform, tta_transforms
from core.decoding import (
    ctc_greedy_decode_with_confidence,
    ctc_beam_decode,
    CharLM,
)

# ── Device & model ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetCRNN(NUM_CLASSES).to(device)

_weights = BEST_WEIGHTS if os.path.exists(BEST_WEIGHTS) else FINAL_WEIGHTS

if os.path.exists(_weights):
    model.load_state_dict(torch.load(_weights, map_location=device))
    model.eval()
    print(f"[predict] Loaded weights: {_weights} on {device}")
else:
    print("[predict] WARNING: No weights found")

# ── Language model ────────────────────────────────────────────────────────────
char_lm = CharLM()
char_lm.load()


# ── WORD SEGMENTATION (resolution-independent) ────────────────────────────────

def _preprocess_for_segmentation(pil_image):
    """Convert to grayscale numpy, normalise contrast, return (gray, binary)."""
    img = np.array(pil_image.convert("L"))

    # CLAHE handles shadows / uneven page lighting that crush Otsu.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    _, binary = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    binary = cv2.medianBlur(binary, 3)
    return img, binary


def _detect_lines(binary):
    """Return list of (y0, y1) line spans via horizontal projection."""
    h = binary.shape[0]
    h_proj = np.sum(binary, axis=1)
    if h_proj.max() == 0:
        return []

    line_thresh = h_proj.max() * 0.15
    lines = []
    in_line, start = False, 0
    for y in range(h):
        if h_proj[y] > line_thresh and not in_line:
            start, in_line = y, True
        elif h_proj[y] <= line_thresh and in_line:
            if y - start > 10:
                lines.append((start, y))
            in_line = False
    if in_line:
        lines.append((start, h))
    return lines


def _segment_words(pil_image):
    """Segment a page image into per-word crops with absolute bboxes.

    Word-spacing thresholds scale with the detected line height instead of
    using fixed pixel constants — fixed values broke on photos at different
    resolutions (e.g. > 20 px gap required for a 1500 px-wide phone photo
    is barely a single space).

    Returns:
        list of (PIL crop, (x0, y0, x1, y1)) — bbox in source-image coords.
    """
    _, binary = _preprocess_for_segmentation(pil_image)
    h, w = binary.shape

    lines = _detect_lines(binary)
    if not lines:
        return [(pil_image, (0, 0, pil_image.size[0], pil_image.size[1]))]

    crops = []
    for y0, y1 in lines:
        line_h = max(y1 - y0, 1)
        # Resolution-independent thresholds:
        word_gap_px = max(int(line_h * 0.6), 6)
        min_word_w  = max(int(line_h * 0.3), 4)

        row = binary[y0:y1, :]
        v_proj = np.sum(row, axis=0)

        col_thresh = v_proj.max() * 0.05 if v_proj.max() > 0 else 0
        gaps = v_proj <= col_thresh

        words = []
        in_word, start_x = False, 0
        for x in range(w):
            if not gaps[x] and not in_word:
                start_x, in_word = x, True
            elif gaps[x] and in_word:
                if x - start_x >= min_word_w:
                    words.append([start_x, x])
                in_word = False
        if in_word:
            words.append([start_x, w])

        # Merge spans separated by less than word_gap_px (these are intra-word gaps).
        merged = []
        for wx0, wx1 in words:
            if merged and wx0 - merged[-1][1] < word_gap_px:
                merged[-1][1] = wx1
            else:
                merged.append([wx0, wx1])

        for wx0, wx1 in merged:
            crop = pil_image.crop((wx0, y0, wx1, y1))
            if crop.size[0] > 12 and crop.size[1] > 12:
                crops.append((crop, (wx0, y0, wx1, y1)))

    if not crops:
        return [(pil_image, (0, 0, pil_image.size[0], pil_image.size[1]))]
    return crops


# ── DECODING ──────────────────────────────────────────────────────────────────

def _decode_with_confidence(output, use_beam, beam_width, lm_weight):
    """Return (text, confidence). Beam path uses the same confidence proxy
    via greedy on the same averaged logits — bounded at [0, 1] and stable."""
    text_g, conf = ctc_greedy_decode_with_confidence(output)
    if not use_beam:
        return text_g, conf

    lm = char_lm if char_lm.loaded and lm_weight > 0 else None
    text_b = ctc_beam_decode(output, beam_width=beam_width,
                             lm_weight=lm_weight, lm=lm)
    # Confidence stays the greedy estimate — it's the model's own certainty,
    # independent of which decoder picked the final string.
    return text_b, conf


# ── SINGLE PREDICTION ─────────────────────────────────────────────────────────

def _predict_single(pil_image, use_beam=True, beam_width=15,
                    use_tta=True, lm_weight=0.4):
    if use_tta:
        outs = []
        for t in tta_transforms:
            x = t(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outs.append(model(x))
        output = torch.stack(outs).mean(0)
    else:
        x = base_transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(x)

    output = output[:, 0:1, :]
    return _decode_with_confidence(output, use_beam, beam_width, lm_weight)


# ── FULL PIPELINE ─────────────────────────────────────────────────────────────

def predict(image, use_beam=True, beam_width=15,
            use_tta=True, lm_weight=0.4):
    """Run segmentation + recognition and return per-word output.

    Args:
        image: PIL.Image or str path to an image file.

    Returns:
        dict with:
            text:       joined transcription
            confidence: mean confidence across non-empty words [0, 1]
            words:      list of {text, confidence, bbox: [x0,y0,x1,y1]}
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    crops = _segment_words(image)

    words = []
    for crop, bbox in crops:
        text, conf = _predict_single(crop, use_beam, beam_width,
                                     use_tta, lm_weight)
        text = text.strip()
        if not text:
            continue
        words.append({
            "text": text,
            "confidence": round(float(conf), 3),
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
        })

    avg_conf = sum(w["confidence"] for w in words) / len(words) if words else 0.0
    return {
        "text": " ".join(w["text"] for w in words),
        "confidence": round(float(avg_conf), 3),
        "words": words,
    }
