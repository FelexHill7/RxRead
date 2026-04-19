"""
inference.py — Inference pipeline for the handwriting recogniser.

Loads the trained model once and exposes:
    - predict_pil(image)   — predict from a PIL Image (with TTA + LM rescoring)
    - predict_file(path)   — predict from an image file path

Delegates decoding to decoding.py, transforms to preprocessing.py,
and word segmentation to segmentation.py.
"""

import os

import cv2
import numpy as np
import torch
from PIL import Image

from config import NUM_CLASSES, BEST_WEIGHTS, FINAL_WEIGHTS
from core.model import ResNetCRNN
from pipeline.preprocessing import base_transform, tta_transforms
from core.decoding import ctc_greedy_decode_single, ctc_beam_decode, CharLM

# ── Device & model (loaded once on import) ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetCRNN(NUM_CLASSES).to(device)

_weights = BEST_WEIGHTS if os.path.exists(BEST_WEIGHTS) else FINAL_WEIGHTS
if os.path.exists(_weights):
    try:
        model.load_state_dict(torch.load(_weights, map_location=device))
        model.eval()
        print(f"[predict] Loaded weights from {_weights} on {device}")
    except RuntimeError:
        print(f"[predict] WARNING: Weights in {_weights} don't match current architecture — retrain the model")
else:
    print("[predict] WARNING: No weights found — run train.py first")

# ── Language model (loaded once) ──────────────────────────────────────────────
char_lm = CharLM()
if char_lm.load():
    print("[predict] Character LM loaded for beam search rescoring")


# ── Word segmentation ─────────────────────────────────────────────────────────

def _remove_ruled_lines(binary, img_w, img_h):
    """Remove horizontal and vertical ruled lines without destroying handwriting.

    Uses a narrow kernel so only true long straight lines are removed,
    not handwriting strokes.
    """
    # Only remove lines that span at least 60% of the image width/height
    # This avoids wiping out handwriting strokes
    h_kernel_len = max(img_w // 5, 40)
    v_kernel_len = max(img_h // 5, 40)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Dilate detected lines slightly before subtracting to clean edges
    h_lines = cv2.dilate(h_lines, np.ones((2, 1), np.uint8), iterations=1)
    v_lines = cv2.dilate(v_lines, np.ones((1, 2), np.uint8), iterations=1)

    cleaned = cv2.subtract(binary, h_lines)
    cleaned = cv2.subtract(cleaned, v_lines)
    return cleaned


def _segment_words(pil_image):
    """Segment a full-page image into individual word crops.

    Strategy: detect text lines via horizontal projection profile,
    then split each line into words using vertical projection gaps.
    """
    img = np.array(pil_image.convert("L"))
    img_h, img_w = img.shape

    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.medianBlur(binary, 3)

    # Remove ruled lines safely
    binary = _remove_ruled_lines(binary, img_w, img_h)

    # Step 1: Find text lines via horizontal projection
    h_proj = binary.sum(axis=1) / 255
    text_thresh = img_w * 0.005  # lowered from 0.01 — more sensitive
    line_regions = []
    in_line, start = False, 0
    for y in range(img_h):
        if h_proj[y] > text_thresh and not in_line:
            start, in_line = y, True
        elif h_proj[y] <= text_thresh and in_line:
            if y - start > img_h * 0.005:  # lowered min line height
                line_regions.append((start, y))
            in_line = False
    if in_line and img_h - start > img_h * 0.005:
        line_regions.append((start, img_h))

    if not line_regions:
        # Fallback: return the whole image as one crop
        return [pil_image]

    # Merge close lines
    merged_lines = [line_regions[0]]
    for start, end in line_regions[1:]:
        prev_start, prev_end = merged_lines[-1]
        if start - prev_end < (prev_end - prev_start) * 0.5:
            merged_lines[-1] = (prev_start, end)
        else:
            merged_lines.append((start, end))

    # Step 2: Split each line into words via vertical projection
    crops = []
    for line_y0, line_y1 in merged_lines:
        pad_y = max(2, int((line_y1 - line_y0) * 0.2))
        ly0 = max(0, line_y0 - pad_y)
        ly1 = min(img_h, line_y1 + pad_y)

        v_proj = binary[line_y0:line_y1, :].sum(axis=0) / 255
        word_thresh = max(1, (line_y1 - line_y0) * 0.01)  # more sensitive
        word_regions = []
        in_word, wx_start = False, 0
        for x in range(img_w):
            if v_proj[x] > word_thresh and not in_word:
                wx_start, in_word = x, True
            elif v_proj[x] <= word_thresh and in_word:
                if x - wx_start > img_w * 0.005:
                    word_regions.append((wx_start, x))
                in_word = False
        if in_word and img_w - wx_start > img_w * 0.005:
            word_regions.append((wx_start, img_w))

        if word_regions:
            merged_words = [word_regions[0]]
            gap_thresh = (line_y1 - line_y0) * 0.6
            for wx0, wx1 in word_regions[1:]:
                prev_x0, prev_x1 = merged_words[-1]
                if wx0 - prev_x1 < gap_thresh:
                    merged_words[-1] = (prev_x0, wx1)
                else:
                    merged_words.append((wx0, wx1))

            for wx0, wx1 in merged_words:
                pad_x = max(3, int((wx1 - wx0) * 0.05))
                cx0, cx1 = max(0, wx0 - pad_x), min(img_w, wx1 + pad_x)
                crop = pil_image.crop((cx0, ly0, cx1, ly1))
                if crop.size[0] > 10 and crop.size[1] > 8:
                    crops.append(crop)

    return crops if crops else [pil_image]


# ── Prediction functions ──────────────────────────────────────────────────────

def _decode(output, use_beam, beam_width, lm_weight):
    """Shared decoding wrapper for greedy vs beam search."""
    if use_beam:
        lm = char_lm if char_lm.loaded and lm_weight > 0 else None
        return ctc_beam_decode(
            output,
            beam_width=beam_width,
            lm_weight=lm_weight,
            lm=lm
        )
    else:
        return ctc_greedy_decode_single(output)
    
    
def _predict_single(pil_image, use_beam=True, beam_width=10, use_tta=True, lm_weight=0.3):
    if use_tta:
        all_outputs = []
        for t in tta_transforms:
            tensor = t(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)
            all_outputs.append(out)
        output = torch.stack(all_outputs, dim=0).mean(0)
    else:
        tensor = base_transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)

    output = output[:, 0:1, :]  # enforce shape (T,1,C)

    return _decode(
        output,
        use_beam=use_beam,
        beam_width=beam_width,
        lm_weight=lm_weight
    )


def predict_pil(pil_image, use_beam=True, beam_width=10, use_tta=True, lm_weight=0.3):
    """Run prediction on a PIL Image and return decoded text.

    If the image is large (likely a full page), segment it into word crops
    first and predict each one.
    """
    w, h = pil_image.size
    crops = _segment_words(pil_image) if (w > 300 or h > 150) else [pil_image]

    words = []
    for crop in crops:
        text = _predict_single(crop, use_beam, beam_width, use_tta, lm_weight)
        if text.strip():
            words.append(text.strip())

    return " ".join(words) if words else ""


def predict_file(image_path, use_beam=True, beam_width=10):
    """Read an image from disk and return decoded text."""
    image = Image.open(image_path).convert("RGB")
    return predict_pil(image, use_beam=use_beam, beam_width=beam_width)