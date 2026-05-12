"""
inference.py — In-house CRNN inference + shared line/word segmentation.

Public entry points:
    predict(image)              — word-level CRNN, segments by word
    detect_text_lines(image)    — line bounding-box detector, also used
                                   by TrOCR backend (no recognition)

Internals:
    _preprocess_for_segmentation  binarize + strip non-text noise
    _detect_lines                 histogram-of-CC-centers line detector
    _segment_words                projection-based word splitter within a line
"""

import glob
import os
import cv2
import numpy as np
import torch
from PIL import Image

from config import (
    NUM_CLASSES, CHECKPOINT_DIR, SEEDS, seed_weights_path,
)
from core.model import ResNetCRNN
from pipeline.preprocessing import base_transform, tta_transforms
from core.decoding import (
    ctc_greedy_decode_with_confidence,
    ctc_beam_decode,
    CharLM,
)


# ── Model loading ────────────────────────────────────────────────────────────

def _discover_checkpoints():
    """Return every per-seed best-CER checkpoint that exists on disk.

    Multiple checkpoints get auto-ensembled at inference (mean of logits).
    """
    paths, seen = [], set()
    for seed in SEEDS:
        p = seed_weights_path(seed)
        if os.path.exists(p) and p not in seen:
            paths.append(p)
            seen.add(p)
    for p in sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "crnn_gnhk_seed*_best.pth"))):
        if p not in seen:
            paths.append(p)
            seen.add(p)
    return paths


def _load_model(weights_path):
    m = ResNetCRNN(NUM_CLASSES).to(device)
    state = torch.load(weights_path, map_location=device)
    m.load_state_dict(state)
    m.eval()
    return m


# ── Device & ensemble ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_checkpoints = _discover_checkpoints()
if _checkpoints:
    models = [_load_model(p) for p in _checkpoints]
    print(
        f"[predict] Loaded {len(models)} checkpoint(s) on {device}: "
        f"{[os.path.basename(p) for p in _checkpoints]}"
    )
else:
    models = [ResNetCRNN(NUM_CLASSES).to(device).eval()]
    print("[predict] WARNING: No weights found — predictions will be random")


def _forward_ensemble(x):
    """Average logits across all loaded checkpoints. Single-model case is a no-op."""
    with torch.no_grad():
        if len(models) == 1:
            return models[0](x)
        outs = [m(x) for m in models]
        return torch.stack(outs).mean(0)


# ── Language model ───────────────────────────────────────────────────────────
char_lm = CharLM()
char_lm.load()


# ── WORD SEGMENTATION (resolution-independent) ────────────────────────────────

def _upscale_if_small(pil_image, min_side=800):
    """Upscale tiny inputs so segmentation has enough pixels.

    Phone screenshots and downscaled uploads can be ~200 px on a side, which
    starves the morphology kernels and makes line/word thresholds unreliable.
    """
    w, h = pil_image.size
    short = min(w, h)
    if short >= min_side:
        return pil_image
    scale = min_side / short
    return pil_image.resize((int(w * scale), int(h * scale)), Image.BICUBIC)


def _strip_non_text_noise(binary):
    """Remove long horizontal rules (lined paper) and giant blobs (pens,
    watermarks, photos) that derail projection-based line detection.

    Without this, a single tall non-text object — e.g. a pen lying on the
    page — pollutes every row's horizontal projection, collapsing the entire
    image into one detected "line".
    """
    h, w = binary.shape

    # 1. Subtract long horizontal lines (paper rules). The kernel is wide
    #    enough to catch a horizontal rule but too long for any handwritten
    #    stroke to survive intact.
    kernel_w = max(w // 4, 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    binary = cv2.subtract(binary, h_lines)

    # 2. Drop connected components too big to plausibly be text. Cursive
    #    handwriting can chain a whole phrase into one CC (>50 % width is
    #    common), so the width cutoff has to be very loose. The height
    #    cutoff catches pens / photo borders / vertical decorations.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    max_cc_w = int(w * 0.85)
    max_cc_h = int(h * 0.4)
    for i in range(1, num_labels):
        if (stats[i, cv2.CC_STAT_WIDTH] > max_cc_w
                or stats[i, cv2.CC_STAT_HEIGHT] > max_cc_h):
            binary[labels == i] = 0

    return binary


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
    binary = _strip_non_text_noise(binary)
    return img, binary


def _detect_lines(binary):
    """Return list of (y0, y1) line spans via histogram-of-CC-centers.

    Why this over greedy-grouping: greedy grouping breaks when descenders
    from one line overlap with ascenders of the next line — they bridge in
    Y-range and merge into one giant group. Histogram-based detection is
    robust to overlap because each component VOTES at its own y-center.
    Real lines are dense votes at the same Y; noise is diffuse.

    Pipeline:
      1. Find all connected components, filter obvious noise
      2. Each CC contributes 1 vote at its y-center
      3. Smooth the vote histogram at ~half the typical line height
      4. Peaks above 30 % of max are line centers
      5. Assign each CC to its nearest peak; span = min/max of group bboxes
    """
    h_img = binary.shape[0]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    components = []
    for i in range(1, num_labels):
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        y = stats[i, cv2.CC_STAT_TOP]
        if area < 10 or ch < 5 or cw < 3:
            continue
        components.append((y, y + ch))

    if not components:
        return []

    cc_heights = sorted(c[1] - c[0] for c in components)
    # 75th-percentile CC height ≈ line height (capitals/descenders span
    # most of a line; the median is dragged down by lowercase letters).
    typical_h = max(cc_heights[(3 * len(cc_heights)) // 4], 15)

    # Build vote histogram of y-centers
    hist = np.zeros(h_img + 1, dtype=np.float32)
    for c in components:
        hist[int((c[0] + c[1]) / 2)] += 1

    smooth_window = max(int(typical_h * 0.4), 5)
    smooth_kernel = np.ones(smooth_window) / smooth_window
    hist_smooth = np.convolve(hist, smooth_kernel, mode="same")

    if hist_smooth.max() == 0:
        return []

    # Find local maxima then NMS. min_separation has to be roughly the
    # actual LINE SPACING, not character height. Two-pass approach:
    # (1) find candidates with loose NMS using typical_h as initial guess,
    # (2) estimate true line spacing as median gap between candidates and
    #     re-NMS with that. This adapts to whatever line spacing the image
    #     actually has without hard-coding it.
    threshold = hist_smooth.max() * 0.25

    candidates = []
    for y in range(1, len(hist_smooth) - 1):
        v = hist_smooth[y]
        if v >= threshold and v > hist_smooth[y - 1] and v >= hist_smooth[y + 1]:
            candidates.append((y, v))

    if not candidates:
        return []

    def _nms(cands, sep):
        cands.sort(key=lambda c: -c[1])
        kept = []
        for y, h in cands:
            if all(abs(y - s) >= sep for s, _ in kept):
                kept.append((y, h))
        return sorted(kept)

    # First pass: NMS at typical_h to deduplicate within-letter jitter
    pass1 = _nms(list(candidates), max(int(typical_h * 0.6), 20))
    if len(pass1) < 2:
        peaks = [y for y, _ in pass1]
    else:
        # Estimate line spacing from gaps between first-pass peaks. Gaps
        # are bimodal: small gaps = within-line jitter, large gaps = real
        # inter-line spacing. Use p75 to land on the larger cluster.
        gaps = sorted(pass1[i + 1][0] - pass1[i][0] for i in range(len(pass1) - 1))
        p75_gap = gaps[(3 * len(gaps)) // 4]
        line_spacing = max(p75_gap, typical_h * 1.2)
        final = _nms(list(candidates), int(line_spacing * 0.7))
        peaks = [y for y, _ in final]

    if not peaks:
        return []

    # Assign each CC to its nearest peak by y-center
    groups = [[] for _ in peaks]
    for c in components:
        c_center = (c[0] + c[1]) / 2
        best_idx = min(range(len(peaks)), key=lambda i: abs(peaks[i] - c_center))
        groups[best_idx].append(c)

    # Filter sparse groups: a real line of handwriting has many CCs (one
    # per letter); a decoration / icon / artifact typically has very few.
    # Drop any group with < 25 % of the average CC count.
    non_empty = [g for g in groups if g]
    if not non_empty:
        return []
    avg_cc_count = sum(len(g) for g in non_empty) / len(non_empty)
    min_cc_count = max(int(avg_cc_count * 0.25), 3)

    spans = []
    for group in non_empty:
        if len(group) < min_cc_count:
            continue
        y0 = min(c[0] for c in group)
        y1 = max(c[1] for c in group)
        if (y1 - y0) >= 15:
            spans.append((y0, y1))

    return spans


def detect_text_lines(pil_image):
    """Return list of (y0, y1) text-line spans for an image using the
    internal CC-based detector."""
    _, binary = _preprocess_for_segmentation(pil_image)
    return _detect_lines(binary)


def _segment_words(pil_image):
    """Segment a page image into per-word crops with absolute bboxes.

    Word-spacing thresholds scale with the detected line height instead of
    using fixed pixel constants — fixed values broke on photos at different
    resolutions (e.g. > 20 px gap required for a 1500 px-wide phone photo
    is barely a single space).

    Returns:
        list of (PIL crop, (x0, y0, x1, y1)) — bbox in source-image coords.
    """
    pil_image = _upscale_if_small(pil_image)
    _, binary = _preprocess_for_segmentation(pil_image)
    h, w = binary.shape

    lines = _detect_lines(binary)
    if not lines:
        return [(pil_image, (0, 0, pil_image.size[0], pil_image.size[1]))]

    crops = []
    for y0, y1 in lines:
        line_h = max(y1 - y0, 1)
        # Resolution-independent thresholds. word_gap_px must stay BELOW
        # the typical inter-word gap (~0.5*line_h) — too large and every
        # word in the line gets merged into one blob.
        word_gap_px = max(int(line_h * 0.25), 4)
        min_word_w  = max(int(line_h * 0.15), 3)

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
            outs.append(_forward_ensemble(x))
        output = torch.stack(outs).mean(0)
    else:
        x = base_transform(pil_image).unsqueeze(0).to(device)
        output = _forward_ensemble(x)

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


