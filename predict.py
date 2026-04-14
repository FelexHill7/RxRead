"""
predict.py — Shared prediction module for the CRNN handwriting recogniser.

Loads the trained model once and exposes:
    - decode(output)            — CTC greedy decode on raw model output
    - beam_decode(output)       — CTC beam search decode (higher accuracy)
    - predict_pil(image)        — predict from a PIL Image (with TTA + LM rescoring)
    - predict_file(path)        — predict from an image file path

Used by both app.py (Flask web UI) and inference.py (CLI).
"""

import os
import json
import math
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from model import CRNN
from dataset import CHARS, idx2char, char2idx, IMG_HEIGHT, IMG_WIDTH

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_CLASSES = len(CHARS) + 1          # +1 for CTC blank
BEST_WEIGHTS = "checkpoints/crnn_gnhk_best.pth"  # saved by train.py (best val loss)
FINAL_WEIGHTS = "checkpoints/crnn_gnhk.pth"      # saved by train.py (final epoch)

# ── Device & model (loaded once on import) ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CRNN(NUM_CLASSES).to(device)

import os
_weights = BEST_WEIGHTS if os.path.exists(BEST_WEIGHTS) else FINAL_WEIGHTS
if os.path.exists(_weights):
    model.load_state_dict(torch.load(_weights, map_location=device))
    model.eval()
    print(f"[predict] Loaded weights from {_weights} on {device}")
else:
    print(f"[predict] WARNING: No weights found — run train.py first")

# ── Preprocessing (must match training) ───────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def decode(output):
    """CTC greedy decode — pick the most likely class at each timestep,
    collapse consecutive duplicates, and strip the CTC blank (index 0)."""
    output = output.argmax(2).squeeze(1).tolist()
    result, prev = [], None
    for idx in output:
        if idx != prev and idx != 0:
            result.append(idx2char.get(idx, ""))
        prev = idx
    return "".join(result)


def beam_decode(output, beam_width=10, lm_weight=0.0, lm=None):
    """CTC beam search decode — explores multiple hypotheses at each timestep.

    Typically 2-5% more accurate than greedy decode. With a language model,
    the beam search can rescore hypotheses based on character-level plausibility.

    Args:
        output: (T, 1, C) raw model logits for a single sample.
        beam_width: number of hypotheses to keep at each step.
        lm_weight: weight for the language model score (0 = no LM).
        lm: CharLM instance for rescoring (optional).
    Returns:
        Best decoded string.
    """
    log_probs = output.squeeze(1).log_softmax(dim=-1).cpu()  # (T, C)
    T, C = log_probs.shape

    # Each beam: (log_prob, prefix_text, last_char_idx)
    beams = [(0.0, [], 0)]  # start with empty prefix, last=blank

    for t in range(T):
        new_beams = {}  # key: (tuple(text), last_char) -> log_prob
        lp = log_probs[t]

        for score, text, last_idx in beams:
            for c in range(C):
                c_lp = lp[c].item()
                new_score = score + c_lp

                if c == 0:
                    # Blank: keep text unchanged, update last to blank
                    key = (tuple(text), 0)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
                elif c == last_idx:
                    # Same as last non-blank: collapse (don't add new char)
                    key = (tuple(text), c)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
                else:
                    # New character: extend prefix
                    new_text = text + [c]
                    key = (tuple(new_text), c)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

        # Keep top-k beams
        sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(score, list(key[0]), key[1]) for key, score in sorted_beams]

    # Rescore with language model if available
    if lm and lm_weight > 0:
        rescored = []
        for score, text_indices, last_idx in beams:
            text = "".join(idx2char.get(idx, "") for idx in text_indices)
            lm_score = lm.score(text)
            combined = score + lm_weight * lm_score
            rescored.append((combined, text_indices))
        rescored.sort(key=lambda x: x[0], reverse=True)
        best_text_indices = rescored[0][1]
    else:
        best_text_indices = beams[0][1]

    return "".join(idx2char.get(idx, "") for idx in best_text_indices)


# ── Character-level language model ────────────────────────────────────────────
class CharLM:
    """Character bigram language model built from training data labels.

    Scores text by summing log-probabilities of character bigrams:
        score("abc") = log P(a|<s>) + log P(b|a) + log P(c|b)

    Built from the training set transcriptions. Provides a soft prior that
    nudges beam search toward plausible character sequences.
    """

    LM_PATH = "checkpoints/char_lm.json"

    def __init__(self):
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.loaded = False

    def build_from_texts(self, texts):
        """Build bigram counts from a list of training transcriptions."""
        counts = defaultdict(lambda: defaultdict(int))
        for text in texts:
            padded = "^" + text + "$"  # start/end tokens
            for a, b in zip(padded[:-1], padded[1:]):
                counts[a][b] += 1

        # Convert to log-probabilities with add-1 smoothing
        vocab_size = len(CHARS) + 2  # +2 for ^ and $
        self.bigrams = {}
        for a, nexts in counts.items():
            total = sum(nexts.values()) + vocab_size
            self.bigrams[a] = {b: math.log((c + 1) / total) for b, c in nexts.items()}
            self.bigrams[a]["_default"] = math.log(1 / total)

        self.loaded = True

    def save(self):
        """Save bigram model to disk."""
        os.makedirs(os.path.dirname(self.LM_PATH), exist_ok=True)
        with open(self.LM_PATH, "w") as f:
            json.dump(self.bigrams, f)

    def load(self):
        """Load bigram model from disk if it exists."""
        if os.path.exists(self.LM_PATH):
            with open(self.LM_PATH) as f:
                self.bigrams = json.load(f)
            self.loaded = True
        return self.loaded

    def score(self, text):
        """Score a text string using character bigram log-probabilities."""
        if not self.loaded or not text:
            return 0.0
        padded = "^" + text + "$"
        total = 0.0
        for a, b in zip(padded[:-1], padded[1:]):
            if a in self.bigrams:
                total += self.bigrams[a].get(b, self.bigrams[a].get("_default", -10.0))
            else:
                total += -10.0
        return total


# ── Load character LM if available ────────────────────────────────────────────
char_lm = CharLM()
_lm_loaded = char_lm.load()
if _lm_loaded:
    print("[predict] Character LM loaded for beam search rescoring")

# ── Word segmentation ─────────────────────────────────────────────────────────
def segment_words(pil_image):
    """Segment a full-page image into individual word crops.

    Strategy: detect text lines first via horizontal projection profile,
    then split each line into words using vertical projection gaps.
    This is much more robust than contour-based approaches on ruled paper.

    Returns a list of PIL Image crops.
    """
    img = np.array(pil_image.convert("L"))  # grayscale
    img_h, img_w = img.shape

    # Binarize: Otsu threshold (works well for document images)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove long horizontal lines (ruled notebook paper)
    h_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_w // 3, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_line_kernel)
    binary = cv2.subtract(binary, h_lines)

    # Remove long vertical lines (margins)
    v_line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_h // 3))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_line_kernel)
    binary = cv2.subtract(binary, v_lines)

    # Clean noise
    binary = cv2.medianBlur(binary, 3)

    # ── Step 1: Find text lines via horizontal projection ─────────────────
    # Sum ink pixels per row
    h_proj = binary.sum(axis=1) / 255

    # Threshold: a row has text if it has more than 1% of width in ink
    text_thresh = img_w * 0.01
    line_regions = []
    in_line = False
    start = 0
    for y in range(img_h):
        if h_proj[y] > text_thresh and not in_line:
            start = y
            in_line = True
        elif h_proj[y] <= text_thresh and in_line:
            if y - start > img_h * 0.008:  # minimum line height
                line_regions.append((start, y))
            in_line = False
    if in_line and img_h - start > img_h * 0.008:
        line_regions.append((start, img_h))

    if not line_regions:
        return [pil_image]

    # Merge lines that are very close (e.g. ascenders/descenders split a line)
    merged_lines = [line_regions[0]]
    for start, end in line_regions[1:]:
        prev_start, prev_end = merged_lines[-1]
        gap = start - prev_end
        prev_height = prev_end - prev_start
        if gap < prev_height * 0.3:
            merged_lines[-1] = (prev_start, end)
        else:
            merged_lines.append((start, end))

    # ── Step 2: Split each line into words via vertical projection ────────
    crops = []
    for line_y0, line_y1 in merged_lines:
        # Add vertical padding
        pad_y = max(2, int((line_y1 - line_y0) * 0.15))
        ly0 = max(0, line_y0 - pad_y)
        ly1 = min(img_h, line_y1 + pad_y)

        line_strip = binary[line_y0:line_y1, :]
        v_proj = line_strip.sum(axis=0) / 255

        # Find word regions: contiguous columns with ink
        word_thresh = (line_y1 - line_y0) * 0.02
        word_regions = []
        in_word = False
        wx_start = 0
        for x in range(img_w):
            if v_proj[x] > word_thresh and not in_word:
                wx_start = x
                in_word = True
            elif v_proj[x] <= word_thresh and in_word:
                if x - wx_start > img_w * 0.008:  # minimum word width
                    word_regions.append((wx_start, x))
                in_word = False
        if in_word and img_w - wx_start > img_w * 0.008:
            word_regions.append((wx_start, img_w))

        # Merge word segments that are very close (same word, thin gap)
        if word_regions:
            merged_words = [word_regions[0]]
            line_height = line_y1 - line_y0
            # Gap threshold: less than ~60% of average line height = same word
            gap_thresh = line_height * 0.6
            for wx0, wx1 in word_regions[1:]:
                prev_x0, prev_x1 = merged_words[-1]
                if wx0 - prev_x1 < gap_thresh:
                    merged_words[-1] = (prev_x0, wx1)
                else:
                    merged_words.append((wx0, wx1))

            for wx0, wx1 in merged_words:
                pad_x = max(3, int((wx1 - wx0) * 0.05))
                cx0 = max(0, wx0 - pad_x)
                cx1 = min(img_w, wx1 + pad_x)
                crop = pil_image.crop((cx0, ly0, cx1, ly1))
                # Skip tiny crops that are likely noise
                if crop.size[0] > 15 and crop.size[1] > 10:
                    crops.append(crop)

    return crops if crops else [pil_image]


# ── TTA transforms ────────────────────────────────────────────────────────────
# Light augmentations for test-time augmentation: slightly different perspectives
# of the same image to reduce prediction variance.
_tta_transforms = [
    # Original (identity)
    transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    # Slight rotation left
    transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(degrees=(-3, -3)),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    # Slight rotation right
    transforms.Compose([
        transforms.Grayscale(),
        transforms.RandomRotation(degrees=(3, 3)),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    # Slight scale up
    transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((int(IMG_HEIGHT * 1.1), int(IMG_WIDTH * 1.1))),
        transforms.CenterCrop((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    # Slight scale down
    transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((int(IMG_HEIGHT * 0.9), int(IMG_WIDTH * 0.9))),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
]


def predict_pil(pil_image, use_beam=True, beam_width=10, use_tta=True, lm_weight=0.3):
    """Run prediction on a PIL Image and return decoded text.

    If the image is large (likely a full page), segment it into word crops
    first and predict each one. Otherwise treat it as a single word crop.

    Args:
        pil_image: PIL Image to recognize.
        use_beam: If True, use beam search (more accurate). If False, greedy.
        beam_width: Number of beams for beam search.
        use_tta: If True, run test-time augmentation (5 views, averaged logits).
        lm_weight: Weight for character LM rescoring in beam search.
    """
    # Segment into word crops if the image is large enough to be a full page
    w, h = pil_image.size
    if w > 300 or h > 150:
        crops = segment_words(pil_image)
    else:
        crops = [pil_image]

    # Predict each crop and join
    words = []
    for crop in crops:
        text = _predict_single(crop, use_beam, beam_width, use_tta, lm_weight)
        if text.strip():
            words.append(text.strip())

    return " ".join(words) if words else ""


def _predict_single(pil_image, use_beam=True, beam_width=10, use_tta=True, lm_weight=0.3):
    """Predict a single word crop."""
    if use_tta:
        # Run inference on multiple augmented views, average the log-probs
        all_outputs = []
        for t in _tta_transforms:
            tensor = t(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(tensor)  # (T, 1, C)
            all_outputs.append(out.log_softmax(2))
        output = torch.stack(all_outputs).mean(0)  # averaged log-probs
        # Convert back to logits for decode functions
        output = output  # already in log-prob space, beam_decode handles this
    else:
        tensor = transform(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)

    if use_beam:
        lm = char_lm if char_lm.loaded and lm_weight > 0 else None
        return beam_decode(output, beam_width=beam_width, lm_weight=lm_weight, lm=lm)
    return decode(output)


def predict_file(image_path, use_beam=True, beam_width=10):
    """Read an image from disk and return decoded text."""
    image = Image.open(image_path).convert("RGB")
    return predict_pil(image, use_beam=use_beam, beam_width=beam_width)
