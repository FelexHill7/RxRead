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

    Args:
        pil_image: PIL Image to recognize.
        use_beam: If True, use beam search (more accurate). If False, greedy.
        beam_width: Number of beams for beam search.
        use_tta: If True, run test-time augmentation (5 views, averaged logits).
        lm_weight: Weight for character LM rescoring in beam search.
    """
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
