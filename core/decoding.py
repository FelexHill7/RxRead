"""
decoding.py — CTC decoding strategies and character language model.

Provides:
  - ctc_greedy_decode()      — fast argmax decode for batches (training eval)
  - ctc_greedy_decode_single — greedy decode for a single sample (inference)
  - ctc_beam_decode()        — beam search with optional LM rescoring (inference)
  - CharLM                   — character bigram language model
"""

import os
import json
import math
from collections import defaultdict

from config import CHARS, IDX2CHAR, CHAR_LM_PATH


# ── Greedy CTC decoding ──────────────────────────────────────────────────────

def ctc_greedy_decode_batch(output):
    """Greedy CTC decode for a full batch.

    Args:
        output: (T, B, C) raw model logits.
    Returns:
        List of decoded strings, one per sample in the batch.
    """
    preds = output.argmax(2)  # (T, B)
    batch_texts = []
    for b in range(preds.size(1)):
        seq = preds[:, b].tolist()
        chars, prev = [], None
        for idx in seq:
            if idx != prev and idx != 0:
                chars.append(IDX2CHAR.get(idx, ""))
            prev = idx
        batch_texts.append("".join(chars))
    return batch_texts


def ctc_greedy_decode_single(output):
    """Greedy CTC decode for a single sample.

    Args:
        output: (T, 1, C) or (T, C) model logits for one sample.
    Returns:
        Decoded string.
    """
    if output.dim() == 3:
        output = output.squeeze(1)
    seq = output.argmax(1).tolist()
    result, prev = [], None
    for idx in seq:
        if idx != prev and idx != 0:
            result.append(IDX2CHAR.get(idx, ""))
        prev = idx
    return "".join(result)


def ctc_beam_decode(output, beam_width=10, lm_weight=0.0, lm=None):
    """CTC beam search decode — explores multiple hypotheses at each timestep.

    Args:
        output: (T, 1, C) raw model logits for a single sample.
        beam_width: Number of hypotheses to keep at each step.
        lm_weight: Weight for the language model score (0 = no LM).
        lm: CharLM instance for rescoring (optional).
    Returns:
        Best decoded string.
    """
    log_probs = output.squeeze(1).log_softmax(dim=-1).cpu()
    T, C = log_probs.shape

    beams = [(0.0, [], 0)]

    for t in range(T):
        new_beams = {}
        lp = log_probs[t]

        for score, text, last_idx in beams:
            for c in range(C):
                c_lp = lp[c].item()
                new_score = score + c_lp

                if c == 0:
                    key = (tuple(text), 0)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
                elif c == last_idx:
                    key = (tuple(text), c)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score
                else:
                    new_text = text + [c]
                    key = (tuple(new_text), c)
                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

        sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(score, list(key[0]), key[1]) for key, score in sorted_beams]

    if lm and lm_weight > 0:
        rescored = []
        for score, text_indices, last_idx in beams:
            text = "".join(IDX2CHAR.get(idx, "") for idx in text_indices)
            lm_score = lm.score(text)
            combined = score + lm_weight * lm_score
            rescored.append((combined, text_indices))
        rescored.sort(key=lambda x: x[0], reverse=True)
        best_text_indices = rescored[0][1]
    else:
        best_text_indices = beams[0][1]

    return "".join(IDX2CHAR.get(idx, "") for idx in best_text_indices)


def ctc_beam_decode_batch(outputs, beam_width=5, lm_weight=0.0, lm=None):
    """Batch wrapper around ctc_beam_decode for use in the validation loop.
 
    Args:
        outputs: (T, B, C) raw model logits.
        beam_width: Number of hypotheses to keep at each step.
        lm_weight: Weight for the language model score (0 = no LM).
        lm: CharLM instance for rescoring (optional).
    Returns:
        List of decoded strings, one per sample in the batch.
    """
    results = []
    for b in range(outputs.size(1)):
        single = outputs[:, b:b+1, :]  # (T, 1, C)
        results.append(ctc_beam_decode(single, beam_width=beam_width, lm_weight=lm_weight, lm=lm))
    return results

# ── Character-level language model ────────────────────────────────────────────

class CharLM:
    """Character bigram language model built from training data labels.

    Scores text by summing log-probabilities of character bigrams:
        score("abc") = log P(a|<s>) + log P(b|a) + log P(c|b)

    Provides a soft prior that nudges beam search toward plausible
    character sequences.
    """

    def __init__(self, path=CHAR_LM_PATH):
        self.path = path
        self.bigrams = defaultdict(lambda: defaultdict(float))
        self.loaded = False

    def build_from_texts(self, texts):
        """Build bigram counts from a list of training transcriptions."""
        counts = defaultdict(lambda: defaultdict(int))
        for text in texts:
            padded = "^" + text + "$"
            for a, b in zip(padded[:-1], padded[1:]):
                counts[a][b] += 1

        vocab_size = len(CHARS) + 2
        self.bigrams = {}
        for a, nexts in counts.items():
            total = sum(nexts.values()) + vocab_size
            self.bigrams[a] = {b: math.log((c + 1) / total) for b, c in nexts.items()}
            self.bigrams[a]["_default"] = math.log(1 / total)

        self.loaded = True

    def save(self):
        """Save bigram model to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.bigrams, f)

    def load(self):
        """Load bigram model from disk if it exists."""
        if os.path.exists(self.path):
            with open(self.path) as f:
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
