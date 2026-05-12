"""
decoding.py — CTC decoding strategies and character language model.

Provides:
  - ctc_greedy_decode_batch()           — fast argmax decode (training eval)
  - ctc_greedy_decode_with_confidence() — single-sample greedy + per-word
                                          confidence (inference)
  - ctc_beam_decode()                   — beam search with LM rescoring
  - ctc_beam_decode_batch()             — batch wrapper for beam decode
  - CharLM                              — character bigram language model
"""

import os
import json
import math
from collections import defaultdict

from config import CHARS, IDX2CHAR, CHAR_LM_PATH


# ── Greedy CTC decoding ───────────────────────────────────────────────────────

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


def ctc_greedy_decode_with_confidence(output):
    """Greedy CTC decode that also returns a per-sample confidence score.

    Confidence = mean softmax probability of the argmax class on non-blank
    frames (i.e. the frames that actually contributed characters). Restricting
    to non-blank frames stops long padded blank regions from inflating the score.

    Args:
        output: (T, 1, C) or (T, C) model logits for one sample.
    Returns:
        (decoded_string, confidence in [0, 1]).
    """
    if output.dim() == 3:
        output = output.squeeze(1)
    probs = output.softmax(dim=-1)
    max_probs, idxs = probs.max(dim=-1)

    seq = idxs.tolist()
    result, prev = [], None
    contributing = []
    for t, idx in enumerate(seq):
        if idx != prev and idx != 0:
            result.append(IDX2CHAR.get(idx, ""))
            contributing.append(float(max_probs[t]))
        prev = idx

    if contributing:
        confidence = sum(contributing) / len(contributing)
    else:
        confidence = 0.0
    return "".join(result), confidence


# ── Beam search CTC decoding ──────────────────────────────────────────────────

def _log_add(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    NEG_INF = float("-inf")
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def ctc_beam_decode(output, beam_width=10, lm_weight=0.3, lm=None):
    """CTC beam search with correct prefix merging and optional per-step LM pruning.

    Maintains separate blank-ending (p_b) and nonblank-ending (p_nb) probabilities
    per prefix, which is required by the CTC algorithm for correct path merging.
    Without this, paths that produce the same text via different blank placements
    are never merged, causing scores to be wrong and garbage output.

    LM influences beam pruning at every timestep (not just final reranking),
    so high-LM-score paths are not pruned before they reach the end.

    Args:
        output:     (T, 1, C) raw model logits for a single sample.
        beam_width: Number of hypotheses to keep at each timestep.
        lm_weight:  Weight applied to LM score during pruning (0 = no LM).
        lm:         CharLM instance (optional).
    Returns:
        Best decoded string.
    """
    NEG_INF = float("-inf")

    log_probs = output.squeeze(1).log_softmax(dim=-1).cpu().numpy()
    T, C = log_probs.shape

    # beams: prefix_tuple -> (log_prob_ending_in_blank, log_prob_ending_in_nonblank)
    beams = {(): (0.0, NEG_INF)}

    def _beam_score(prefix, p_b, p_nb):
        acoustic = _log_add(p_b, p_nb)
        if lm is not None and lm_weight > 0 and prefix:
            text = "".join(IDX2CHAR.get(i, "") for i in prefix)
            return acoustic + lm_weight * lm.score(text)
        return acoustic

    def _get(d, key):
        return d.get(key, (NEG_INF, NEG_INF))

    for t in range(T):
        lp = log_probs[t]
        new_beams = {}

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _log_add(p_b, p_nb)

            # ── Emit blank: prefix unchanged, only blank path extends ─────────
            nb, nnb = _get(new_beams, prefix)
            new_beams[prefix] = (_log_add(nb, p_total + lp[0]), nnb)

            # ── Emit each non-blank character ─────────────────────────────────
            for c in range(1, C):
                c_lp = float(lp[c])
                new_prefix = prefix + (c,)

                if prefix and prefix[-1] == c:
                    # Same char as last: two cases must be tracked separately.
                    # Case 1 — extend new_prefix (add another c):
                    #   only possible via a blank-ending path.
                    nb, nnb = _get(new_beams, new_prefix)
                    new_beams[new_prefix] = (nb, _log_add(nnb, p_b + c_lp))
                    # Case 2 — stay on same prefix (repeated c collapses):
                    #   only possible via a nonblank-ending path.
                    nb, nnb = _get(new_beams, prefix)
                    new_beams[prefix] = (nb, _log_add(nnb, p_nb + c_lp))
                else:
                    # Different character: either path can extend.
                    nb, nnb = _get(new_beams, new_prefix)
                    new_beams[new_prefix] = (nb, _log_add(nnb, p_total + c_lp))

        # Prune to beam_width using acoustic + LM score
        beams = dict(
            sorted(
                new_beams.items(),
                key=lambda x: _beam_score(x[0], x[1][0], x[1][1]),
                reverse=True,
            )[:beam_width]
        )

    # ── Final selection ───────────────────────────────────────────────────────
    candidates = []
    for prefix, (p_b, p_nb) in beams.items():
        acoustic = _log_add(p_b, p_nb)
        decoded = "".join(IDX2CHAR.get(i, "") for i in prefix)
        lm_score = (lm.score(decoded) if lm is not None and lm_weight > 0 else 0.0)
        candidates.append((acoustic + lm_weight * lm_score, decoded))

    candidates.sort(reverse=True)
    return candidates[0][1] if candidates else ""


def ctc_beam_decode_batch(outputs, beam_width=10, lm_weight=0.3, lm=None):
    """Batch wrapper around ctc_beam_decode.

    Used at inference time only — training uses ctc_greedy_decode_batch.

    Args:
        outputs:    (T, B, C) raw model logits.
        beam_width: Number of hypotheses to keep at each step.
        lm_weight:  Weight for the language model score (0 = no LM).
        lm:         CharLM instance for rescoring (optional).
    Returns:
        List of decoded strings, one per sample in the batch.
    """
    results = []
    for b in range(outputs.size(1)):
        single = outputs[:, b:b+1, :]  # (T, 1, C)
        results.append(ctc_beam_decode(single, beam_width=beam_width,
                                       lm_weight=lm_weight, lm=lm))
    return results


# ── Character-level language model ────────────────────────────────────────────

class CharLM:
    """Character trigram language model with bigram backoff (Stupid Backoff).

    Scores text by summing log-probabilities of character trigrams:
        score("abc") = log P(a|<s><s>) + log P(b|<s>a) + log P(c|ab)

    When a trigram context is unseen, falls back to the bigram count
    (multiplied by a fixed backoff penalty) and finally to a uniform prior.
    Provides a stronger soft prior than pure bigrams: a trigram trained on
    handwriting transcripts captures common letter triples ("ing", "tion",
    "the") that bigrams alone can't model.

    Storage format on disk (JSON):
        {
            "trigrams": {"ab": {"c": logp, "_default": logp}, ...},
            "bigrams":  {"a":  {"b": logp, "_default": logp}, ...},
            "version":  3
        }
    """

    BACKOFF_PENALTY = math.log(0.4)  # Stupid Backoff α

    def __init__(self, path=CHAR_LM_PATH):
        self.path = path
        self.trigrams = {}
        self.bigrams = {}
        self.loaded = False

    def build_from_texts(self, texts):
        """Build trigram + bigram counts from training transcriptions."""
        tri_counts = defaultdict(lambda: defaultdict(int))
        bi_counts = defaultdict(lambda: defaultdict(int))

        for text in texts:
            padded = "^^" + text + "$"
            # Bigrams (skip the leading ^^ pad bigram)
            for a, b in zip(padded[1:-1], padded[2:]):
                bi_counts[a][b] += 1
            # Trigrams: context = previous 2 chars
            for a, b, c in zip(padded[:-2], padded[1:-1], padded[2:]):
                tri_counts[a + b][c] += 1

        vocab_size = len(CHARS) + 2  # +1 for ^, +1 for $
        self.trigrams = {}
        for ctx, nexts in tri_counts.items():
            total = sum(nexts.values()) + vocab_size
            self.trigrams[ctx] = {c: math.log((n + 1) / total) for c, n in nexts.items()}
            self.trigrams[ctx]["_default"] = math.log(1 / total)

        self.bigrams = {}
        for a, nexts in bi_counts.items():
            total = sum(nexts.values()) + vocab_size
            self.bigrams[a] = {b: math.log((n + 1) / total) for b, n in nexts.items()}
            self.bigrams[a]["_default"] = math.log(1 / total)

        self.loaded = True

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(
                {"version": 3, "trigrams": self.trigrams, "bigrams": self.bigrams},
                f,
            )

    def load(self):
        if not os.path.exists(self.path):
            return False
        with open(self.path) as f:
            data = json.load(f)

        # Backward-compat: an older bigram-only file is a flat dict of
        # {char: {next_char: logp}}; treat it as the bigram table.
        if isinstance(data, dict) and "version" in data:
            self.trigrams = data.get("trigrams", {})
            self.bigrams = data.get("bigrams", {})
        else:
            self.trigrams = {}
            self.bigrams = data
            print("[CharLM] Loaded legacy bigram-only file; rebuild for trigram benefit.")

        self.loaded = True
        return True

    def _bigram_logp(self, a, b):
        if a in self.bigrams:
            return self.bigrams[a].get(b, self.bigrams[a].get("_default", -10.0))
        return -10.0

    def score(self, text):
        """Score a text string using trigram log-probabilities + bigram backoff."""
        if not self.loaded or not text:
            return 0.0
        padded = "^^" + text + "$"
        total = 0.0
        for a, b, c in zip(padded[:-2], padded[1:-1], padded[2:]):
            ctx = a + b
            if ctx in self.trigrams:
                total += self.trigrams[ctx].get(
                    c, self.trigrams[ctx].get("_default", -10.0)
                )
            else:
                # Stupid Backoff to bigram with fixed penalty.
                total += self.BACKOFF_PENALTY + self._bigram_logp(b, c)
        return total