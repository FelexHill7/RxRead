# RxRead — Handwriting OCR Comparison Platform

A web app that lets you upload a handwriting image and compare four OCR
backends side-by-side: a from-scratch **ResNet-CRNN**, Microsoft **TrOCR**
(line and whole-image), and **Google Gemini** vision. An optional
**prescription mode** post-processes any backend's output by fuzzy-matching
predicted words against a curated drug-name vocabulary.

The point is **comparison**, not winning. The in-house CRNN (~24 % CER)
shows what one developer can build from scratch; the pretrained / cloud
backends (~5–10 % CER for TrOCR, ~95 %+ word accuracy for Gemini) show
what production-grade looks like.

---

## Quick start

```powershell
python main.py setup        # download fonts, report missing datasets
python main.py train        # ~3–4 hr on a CUDA GPU; saves best-CER checkpoint
python main.py serve        # http://localhost:5000
```

Optional backends:

```powershell
pip install transformers                  # TrOCR
pip install google-genai                  # Gemini
$env:GEMINI_API_KEY = "AIza..."           # free key at aistudio.google.com/apikey
```

---

## Backends

| Backend | What it is | Speed | Accuracy on diverse handwriting |
| --- | --- | --- | --- |
| **CRNN · Words** | Trained ResNet-18 + BiLSTM + attention + CTC, word-level | ~50 ms/word | ~50 % word acc / 24 % CER |
| **TrOCR · Lines** | Microsoft TrOCR per-line | ~500 ms/line | ~70–80 % |
| **TrOCR · Page** | Microsoft TrOCR whole-image | ~500 ms | OK on single line, poor on multi-line |
| **Gemini Vision** | Google `gemini-2.5-flash` | ~2–3 s | ~95 %+ |

---

## Architecture

**Word-level CRNN.** grayscale → ResNet-18 (ImageNet, layer3/4 strides
relaxed) → gated conv → mean+max column-pool → linear → 2-layer BiLSTM →
CTC head. Inference uses beam search + character-bigram LM rescoring +
test-time augmentation, with optional multi-seed ensembling.

**Training.** 60 epochs on GNHK + IAM + Imgur5K + 30K synthetic
font-rendered samples (~87K total) with AMP, OneCycleLR, label smoothing,
and elastic distortion. **Hard-example mining** kicks in at 80 % of
training — the sampler is rebuilt around per-sample loss to focus on the
residual hard set.

**Line detection (TrOCR backend).** Connected components → Y-center
histogram → smoothed local maxima with two-pass NMS. Robust to
descender/ascender overlap; falls back to TrOCR whole-image or Gemini on
heavily decorated inputs.

---

## Project structure

```text
core/        — model, CTC decoders, metrics
pipeline/    — dataset loaders + GPU augmentation
services/    — training, inference (CRNN/TrOCR/Gemini), drug-bias, LM builder
web/         — Flask app + single-page UI
tests/       — 27 tests (drug bias, dispatch, line detection)
```

All paths and hyperparameters live in `config.py`. Single entry point: `main.py`.

---

## Prescription mode

Drug names use orthography that confuses general OCR
(`Hydrochlorothiazide`, `Atorvastatin`). When prescription mode is on, the
post-processor compares every predicted word against ~280
most-prescribed US drugs and snaps it to the canonical spelling if
Levenshtein distance ≤ 2. Short stopwords (`mg`, `tab`, `qd`) are skipped
as match targets so `10mg` doesn't collapse to `mg`. Drop a custom list
at `data/drug_names.txt` to override.

---

## Tests

```powershell
pip install pytest
python -m pytest tests/ -v
```

---

## Acknowledgements

[GNHK](https://goodnotes.com/gnhk) · [IAM](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) · [TrOCR](https://huggingface.co/microsoft/trocr-base-handwritten) · [Gemini](https://aistudio.google.com)
