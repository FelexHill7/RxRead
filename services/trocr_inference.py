"""
trocr_inference.py — Microsoft TrOCR inference path (opt-in).

Microsoft's TrOCR was pretrained on millions of handwritten samples; on
diverse handwriting it routinely lands at 5–10 % CER vs. ~24 % for the
in-house CRNN. Use this when accuracy matters more than latency.

Trade-offs vs. the CRNN path:
- Model size: ~330 MB (vs. ~50 MB CRNN)
- Per-line inference: ~300–500 ms on CUDA (vs. ~50 ms per word for CRNN)
- First call downloads weights (~330 MB) into the HF cache, then cached locally
- Requires `pip install transformers`

Uses the shared line segmenter (services.inference.detect_text_lines).
TrOCR was trained on whole IAM lines, so feeding it line crops matches
its training distribution; word-crops cause hallucination.
"""

import os
import torch
from PIL import Image

from services.inference import detect_text_lines, _upscale_if_small


_processor = None
_model = None
_device = None


def _ensure_loaded():
    """Lazy-load TrOCR on first call. Caches in module globals."""
    global _processor, _model, _device

    if _model is not None:
        return _processor, _model, _device

    try:
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    except ImportError as e:
        raise RuntimeError(
            "TrOCR requires the `transformers` package. Install with:\n"
            "  pip install transformers\n"
            f"(original error: {e})"
        ) from e

    model_id = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-handwritten")
    print(f"[trocr] Loading {model_id} (first call only — ~330 MB download if not cached)...")

    _processor = TrOCRProcessor.from_pretrained(model_id)
    _model = VisionEncoderDecoderModel.from_pretrained(model_id)

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = _model.to(_device)
    _model.eval()

    print(f"[trocr] Loaded on {_device}")
    return _processor, _model, _device


def _decode_one(crop, processor, model, device):
    """Run TrOCR on a single PIL crop, return (text, confidence)."""
    # TrOCR expects an RGB PIL image. Our crops are already RGB.
    pixel_values = processor(crop, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        outputs = model.generate(
            pixel_values,
            max_new_tokens=24,
            num_beams=4,
            return_dict_in_generate=True,
            output_scores=True,
        )

    text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()

    # Confidence proxy: mean exp of per-step beam scores. Bounded [0, 1].
    if outputs.sequences_scores is not None:
        # sequences_scores is log-prob averaged over length, take exp.
        score = float(outputs.sequences_scores[0].exp().clamp(0.0, 1.0))
    else:
        score = 1.0

    return text, score


def predict(image, segment=True):
    """Run TrOCR on a full image. Returns the same dict shape as services.inference.predict.

    Args:
        image:    PIL.Image, str path, or numpy array
        segment:  if True (default), line-segment the image and decode each
                  line separately. TrOCR was trained on whole IAM lines —
                  feeding it single-word crops makes it hallucinate plausible
                  multi-word sequences. If False, decode the whole image at
                  once (only OK for single-line crops).

    Returns:
        dict with `text`, `confidence`, `lines`
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    processor, model, device = _ensure_loaded()

    if not segment:
        text, conf = _decode_one(image, processor, model, device)
        return {
            "text": text,
            "confidence": round(conf, 3),
            "lines": [{
                "text": text,
                "confidence": round(conf, 3),
                "bbox": [0, 0, image.size[0], image.size[1]],
            }] if text else [],
        }

    image = _upscale_if_small(image)
    line_spans = detect_text_lines(image)
    if not line_spans:
        line_spans = [(0, image.size[1])]

    h_img = image.size[1]
    lines_out = []
    for y0, y1 in line_spans:
        pad = max(int((y1 - y0) * 0.05), 2)
        ya = max(y0 - pad, 0)
        yb = min(y1 + pad, h_img)
        crop = image.crop((0, ya, image.size[0], yb))
        if crop.size[0] < 16 or crop.size[1] < 16:
            continue
        try:
            text, conf = _decode_one(crop, processor, model, device)
        except Exception:
            continue
        text = text.strip()
        if not text:
            continue
        lines_out.append({
            "text": text,
            "confidence": round(conf, 3),
            "bbox": [0, int(ya), image.size[0], int(yb)],
        })

    avg_conf = sum(l["confidence"] for l in lines_out) / len(lines_out) if lines_out else 0.0
    return {
        "text": "\n".join(l["text"] for l in lines_out),
        "confidence": round(avg_conf, 3),
        "lines": lines_out,
    }
