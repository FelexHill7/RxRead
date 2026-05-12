"""
gemini_inference.py — Google Gemini vision backend (opt-in, free).

Sends the full image to Gemini and asks for a verbatim transcription.
Quality on handwriting is comparable to Claude vision; the free tier is
generous (15 req/min, 1M tokens/day) so this is the recommended default
free backend.

Trade-offs vs other backends:
- Highest accuracy on diverse handwriting (~95 %+ typical), free tier
- Latency ~1-3 s per image
- Free quota: 15 RPM, 1M tokens/day on gemini-2.5-flash
- Requires `pip install google-genai` + GEMINI_API_KEY env var
- Requires internet

Get a key at https://aistudio.google.com/apikey (no credit card needed).
Model can be overridden via the GEMINI_MODEL env var; defaults to
gemini-2.5-flash.
"""

import os

from PIL import Image


_client = None


def _get_model_id():
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


def _ensure_client():
    """Lazy-init the Gemini client. Raises with actionable error if missing."""
    global _client
    if _client is not None:
        return _client

    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError(
            "Gemini backend requires google-genai. Install with:\n"
            "  pip install google-genai"
        ) from e

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY in your environment. "
            "Get a key (free, no credit card) at https://aistudio.google.com/apikey"
        )

    _client = genai.Client(api_key=api_key)
    return _client


_PROMPT = (
    "Transcribe the handwriting in this image verbatim. Output only the "
    "transcribed text, one line per detected line. Preserve original "
    "punctuation and capitalization. If you cannot read part of it, "
    "write [unreadable] for that part. No commentary, no formatting."
)


def predict(image):
    """Run Gemini vision on the image. Returns the same dict shape as the
    other backends so the web UI / CLI can treat all backends uniformly."""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    client = _ensure_client()
    response = client.models.generate_content(
        model=_get_model_id(),
        contents=[_PROMPT, image],
    )
    text = (response.text or "").strip()

    # Per-line confidence isn't exposed by Gemini — synthesize 1.0
    lines_out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        lines_out.append({
            "text": line,
            "confidence": 1.0,
            "bbox": [0, 0, image.size[0], image.size[1]],
        })

    return {
        "text": text,
        "confidence": 1.0,
        "lines": lines_out,
        "model": _get_model_id(),
    }
