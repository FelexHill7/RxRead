"""
app.py — Flask web UI: handwriting OCR comparison platform.

Serves a single-page app where users drag-and-drop a handwriting image and
pick one of four backends to transcribe it. Optional drug-name post-bias
applies to whichever backend is chosen.

Backends:
    words        — in-house CRNN, word-level (services.inference.predict)
    trocr        — Microsoft TrOCR, line-segmented
    trocr-whole  — Microsoft TrOCR, whole-image
    gemini       — Google Gemini vision (free tier)

Run via:    python main.py serve
Visit:      http://localhost:5000
"""

import io
import os
import sys

# Ensure project root is on the path regardless of where Flask is invoked from
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
os.chdir(_project_root)

from flask import Flask, request, jsonify, render_template
from PIL import Image

from services.inference import predict as run_word_prediction


def _run_trocr_prediction(image):
    from services.trocr_inference import predict as trocr_predict
    return trocr_predict(image, segment=True)


def _run_trocr_whole_prediction(image):
    from services.trocr_inference import predict as trocr_predict
    return trocr_predict(image, segment=False)


def _run_gemini_prediction(image):
    from services.gemini_inference import predict as gemini_predict
    return gemini_predict(image)


_BACKENDS = {
    "words": run_word_prediction,
    "trocr": _run_trocr_prediction,
    "trocr-whole": _run_trocr_whole_prediction,
    "gemini": _run_gemini_prediction,
}

# ── Flask app ─────────────────────────────────────────────────────────────────
_web_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(_web_dir, "templates"),
    static_folder=os.path.join(_web_dir, "static"),
)


@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accept an uploaded image and return the transcribed text as JSON.

    Form fields:
        image        — file upload (required)
        mode         — backend key from _BACKENDS (default "words")
        drug_bias    — "true" to apply prescription-mode post-processing
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    mode = (request.form.get("mode") or "words").lower()
    backend = _BACKENDS.get(mode)
    if backend is None:
        return jsonify({"error": f"Unknown mode '{mode}'"}), 400

    drug_bias = request.form.get("drug_bias") == "true"

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        result = backend(image)
        result["mode"] = mode
        if drug_bias:
            from services.drug_bias import bias_result
            bias_result(result)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "mode": mode}), 500


def run_server(host="0.0.0.0", port=5000):
    """Start the Flask development server."""
    print(f"Starting RxRead at http://localhost:{port}")
    app.run(debug=False, host=host, port=port)


if __name__ == "__main__":
    run_server()