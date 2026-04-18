"""
app.py — Flask web UI for doctor handwriting recognition.

Serves a single-page app where users drag-and-drop or upload a prescription
image.  The image is sent to the /predict endpoint, passed through the
CRNN model, and the decoded text is returned as JSON.

Delegates model loading and inference to services/inference.py.
HTML template lives in web/templates/index.html.

Run:   python app.py
Visit: http://localhost:5000
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

from services.inference import predict_pil

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
    """Accept an uploaded image and return the transcribed text as JSON."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        text = predict_pil(image)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_server(host="0.0.0.0", port=5000):
    """Start the Flask development server."""
    print(f"Starting RxRead at http://localhost:{port}")
    app.run(debug=False, host=host, port=port)


if __name__ == "__main__":
    run_server()