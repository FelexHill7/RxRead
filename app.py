"""
app.py — Flask web UI for doctor handwriting recognition.

Serves a single-page app where users drag-and-drop or upload a prescription
image.  The image is sent to the /predict endpoint, passed through the
CRNN model, and the decoded text is returned as JSON.

Delegates model loading and inference to predict.py.
HTML template lives in templates/index.html.

Run:   python app.py
Visit: http://localhost:5000
"""

import io
from flask import Flask, request, jsonify, render_template
from PIL import Image

from predict import predict_pil

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)


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