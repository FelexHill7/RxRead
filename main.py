"""
main.py — Single entry point for the RxRead pipeline.

Commands:
    python main.py                       Full pipeline:
                                           train if no weights → build LM → serve
    python main.py setup                 Bootstrap: download fonts, check datasets,
                                           print instructions for any missing data
    python main.py train                 Train the CRNN (saves only best-CER checkpoint)
                                           and rebuild LM, then serve
    python main.py lm                    (Re)build the character LM from training data
    python main.py serve                 Launch the web UI only
    python main.py predict <image>       Run in-house CRNN inference on an image
    python main.py predict-trocr <image> Run Microsoft TrOCR inference on an image
                                           (requires `pip install transformers`)

All paths and hyperparameters live in config.py. Never run the service
modules directly — go through this file so working directory and import
paths stay consistent.
"""

import json
import os
import sys

# Pin cwd to project root so relative paths in config.py resolve no matter
# where main.py is invoked from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


# ── Sub-commands ─────────────────────────────────────────────────────────────

def _cmd_train():
    from services.training import train
    train()


def _cmd_lm():
    from services.lm_builder import build_lm
    build_lm()


def _cmd_serve():
    from web.app import run_server
    run_server()


def _cmd_predict(path):
    from services.inference import predict
    result = predict(path)
    print(json.dumps(result, indent=2))


def _cmd_predict_trocr(path):
    from services.trocr_inference import predict
    result = predict(path)
    print(json.dumps(result, indent=2))


def _cmd_setup():
    """Bootstrap data folder: download fonts, report dataset status.

    Idempotent — safe to run repeatedly. Won't re-download anything that's
    already in place.
    """
    import importlib.util
    import urllib.request
    from pipeline.dataset import _SYNTH_FONT_URLS
    from config import TRAIN_DIR, IAM_DIR, IMGUR5K_DIR

    print("Setup: ensuring data/ is ready for training\n")

    # Fonts (auto-downloadable)
    fonts_dir = os.path.join("data", "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    print(f"[1/4] Fonts -> {fonts_dir}/")
    downloaded, skipped, failed = 0, 0, 0
    for fname, url in _SYNTH_FONT_URLS:
        dst = os.path.join(fonts_dir, fname)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skipped += 1
            continue
        try:
            urllib.request.urlretrieve(url, dst)
            downloaded += 1
            print(f"  [OK]   {fname}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {fname}: {e}")
    print(f"  -> {downloaded} downloaded, {skipped} already present, {failed} failed\n")

    # GNHK (manual download — print instructions if missing)
    print(f"[2/4] GNHK -> {TRAIN_DIR}/")
    if os.path.isdir(TRAIN_DIR) and os.listdir(TRAIN_DIR):
        n_jsons = sum(
            1 for _, _, files in os.walk(TRAIN_DIR) for f in files if f.endswith(".json")
        )
        print(f"  [OK] Present ({n_jsons} annotation files)\n")
    else:
        print("  [MISSING]")
        print("    Download from: https://goodnotes.com/gnhk")
        print(f"    Extract train_data/ and test_data/ into {os.path.dirname(TRAIN_DIR)}/\n")

    # IAM words (manual download — registration required)
    print(f"[3/4] IAM words -> {IAM_DIR}/")
    if os.path.isdir(IAM_DIR) and os.path.isdir(os.path.join(IAM_DIR, "words")):
        n_pngs = sum(
            1 for _, _, files in os.walk(os.path.join(IAM_DIR, "words"))
            for f in files if f.endswith(".png")
        )
        print(f"  [OK] Present ({n_pngs} word images)\n")
    else:
        print("  [MISSING] (optional but recommended for handwriting diversity)")
        print("    Source 1 (FKI, requires registration):")
        print("      https://fki.tic.heia-fr.ch/databases/iam-handwriting-database")
        print("    Source 2 (Kaggle mirror, no registration):")
        print("      https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database")
        print(f"    Extract into {IAM_DIR}/ so you have {IAM_DIR}/words/ + {IAM_DIR}/words.txt\n")

    # Imgur5K (manual download via Meta's scraper)
    print(f"[4/4] Imgur5K -> {IMGUR5K_DIR}/")
    info_present = (
        os.path.isdir(IMGUR5K_DIR)
        and (
            os.path.exists(os.path.join(IMGUR5K_DIR, "dataset_info", "imgur5k_data.json"))
            or os.path.exists(os.path.join(IMGUR5K_DIR, "imgur5k_data.json"))
        )
        and os.path.isdir(os.path.join(IMGUR5K_DIR, "images"))
    )
    if info_present:
        n_imgs = sum(
            1 for f in os.listdir(os.path.join(IMGUR5K_DIR, "images"))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"  [OK] Present ({n_imgs} images)\n")
    else:
        print("  [MISSING] (optional, adds real-world handwriting diversity)")
        print("    1. git clone https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset")
        print("    2. cd into it and run their download_imgur5k.py")
        print("       (downloads ~5k images from Imgur; ~50% URLs are dead in 2026)")
        print(f"    3. Move images/ and dataset_info/ into {IMGUR5K_DIR}/\n")

    # TrOCR (optional, much higher accuracy)
    print("[+] TrOCR (optional, drops CER from ~24% to ~5-10%)")
    if importlib.util.find_spec("transformers") is not None:
        print("  [OK] transformers installed -- run: python main.py predict-trocr <image>")
        print("       (first call downloads ~330 MB model into HF cache)\n")
    else:
        print("  [MISSING] Install with: pip install transformers")
        print("            Then: python main.py predict-trocr <image>\n")

    # Gemini API backend (optional, free tier, highest accuracy)
    print("[+] Gemini Vision (optional, ~95%+ accuracy, free tier)")
    has_genai = importlib.util.find_spec("google.generativeai") is not None
    has_key = bool(os.environ.get("GEMINI_API_KEY"))
    if has_genai and has_key:
        model_id = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-exp")
        print(f"  [OK] google-generativeai installed + GEMINI_API_KEY set (model: {model_id})")
        print("       Run: python main.py serve, then pick 'Gemini Vision' in the UI\n")
    elif has_genai and not has_key:
        print("  [PARTIAL] google-generativeai installed but GEMINI_API_KEY not set")
        print("            Get a key (free, no credit card): https://aistudio.google.com/apikey")
        print("            Then in PowerShell: $env:GEMINI_API_KEY = 'AIza...'\n")
    else:
        print("  [MISSING] Install in two steps:")
        print("    1. pip install google-generativeai")
        print("    2. Set GEMINI_API_KEY env var (free key at aistudio.google.com/apikey)\n")

    print("Setup complete. Run: python main.py train")


def _weights_exist():
    import glob
    from config import BEST_WEIGHTS, FINAL_WEIGHTS, CHECKPOINT_DIR, SEEDS, seed_weights_path
    if os.path.exists(BEST_WEIGHTS) or os.path.exists(FINAL_WEIGHTS):
        return True
    for seed in SEEDS:
        if os.path.exists(seed_weights_path(seed, "best")) or os.path.exists(seed_weights_path(seed, "final")):
            return True
    return bool(glob.glob(os.path.join(CHECKPOINT_DIR, "crnn_gnhk_seed*_best.pth")))


def _lm_exists():
    from config import CHAR_LM_PATH
    return os.path.exists(CHAR_LM_PATH)


# ── Pipeline orchestration ───────────────────────────────────────────────────

def _run_full_pipeline(force_train=False):
    """Train (if needed) → build LM (if needed) → launch the web UI."""
    if force_train or not _weights_exist():
        print("→ Training model...\n")
        _cmd_train()
        print()
    else:
        print("→ Trained model found — skipping training.")

    if force_train or not _lm_exists():
        print("→ Building character LM...\n")
        _cmd_lm()
        print()
    else:
        print("→ Character LM found — skipping rebuild.")

    print("→ Launching web UI...\n")
    _cmd_serve()


# ── CLI dispatch ─────────────────────────────────────────────────────────────

def main():
    command = sys.argv[1].lower() if len(sys.argv) >= 2 else "all"

    if command == "all":
        _run_full_pipeline(force_train=False)

    elif command == "setup":
        _cmd_setup()

    elif command == "train":
        _run_full_pipeline(force_train=True)

    elif command == "lm":
        _cmd_lm()

    elif command == "serve":
        _cmd_serve()

    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: python main.py predict <image_path>")
            sys.exit(1)
        _cmd_predict(sys.argv[2])

    elif command == "predict-trocr":
        if len(sys.argv) < 3:
            print("Usage: python main.py predict-trocr <image_path>")
            sys.exit(1)
        _cmd_predict_trocr(sys.argv[2])

    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [setup | train | lm | serve | predict <image> | predict-trocr <image>]")
        sys.exit(1)


if __name__ == "__main__":
    main()
