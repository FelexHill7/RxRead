"""
main.py — Single entry point for the RxRead pipeline.

Commands:
    python main.py                  Full pipeline:
                                      train if no weights → build LM → serve
    python main.py train            Train (and rebuild the LM), then serve
    python main.py lm               (Re)build the character LM from training data
    python main.py serve            Launch the web UI only
    python main.py predict <image>  Run inference on a single image (CLI)

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


def _weights_exist():
    from config import BEST_WEIGHTS, FINAL_WEIGHTS
    return os.path.exists(BEST_WEIGHTS) or os.path.exists(FINAL_WEIGHTS)


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

    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [train | lm | serve | predict <image>]")
        sys.exit(1)


if __name__ == "__main__":
    main()
