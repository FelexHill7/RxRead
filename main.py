"""
main.py — Single entry point for the RxRead pipeline.

Delegates to train.py, predict.py, and app.py so nothing else needs to be
run directly.  All paths and configuration live here.

Usage:
    python main.py                    Train the model, then launch the web UI
    python main.py train              Train the CRNN, then launch the web UI
    python main.py serve              Launch the web UI only (skip training)
    python main.py predict <image>    Quick CLI inference on a single image
"""

import os
import sys

# Ensure project root is the working directory regardless of where the script
# is invoked from — keeps data/ and weight paths consistent.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)


def main():
    command = sys.argv[1].lower() if len(sys.argv) >= 2 else "all"

    if command == "train":
        from train import train
        train()
        print("\nTraining complete. Launching web UI...\n")
        from app import run_server
        run_server()

    elif command == "serve":
        from app import run_server
        run_server()

    elif command == "predict":
        if len(sys.argv) < 3:
            print("Usage: python main.py predict <image_path>")
            return
        from predict import predict_file
        print(predict_file(sys.argv[2]))

    elif command == "all":
        # Full pipeline: train (if no weights exist) → launch web UI
        weights_exist = (
            os.path.exists("checkpoints/crnn_gnhk_best.pth")
            or os.path.exists("checkpoints/crnn_gnhk.pth")
        )

        if not weights_exist:
            print("No trained model found — starting training...\n")
            from train import train
            train()
            print()
        else:
            print("Trained model found — skipping training.")

        print("Launching web UI...\n")
        from app import run_server
        run_server()

    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [train | serve | predict <image>]")


if __name__ == "__main__":
    main()
