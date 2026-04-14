"""
generate_synthetic.py — Generate synthetic handwriting word images for training.

Uses trdg (TextRecognitionDataGenerator) to produce distorted handwriting-style
word images with random fonts, skew, blur, and background noise.  Output is a
folder of images + a labels.json mapping filename → text.

Usage:
    python generate_synthetic.py [--count 50000] [--output data/synthetic]

The generated data is read by dataset.py alongside the real GNHK data.
"""

import os
import json
import argparse
import random
import string

from trdg.generators import GeneratorFromStrings


# ── Common English words + medical/prescription terms for realistic mix ──────
COMMON_WORDS = [
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "her",
    "was", "one", "our", "out", "day", "had", "has", "his", "how", "its",
    "may", "new", "now", "old", "see", "way", "who", "did", "get", "let",
    "say", "she", "too", "use", "take", "once", "twice", "daily", "after",
    "before", "meals", "tablet", "capsule", "syrup", "drops", "cream",
    "ointment", "injection", "dose", "morning", "evening", "night", "oral",
    "apply", "every", "hours", "days", "weeks", "months", "patient", "name",
    "date", "doctor", "prescription", "pharmacy", "medicine", "treatment",
    "diagnosis", "symptoms", "pain", "fever", "cough", "cold", "headache",
    "blood", "pressure", "sugar", "heart", "liver", "kidney", "lung",
    "aspirin", "ibuprofen", "paracetamol", "amoxicillin", "metformin",
    "omeprazole", "amlodipine", "atorvastatin", "lisinopril", "metoprolol",
    "please", "thank", "with", "from", "this", "that", "have", "been",
    "will", "each", "make", "like", "long", "look", "many", "some", "them",
    "then", "than", "first", "water", "food", "left", "right", "hand",
    "high", "last", "need", "still", "between", "never", "under", "while",
    "house", "world", "below", "asked", "going", "large", "until", "along",
    "shall", "being", "often", "earth", "began", "since", "study", "might",
    "should", "would", "could", "about", "other", "which", "their", "there",
    "these", "those", "where", "when", "what", "much", "only", "also",
]


def random_word():
    """Return a random word — mix of dictionary words and random strings."""
    r = random.random()
    if r < 0.6:
        return random.choice(COMMON_WORDS)
    elif r < 0.8:
        # Random alphanumeric (simulates IDs, codes, dosages like "500mg")
        length = random.randint(2, 8)
        return "".join(random.choices(string.ascii_letters + string.digits, k=length))
    elif r < 0.9:
        # Number with unit
        num = random.randint(1, 999)
        unit = random.choice(["mg", "ml", "mcg", "g", "kg", "mm", "cm", "%"])
        return f"{num}{unit}"
    else:
        # Short phrase (2-3 words)
        count = random.randint(2, 3)
        return " ".join(random.choice(COMMON_WORDS) for _ in range(count))


def generate(count, output_dir):
    """Generate synthetic handwriting word images using trdg."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate random strings
    words = [random_word() for _ in range(count)]

    generator = GeneratorFromStrings(
        strings=words,
        count=count,
        fonts=[],                    # empty = use trdg's built-in handwriting fonts
        language="en",
        size=64,                     # font size (will be resized to 32×128 by dataset)
        skewing_angle=5,             # random skew up to ±5°
        random_skew=True,
        blur=1,                      # slight blur
        random_blur=True,
        distorsio_type=3,            # 3 = random distortion type
        distorsio_orientation=2,     # 2 = both horizontal and vertical
        background_type=1,           # 1 = gaussian noise background
        width=-1,                    # auto width based on text length
        alignment=1,                 # center
        text_color="#000000,#404040,#202020",  # dark ink variants
        is_handwritten=True,         # use handwriting-style rendering
        space_width=1.0,
    )

    labels = {}
    for i, (img, text) in enumerate(generator):
        if i >= count:
            break
        fname = f"syn_{i:06d}.png"
        img.save(os.path.join(output_dir, fname))
        labels[fname] = text

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1}/{count} images...")

    # Save labels mapping
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Done! Generated {len(labels)} images in {output_dir}")
    print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic handwriting data")
    parser.add_argument("--count", type=int, default=50000,
                        help="Number of images to generate (default: 50000)")
    parser.add_argument("--output", type=str, default="data/synthetic",
                        help="Output directory (default: data/synthetic)")
    args = parser.parse_args()

    generate(args.count, args.output)
