"""
dataset.py — Dataset classes and data loading.

Each dataset class reads its source format, crops/loads word images,
and pre-caches tensors at init for I/O-free training.

Public API:
    - GNHKDataset              — GNHK handwriting corpus (target domain)
    - IAMDataset               — IAM handwriting database (regulariser)
    - build_weighted_train_set — combine sources with GNHK-biased sampling
    - build_dataloader         — construct a DataLoader with project settings
    - collate_fn               — CTC-compatible batch collation
"""

import os
import json
import unicodedata
import cv2
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
from PIL import Image

from config import encode_text, BATCH_SIZE, NUM_WORKERS
from .preprocessing import base_transform


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
MAPPING_FILES = (
    "labels.tsv",
    "labels.txt",
    "labels.csv",
    "ground_truth.txt",
    "gt.txt",
    "train_gt.txt",
    "val_gt.txt",
    "linux_gt.txt",
)

# Sampling weights per source — GNHK is the target domain so it gets
# oversampled, but IAM's character coverage regularises rare letters; previously
# 7:3 was over-biased and underused IAM's diversity once GNHK fit.
_GNHK_WEIGHT = 5.0
_IAM_WEIGHT = 4.0

def _normalize_text_for_charset(text):
    """Normalize labels so accented IAM/French text can map into ASCII charset."""
    if text is None:
        return ""
    text = text.strip().replace("|", " ")
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    return " ".join(text.split())


class GNHKDataset(Dataset):
    """
    PyTorch Dataset for the GNHK handwriting corpus.

    Each GNHK sample is a full-page image plus a JSON file that contains a
    flat list of word-level annotations with polygon coordinates.  This class
    walks the directory tree, crops every annotated word, and stores the
    (crop, text) pairs in memory so training is I/O-free after init.
    """

    def __init__(self, root_dir):
        samples = []
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.endswith(".json"):
                    continue

                json_path = os.path.join(dirpath, fname)
                img_path = json_path.replace(".json", ".jpg")

                if not os.path.exists(img_path):
                    continue

                with open(json_path) as f:
                    data = json.load(f)

                image = cv2.imread(img_path)
                if image is None:
                    continue

                for obj in data:
                    text = obj.get("text", "").strip()
                    if not text or text in ["###", ""]:
                        continue

                    polygon = obj.get("polygon", None)
                    if polygon is None:
                        continue

                    xs = [polygon[k] for k in ("x0", "x1", "x2", "x3")]
                    ys = [polygon[k] for k in ("y0", "y1", "y2", "y3")]
                    x_min, x_max = max(min(xs), 0), max(xs)
                    y_min, y_max = max(min(ys), 0), max(ys)

                    crop = image[y_min:y_max, x_min:x_max]
                    if crop.size == 0:
                        continue

                    samples.append((crop, text))

        self.cached = []
        for crop, text in samples:
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensor = base_transform(pil_img)
            label = torch.tensor(encode_text(text), dtype=torch.long)
            self.cached.append((tensor, label, text))

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, idx):
        return self.cached[idx]

class IAMDataset(Dataset):
    """
    PyTorch Dataset for the IAM Handwriting Database.

    Expects the standard IAM layout:
        root_dir/
            words/       — word images in IAM folder hierarchy
            words.txt    — annotation file

    Each line of words.txt (non-comment) has the format:
        word_id seg_result graylevel #components x y w h tag transcription

    Word images live at: words/<form_part1>/<form_id>/<word_id>.png
    Pre-caches all tensors at init, same format as GNHKDataset.
    """

    def __init__(self, root_dir):
        self.cached = []
        self.total_images_found = 0

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    self.total_images_found += 1

        added = self._load_standard_iam_layout(root_dir)
        source = "words.txt"

        if added == 0:
            added = self._load_mapping_layout(root_dir)
            source = "labels mapping"

        if added == 0:
            added = self._load_sidecar_layout(root_dir)
            source = "image+txt sidecar"

        if added > 0:
            print(f"  IAM loaded: {len(self.cached)} word samples ({source})")
        elif self.total_images_found > 0:
            print(
                "  IAM images found but no transcriptions detected. "
                "Add one of: words.txt, labels.tsv/labels.txt/labels.csv, or per-image .txt sidecars"
            )

    def _append_sample(self, img_path, raw_text):
        text = _normalize_text_for_charset(raw_text)
        if not text or text == "###":
            return False

        encoded = encode_text(text)
        if not encoded:
            return False

        try:
            pil_img = Image.open(img_path).convert("RGB")
            tensor = base_transform(pil_img)
            label = torch.tensor(encoded, dtype=torch.long)
            self.cached.append((tensor, label, text))
            return True
        except Exception:
            return False

    def _load_standard_iam_layout(self, root_dir):
        before = len(self.cached)
        words_file = os.path.join(root_dir, "words.txt")
        words_dir = os.path.join(root_dir, "words")

        if not (os.path.exists(words_file) and os.path.isdir(words_dir)):
            return 0

        with open(words_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) < 9:
                    continue

                word_id = parts[0]
                seg_result = parts[1]
                text = " ".join(parts[8:])

                if seg_result == "err":
                    continue

                parts_id = word_id.split("-")
                if len(parts_id) < 3:
                    continue

                folder1 = parts_id[0]
                folder2 = f"{parts_id[0]}-{parts_id[1]}"
                img_path = os.path.join(words_dir, folder1, folder2, f"{word_id}.png")

                if not os.path.exists(img_path):
                    continue

                self._append_sample(img_path, text)

        return len(self.cached) - before

    def _load_mapping_layout(self, root_dir):
        before = len(self.cached)
        mapping_paths = []
        for dirpath, _, filenames in os.walk(root_dir):
            for name in MAPPING_FILES:
                if name in filenames:
                    mapping_paths.append(os.path.join(dirpath, name))
            for fname in filenames:
                if fname.lower().endswith("_gt.txt"):
                    mapping_paths.append(os.path.join(dirpath, fname))

        if not mapping_paths:
            return 0

        image_by_rel = {}
        image_by_stem = {}
        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                full = os.path.join(dirpath, fname)
                rel = os.path.relpath(full, root_dir).replace("\\", "/")
                stem = os.path.splitext(os.path.basename(full))[0]
                image_by_rel[rel] = full
                image_by_stem.setdefault(stem, []).append(full)

        for mapping_path in mapping_paths:
            with open(mapping_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    pair = None
                    for sep in ("\t", ";", ","):
                        if sep in line:
                            pair = line.split(sep, 1)
                            break

                    if not pair or len(pair) != 2:
                        continue

                    key, text = pair[0].strip(), pair[1].strip()
                    if not key:
                        continue

                    normalized_key = key.replace("\\", "/")
                    stem_key = os.path.splitext(os.path.basename(normalized_key))[0]

                    img_path = None
                    if os.path.isabs(key) and os.path.exists(key):
                        img_path = key
                    elif normalized_key in image_by_rel:
                        img_path = image_by_rel[normalized_key]
                    elif stem_key in image_by_stem and len(image_by_stem[stem_key]) == 1:
                        img_path = image_by_stem[stem_key][0]

                    if img_path is None:
                        continue

                    self._append_sample(img_path, text)

        return len(self.cached) - before

    def _load_sidecar_layout(self, root_dir):
        before = len(self.cached)

        for dirpath, _, filenames in os.walk(root_dir):
            for fname in filenames:
                if not fname.lower().endswith(IMAGE_EXTENSIONS):
                    continue

                img_path = os.path.join(dirpath, fname)
                sidecar = os.path.splitext(img_path)[0] + ".txt"
                if not os.path.exists(sidecar):
                    continue

                try:
                    with open(sidecar, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read().strip()
                except Exception:
                    continue

                self._append_sample(img_path, text)

        return len(self.cached) - before

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, idx):
        return self.cached[idx]


def collate_fn(batch):
    """Custom collate: stack images and concatenate variable-length labels
    into the flat format required by CTCLoss."""
    images, labels, texts = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels])
    labels_concat = torch.cat(labels)
    return images, labels_concat, label_lengths, texts


def build_weighted_train_set(gnhk_dir, iam_dir=None):
    """Build a weighted training dataset biased toward the target domain (GNHK).

    GNHK is oversampled since the val set is pure GNHK. IAM contributes
    character-level diversity and acts as a regulariser.

    Returns:
        train_set: ConcatDataset (or single Dataset if only GNHK present)
        sampler:   WeightedRandomSampler for use with DataLoader
    """
    gnhk = GNHKDataset(gnhk_dir)
    print(f"  GNHK samples: {len(gnhk)}")

    datasets = [gnhk]
    weights = [_GNHK_WEIGHT] * len(gnhk)

    if iam_dir and os.path.isdir(iam_dir):
        iam = IAMDataset(iam_dir)
        if len(iam) > 0:
            print(f"  IAM samples: {len(iam)}")
            datasets.append(iam)
            weights += [_IAM_WEIGHT] * len(iam)

    train_set = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"  Total combined: {len(train_set)}")

    sampler = WeightedRandomSampler(weights, num_samples=len(train_set), replacement=True)
    return train_set, sampler


def build_dataloader(dataset, sampler=None, shuffle=False):
    """Construct a DataLoader with project-standard settings."""
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )