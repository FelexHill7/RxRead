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
import random
import unicodedata
import urllib.request
import cv2
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, Subset, WeightedRandomSampler
from PIL import Image, ImageDraw, ImageFilter, ImageFont

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

# Sampling weights per source. GNHK has ~5× more raw samples than IAM, so
# equal weights still leave GNHK dominating ~80 % of effective sampling. To
# get general-purpose handwriting recognition (clean + messy across writers),
# we down-weight GNHK and up-weight IAM until effective sampling is closer to
# 35 % GNHK / 55 % IAM / 10 % synthetic.
_GNHK_WEIGHT = 2.0
_IAM_WEIGHT = 8.0
# Synthetic up-weighted from 2 → 4 now that we generate 30k+ samples — at
# this volume it's a meaningful regularizer, not just background noise.
_SYNTH_WEIGHT = 4.0


# Free Google Fonts handwriting fonts — auto-downloaded on first synthetic run.
# Apache-licensed fonts live under /apache/ in the Google Fonts repo, OFL ones
# under /ofl/. URLs that 404 are skipped silently; 5+ fonts is enough variety.
_SYNTH_FONT_URLS = [
    ("Kalam-Regular.ttf",             "https://raw.githubusercontent.com/google/fonts/main/ofl/kalam/Kalam-Regular.ttf"),
    ("PatrickHand-Regular.ttf",       "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf"),
    ("IndieFlower-Regular.ttf",       "https://raw.githubusercontent.com/google/fonts/main/ofl/indieflower/IndieFlower-Regular.ttf"),
    ("ArchitectsDaughter-Regular.ttf","https://raw.githubusercontent.com/google/fonts/main/ofl/architectsdaughter/ArchitectsDaughter-Regular.ttf"),
    ("Sacramento-Regular.ttf",        "https://raw.githubusercontent.com/google/fonts/main/ofl/sacramento/Sacramento-Regular.ttf"),
    ("ReenieBeanie.ttf",              "https://raw.githubusercontent.com/google/fonts/main/ofl/reeniebeanie/ReenieBeanie.ttf"),
    ("AnnieUseYourTelescope-Regular.ttf", "https://raw.githubusercontent.com/google/fonts/main/ofl/annieuseyourtelescope/AnnieUseYourTelescope-Regular.ttf"),
    ("PermanentMarker-Regular.ttf",   "https://raw.githubusercontent.com/google/fonts/main/apache/permanentmarker/PermanentMarker-Regular.ttf"),
    ("HomemadeApple-Regular.ttf",     "https://raw.githubusercontent.com/google/fonts/main/apache/homemadeapple/HomemadeApple-Regular.ttf"),
    ("ComingSoon-Regular.ttf",        "https://raw.githubusercontent.com/google/fonts/main/apache/comingsoon/ComingSoon-Regular.ttf"),
]

_SYNTH_FALLBACK_WORDS = (
    "the and of to a in is you that it he was for on are with as I his they "
    "be at one have this from or had by hot word but what some we can out "
    "other were all your when up use word how said an each she which do their "
    "time if will way about many then them write would like so these her long "
    "make thing see him two has look more day could go come did my sound no "
    "most number who over know water than call first people may down side been "
    "now find any new work part take get place made live where after back little "
    "only round man year came show every good me give our under name patient "
    "doctor nurse prescription medicine tablet capsule mg ml dose daily morning "
    "evening night Hello World Today Tomorrow Yesterday Monday Tuesday Wednesday "
    "Thursday Friday Saturday Sunday January February March April May June July "
    "August September October November December"
)

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
        # Dedup by absolute image path: IAM ships multiple gt files
        # (train_gt.txt, val_gt.txt, linux_gt.txt) that overlap, and the
        # mapping loader walks every gt file, so the same image+label would
        # otherwise be cached multiple times — inflating IAM's effective
        # weight and leaking duplicates across train/val splits.
        self._seen_paths = set()

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
            print(f"  IAM loaded: {len(self.cached)} unique word samples ({source})")
        elif self.total_images_found > 0:
            print(
                "  IAM images found but no transcriptions detected. "
                "Add one of: words.txt, labels.tsv/labels.txt/labels.csv, or per-image .txt sidecars"
            )

    def _append_sample(self, img_path, raw_text):
        canonical = os.path.normcase(os.path.abspath(img_path))
        if canonical in self._seen_paths:
            return False
        self._seen_paths.add(canonical)

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


class SyntheticDataset(Dataset):
    """Generate handwriting word images from free fonts.

    Bootstraps a small set of Google-Fonts handwriting fonts on first use
    (cached in `data/fonts/`), then renders `num_samples` random word
    images. Pre-caches tensors in RAM, same pattern as the other datasets.

    Vocabulary is reused from existing IAM label files when present so the
    synthetic word distribution matches real training data; falls back to a
    short common-words list if no labels are available.
    """

    def __init__(self, num_samples=8000, fonts_dir="data/fonts", seed=42):
        self.cached = []
        os.makedirs(fonts_dir, exist_ok=True)

        font_paths = self._ensure_fonts(fonts_dir)
        if not font_paths:
            print("  Synthetic: no fonts available, skipping")
            return

        words = self._build_word_list()
        if not words:
            print("  Synthetic: empty word list, skipping")
            return

        rng = random.Random(seed)
        attempts = 0
        max_attempts = num_samples * 3
        while len(self.cached) < num_samples and attempts < max_attempts:
            attempts += 1
            text = rng.choice(words)
            font_path = rng.choice(font_paths)
            try:
                pil = self._render_word(text, font_path, rng)
                if pil is None:
                    continue
                tensor = base_transform(pil)
                encoded = encode_text(text)
                if not encoded:
                    continue
                label = torch.tensor(encoded, dtype=torch.long)
                self.cached.append((tensor, label, text))
            except Exception:
                continue

        print(f"  Synthetic samples: {len(self.cached)} "
              f"(from {len(font_paths)} fonts, {len(words)} words)")

    def _ensure_fonts(self, fonts_dir):
        existing = [
            os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir)
            if f.lower().endswith((".ttf", ".otf"))
        ]
        if existing:
            return existing

        print(f"  Synthetic: bootstrapping fonts to {fonts_dir}/ ...")
        for fname, url in _SYNTH_FONT_URLS:
            dst = os.path.join(fonts_dir, fname)
            try:
                urllib.request.urlretrieve(url, dst)
            except Exception as e:
                print(f"    failed: {fname} ({e})")

        return [
            os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir)
            if f.lower().endswith((".ttf", ".otf"))
        ]

    def _build_word_list(self):
        wordset = set()
        for path in (
            os.path.join("data", "iam", "train_gt.txt"),
            os.path.join("data", "iam", "val_gt.txt"),
            os.path.join("data", "iam", "linux_gt.txt"),
        ):
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t", 1) if "\t" in line else line.split(None, 1)
                    text = _normalize_text_for_charset(parts[-1])
                    for word in text.split():
                        if 1 <= len(word) <= 20 and encode_text(word):
                            wordset.add(word)

        if not wordset:
            wordset = {w for w in _SYNTH_FALLBACK_WORDS.split() if encode_text(w)}

        # Mix in drug names so the model directly learns prescription-domain
        # vocabulary. Without this, drug names are out-of-distribution and
        # need the post-processing bias to fix them. With this, the model
        # itself outputs them — bias becomes a backstop, not a crutch.
        try:
            from services.drug_bias import _DRUGS
            for d in _DRUGS:
                if 4 <= len(d) <= 20 and encode_text(d):
                    wordset.add(d)
        except Exception:
            pass

        return list(wordset)

    def _render_word(self, text, font_path, rng):
        font_size = rng.randint(28, 56)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            return None

        bbox = font.getbbox(text)
        if bbox is None:
            return None
        w = max(bbox[2] - bbox[0] + 30, 40)
        h = max(bbox[3] - bbox[1] + 20, 30)

        bg = rng.randint(245, 255)
        ink = rng.randint(0, 70)
        img = Image.new("RGB", (w, h), color=(bg, bg, bg))
        draw = ImageDraw.Draw(img)
        x = 15 - bbox[0]
        y = 10 - bbox[1] + rng.randint(-2, 2)
        draw.text((x, y), text, font=font, fill=(ink, ink, ink))

        if rng.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
        return img

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, idx):
        return self.cached[idx]


class Imgur5KDataset(Dataset):
    """Loader for Meta's IMGUR5K Handwriting dataset.

    Real-world handwriting images scraped from Imgur. Genuinely diverse
    (clean to chaotic) — complements GNHK and IAM well for general HTR.

    Expected layout (matches Meta's `download_imgur5k.py` script output):
        root_dir/
            images/<image_hash>.jpg
            dataset_info/imgur5k_data.json   (or just imgur5k_data.json at root)

    The JSON is a dict keyed by image_hash; per-image entries carry an
    `annotations` list of word-level boxes + transcripts.

    Setup:
        1. git clone https://github.com/facebookresearch/IMGUR5K-Handwriting-Dataset
        2. Run their `download_imgur5k.py` (downloads ~5k images from Imgur;
           ~50 % URLs are dead in 2026 so expect 2.5–3k actual images)
        3. Move/symlink the resulting `images/` and `dataset_info/` into
           data/imgur5k/
    """

    def __init__(self, root_dir):
        self.cached = []
        if not root_dir or not os.path.isdir(root_dir):
            return

        info_path = None
        for candidate in (
            os.path.join(root_dir, "dataset_info", "imgur5k_data.json"),
            os.path.join(root_dir, "imgur5k_data.json"),
            os.path.join(root_dir, "dataset_info.json"),
        ):
            if os.path.exists(candidate):
                info_path = candidate
                break

        images_dir = os.path.join(root_dir, "images")
        if not info_path or not os.path.isdir(images_dir):
            return

        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            return

        for image_id, meta in info.items():
            if not isinstance(meta, dict):
                continue
            annotations = meta.get("annotations") or meta.get("words") or []
            if not annotations:
                continue

            img_path = None
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = os.path.join(images_dir, image_id + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if not img_path:
                continue

            full = cv2.imread(img_path)
            if full is None:
                continue

            for ann in annotations:
                text = (ann.get("word") or ann.get("text") or "").strip()
                if not text or text in ("###", "."):
                    continue

                bbox = ann.get("bbox") or ann.get("xywh") or ann.get("polygon")
                if not bbox:
                    continue
                if isinstance(bbox, dict):
                    xs = [bbox.get(k, 0) for k in ("x0", "x1", "x2", "x3")]
                    ys = [bbox.get(k, 0) for k in ("y0", "y1", "y2", "y3")]
                    x0, y0 = max(int(min(xs)), 0), max(int(min(ys)), 0)
                    x1, y1 = int(max(xs)), int(max(ys))
                elif len(bbox) >= 4:
                    x, y, w, h = (int(v) for v in bbox[:4])
                    x0, y0 = max(x, 0), max(y, 0)
                    x1, y1 = x + w, y + h
                else:
                    continue

                crop = full[y0:y1, x0:x1]
                if crop.size == 0:
                    continue

                normalized = _normalize_text_for_charset(text)
                if not normalized:
                    continue
                encoded = encode_text(normalized)
                if not encoded:
                    continue

                try:
                    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    tensor = base_transform(pil)
                    label = torch.tensor(encoded, dtype=torch.long)
                    self.cached.append((tensor, label, normalized))
                except Exception:
                    continue

        if self.cached:
            print(f"  Imgur5K samples: {len(self.cached)}")

    def __len__(self):
        return len(self.cached)

    def __getitem__(self, idx):
        return self.cached[idx]


def _interleaved_split(dataset, val_frac):
    """Deterministic interleaved train/val split — every Nth sample to val.

    Caveat: not writer-disjoint. IAM's word_id encodes a form ID but multiple
    forms exist per writer; a true Aachen-style writer-disjoint split would
    need ascii/forms.txt to map form→writer. This is good enough to measure
    "does the model generalize beyond what it saw in train" but slightly
    overestimates accuracy because val may contain seen-writer samples.
    """
    n = len(dataset)
    if n == 0 or val_frac <= 0:
        return dataset, None
    step = max(int(round(1 / val_frac)), 2)
    val_idx = list(range(0, n, step))
    val_set = set(val_idx)
    train_idx = [i for i in range(n) if i not in val_set]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


class IndexedDataset(Dataset):
    """Wrap a dataset so __getitem__ also returns the global sample index.

    Needed for hard-example mining: the training loop tracks per-sample
    losses and rebuilds the sampler from them, which only works if every
    item is identifiable when it comes back from the DataLoader.
    """

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return (*item, idx)


def collate_fn(batch):
    """Custom collate: stack images and concatenate variable-length labels
    into the flat format required by CTCLoss. Carries through optional
    sample indices appended by IndexedDataset."""
    if len(batch[0]) == 4:
        images, labels, texts, indices = zip(*batch)
        indices = torch.tensor(indices, dtype=torch.long)
    else:
        images, labels, texts = zip(*batch)
        indices = None

    images = torch.stack(images)
    label_lengths = torch.tensor([len(l) for l in labels])
    labels_concat = torch.cat(labels)

    if indices is None:
        return images, labels_concat, label_lengths, texts
    return images, labels_concat, label_lengths, texts, indices


def build_weighted_train_set(
    gnhk_dir, iam_dir=None, imgur5k_dir=None,
    synthetic_samples=0, iam_val_frac=0.1,
):
    """Build a weighted training dataset combining real + synthetic sources.

    Args:
        gnhk_dir:           path to GNHK training data
        iam_dir:            optional path to IAM dataset
        imgur5k_dir:        optional path to IMGUR5K dataset (Meta's layout)
        synthetic_samples:  if > 0, generate this many font-rendered word
                            images and mix them in at _SYNTH_WEIGHT
        iam_val_frac:       fraction of IAM to hold out for validation. The
                            held-out portion is returned separately; the
                            training set sees only the remaining 1-frac.

    Returns:
        train_set:    ConcatDataset (or single Dataset if only one source)
        sampler:      WeightedRandomSampler for use with DataLoader
        base_weights: numpy array of per-sample source weights, same length
                      as train_set. Used by hard-example mining to construct
                      a new sampler = base_weights × per_sample_loss.
        iam_val:      held-out IAM Subset (or None if IAM unavailable / no
                      split). Caller appends this to the validation set.
    """
    import numpy as np

    gnhk = GNHKDataset(gnhk_dir)
    print(f"  GNHK samples: {len(gnhk)}")

    datasets = [gnhk]
    weights = [_GNHK_WEIGHT] * len(gnhk)

    iam_val = None
    if iam_dir and os.path.isdir(iam_dir):
        iam = IAMDataset(iam_dir)
        if len(iam) > 0:
            iam_train, iam_val = _interleaved_split(iam, iam_val_frac)
            print(
                f"  IAM split: {len(iam_train)} train, "
                f"{0 if iam_val is None else len(iam_val)} val"
            )
            datasets.append(iam_train)
            weights += [_IAM_WEIGHT] * len(iam_train)

    if imgur5k_dir and os.path.isdir(imgur5k_dir):
        imgur = Imgur5KDataset(imgur5k_dir)
        if len(imgur) > 0:
            datasets.append(imgur)
            weights += [_IAM_WEIGHT] * len(imgur)  # treat as same tier as IAM

    if synthetic_samples and synthetic_samples > 0:
        synth = SyntheticDataset(num_samples=synthetic_samples)
        if len(synth) > 0:
            datasets.append(synth)
            weights += [_SYNTH_WEIGHT] * len(synth)

    train_set = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"  Total combined: {len(train_set)}")

    base_weights = np.asarray(weights, dtype=np.float64)
    sampler = WeightedRandomSampler(
        base_weights.tolist(),
        num_samples=len(train_set),
        replacement=True,
    )
    return train_set, sampler, base_weights, iam_val


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