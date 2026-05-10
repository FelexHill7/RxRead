"""
preprocessing.py — Image transforms and GPU batch augmentation.

Fixes applied:
- FIXED: center padding (removes left-alignment bias that hurts CTC)
- FIXED: safe grayscale conversion (no silent channel dropping)
- FIXED: consistent preprocessing distribution for training/inference
"""

import math
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF

from config import IMG_HEIGHT, IMG_WIDTH


# ── Aspect ratio preserving resize ───────────────────────────────────────────
def resize_with_padding(img, target_h=IMG_HEIGHT, target_w=IMG_WIDTH):
    """
    Resize image while preserving aspect ratio and center-pad to target size.
    FIX: center padding improves CTC alignment significantly.
    """

    img = np.array(img)

    # ── SAFE grayscale handling ─────────────────────────────
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    h, w = img.shape
    scale = target_h / h
    new_w = int(w * scale)

    img = cv2.resize(img, (new_w, target_h))

    # ── FIXED: center padding (IMPORTANT for CTC) ───────────
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255

    start_x = max(0, (target_w - new_w) // 2)
    end_x = start_x + min(new_w, target_w)

    padded[:, start_x:end_x] = img[:, :min(new_w, target_w)]

    return Image.fromarray(padded)


# ── Base transform (used everywhere) ─────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(resize_with_padding),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # keep consistent with training
])


# ── Tensor-space rotation for TTA ────────────────────────────────────────────
# Rotation runs AFTER resize_with_padding so it does not change the effective
# scale of the cropped word — previously, expanding the bbox via PIL rotation
# before resize made the rotated TTA branches see a different scale than base.
def _rotate_tensor(t, angle):
    return TF.rotate(t, angle=angle, fill=1.0)

def _tta_neg3(img):
    return _rotate_tensor(base_transform(img), -3)

def _tta_pos3(img):
    return _rotate_tensor(base_transform(img), 3)


# ── Test-time augmentation ───────────────────────────────────────────────────
# Each entry is a callable PIL → tensor (1, H, W).
tta_transforms = [
    base_transform,
    transforms.Lambda(_tta_neg3),
    transforms.Lambda(_tta_pos3),
]


# ── GPU batch augmentation (training only) ───────────────────────────────────
@torch.no_grad()
def gpu_augment(images):
    B, C, H, W = images.shape
    device = images.device

    # ── affine ─────────────────────────────
    angles = (torch.rand(B, device=device) * 2 - 1) * 5 * (math.pi / 180)
    shear  = (torch.rand(B, device=device) * 2 - 1) * 5 * (math.pi / 180)
    tx     = (torch.rand(B, device=device) * 2 - 1) * 0.05
    ty     = (torch.rand(B, device=device) * 2 - 1) * 0.05
    # Horizontal stretch — handwriting has high width variance; this is the
    # single augmentation that consistently helps OCR. Range 0.85..1.15.
    sx     = 1.0 + (torch.rand(B, device=device) * 2 - 1) * 0.15

    cos_a, sin_a = torch.cos(angles), torch.sin(angles)

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = (cos_a + shear * sin_a) * sx
    theta[:, 0, 1] = -sin_a + shear * cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a * sx
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty

    grid = torch.nn.functional.affine_grid(theta, images.shape, align_corners=False)
    images = torch.nn.functional.grid_sample(
        images, grid, align_corners=False, padding_mode="border"
    )

    # ── elastic distortion ──────────────────
    if torch.rand(1).item() > 0.7:
        alpha = 3.0
        sigma = 4.0

        dx = torch.rand(B, 1, H, W, device=device) * 2 - 1
        dy = torch.rand(B, 1, H, W, device=device) * 2 - 1

        k = 7
        ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
        kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel = kernel / kernel.sum()
        kernel2d = (kernel[:, None] * kernel[None, :]).view(1, 1, k, k)

        dx = torch.nn.functional.conv2d(dx, kernel2d, padding=k // 2) * alpha
        dy = torch.nn.functional.conv2d(dy, kernel2d, padding=k // 2) * alpha

        dx = dx.squeeze(1) / (W / 2)
        dy = dy.squeeze(1) / (H / 2)

        base_grid = torch.nn.functional.affine_grid(
            torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1),
            images.shape,
            align_corners=False
        )

        base_grid[..., 0] += dx
        base_grid[..., 1] += dy

        images = torch.nn.functional.grid_sample(
            images, base_grid, align_corners=False, padding_mode="border"
        )

    # ── brightness / contrast ───────────────
    brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.3
    contrast   = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.3

    mean = images.mean(dim=(2, 3), keepdim=True)
    images = contrast * (images - mean) + mean + (brightness - 1.0)

    # ── blur ────────────────────────────────
    sigma = 0.1 + torch.rand(1, device=device).item() * 0.9
    k = 3
    ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel2d = (kernel[:, None] * kernel[None, :]).view(1, 1, k, k)

    images = torch.nn.functional.conv2d(images, kernel2d, padding=k // 2)

    # ── morphology ─────────────────────────
    morph_roll = torch.rand(1).item()
    if morph_roll < 0.3:
        images = -torch.nn.functional.max_pool2d(-images, 3, 1, 1)
    elif morph_roll < 0.6:
        images = torch.nn.functional.max_pool2d(images, 3, 1, 1)

    return images.clamp(-1, 1)