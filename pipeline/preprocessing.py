"""
preprocessing.py — Image transforms and GPU batch augmentation.

Improved version:
- Aspect ratio preserving resize (CRITICAL for handwriting)
- Cleaner TTA pipeline
- Same GPU augmentation (already strong)
"""

import math
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from config import IMG_HEIGHT, IMG_WIDTH


# ── Aspect ratio preserving resize ───────────────────────────────────────────
def resize_with_padding(img, target_h=IMG_HEIGHT, target_w=IMG_WIDTH):
    """Resize image while preserving aspect ratio and pad to target size."""
    img = np.array(img)

    h, w = img.shape
    scale = target_h / h
    new_w = int(w * scale)

    img = cv2.resize(img, (new_w, target_h))

    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    padded[:, :min(new_w, target_w)] = img[:, :target_w]

    return Image.fromarray(padded)


# ── Base transform (used everywhere) ─────────────────────────────────────────
base_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(resize_with_padding),  # 🔥 FIXED
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


# ── Test-time augmentation ───────────────────────────────────────────────────
tta_transforms = [
    base_transform,

    transforms.Compose([
        transforms.Grayscale(),
        transforms.Lambda(resize_with_padding),
        transforms.RandomRotation(degrees=(-3, -3)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),

    transforms.Compose([
        transforms.Grayscale(),
        transforms.Lambda(resize_with_padding),
        transforms.RandomRotation(degrees=(3, 3)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]),
]


# ── GPU batch augmentation (UNCHANGED — already strong) ──────────────────────
@torch.no_grad()
def gpu_augment(images):
    B, C, H, W = images.shape
    device = images.device

    # Random affine
    angles = (torch.rand(B, device=device) * 2 - 1) * 5 * (math.pi / 180)
    shear  = (torch.rand(B, device=device) * 2 - 1) * 5 * (math.pi / 180)
    tx     = (torch.rand(B, device=device) * 2 - 1) * 0.05
    ty     = (torch.rand(B, device=device) * 2 - 1) * 0.05

    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = cos_a + shear * sin_a
    theta[:, 0, 1] = -sin_a + shear * cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 1, 2] = ty

    grid = torch.nn.functional.affine_grid(theta, images.shape, align_corners=False)
    images = torch.nn.functional.grid_sample(images, grid, align_corners=False, padding_mode="border")

    # Elastic distortion
    if torch.rand(1).item() > 0.5:
        alpha = 3.0
        sigma = 4.0
        dx = torch.rand(B, 1, H, W, device=device) * 2 - 1
        dy = torch.rand(B, 1, H, W, device=device) * 2 - 1

        ek = 7
        eax = torch.arange(ek, dtype=torch.float32, device=device) - ek // 2
        ekernel = torch.exp(-0.5 * (eax / sigma) ** 2)
        ekernel = ekernel / ekernel.sum()
        ek2d = (ekernel[:, None] * ekernel[None, :]).view(1, 1, ek, ek)

        dx = torch.nn.functional.conv2d(dx, ek2d, padding=ek // 2) * alpha
        dy = torch.nn.functional.conv2d(dy, ek2d, padding=ek // 2) * alpha

        dx = dx.squeeze(1) / (W / 2)
        dy = dy.squeeze(1) / (H / 2)

        base_grid = torch.nn.functional.affine_grid(
            torch.eye(2, 3, device=device).unsqueeze(0).expand(B, -1, -1),
            images.shape, align_corners=False
        )

        base_grid[..., 0] += dx
        base_grid[..., 1] += dy

        images = torch.nn.functional.grid_sample(images, base_grid, align_corners=False, padding_mode="border")

    # Brightness + contrast
    brightness = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6
    contrast   = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.6

    mean = images.mean(dim=(2, 3), keepdim=True)
    images = contrast * (images - mean) + mean + (brightness - 1.0)

    # Gaussian blur
    sigma = 0.1 + torch.rand(1, device=device).item() * 0.9
    k = 3
    ax = torch.arange(k, dtype=torch.float32, device=device) - k // 2
    kernel = torch.exp(-0.5 * (ax / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel_2d = (kernel[:, None] * kernel[None, :]).view(1, 1, k, k)

    images = torch.nn.functional.conv2d(images, kernel_2d, padding=k // 2)

    # Morphology
    morph_roll = torch.rand(1).item()
    if morph_roll < 0.3:
        images = -torch.nn.functional.max_pool2d(-images, 3, 1, 1)
    elif morph_roll < 0.6:
        images = torch.nn.functional.max_pool2d(images, 3, 1, 1)

    return images.clamp(-1, 1)