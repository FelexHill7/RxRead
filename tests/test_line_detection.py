"""CC-based line detector — sanity checks on synthetic binary inputs."""

import numpy as np
import pytest


def _make_binary_with_lines(line_centers, image_h=400, image_w=600,
                            line_height=40, blob_width=60, blobs_per_line=8):
    """Construct a fake binary image with known line positions.

    Each "line" is a horizontal row of CC blobs at the given Y-center. Lets
    us test line detection without needing real text images.
    """
    binary = np.zeros((image_h, image_w), dtype=np.uint8)
    for cy in line_centers:
        y0 = max(cy - line_height // 2, 0)
        y1 = min(cy + line_height // 2, image_h)
        for i in range(blobs_per_line):
            x0 = 30 + i * (blob_width + 20)
            x1 = x0 + blob_width
            if x1 > image_w:
                break
            binary[y0:y1, x0:x1] = 255
    return binary


def test_detects_three_lines():
    from services.inference import _detect_lines
    binary = _make_binary_with_lines([60, 180, 300], image_h=400)
    spans = _detect_lines(binary)
    assert len(spans) == 3, f"expected 3 lines, got {len(spans)}: {spans}"


def test_detects_six_lines():
    from services.inference import _detect_lines
    binary = _make_binary_with_lines([50, 110, 170, 230, 290, 350], image_h=400)
    spans = _detect_lines(binary)
    assert len(spans) == 6, f"expected 6 lines, got {len(spans)}: {spans}"


def test_returns_empty_for_blank_image():
    from services.inference import _detect_lines
    binary = np.zeros((400, 600), dtype=np.uint8)
    assert _detect_lines(binary) == []


def test_spans_are_sorted_top_to_bottom():
    from services.inference import _detect_lines
    binary = _make_binary_with_lines([300, 60, 180], image_h=400)
    spans = _detect_lines(binary)
    centers = [(y0 + y1) / 2 for y0, y1 in spans]
    assert centers == sorted(centers)


def test_spans_roughly_match_centers():
    """Detected line centers should be within ±20px of the real centers."""
    from services.inference import _detect_lines
    truth_centers = [80, 200, 320]
    binary = _make_binary_with_lines(truth_centers, image_h=400)
    spans = _detect_lines(binary)
    detected_centers = [(y0 + y1) / 2 for y0, y1 in spans]
    assert len(detected_centers) == len(truth_centers)
    for true_c, det_c in zip(truth_centers, detected_centers):
        assert abs(true_c - det_c) < 20, \
            f"detected center {det_c} too far from truth {true_c}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
