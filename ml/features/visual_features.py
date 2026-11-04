"""
PhishGuard – Visual Feature Extraction

Extracts numerical visual features from images/screenshots for phishing detection
using OpenCV and NumPy.
Does not make network requests or invent phishing classifications.
"""

from __future__ import annotations

import os
from typing import Any, Union

import cv2
import numpy as np

# ── Feature names (canonical order) ─────────────────────────────

VISUAL_FEATURE_NAMES: list[str] = [
    # Geometry & Dimensions
    "width",
    "height",
    "aspect_ratio",
    # Brightness / Luminance
    "mean_brightness",
    "std_brightness",
    # Color Distribution (BGR channels)
    "mean_blue",
    "mean_green",
    "mean_red",
    "std_blue",
    "std_green",
    "std_red",
    "color_variance",
    # Edge & Texture
    "edge_density",
    "image_entropy",
    # Dominant Colors (Primary cluster)
    "dominant_color_b",
    "dominant_color_g",
    "dominant_color_r",
    "dominant_color_proportion",
    "dominant_color_1_b",
    "dominant_color_1_g",
    "dominant_color_1_r",
    "dominant_color_1_proportion",
    # Secondary Dominant Color
    "dominant_color_2_b",
    "dominant_color_2_g",
    "dominant_color_2_r",
    "dominant_color_2_proportion",
    # Tertiary Dominant Color
    "dominant_color_3_b",
    "dominant_color_3_g",
    "dominant_color_3_r",
    "dominant_color_3_proportion",
]


# ── Public API ──────────────────────────────────────────────────

def extract_visual_features(
    image: Union[np.ndarray, str, bytes, os.PathLike, Any],
) -> dict[str, int | float]:
    """Extract numerical visual features from an image.

    Parameters
    ----------
    image : np.ndarray | str | bytes | os.PathLike
        The input image as a NumPy array (BGR, Grayscale, or BGRA),
        a valid file path string/PathLike, or raw image bytes.

    Returns
    -------
    dict[str, int | float]
        Dictionary of visual features with canonical keys defined in
        `VISUAL_FEATURE_NAMES`. If the image is invalid, empty, or
        unreadable, a dictionary with safe zero values is returned.
    """
    bgr = _load_image(image)
    if bgr is None:
        return _empty_features()

    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return _empty_features()

    # 1. Dimensions and Aspect Ratio
    aspect_ratio = round(float(w / h), 4)

    # 2. Grayscale conversion for brightness, edge, and entropy
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = round(float(np.mean(gray)), 4)
    std_brightness = round(float(np.std(gray)), 4)

    # 3. Color distribution across BGR channels
    b_channel = bgr[:, :, 0]
    g_channel = bgr[:, :, 1]
    r_channel = bgr[:, :, 2]

    mean_b = round(float(np.mean(b_channel)), 4)
    mean_g = round(float(np.mean(g_channel)), 4)
    mean_r = round(float(np.mean(r_channel)), 4)

    std_b = round(float(np.std(b_channel)), 4)
    std_g = round(float(np.std(g_channel)), 4)
    std_r = round(float(np.std(r_channel)), 4)

    var_b = float(np.var(b_channel))
    var_g = float(np.var(g_channel))
    var_r = float(np.var(r_channel))
    color_variance = round(float((var_b + var_g + var_r) / 3.0), 4)

    # 4. Edge density via Canny
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    edge_density = round(float(np.count_nonzero(edges) / (w * h)), 4)

    # 5. Image Shannon Entropy
    image_entropy = round(_compute_entropy(gray), 4)

    # 6. Dominant Colors via Color Quantization / K-Means
    dominant = _extract_dominant_colors(bgr)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "mean_brightness": mean_brightness,
        "std_brightness": std_brightness,
        "mean_blue": mean_b,
        "mean_green": mean_g,
        "mean_red": mean_r,
        "std_blue": std_b,
        "std_green": std_g,
        "std_red": std_r,
        "color_variance": color_variance,
        "edge_density": edge_density,
        "image_entropy": image_entropy,
        # Primary Dominant Color
        "dominant_color_b": dominant[0]["b"],
        "dominant_color_g": dominant[0]["g"],
        "dominant_color_r": dominant[0]["r"],
        "dominant_color_proportion": dominant[0]["proportion"],
        "dominant_color_1_b": dominant[0]["b"],
        "dominant_color_1_g": dominant[0]["g"],
        "dominant_color_1_r": dominant[0]["r"],
        "dominant_color_1_proportion": dominant[0]["proportion"],
        # Secondary Dominant Color
        "dominant_color_2_b": dominant[1]["b"],
        "dominant_color_2_g": dominant[1]["g"],
        "dominant_color_2_r": dominant[1]["r"],
        "dominant_color_2_proportion": dominant[1]["proportion"],
        # Tertiary Dominant Color
        "dominant_color_3_b": dominant[2]["b"],
        "dominant_color_3_g": dominant[2]["g"],
        "dominant_color_3_r": dominant[2]["r"],
        "dominant_color_3_proportion": dominant[2]["proportion"],
    }


# ── Internal Helpers ────────────────────────────────────────────

def _empty_features() -> dict[str, int | float]:
    """Return dictionary of zeroed features for invalid or missing inputs."""
    res: dict[str, int | float] = {}
    for name in VISUAL_FEATURE_NAMES:
        if name in ("width", "height"):
            res[name] = 0
        else:
            res[name] = 0.0
    return res


def _load_image(image: Any) -> np.ndarray | None:
    """Normalize and convert input image into a 3-channel BGR uint8 NumPy array."""
    if image is None:
        return None

    # Case 1: File path (str or os.PathLike)
    if isinstance(image, (str, os.PathLike)):
        path_str = os.fspath(image)
        if not os.path.isfile(path_str):
            return None
        try:
            loaded = cv2.imread(path_str, cv2.IMREAD_COLOR)
            return loaded if loaded is not None else None
        except Exception:
            return None

    # Case 2: Raw bytes
    if isinstance(image, (bytes, bytearray, memoryview)):
        try:
            nparr = np.frombuffer(image, np.uint8)
            if nparr.size == 0:
                return None
            decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return decoded if decoded is not None else None
        except Exception:
            return None

    # Case 3: NumPy array
    if isinstance(image, np.ndarray):
        if image.size == 0 or image.ndim < 2:
            return None

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            return None

        arr = image

        # Handle float arrays (convert to uint8 in range 0..255)
        if np.issubdtype(arr.dtype, np.floating):
            if np.isnan(arr).any() or np.isinf(arr).any():
                return None
            max_val = float(arr.max()) if arr.size > 0 else 0.0
            if max_val <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        # Grayscale (2D or 3D with 1 channel)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3:
            channels = arr.shape[2]
            if channels == 1:
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if channels == 3:
                return arr
            if channels == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        return None

    return None


def _compute_entropy(gray: np.ndarray) -> float:
    """Compute Shannon entropy of a grayscale image."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    total = hist.sum()
    if total <= 0:
        return 0.0

    probs = hist[hist > 0] / total
    return -float(np.sum(probs * np.log2(probs)))


def _extract_dominant_colors(
    bgr: np.ndarray,
    max_k: int = 3,
) -> list[dict[str, float]]:
    """Extract top dominant colors using K-Means clustering.

    Downsamples large images to ensure sub-10ms extraction time.
    Returns a list of 3 dicts: ``[{'b': ..., 'g': ..., 'r': ..., 'proportion': ...}, ...]``.
    """
    default_entry = {"b": 0.0, "g": 0.0, "r": 0.0, "proportion": 0.0}

    h, w = bgr.shape[:2]
    # Downsample for fast clustering if larger than 128px on any side
    max_dim = max(h, w)
    if max_dim > 128:
        scale = 128.0 / max_dim
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        sample = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        sample = bgr

    pixels = sample.reshape((-1, 3)).astype(np.float32)
    n_pixels = len(pixels)
    if n_pixels == 0:
        return [dict(default_entry) for _ in range(max_k)]

    unique_colors = len(np.unique(pixels, axis=0))
    k = min(max_k, unique_colors)

    if k <= 0:
        return [dict(default_entry) for _ in range(max_k)]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    try:
        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            attempts=3,
            flags=cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=k)
        order = np.argsort(-counts)

        results: list[dict[str, float]] = []
        for idx in order:
            prop = round(float(counts[idx] / n_pixels), 4)
            b = round(float(centers[idx][0]), 2)
            g = round(float(centers[idx][1]), 2)
            r = round(float(centers[idx][2]), 2)
            results.append({"b": b, "g": g, "r": r, "proportion": prop})

        # Pad with zeros if fewer than max_k clusters were formed
        while len(results) < max_k:
            results.append(dict(default_entry))

        return results
    except Exception:
        # Fallback to mean color if kmeans encounters an issue
        mean_b = round(float(np.mean(pixels[:, 0])), 2)
        mean_g = round(float(np.mean(pixels[:, 1])), 2)
        mean_r = round(float(np.mean(pixels[:, 2])), 2)
        res = [{"b": mean_b, "g": mean_g, "r": mean_r, "proportion": 1.0}]
        while len(res) < max_k:
            res.append(dict(default_entry))
        return res

