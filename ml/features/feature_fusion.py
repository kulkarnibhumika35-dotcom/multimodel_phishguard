"""
PhishGuard – Multimodal Feature Fusion

Combines URL lexical features and screenshot visual features into a unified,
fixed-order numerical feature vector for multimodal phishing detection.

Guarantees:
- Preserves all 20 URL features from ml/features/url_features.py.
- Preserves all 30 documented visual features from ml/features/visual_features.py.
- Uses an explicit `has_visual_features` indicator so missing visual data is never
  silently masked or confused with legitimate zero values.
- Validates missing, non-numeric, NaN, and infinite values safely.
- Does not perform phishing classification or invent training data.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Mapping, Union

import numpy as np

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES
from ml.features.url_features import extract_url_features
from ml.features.visual_features import VISUAL_FEATURE_NAMES, extract_visual_features

# ── Canonical Fused Feature Names (Fixed Order) ────────────────

# Visual availability indicator placed between modalities
VISUAL_AVAILABILITY_FEATURE = "has_visual_features"

FUSED_FEATURE_NAMES: list[str] = (
    list(URL_FEATURE_NAMES)
    + [VISUAL_AVAILABILITY_FEATURE]
    + list(VISUAL_FEATURE_NAMES)
)


# ── Public API ──────────────────────────────────────────────────

def fuse_features(
    url_input: Union[Mapping[str, Any], str, None],
    visual_input: Union[Mapping[str, Any], np.ndarray, str, bytes, os.PathLike, None] = None,
) -> dict[str, int | float]:
    """Combine URL features and visual features into an ordered dictionary.

    Parameters
    ----------
    url_input : Mapping[str, Any] | str | None
        Either a pre-extracted URL features dictionary or a raw URL string.
        If invalid or None, safe zero defaults are used.
    visual_input : Mapping[str, Any] | np.ndarray | str | bytes | os.PathLike | None
        Either a pre-extracted visual features dictionary, a NumPy image,
        a file path, or image bytes. If unavailable or invalid,
        `has_visual_features` is explicitly set to 0 and all visual
        features are safely zeroed.

    Returns
    -------
    dict[str, int | float]
        Dictionary matching the exact keys and order of `FUSED_FEATURE_NAMES`.
    """
    url_feats = _process_url_input(url_input)
    visual_feats, has_visual = _process_visual_input(visual_input)

    fused: dict[str, int | float] = {}

    # 1. All 20 URL features
    for name in URL_FEATURE_NAMES:
        fused[name] = url_feats[name]

    # 2. Explicit visual availability indicator (1 if present, 0 if missing/invalid)
    fused[VISUAL_AVAILABILITY_FEATURE] = has_visual

    # 3. All documented visual features
    for name in VISUAL_FEATURE_NAMES:
        fused[name] = visual_feats[name]

    return fused


def fuse_feature_vector(
    url_input: Union[Mapping[str, Any], str, None],
    visual_input: Union[Mapping[str, Any], np.ndarray, str, bytes, os.PathLike, None] = None,
) -> np.ndarray:
    """Combine URL and visual features into a 1D NumPy float64 vector.

    Matches the exact fixed ordering of `FUSED_FEATURE_NAMES`.

    Returns
    -------
    np.ndarray
        1D array of shape `(len(FUSED_FEATURE_NAMES),)` with dtype `float64`.
    """
    fused_dict = fuse_features(url_input, visual_input)
    return fused_dict_to_vector(fused_dict)


def fused_dict_to_vector(fused_dict: Mapping[str, Any]) -> np.ndarray:
    """Convert a fused feature dictionary to a 1D NumPy float64 vector."""
    values: list[float] = []
    for name in FUSED_FEATURE_NAMES:
        val = fused_dict.get(name, 0.0)
        values.append(_sanitize_float(val))
    return np.array(values, dtype=np.float64)


def get_fused_feature_names() -> list[str]:
    """Return a copy of the canonical fixed feature order list."""
    return list(FUSED_FEATURE_NAMES)


# ── Internal Sanitization & Processing ──────────────────────────

def _sanitize_float(value: Any, default: float = 0.0) -> float:
    """Convert *value* to a finite float, falling back to *default* on error/NaN/Inf."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def _sanitize_int(value: Any, default: int = 0) -> int:
    """Convert *value* to an int, falling back to *default* on error/NaN/Inf."""
    if value is None:
        return default
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return int(f)
    except (ValueError, TypeError):
        return default


def _process_url_input(url_input: Any) -> dict[str, int | float]:
    """Extract or validate URL features."""
    if isinstance(url_input, str):
        # Raw URL string provided -> extract features
        return extract_url_features(url_input)

    if isinstance(url_input, Mapping):
        # Pre-extracted dictionary provided -> validate and sanitize each key
        res: dict[str, int | float] = {}
        for name in URL_FEATURE_NAMES:
            raw_val = url_input.get(name)
            if name == "url_entropy":
                res[name] = round(_sanitize_float(raw_val, 0.0), 4)
            else:
                res[name] = _sanitize_int(raw_val, 0)
        return res

    # Missing, invalid or wrong type -> zero defaults
    return {
        name: 0.0 if name == "url_entropy" else 0
        for name in URL_FEATURE_NAMES
    }


def _process_visual_input(visual_input: Any) -> tuple[dict[str, int | float], int]:
    """Extract or validate visual features and return (features_dict, has_visual_flag)."""
    default_visual = {
        name: 0 if name in ("width", "height") else 0.0
        for name in VISUAL_FEATURE_NAMES
    }

    if visual_input is None:
        return default_visual, 0

    # Case 1: Image array, raw bytes, or file path
    if isinstance(visual_input, (np.ndarray, bytes, bytearray, memoryview, os.PathLike)):
        raw_feats = extract_visual_features(visual_input)
        is_available = int(raw_feats.get("width", 0) > 0 and raw_feats.get("height", 0) > 0)
        return raw_feats, is_available

    # Case 2: String path to image file (distinguish from non-existent string)
    if isinstance(visual_input, str):
        if os.path.isfile(visual_input):
            raw_feats = extract_visual_features(visual_input)
            is_available = int(raw_feats.get("width", 0) > 0 and raw_feats.get("height", 0) > 0)
            return raw_feats, is_available
        # String is not a valid file
        return default_visual, 0

    # Case 3: Pre-extracted dictionary
    if isinstance(visual_input, Mapping):
        # Check explicit availability overrides if provided
        explicit_flag = visual_input.get(VISUAL_AVAILABILITY_FEATURE)
        if explicit_flag is None:
            explicit_flag = visual_input.get("visual_available")

        if explicit_flag is not None:
            is_available = 1 if bool(explicit_flag) and str(explicit_flag) not in ("0", "False", "false") else 0
        else:
            # Infer from dimensions: if both width and height > 0, visual data is available
            w_raw = _sanitize_int(visual_input.get("width"), 0)
            h_raw = _sanitize_int(visual_input.get("height"), 0)
            is_available = 1 if (w_raw > 0 and h_raw > 0) else 0

        if not is_available:
            return default_visual, 0

        # Sanitize all visual features
        res: dict[str, int | float] = {}
        for name in VISUAL_FEATURE_NAMES:
            raw_val = visual_input.get(name)
            if name in ("width", "height"):
                res[name] = _sanitize_int(raw_val, 0)
            else:
                res[name] = round(_sanitize_float(raw_val, 0.0), 4)

        return res, 1

    return default_visual, 0

