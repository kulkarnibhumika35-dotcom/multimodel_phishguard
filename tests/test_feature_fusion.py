"""Unit tests for ml.features.feature_fusion."""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from ml.features.feature_fusion import (
    FUSED_FEATURE_NAMES,
    VISUAL_AVAILABILITY_FEATURE,
    fuse_feature_vector,
    fuse_features,
    fused_dict_to_vector,
    get_fused_feature_names,
)
from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES
from ml.features.url_features import extract_url_features
from ml.features.visual_features import VISUAL_FEATURE_NAMES, extract_visual_features


# ── Helpers ─────────────────────────────────────────────────────

def _make_dummy_url_features():
    """Generate a clean dictionary of 20 URL features."""
    return extract_url_features("https://secure.example.com/login?token=abc")


def _make_dummy_visual_features():
    """Generate a clean dictionary of 30 visual features from a synthetic image."""
    img = np.full((100, 200, 3), (120, 180, 240), dtype=np.uint8)
    return extract_visual_features(img)


# ── 1. Canonical Ordering & Counts ─────────────────────────────

class TestFeatureOrderAndCounts:
    def test_total_feature_count(self):
        # 20 URL features + 1 availability indicator + 30 Visual features = 51
        assert len(FUSED_FEATURE_NAMES) == 51
        assert len(URL_FEATURE_NAMES) == 20
        assert len(VISUAL_FEATURE_NAMES) == 30

    def test_feature_order_structure(self):
        # First 20 are URL features
        assert FUSED_FEATURE_NAMES[:20] == list(URL_FEATURE_NAMES)
        # 21st is the availability indicator
        assert FUSED_FEATURE_NAMES[20] == VISUAL_AVAILABILITY_FEATURE
        # Last 30 are Visual features
        assert FUSED_FEATURE_NAMES[21:] == list(VISUAL_FEATURE_NAMES)

    def test_get_fused_feature_names_returns_copy(self):
        names = get_fused_feature_names()
        assert names == FUSED_FEATURE_NAMES
        names.append("extra")
        assert len(FUSED_FEATURE_NAMES) == 51  # Unchanged


# ── 2. Valid Input Fusion ──────────────────────────────────────

class TestValidFusion:
    def test_fuse_pre_extracted_dicts(self):
        url_f = _make_dummy_url_features()
        vis_f = _make_dummy_visual_features()

        fused = fuse_features(url_f, vis_f)
        assert len(fused) == 51
        assert list(fused.keys()) == FUSED_FEATURE_NAMES

        # Verify URL features preserved
        for name in URL_FEATURE_NAMES:
            assert fused[name] == url_f[name]

        # Verify visual availability indicator is active
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 1

        # Verify visual features preserved
        for name in VISUAL_FEATURE_NAMES:
            assert fused[name] == vis_f[name]

    def test_fuse_raw_url_and_visual_dict(self):
        raw_url = "http://phish-login.xyz/verify?acc=123"
        vis_f = _make_dummy_visual_features()

        fused = fuse_features(raw_url, vis_f)
        assert fused["is_https"] == 0
        assert fused["has_suspicious_tld"] == 1
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 1
        assert fused["width"] == vis_f["width"]

    def test_fuse_url_dict_and_raw_image(self):
        url_f = _make_dummy_url_features()
        img = np.full((80, 160, 3), (50, 100, 150), dtype=np.uint8)

        fused = fuse_features(url_f, img)
        assert fused["url_length"] == url_f["url_length"]
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 1
        assert fused["width"] == 160
        assert fused["height"] == 80
        assert fused["aspect_ratio"] == 2.0

    def test_fuse_raw_url_and_raw_image(self):
        raw_url = "https://example.org"
        img = np.zeros((50, 50, 3), dtype=np.uint8)

        fused = fuse_features(raw_url, img)
        assert fused["is_https"] == 1
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 1
        assert fused["width"] == 50
        assert fused["height"] == 50


# ── 3. Missing & Unavailable Visual Features ───────────────────

class TestUnavailableVisualFeatures:
    def test_none_visual_input(self):
        url_f = _make_dummy_url_features()
        fused = fuse_features(url_f, None)

        # URL features intact
        assert fused["url_length"] == url_f["url_length"]
        # Explicit indicator tells model visual features are unavailable
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 0
        # Visual features safely zeroed
        assert fused["width"] == 0
        assert fused["height"] == 0
        assert fused["aspect_ratio"] == 0.0
        assert fused["mean_brightness"] == 0.0
        assert fused["dominant_color_proportion"] == 0.0

    def test_empty_visual_dict(self):
        url_f = _make_dummy_url_features()
        fused = fuse_features(url_f, {})
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 0
        assert fused["width"] == 0

    def test_zero_dimensions_visual_dict(self):
        url_f = _make_dummy_url_features()
        vis_f = {"width": 0, "height": 0, "mean_brightness": 0.0}
        fused = fuse_features(url_f, vis_f)
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 0

    def test_explicit_flag_disabled(self):
        url_f = _make_dummy_url_features()
        vis_f = _make_dummy_visual_features()
        vis_f[VISUAL_AVAILABILITY_FEATURE] = 0

        fused = fuse_features(url_f, vis_f)
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 0
        assert fused["width"] == 0
        assert fused["height"] == 0

    def test_explicit_visual_available_false(self):
        url_f = _make_dummy_url_features()
        vis_f = _make_dummy_visual_features()
        vis_f["visual_available"] = False

        fused = fuse_features(url_f, vis_f)
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 0
        assert fused["width"] == 0

    def test_nonexistent_image_file_path(self):
        url_f = _make_dummy_url_features()
        fused = fuse_features(url_f, "missing_screenshot_12345.png")
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 0
        assert fused["width"] == 0


# ── 4. Robust Validation of Invalid & Non-numeric Values ───────

class TestInputSanitization:
    def test_non_numeric_values_in_url_features(self):
        bad_url_f = {
            "url_length": "invalid_length",
            "is_https": None,
            "url_entropy": "nan",
            "num_dots": [1, 2, 3],
        }
        fused = fuse_features(bad_url_f, None)
        assert fused["url_length"] == 0
        assert fused["is_https"] == 0
        assert fused["url_entropy"] == 0.0
        assert fused["num_dots"] == 0

    def test_nan_and_inf_in_visual_features(self):
        vis_f = {
            "width": 100,
            "height": 100,
            "mean_brightness": float("nan"),
            "aspect_ratio": float("inf"),
            "edge_density": -float("inf"),
        }
        fused = fuse_features(_make_dummy_url_features(), vis_f)
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 1
        assert fused["mean_brightness"] == 0.0
        assert fused["aspect_ratio"] == 0.0
        assert fused["edge_density"] == 0.0

    def test_none_url_and_none_visual(self):
        fused = fuse_features(None, None)
        assert len(fused) == 51
        assert all(isinstance(v, (int, float)) for v in fused.values())
        assert all(v == 0 or v == 0.0 for v in fused.values())

    def test_arbitrary_types_as_inputs(self):
        fused = fuse_features(12345, object())
        assert len(fused) == 51
        assert fused[VISUAL_AVAILABILITY_FEATURE] == 0
        assert all(isinstance(v, (int, float)) for v in fused.values())


# ── 5. Vector Output & Formatting ──────────────────────────────

class TestVectorOutput:
    def test_fuse_feature_vector_shape_and_dtype(self):
        url_f = _make_dummy_url_features()
        vis_f = _make_dummy_visual_features()

        vec = fuse_feature_vector(url_f, vis_f)
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (51,)
        assert vec.dtype == np.float64
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

    def test_vector_matches_dict_values(self):
        url_f = _make_dummy_url_features()
        vis_f = _make_dummy_visual_features()

        fused_dict = fuse_features(url_f, vis_f)
        vec = fused_dict_to_vector(fused_dict)

        for i, name in enumerate(FUSED_FEATURE_NAMES):
            assert vec[i] == pytest.approx(float(fused_dict[name]), abs=1e-4)

    def test_vector_unavailable_visual(self):
        vec = fuse_feature_vector("https://example.com", None)
        assert vec.shape == (51,)
        # Availability indicator at index 20
        assert vec[20] == 0.0


# ── 6. Classification Honesty ──────────────────────────────────

class TestNoClassificationInvented:
    def test_does_not_invent_phishing_labels(self):
        fused = fuse_features("https://phishing-site.xyz/login", None)
        forbidden_keys = {"label", "prediction", "is_phishing", "phishing", "risk_score", "status"}
        assert forbidden_keys.isdisjoint(fused.keys())

