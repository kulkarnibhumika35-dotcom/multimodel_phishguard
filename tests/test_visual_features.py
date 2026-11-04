"""Unit tests for ml.features.visual_features.extract_visual_features."""

import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pytest

from ml.features.visual_features import VISUAL_FEATURE_NAMES, extract_visual_features


# ── Helpers ─────────────────────────────────────────────────────

def _assert_valid_feature_dict(features: dict):
    """Verify that all canonical feature keys are present and have finite numeric values."""
    assert isinstance(features, dict)
    assert set(features.keys()) == set(VISUAL_FEATURE_NAMES)
    for name, value in features.items():
        assert isinstance(value, (int, float)), f"Feature '{name}' is not numeric: {type(value)}"
        assert not np.isnan(value), f"Feature '{name}' is NaN"
        assert not np.isinf(value), f"Feature '{name}' is Inf"


# ── 1. Geometry & Dimensions ───────────────────────────────────

class TestDimensionsAndAspectRatio:
    def test_horizontal_dimensions(self):
        img = np.zeros((100, 300, 3), dtype=np.uint8)
        feats = extract_visual_features(img)
        _assert_valid_feature_dict(feats)
        assert feats["width"] == 300
        assert feats["height"] == 100
        assert feats["aspect_ratio"] == 3.0

    def test_vertical_dimensions(self):
        img = np.zeros((400, 100, 3), dtype=np.uint8)
        feats = extract_visual_features(img)
        _assert_valid_feature_dict(feats)
        assert feats["width"] == 100
        assert feats["height"] == 400
        assert feats["aspect_ratio"] == 0.25

    def test_square_dimensions(self):
        img = np.zeros((120, 120, 3), dtype=np.uint8)
        feats = extract_visual_features(img)
        assert feats["width"] == 120
        assert feats["height"] == 120
        assert feats["aspect_ratio"] == 1.0


# ── 2. Brightness & Color Distribution ─────────────────────────

class TestBrightnessAndColorDistribution:
    def test_pure_black_image(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        feats = extract_visual_features(img)
        _assert_valid_feature_dict(feats)
        assert feats["mean_brightness"] == 0.0
        assert feats["std_brightness"] == 0.0
        assert feats["mean_blue"] == 0.0
        assert feats["mean_green"] == 0.0
        assert feats["mean_red"] == 0.0
        assert feats["color_variance"] == 0.0

    def test_pure_white_image(self):
        img = np.full((50, 50, 3), 255, dtype=np.uint8)
        feats = extract_visual_features(img)
        _assert_valid_feature_dict(feats)
        assert feats["mean_brightness"] == 255.0
        assert feats["mean_blue"] == 255.0
        assert feats["mean_green"] == 255.0
        assert feats["mean_red"] == 255.0
        assert feats["color_variance"] == 0.0

    def test_pure_red_image(self):
        # BGR: (0, 0, 255)
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, :, 2] = 255
        feats = extract_visual_features(img)
        assert feats["mean_red"] == 255.0
        assert feats["mean_green"] == 0.0
        assert feats["mean_blue"] == 0.0

    def test_pure_green_image(self):
        # BGR: (0, 255, 0)
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, :, 1] = 255
        feats = extract_visual_features(img)
        assert feats["mean_green"] == 255.0
        assert feats["mean_red"] == 0.0
        assert feats["mean_blue"] == 0.0

    def test_pure_blue_image(self):
        # BGR: (255, 0, 0)
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[:, :, 0] = 255
        feats = extract_visual_features(img)
        assert feats["mean_blue"] == 255.0
        assert feats["mean_green"] == 0.0
        assert feats["mean_red"] == 0.0


# ── 3. Edge Density ────────────────────────────────────────────

class TestEdgeDensity:
    def test_solid_image_has_zero_edge_density(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        feats = extract_visual_features(img)
        assert feats["edge_density"] == 0.0

    def test_checkerboard_pattern_has_positive_edge_density(self):
        # High frequency edges: alternating black and white bars
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, ::4] = 255
        img[:, 1::4] = 255
        feats = extract_visual_features(img)
        assert feats["edge_density"] > 0.0
        assert 0.0 < feats["edge_density"] <= 1.0


# ── 4. Image Entropy ───────────────────────────────────────────

class TestImageEntropy:
    def test_uniform_image_has_zero_entropy(self):
        img = np.full((80, 80, 3), 200, dtype=np.uint8)
        feats = extract_visual_features(img)
        assert feats["image_entropy"] == 0.0

    def test_random_noise_image_has_high_entropy(self):
        rng = np.random.RandomState(42)
        img = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        feats = extract_visual_features(img)
        assert feats["image_entropy"] > 6.0


# ── 5. Dominant Colors ─────────────────────────────────────────

class TestDominantColors:
    def test_solid_color_dominant(self):
        # Solid cyan: B=255, G=255, R=0
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :, 0] = 255
        img[:, :, 1] = 255
        feats = extract_visual_features(img)

        assert feats["dominant_color_b"] == 255.0
        assert feats["dominant_color_g"] == 255.0
        assert feats["dominant_color_r"] == 0.0
        assert feats["dominant_color_proportion"] == 1.0

    def test_two_color_proportions(self):
        # 75% red, 25% blue
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:75, :] = [0, 0, 255]  # Red
        img[75:, :] = [255, 0, 0]  # Blue
        feats = extract_visual_features(img)

        # Primary dominant color should be red with proportion ~0.75
        assert feats["dominant_color_r"] > 200
        assert feats["dominant_color_proportion"] >= 0.70

        # Secondary dominant color should be blue with proportion ~0.25
        assert feats["dominant_color_2_b"] > 200
        assert feats["dominant_color_2_proportion"] >= 0.20


# ── 6. Diverse Image Representations ───────────────────────────

class TestImageRepresentations:
    def test_grayscale_2d_array(self):
        img = np.full((60, 80), 120, dtype=np.uint8)
        feats = extract_visual_features(img)
        _assert_valid_feature_dict(feats)
        assert feats["width"] == 80
        assert feats["height"] == 60
        assert feats["mean_brightness"] == 120.0

    def test_bgra_4_channel_array(self):
        img = np.full((50, 50, 4), 100, dtype=np.uint8)
        feats = extract_visual_features(img)
        _assert_valid_feature_dict(feats)
        assert feats["width"] == 50
        assert feats["height"] == 50

    def test_float_normalized_array(self):
        # Range 0.0 to 1.0
        img = np.ones((40, 40, 3), dtype=np.float32)
        feats = extract_visual_features(img)
        _assert_valid_feature_dict(feats)
        assert feats["mean_brightness"] == 255.0

    def test_image_file_path(self, tmp_path):
        file_path = str(tmp_path / "sample.png")
        sample = np.full((70, 90, 3), 150, dtype=np.uint8)
        cv2.imwrite(file_path, sample)

        feats = extract_visual_features(file_path)
        _assert_valid_feature_dict(feats)
        assert feats["width"] == 90
        assert feats["height"] == 70
        assert feats["mean_brightness"] == 150.0

    def test_image_bytes(self):
        sample = np.full((60, 60, 3), 80, dtype=np.uint8)
        success, encoded = cv2.imencode(".png", sample)
        assert success
        img_bytes = encoded.tobytes()

        feats = extract_visual_features(img_bytes)
        _assert_valid_feature_dict(feats)
        assert feats["width"] == 60
        assert feats["height"] == 60
        assert feats["mean_brightness"] == 80.0


# ── 7. Invalid Inputs & Robust Error Handling ───────────────────

class TestInvalidInputs:
    def test_none_input(self):
        feats = extract_visual_features(None)
        _assert_valid_feature_dict(feats)
        assert all(v == 0 or v == 0.0 for v in feats.values())

    def test_empty_string(self):
        feats = extract_visual_features("")
        _assert_valid_feature_dict(feats)
        assert all(v == 0 or v == 0.0 for v in feats.values())

    def test_nonexistent_file(self):
        feats = extract_visual_features("nonexistent_path_404.png")
        _assert_valid_feature_dict(feats)
        assert all(v == 0 or v == 0.0 for v in feats.values())

    def test_empty_numpy_array(self):
        feats = extract_visual_features(np.array([]))
        _assert_valid_feature_dict(feats)
        assert all(v == 0 or v == 0.0 for v in feats.values())

    def test_zero_dimension_array(self):
        feats = extract_visual_features(np.zeros((0, 100, 3), dtype=np.uint8))
        _assert_valid_feature_dict(feats)
        assert all(v == 0 or v == 0.0 for v in feats.values())

    def test_1d_array(self):
        feats = extract_visual_features(np.array([1, 2, 3, 4]))
        _assert_valid_feature_dict(feats)
        assert all(v == 0 or v == 0.0 for v in feats.values())

    def test_array_with_nan(self):
        arr = np.full((20, 20, 3), np.nan, dtype=np.float32)
        feats = extract_visual_features(arr)
        _assert_valid_feature_dict(feats)
        assert all(v == 0 or v == 0.0 for v in feats.values())

    def test_non_image_types(self):
        for bad_input in (12345, [1, 2, 3], {"key": "val"}, object()):
            feats = extract_visual_features(bad_input)
            _assert_valid_feature_dict(feats)
            assert all(v == 0 or v == 0.0 for v in feats.values())


# ── 8. No Classification Rule ──────────────────────────────────

class TestNoClassificationInvented:
    def test_does_not_invent_phishing_labels(self):
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        feats = extract_visual_features(img)
        forbidden_keys = {"label", "prediction", "is_phishing", "phishing", "risk_score", "status"}
        assert forbidden_keys.isdisjoint(feats.keys())

