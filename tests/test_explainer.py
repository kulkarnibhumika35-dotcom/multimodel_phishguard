"""
Tests for PhishGuard Explainable AI (XAI) Module

Covers:
- SHAP TreeExplainer integration for trained models.
- Identification of most influential features (top 5 by default).
- Distinguishing positive contributions (phishing risk) vs negative (legitimacy).
- Human-readable explanations and contribution values.
- Safe handling of missing models and unavailable SHAP.
- Ensuring explanations are never generated for predictions that were not made.
- Ensuring feature importance values are never invented.
- Input validation (dict, 1D/2D array, list).
- Strict 20 URL feature constraint (no false claims of multimodal explainability).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.explainability.explainer import (
    CLASS_LEGITIMATE,
    CLASS_MALWARE,
    CLASS_MAP,
    CLASS_PHISHING,
    DEFAULT_TOP_K,
    EXPLAINABLE_FEATURE_NAMES,
    PhishGuardExplainer,
    _extract_base_value,
    _extract_sample_shap_values,
    _generate_feature_explanation,
    explain_prediction,
    load_class_mapping,
    load_trained_model,
)
from ml.features.url_features import FEATURE_NAMES, extract_url_features


# ── Fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_rf_model() -> RandomForestClassifier:
    """Train a small deterministic RandomForest on synthetic 20 URL features."""
    rng = np.random.RandomState(42)
    # 60 synthetic samples with 20 features
    X = rng.randn(60, len(FEATURE_NAMES))
    # Make feature 14 (has_ip_address) and feature 13 (is_https) strongly correlate with class
    y = ((X[:, 14] > 0) | (X[:, 13] < 0)).astype(int)
    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture
def sample_features_dict() -> dict[str, int | float]:
    """Sample dictionary containing all 20 URL features."""
    return {
        "url_length": 45,
        "hostname_length": 15,
        "num_dots": 2,
        "num_hyphens": 0,
        "num_underscores": 0,
        "num_slashes": 2,
        "num_at": 0,
        "num_question_marks": 0,
        "num_ampersands": 0,
        "num_equals": 0,
        "num_percent": 0,
        "num_digits": 0,
        "num_subdomains": 1,
        "is_https": 1,
        "has_ip_address": 0,
        "has_suspicious_tld": 0,
        "has_suspicious_keyword": 0,
        "path_length": 12,
        "query_length": 0,
        "url_entropy": 3.42,
    }


@pytest.fixture
def phishing_like_features_dict() -> dict[str, int | float]:
    """Sample dictionary with suspicious indicators."""
    return {
        "url_length": 115,
        "hostname_length": 25,
        "num_dots": 5,
        "num_hyphens": 3,
        "num_underscores": 2,
        "num_slashes": 4,
        "num_at": 1,
        "num_question_marks": 1,
        "num_ampersands": 2,
        "num_equals": 2,
        "num_percent": 3,
        "num_digits": 12,
        "num_subdomains": 4,
        "is_https": 0,
        "has_ip_address": 1,
        "has_suspicious_tld": 1,
        "has_suspicious_keyword": 1,
        "path_length": 45,
        "query_length": 30,
        "url_entropy": 4.85,
    }


# ── Unit Tests ──────────────────────────────────────────────────────

def test_explainer_initialization(small_rf_model: RandomForestClassifier):
    """Test explainer initialization with a model and default feature names."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    assert explainer.model is small_rf_model
    assert explainer.feature_names == FEATURE_NAMES
    assert len(explainer.feature_names) == 20


def test_explainer_initialization_without_model():
    """Test explainer initialization with None."""
    explainer = PhishGuardExplainer(model=None)
    assert explainer.model is None


def test_missing_model_returns_model_unavailable(sample_features_dict: dict):
    """Requirement 6 & 8: Missing model safely returns model_unavailable without fake values."""
    explainer = PhishGuardExplainer(model=None)
    result = explainer.explain(sample_features_dict)

    assert result["status"] == "model_unavailable"
    assert result["prediction"] is None
    assert result["risk_score"] is None
    assert result["base_value"] is None
    assert result["top_features"] == []
    assert result["all_contributions"] == {}
    assert "unavailable" in result["error"].lower() or "not provided" in result["error"].lower()


def test_never_generate_explanations_when_prediction_not_made(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
):
    """Requirement 7: Never generate explanations for predictions that were not made."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    # Explicitly pass prediction=None to simulate an unpredicted or missing prediction
    result = explainer.explain(sample_features_dict, prediction=None)

    assert result["status"] == "no_prediction"
    assert result["prediction"] is None
    assert result["risk_score"] is None
    assert result["top_features"] == []
    assert result["all_contributions"] == {}
    assert "no prediction was made" in result["error"].lower()


def test_explain_with_dict_features(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
):
    """Requirement 1, 2, 3, 4: Explains prediction and returns top 5 features."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    result = explainer.explain(sample_features_dict)

    assert result["status"] == "explained"
    assert result["prediction"] in (0, 1)
    assert isinstance(result["risk_score"], float)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert isinstance(result["base_value"], float)
    assert result["error"] is None

    top_features = result["top_features"]
    assert len(top_features) == DEFAULT_TOP_K

    for item in top_features:
        assert item["feature_name"] in FEATURE_NAMES
        assert isinstance(item["feature_value"], (int, float))
        assert isinstance(item["contribution"], float)
        assert item["direction"] in ("positive", "negative", "neutral")
        assert item["impact"] in (
            "increases_phishing_risk",
            "decreases_phishing_risk",
            "neutral",
        )
        assert isinstance(item["explanation"], str)
        assert len(item["explanation"]) > 0


def test_top_features_ordered_by_absolute_magnitude(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
):
    """Requirement 2 & 3: Top features must be ordered descending by |SHAP value|."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    result = explainer.explain(sample_features_dict)

    top_features = result["top_features"]
    magnitudes = [abs(f["contribution"]) for f in top_features]
    for i in range(len(magnitudes) - 1):
        assert magnitudes[i] >= magnitudes[i + 1], (
            f"Expected {magnitudes[i]} >= {magnitudes[i + 1]}"
        )


def test_distinguish_positive_and_negative_contributions(
    small_rf_model: RandomForestClassifier,
    phishing_like_features_dict: dict,
):
    """Requirement 5: Clearly distinguish positive and negative contributions."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    result = explainer.explain(phishing_like_features_dict)

    assert result["status"] == "explained"
    top_features = result["top_features"]

    for item in top_features:
        contrib = item["contribution"]
        if contrib > 0:
            assert item["direction"] == "positive"
            assert item["impact"] == "increases_phishing_risk"
            assert "increases phishing risk" in item["explanation"]
        elif contrib < 0:
            assert item["direction"] == "negative"
            assert item["impact"] == "decreases_phishing_risk"
            assert (
                "decreases phishing risk" in item["explanation"]
                or "legitimacy" in item["explanation"]
            )
        else:
            assert item["direction"] == "neutral"
            assert item["impact"] == "neutral"


def test_all_contributions_mapping(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
):
    """Test that all_contributions contains all 20 URL features."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    result = explainer.explain(sample_features_dict)

    all_contribs = result["all_contributions"]
    assert len(all_contribs) == 20
    for name in FEATURE_NAMES:
        assert name in all_contribs
        assert isinstance(all_contribs[name], float)


def test_explain_url_convenience_method(small_rf_model: RandomForestClassifier):
    """Test explain_url extracts URL features and runs explanation."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    result = explainer.explain_url("https://secure.banking.example.com/account/login?verify=1")

    assert result["status"] == "explained"
    assert len(result["top_features"]) == 5
    assert result["prediction"] in (0, 1)


def test_standalone_explain_prediction_function(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
):
    """Test standalone explain_prediction function wrapper."""
    result = explain_prediction(
        model=small_rf_model,
        features=sample_features_dict,
        top_k=3,
    )
    assert result["status"] == "explained"
    assert len(result["top_features"]) == 3


def test_custom_top_k(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
):
    """Test configurable top_k returns requested number of features."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    result_3 = explainer.explain(sample_features_dict, top_k=3)
    assert len(result_3["top_features"]) == 3

    result_7 = explainer.explain(sample_features_dict, top_k=7)
    assert len(result_7["top_features"]) == 7


def test_input_formats_1d_array(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
):
    """Test passing 1D numpy array."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    arr_1d = np.array([sample_features_dict[name] for name in FEATURE_NAMES])
    result = explainer.explain(arr_1d)
    assert result["status"] == "explained"
    assert len(result["top_features"]) == 5


def test_input_formats_2d_array(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
):
    """Test passing 2D numpy array of shape (1, 20)."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    arr_2d = np.array([[sample_features_dict[name] for name in FEATURE_NAMES]])
    result = explainer.explain(arr_2d)
    assert result["status"] == "explained"
    assert len(result["top_features"]) == 5


def test_input_formats_list(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
):
    """Test passing list of 20 floats."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    lst = [float(sample_features_dict[name]) for name in FEATURE_NAMES]
    result = explainer.explain(lst)
    assert result["status"] == "explained"
    assert len(result["top_features"]) == 5


def test_missing_keys_in_dict_rejected(small_rf_model: RandomForestClassifier):
    """Test that a dictionary with missing URL features is safely rejected."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    incomplete_dict = {"url_length": 50, "is_https": 1}  # Missing other 18 features
    result = explainer.explain(incomplete_dict)
    assert result["status"] == "invalid_input"
    assert "Missing required" in result["error"]
    assert result["top_features"] == []


def test_multimodal_explainability_not_claimed_or_supported(
    small_rf_model: RandomForestClassifier,
):
    """Important constraint: Do not claim multimodal explainability until a compatible model exists."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    # Multimodal feature vector (e.g. 29 features from Step 11)
    multimodal_vector = [0.0] * 29
    result = explainer.explain(multimodal_vector)

    assert result["status"] == "invalid_input"
    assert "Multimodal explainability is not supported" in result["error"]
    assert result["top_features"] == []


def test_wrong_dimension_array_rejected(small_rf_model: RandomForestClassifier):
    """Test that array with incorrect feature count is rejected."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    wrong_arr = np.zeros((1, 10))
    result = explainer.explain(wrong_arr)
    assert result["status"] == "invalid_input"
    assert "Expected 20" in result["error"]


def test_nan_or_inf_in_features_rejected(small_rf_model: RandomForestClassifier):
    """Test that NaN or Inf feature values are rejected."""
    explainer = PhishGuardExplainer(model=small_rf_model)
    nan_arr = np.zeros(20)
    nan_arr[0] = np.nan
    result = explainer.explain(nan_arr)
    assert result["status"] == "invalid_input"
    assert "NaN or Inf" in result["error"]


def test_shap_unavailable_handled_safely(
    small_rf_model: RandomForestClassifier,
    sample_features_dict: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """Requirement 6: Safely handle unavailable SHAP library without crashing."""
    import ml.explainability.explainer as explainer_module

    monkeypatch.setattr(explainer_module, "_SHAP_AVAILABLE", False)

    explainer = PhishGuardExplainer(model=small_rf_model)
    result = explainer.explain(sample_features_dict)

    assert result["status"] == "shap_unavailable"
    assert result["top_features"] == []
    assert result["all_contributions"] == {}
    assert "SHAP library is not installed" in result["error"]


def test_model_without_predict_handled_safely(sample_features_dict: dict):
    """Test that an object without predict method is handled safely."""
    dummy_object = object()
    explainer = PhishGuardExplainer(model=dummy_object)
    result = explainer.explain(sample_features_dict)
    assert result["status"] in ("no_prediction", "unsupported_model")
    assert result["top_features"] == []


def test_feature_explanation_generator_all_20_features():
    """Verify all 20 URL features produce valid explanations for positive and negative contributions."""
    for name in FEATURE_NAMES:
        exp_pos = _generate_feature_explanation(name, 1, 0.25)
        assert "increases phishing risk" in exp_pos

        exp_neg = _generate_feature_explanation(name, 0, -0.25)
        assert "decreases phishing risk" in exp_neg or "legitimacy" in exp_neg

    exp_neutral = _generate_feature_explanation("url_length", 50, 0.0)
    assert "neutral" in exp_neutral


def test_extract_sample_shap_values_helper():
    """Test _extract_sample_shap_values helper with 2D, 3D, and list inputs."""
    # 3D: (1, 20, 2)
    arr_3d = np.ones((1, 20, 2))
    extracted = _extract_sample_shap_values(arr_3d, sample_idx=0, target_class=1)
    assert extracted.shape == (20,)

    # 2D: (1, 20)
    arr_2d = np.ones((1, 20))
    extracted_2d = _extract_sample_shap_values(arr_2d, sample_idx=0)
    assert extracted_2d.shape == (20,)

    # List of 2 arrays
    arr_list = [np.zeros((1, 20)), np.ones((1, 20))]
    extracted_list = _extract_sample_shap_values(arr_list, sample_idx=0, target_class=1)
    assert extracted_list.shape == (20,)
    assert (extracted_list == 1.0).all()


def test_extract_base_value_helper():
    """Test _extract_base_value helper handles array, list, and float."""
    assert _extract_base_value([0.4, 0.6], target_class=1) == 0.6
    assert _extract_base_value(np.array([0.3, 0.7]), target_class=1) == 0.7
    assert _extract_base_value(0.55) == 0.55
# ── Multiclass SHAP Unit Tests (Three-Class Architecture) ───────────

@pytest.fixture(scope="module")
def three_class_rf_model() -> RandomForestClassifier:
    """Train a deterministic 3-class RandomForest on synthetic 20 URL features."""
    rng = np.random.RandomState(42)
    # 90 synthetic samples: 30 legit (0), 30 phish (1), 30 malware (2)
    X = rng.randn(90, len(FEATURE_NAMES))
    # Make feature 14 (has_ip) strongly indicate malware (2)
    # Make feature 16 (has_suspicious_keyword) indicate phishing (1)
    y = np.zeros(90, dtype=int)
    y[30:60] = 1  # phishing
    y[60:90] = 2  # malware
    X[30:60, 16] = 2.0   # phishing has keywords
    X[60:90, 14] = 3.0   # malware has IP
    X[60:90, 13] = -2.0  # malware lacks HTTPS
    clf = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42)
    clf.fit(X, y)
    return clf


class TestMulticlassSHAPExplainer:
    def test_explainer_from_trained_model(self):
        """Requirement 1: Load the trained three-class Random Forest model from disk."""
        explainer = PhishGuardExplainer.from_trained_model()
        if explainer.model is not None:
            assert hasattr(explainer.model, "predict")
            assert hasattr(explainer.model, "classes_")
            assert len(explainer.model.classes_) == 3
            assert explainer.class_mapping[0] == "legitimate"
            assert explainer.class_mapping[1] == "phishing"
            assert explainer.class_mapping[2] == "malware_distribution"

    def test_explain_legitimate_url_never_describes_as_malicious(
        self,
        three_class_rf_model: RandomForestClassifier,
        sample_features_dict: dict,
    ):
        """Requirements 2, 3, 5, 7: Explains class 0 and NEVER describes legitimate URL as malicious."""
        explainer = PhishGuardExplainer(model=three_class_rf_model)
        # Explain with target_class=0 (Legitimate)
        result = explainer.explain(sample_features_dict, prediction=0, target_class=0)

        assert result["status"] == "explained"
        assert result["target_class"] == 0
        assert result["threat_category"] == "legitimate"
        assert result["target_class_name"] == "legitimate"

        top_features = result["top_features"]
        assert len(top_features) == 5

        for feat in top_features:
            explanation = feat["explanation"].lower()
            # Requirement 7: Never describe a legitimate URL as malicious
            assert "phishing" not in explanation, f"Legitimate explanation contained 'phishing': {feat['explanation']}"
            assert "malware" not in explanation, f"Legitimate explanation contained 'malware': {feat['explanation']}"
            assert "malicious" not in explanation, f"Legitimate explanation contained 'malicious': {feat['explanation']}"

            if feat["contribution"] > 0:
                assert feat["impact"] == "supports_legitimate"
                assert "supports legitimate classification" in explanation
            elif feat["contribution"] < 0:
                assert feat["impact"] == "reduces_legitimate_confidence"
                assert "reduces legitimate confidence" in explanation

    def test_explain_phishing_url_distinguishes_phishing_indicators(
        self,
        three_class_rf_model: RandomForestClassifier,
        phishing_like_features_dict: dict,
    ):
        """Requirements 2, 3, 5, 6: Explains class 1 and highlights phishing-specific lures."""
        explainer = PhishGuardExplainer(model=three_class_rf_model)
        result = explainer.explain(phishing_like_features_dict, prediction=1, target_class=1)

        assert result["status"] == "explained"
        assert result["target_class"] == 1
        assert result["threat_category"] == "phishing"

        top_features = result["top_features"]
        for feat in top_features:
            if feat["contribution"] > 0:
                assert feat["impact"] == "increases_phishing_risk"
                assert "increases phishing risk" in feat["explanation"]

    def test_explain_malware_url_distinguishes_malware_indicators(
        self,
        three_class_rf_model: RandomForestClassifier,
    ):
        """Requirements 2, 3, 5, 6: Explains class 2 and highlights malware-specific indicators."""
        malware_features = {
            "url_length": 65,
            "hostname_length": 15,
            "num_dots": 3,
            "num_hyphens": 0,
            "num_underscores": 1,
            "num_slashes": 3,
            "num_at": 0,
            "num_question_marks": 0,
            "num_ampersands": 0,
            "num_equals": 0,
            "num_percent": 0,
            "num_digits": 12,
            "num_subdomains": 0,
            "is_https": 0,
            "has_ip_address": 1,
            "has_suspicious_tld": 0,
            "has_suspicious_keyword": 0,
            "path_length": 28,
            "query_length": 0,
            "url_entropy": 4.92,
        }

        explainer = PhishGuardExplainer(model=three_class_rf_model)
        result = explainer.explain(malware_features, prediction=2, target_class=2)

        assert result["status"] == "explained"
        assert result["target_class"] == 2
        assert result["threat_category"] == "malware_distribution"
        assert result["target_class_name"] == "malware_distribution"

        top_features = result["top_features"]
        for feat in top_features:
            if feat["contribution"] > 0:
                assert feat["impact"] == "increases_malware_risk"
                assert "increases malware distribution risk" in feat["explanation"]
            elif feat["contribution"] < 0:
                assert feat["impact"] == "decreases_malware_risk"
                assert "decreases malware distribution risk" in feat["explanation"]

    def test_dynamic_class_selection(
        self,
        three_class_rf_model: RandomForestClassifier,
        phishing_like_features_dict: dict,
    ):
        """Requirement 3: Dynamically select the predicted class when target_class is omitted."""
        explainer = PhishGuardExplainer(model=three_class_rf_model)
        # Omit target_class -> should automatically match predicted class
        result = explainer.explain(phishing_like_features_dict)

        assert result["status"] == "explained"
        assert result["target_class"] == result["prediction"]
        assert result["target_class_name"] == explainer.class_mapping[result["prediction"]]

    def test_three_class_shap_3d_array_extraction(self):
        """Requirement 8: Handle 3D SHAP output (N, 20, 3) across all classes."""
        # 3D array: (1 sample, 20 features, 3 classes)
        shap_3d = np.zeros((1, 20, 3))
        shap_3d[0, :, 0] = 0.1  # Class 0
        shap_3d[0, :, 1] = 0.5  # Class 1
        shap_3d[0, :, 2] = 0.9  # Class 2

        v0 = _extract_sample_shap_values(shap_3d, sample_idx=0, target_class=0)
        assert (v0 == 0.1).all()

        v1 = _extract_sample_shap_values(shap_3d, sample_idx=0, target_class=1)
        assert (v1 == 0.5).all()

        v2 = _extract_sample_shap_values(shap_3d, sample_idx=0, target_class=2)
        assert (v2 == 0.9).all()

    def test_three_class_base_value_extraction(self):
        """Requirement 8: Handle multiclass base value list/array."""
        expected_vals = [0.44, 0.48, 0.08]
        assert _extract_base_value(expected_vals, target_class=0) == 0.44
        assert _extract_base_value(expected_vals, target_class=1) == 0.48
        assert _extract_base_value(expected_vals, target_class=2) == 0.08

    def test_out_of_bounds_target_class_clamped_safely(self):
        """Edge case: out-of-bounds target_class does not crash and clamps safely."""
        shap_3d = np.ones((1, 20, 3))
        extracted = _extract_sample_shap_values(shap_3d, sample_idx=0, target_class=99)
        assert extracted.shape == (20,)

        base_val = _extract_base_value([0.3, 0.5, 0.2], target_class=99)
        assert base_val == 0.2

    def test_class_probabilities_included(
        self,
        three_class_rf_model: RandomForestClassifier,
        sample_features_dict: dict,
    ):
        """Test class_probabilities dictionary contains probabilities for all 3 classes."""
        explainer = PhishGuardExplainer(model=three_class_rf_model)
        result = explainer.explain(sample_features_dict)

        probs = result["class_probabilities"]
        assert "legitimate" in probs
        assert "phishing" in probs
        assert "malware_distribution" in probs
        assert abs(sum(probs.values()) - 1.0) < 0.01
