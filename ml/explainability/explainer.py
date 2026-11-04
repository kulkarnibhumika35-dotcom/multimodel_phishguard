"""
PhishGuard – Explainable AI (XAI) Engine (Three-Class Architecture)

Provides local feature-attribution explainability for PhishGuard predictions
across three canonical threat categories:
    0 = legitimate
    1 = phishing
    2 = malware_distribution

Requirements:
- Explains predictions from trained 3-class and binary models using TreeSHAP.
- Dynamically selects the predicted class (or explicit target class).
- Identifies the top K most influential features per prediction.
- Generates tailored, human-readable explanations distinguishing:
    * Phishing indicators (credential lures, brand spoofing, subdomain depth)
    * Malware indicators (direct IP hosting, C2 patterns, payload hashes, plain HTTP)
    * Legitimate indicators (reputable TLD, HTTPS, shallow structure, no lures)
- Never describes a legitimate URL as malicious.
- Safely handles multidimensional SHAP output shapes (3D array, list of arrays, Explanation).
- Safely handles missing models, unavailable SHAP, and unpredicted samples.
- Strictly compatible with the 20 canonical URL features.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover
    shap = None
    _SHAP_AVAILABLE = False

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.features.url_features import FEATURE_NAMES, extract_url_features
from ml.training.dataset_pipeline import canonical_normalize_url

logger = logging.getLogger(__name__)

# Canonical list of 20 URL feature names
EXPLAINABLE_FEATURE_NAMES: list[str] = list(FEATURE_NAMES)
DEFAULT_TOP_K: int = 5

# Sentinel to distinguish an omitted prediction from an explicit None
_UNSET: Any = object()

# ── Class Mapping (3-Class Architecture) ─────────────────────────────
CLASS_LEGITIMATE = 0
CLASS_PHISHING = 1
CLASS_MALWARE = 2

CLASS_MAP = {
    CLASS_LEGITIMATE: "legitimate",
    CLASS_PHISHING: "phishing",
    CLASS_MALWARE: "malware_distribution",
}

LABEL_NAME_TO_INT = {
    "legitimate": CLASS_LEGITIMATE,
    "phishing": CLASS_PHISHING,
    "malware_distribution": CLASS_MALWARE,
}

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "url_model.joblib")
DEFAULT_CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "class_mapping.json")
DEFAULT_FEATURES_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "feature_names.json")


# ── Feature Explanation Generator ─────────────────────────────────

def _generate_feature_explanation(
    feature_name: str,
    value: int | float,
    contribution: float,
    target_class: int = CLASS_PHISHING,
) -> str:
    """Generate tailored, human-readable explanation for a feature's contribution.

    Distinguishes:
    - Target Class 0 (Legitimate): Explains why the URL is benign. Never calls it malicious.
    - Target Class 1 (Phishing): Highlights credential spoofing, deceptive TLDs, and domain depth.
    - Target Class 2 (Malware Distribution): Highlights direct IP/C2 hosting, payload patterns, plain HTTP.
    """
    clean_val: int | float = int(value) if isinstance(value, float) and value.is_integer() else value

    # Neutral impact handling
    if contribution == 0:
        if target_class == CLASS_LEGITIMATE:
            return f"Feature '{feature_name}' (value: {clean_val}) has neutral impact on legitimacy assessment."
        if target_class == CLASS_MALWARE:
            return f"Feature '{feature_name}' (value: {clean_val}) has neutral impact on malware distribution risk."
        return f"Feature '{feature_name}' (value: {clean_val}) has neutral impact on phishing risk."

    # Determine impact suffix based on target class
    if target_class == CLASS_LEGITIMATE:
        impact_text = "supports legitimate classification" if contribution > 0 else "reduces legitimate confidence (without indicating a confirmed threat)"
    elif target_class == CLASS_MALWARE:
        impact_text = "increases malware distribution risk" if contribution > 0 else "decreases malware distribution risk"
    else:  # CLASS_PHISHING (1) or binary threat
        impact_text = "increases phishing risk" if contribution > 0 else "decreases phishing risk (indicates legitimacy)"

    # 1. Direct IP Address
    if feature_name == "has_ip_address":
        if target_class == CLASS_MALWARE:
            if clean_val:
                return f"Use of direct IP address instead of domain name strongly indicates malware distribution host / command-and-control server ({impact_text})."
            return f"Standard registered domain used instead of raw IP ({impact_text})."
        if target_class == CLASS_LEGITIMATE:
            if not clean_val:
                return f"Standard registered domain name used instead of raw IP, {impact_text}."
            return f"Use of direct IP address deviates from typical legitimate web branding ({impact_text})."
        # Phishing
        if clean_val:
            return f"Use of direct IP address instead of domain name {impact_text}."
        return f"Standard domain name used instead of raw IP {impact_text}."

    # 2. HTTPS Protocol
    if feature_name == "is_https":
        if target_class == CLASS_MALWARE:
            if not clean_val:
                return f"Lack of HTTPS protocol (plain HTTP) is common in automated malware payload droppers ({impact_text})."
            return f"HTTPS protocol is enabled ({impact_text})."
        if target_class == CLASS_LEGITIMATE:
            if clean_val:
                return f"HTTPS protocol is enabled, {impact_text}."
            return f"Lack of HTTPS protocol {impact_text}."
        # Phishing
        if clean_val:
            return f"HTTPS protocol is enabled {impact_text}."
        return f"Lack of HTTPS protocol {impact_text}."

    # 3. Suspicious TLD
    if feature_name == "has_suspicious_tld":
        if target_class == CLASS_MALWARE:
            if clean_val:
                return f"Top-level domain belongs to high-risk registry frequently associated with malware campaigns ({impact_text})."
            return f"Standard reputable top-level domain ({impact_text})."
        if target_class == CLASS_LEGITIMATE:
            if not clean_val:
                return f"Top-level domain is standard and reputable, {impact_text}."
            return f"Non-standard TLD deviates from typical enterprise domain patterns ({impact_text})."
        # Phishing
        if clean_val:
            return f"Top-level domain belongs to known high-risk TLD list {impact_text}."
        return f"Top-level domain is standard and reputable {impact_text}."

    # 4. Suspicious Keywords
    if feature_name == "has_suspicious_keyword":
        if target_class == CLASS_MALWARE:
            if clean_val:
                return f"Contains lure keywords associated with payload delivery ({impact_text})."
            return f"No suspicious lure keywords detected ({impact_text})."
        if target_class == CLASS_LEGITIMATE:
            if not clean_val:
                return f"Absence of deceptive credential lures {impact_text}."
            return f"Security or account keyword present in URL structure ({impact_text})."
        # Phishing
        if clean_val:
            return f"Contains high-risk security/credential keywords {impact_text}."
        return f"No suspicious security or credential keywords detected {impact_text}."

    # 5. Number of Subdomains
    if feature_name == "num_subdomains":
        sub_cnt = int(clean_val)
        if target_class == CLASS_MALWARE:
            if sub_cnt > 2:
                return f"Deep subdomain hierarchy ({sub_cnt} subdomains) ({impact_text})."
            return f"Standard subdomain hierarchy ({sub_cnt} subdomains) ({impact_text})."
        if target_class == CLASS_LEGITIMATE:
            if sub_cnt <= 2:
                return f"Standard subdomain hierarchy ({sub_cnt} subdomains), {impact_text}."
            return f"Multi-tier subdomain structure ({sub_cnt} subdomains) {impact_text}."
        # Phishing
        if sub_cnt > 2:
            return f"Elevated subdomain depth ({sub_cnt} subdomains) {impact_text}."
        return f"Normal subdomain count ({sub_cnt}) {impact_text}."

    # 6. Number of Dots
    if feature_name == "num_dots":
        dot_cnt = int(clean_val)
        if dot_cnt > 3:
            return f"Excessive dots ({dot_cnt}) in URL structure {impact_text}."
        return f"Dot count ({dot_cnt}) in URL {impact_text}."

    # 7. Number of Hyphens
    if feature_name == "num_hyphens":
        hyph_cnt = int(clean_val)
        if hyph_cnt > 0:
            return f"Presence of hyphens ({hyph_cnt}) in URL {impact_text}."
        return f"No hyphens present {impact_text}."

    # 8. Number of @ Characters
    if feature_name == "num_at":
        at_cnt = int(clean_val)
        if at_cnt > 0:
            return f"Presence of '@' character ({at_cnt}) indicating URL spoofing {impact_text}."
        return f"No '@' spoofing character detected {impact_text}."

    # 9. Number of Percent Encoded Characters
    if feature_name == "num_percent":
        pct_cnt = int(clean_val)
        if pct_cnt > 0:
            return f"Percent-encoded characters ({pct_cnt}) indicating URL obfuscation {impact_text}."
        return f"No percent-encoded obfuscation characters {impact_text}."

    # 10. Number of Digits
    if feature_name == "num_digits":
        dig_cnt = int(clean_val)
        if target_class == CLASS_MALWARE and dig_cnt > 5:
            return f"High digit count ({dig_cnt} digits) typical of payload hashes or raw IP endpoints ({impact_text})."
        return f"Digit count in URL ({dig_cnt} digits) {impact_text}."

    # 11. Shannon Entropy
    if feature_name == "url_entropy":
        entropy_val = float(clean_val)
        if target_class == CLASS_MALWARE and entropy_val > 4.5:
            return f"High Shannon entropy ({entropy_val:.2f}) indicating randomized payload hash or obfuscation ({impact_text})."
        return f"URL Shannon entropy ({entropy_val:.2f}) {impact_text}."

    # 12. Structural Length Features
    if feature_name == "url_length":
        return f"Total URL length ({int(clean_val)} characters) {impact_text}."

    if feature_name == "hostname_length":
        return f"Hostname length ({int(clean_val)} characters) {impact_text}."

    if feature_name == "path_length":
        if target_class == CLASS_MALWARE and clean_val > 25:
            return f"URL path length ({int(clean_val)} characters) typical of binary payload locations ({impact_text})."
        return f"URL path length ({int(clean_val)} characters) {impact_text}."

    if feature_name == "query_length":
        return f"URL query string length ({int(clean_val)} characters) {impact_text}."

    if feature_name == "num_slashes":
        return f"Path delimiter count ({int(clean_val)} slashes) {impact_text}."

    if feature_name == "num_underscores":
        return f"Underscore count in URL ({int(clean_val)}) {impact_text}."

    if feature_name == "num_question_marks":
        return f"Question mark count ({int(clean_val)}) {impact_text}."

    if feature_name == "num_ampersands":
        return f"Query parameter separator count ({int(clean_val)} ampersands) {impact_text}."

    if feature_name == "num_equals":
        return f"Query parameter assignment count ({int(clean_val)} equals) {impact_text}."

    # Generic Fallback
    val_repr = f"{clean_val:.2f}" if isinstance(clean_val, float) else f"{clean_val}"
    return f"Feature '{feature_name}' (value: {val_repr}) {impact_text}."


# ── Helper functions for SHAP shape handling ──────────────────────

def _extract_sample_shap_values(
    shap_vals: Any,
    sample_idx: int = 0,
    target_class: int = 1,
) -> np.ndarray:
    """Extract 1D array of SHAP values for sample_idx and target_class safely.

    Handles:
    - List of arrays: [class_0_values, class_1_values, class_2_values]
    - 3D numpy array: (N, num_features, num_classes)
    - 2D numpy array: (N, num_features)
    - 1D numpy array: (num_features,)
    - shap.Explanation object
    """
    # Case 1: list of arrays [class_0_values, class_1_values, ...]
    if isinstance(shap_vals, list):
        if len(shap_vals) == 0:
            raise ValueError("Empty SHAP values list")
        safe_idx = target_class if (0 <= target_class < len(shap_vals)) else (len(shap_vals) - 1)
        arr = shap_vals[safe_idx]
        return np.asarray(arr[sample_idx], dtype=np.float64)

    # Case 2: shap.Explanation object
    if hasattr(shap_vals, "values"):
        shap_vals = shap_vals.values

    arr = np.asarray(shap_vals)

    # Case 3: 3D array (N, num_features, num_classes)
    if arr.ndim == 3:
        num_classes = arr.shape[2]
        safe_idx = target_class if (0 <= target_class < num_classes) else (num_classes - 1)
        return np.asarray(arr[sample_idx, :, safe_idx], dtype=np.float64)

    # Case 4: 2D array (N, num_features)
    if arr.ndim == 2:
        return np.asarray(arr[sample_idx, :], dtype=np.float64)

    # Case 5: 1D array (num_features,)
    if arr.ndim == 1:
        return np.asarray(arr, dtype=np.float64)

    raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")


def _extract_base_value(expected_val: Any, target_class: int = 1) -> float:
    """Safely extract the expected base value for target_class."""
    if isinstance(expected_val, (list, tuple, np.ndarray)):
        arr = np.asarray(expected_val)
        if arr.ndim > 0 and len(arr) > 0:
            safe_idx = target_class if (0 <= target_class < len(arr)) else (len(arr) - 1)
            return round(float(arr[safe_idx]), 4)
    if expected_val is None:
        return 0.0
    return round(float(expected_val), 4)


# ── Model & Class Mapping Loaders ─────────────────────────────────

def load_trained_model(path: str = DEFAULT_MODEL_PATH) -> Any | None:
    """Load the trained Random Forest model from disk."""
    if not os.path.isfile(path):
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as exc:
        logger.warning("Failed to load model from %s: %s", path, exc)
        return None


def load_class_mapping(path: str = DEFAULT_CLASS_MAPPING_PATH) -> dict[int, str]:
    """Load class mapping dictionary from JSON."""
    if not os.path.isfile(path):
        return dict(CLASS_MAP)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            c2n = data.get("class_to_name", {})
            return {int(k): v for k, v in c2n.items()}
    except Exception as exc:
        logger.warning("Failed to load class mapping from %s: %s", path, exc)
        return dict(CLASS_MAP)


# ── PhishGuard Explainer Class ────────────────────────────────────

class PhishGuardExplainer:
    """Local feature attribution explainer for PhishGuard predictions using TreeSHAP."""

    def __init__(
        self,
        model: Any | None = None,
        feature_names: list[str] | None = None,
        class_mapping: dict[int, str] | None = None,
    ) -> None:
        """Initialize explainer with optional model, feature names, and class mapping."""
        self.feature_names: list[str] = list(feature_names or EXPLAINABLE_FEATURE_NAMES)
        self.class_mapping: dict[int, str] = dict(class_mapping or CLASS_MAP)
        self._model: Any | None = None
        self._shap_explainer: Any | None = None
        self._init_error: str | None = None

        if model is not None:
            self.set_model(model)

    @classmethod
    def from_trained_model(
        cls,
        model_path: str = DEFAULT_MODEL_PATH,
        class_mapping_path: str = DEFAULT_CLASS_MAPPING_PATH,
        features_path: str = DEFAULT_FEATURES_PATH,
    ) -> "PhishGuardExplainer":
        """Factory method to load the persisted three-class model and metadata."""
        model = load_trained_model(model_path)
        class_mapping = load_class_mapping(class_mapping_path)
        feature_names = None
        if os.path.isfile(features_path):
            try:
                with open(features_path, "r", encoding="utf-8") as f:
                    feature_names = json.load(f)
            except Exception:
                feature_names = None
        return cls(model=model, feature_names=feature_names, class_mapping=class_mapping)

    @property
    def model(self) -> Any | None:
        """Return the current model."""
        return self._model

    def set_model(self, model: Any | None) -> None:
        """Set or update the active model and prepare the SHAP explainer."""
        self._model = model
        self._shap_explainer = None
        self._init_error = None

        if model is None:
            return

        if not _SHAP_AVAILABLE:
            self._init_error = "SHAP library is not installed or available."
            return

        try:
            self._shap_explainer = shap.TreeExplainer(model)
        except Exception as exc:
            try:
                self._shap_explainer = shap.Explainer(model)
            except Exception as exc2:
                self._shap_explainer = None
                self._init_error = f"Model is not compatible with SHAP: {exc} | {exc2}"
                logger.warning("Could not initialize SHAP explainer: %s", self._init_error)

    def _prepare_input_vector(
        self,
        features: dict[str, Any] | list | tuple | np.ndarray,
    ) -> np.ndarray:
        """Validate and convert features into a 2D numpy array of shape (1, 20)."""
        expected_len = len(self.feature_names)

        if isinstance(features, dict):
            missing = [name for name in self.feature_names if name not in features]
            if missing:
                raise ValueError(f"Missing required URL feature(s): {missing}")
            vector = [features[name] for name in self.feature_names]
            arr = np.array([vector], dtype=np.float64)
        elif isinstance(features, (list, tuple)):
            if len(features) > expected_len:
                raise ValueError(
                    f"Input has {len(features)} features. Multimodal explainability "
                    f"is not supported until a compatible multimodal model exists. "
                    f"Expected {expected_len} URL features."
                )
            if len(features) != expected_len:
                raise ValueError(
                    f"Expected {expected_len} features, got {len(features)}."
                )
            arr = np.array([features], dtype=np.float64)
        elif isinstance(features, np.ndarray):
            if features.ndim == 1:
                arr = features.reshape(1, -1).astype(np.float64)
            elif features.ndim == 2:
                arr = features.astype(np.float64)
            else:
                raise ValueError(f"Array must be 1D or 2D, got {features.ndim}D.")

            if arr.shape[1] > expected_len:
                raise ValueError(
                    f"Input has {arr.shape[1]} features. Multimodal explainability "
                    f"is not supported until a compatible multimodal model exists. "
                    f"Expected {expected_len} URL features."
                )
            if arr.shape[1] != expected_len:
                raise ValueError(
                    f"Expected {expected_len} features, got {arr.shape[1]}."
                )
        else:
            raise TypeError(
                f"features must be a dict, list, tuple, or numpy ndarray, got {type(features).__name__}."
            )

        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError("Feature vector contains NaN or Inf values.")

        return arr

    def explain(
        self,
        features: dict[str, Any] | list | tuple | np.ndarray,
        prediction: int | None = _UNSET,
        target_class: int | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """Explain a prediction on the given features.

        Parameters
        ----------
        features : dict, list, tuple, or np.ndarray
            The 20 URL features.
        prediction : int, None, or _UNSET
            If explicitly None: indicates no prediction was made. Returns
            status="no_prediction" without generating explanations.
            If omitted (_UNSET): runs inference using the model.
            If int: explains the specified prediction.
        target_class : int | None
            Class to explain. If None, dynamically defaults to the predicted class.
        top_k : int
            Number of top contributing features to return (default 5).

        Returns
        -------
        dict[str, Any]
            Explanation payload containing status, prediction, threat_category,
            target_class, risk_score, base_value, top_features, and all_contributions.
        """
        # 1. Handle missing model safely
        if self._model is None:
            return {
                "status": "model_unavailable",
                "prediction": None,
                "threat_category": None,
                "target_class": None,
                "target_class_name": None,
                "risk_score": None,
                "class_probabilities": {},
                "base_value": None,
                "top_features": [],
                "all_contributions": {},
                "error": "Model is not provided or unavailable.",
            }

        # 2. Never generate explanations when prediction was not made
        if prediction is None:
            return {
                "status": "no_prediction",
                "prediction": None,
                "threat_category": None,
                "target_class": None,
                "target_class_name": None,
                "risk_score": None,
                "class_probabilities": {},
                "base_value": None,
                "top_features": [],
                "all_contributions": {},
                "error": "No prediction was made. Explanations cannot be generated for unpredicted samples.",
            }

        # 3. Validate feature vector
        try:
            X = self._prepare_input_vector(features)
        except Exception as exc:
            return {
                "status": "invalid_input",
                "prediction": None,
                "threat_category": None,
                "target_class": None,
                "target_class_name": None,
                "risk_score": None,
                "class_probabilities": {},
                "base_value": None,
                "top_features": [],
                "all_contributions": {},
                "error": str(exc),
            }

        # 4. Determine prediction and risk score
        pred: int
        risk_score: float | None = None
        class_probabilities: dict[str, float] = {}

        if prediction is _UNSET:
            if not hasattr(self._model, "predict"):
                return {
                    "status": "no_prediction",
                    "prediction": None,
                    "threat_category": None,
                    "target_class": None,
                    "target_class_name": None,
                    "risk_score": None,
                    "class_probabilities": {},
                    "base_value": None,
                    "top_features": [],
                    "all_contributions": {},
                    "error": "Model has no predict method; cannot generate prediction.",
                }
            try:
                pred = int(self._model.predict(X)[0])
            except Exception as exc:
                return {
                    "status": "no_prediction",
                    "prediction": None,
                    "threat_category": None,
                    "target_class": None,
                    "target_class_name": None,
                    "risk_score": None,
                    "class_probabilities": {},
                    "base_value": None,
                    "top_features": [],
                    "all_contributions": {},
                    "error": f"Model inference failed: {exc}",
                }
        else:
            pred = int(prediction)

        if hasattr(self._model, "predict_proba"):
            try:
                proba = self._model.predict_proba(X)[0]
                for idx, p in enumerate(proba):
                    cat_name = self.class_mapping.get(idx, f"class_{idx}")
                    class_probabilities[cat_name] = round(float(p), 4)

                if len(proba) == 2:
                    risk_score = round(float(proba[1]), 4)
                elif len(proba) >= 3:
                    # Overall threat risk: 1 - P(legitimate)
                    risk_score = round(float(1.0 - proba[0]), 4)
                else:
                    risk_score = round(float(proba[0]), 4)
            except Exception:
                risk_score = float(pred)
        else:
            risk_score = float(pred)

        # 5. Dynamically select the target class to explain
        if target_class is not None:
            effective_target_class = int(target_class)
        else:
            effective_target_class = pred

        threat_category = self.class_mapping.get(pred, str(pred))
        target_class_name = self.class_mapping.get(effective_target_class, str(effective_target_class))

        # 6. Handle unavailable SHAP safely
        if not _SHAP_AVAILABLE:
            return {
                "status": "shap_unavailable",
                "prediction": pred,
                "threat_category": threat_category,
                "target_class": effective_target_class,
                "target_class_name": target_class_name,
                "risk_score": risk_score,
                "class_probabilities": class_probabilities,
                "base_value": None,
                "top_features": [],
                "all_contributions": {},
                "error": "SHAP library is not installed or available.",
            }

        # 7. Initialize SHAP Explainer if not ready
        if self._shap_explainer is None:
            try:
                self._shap_explainer = shap.TreeExplainer(self._model)
            except Exception as exc:
                try:
                    self._shap_explainer = shap.Explainer(self._model)
                except Exception as exc2:
                    return {
                        "status": "unsupported_model",
                        "prediction": pred,
                        "threat_category": threat_category,
                        "target_class": effective_target_class,
                        "target_class_name": target_class_name,
                        "risk_score": risk_score,
                        "class_probabilities": class_probabilities,
                        "base_value": None,
                        "top_features": [],
                        "all_contributions": {},
                        "error": f"SHAP explainer could not be initialized: {exc} | {exc2}",
                    }

        # 8. Compute SHAP values
        try:
            raw_shap = self._shap_explainer.shap_values(X)
        except Exception as exc:
            return {
                "status": "explanation_error",
                "prediction": pred,
                "threat_category": threat_category,
                "target_class": effective_target_class,
                "target_class_name": target_class_name,
                "risk_score": risk_score,
                "class_probabilities": class_probabilities,
                "base_value": None,
                "top_features": [],
                "all_contributions": {},
                "error": f"SHAP value computation failed: {exc}",
            }

        # 9. Extract sample SHAP values and base value for effective target class
        try:
            sample_shap = _extract_sample_shap_values(
                raw_shap,
                sample_idx=0,
                target_class=effective_target_class,
            )
            base_value = _extract_base_value(
                getattr(self._shap_explainer, "expected_value", 0.0),
                target_class=effective_target_class,
            )
        except Exception as exc:
            return {
                "status": "explanation_error",
                "prediction": pred,
                "threat_category": threat_category,
                "target_class": effective_target_class,
                "target_class_name": target_class_name,
                "risk_score": risk_score,
                "class_probabilities": class_probabilities,
                "base_value": None,
                "top_features": [],
                "all_contributions": {},
                "error": f"Failed to extract SHAP values: {exc}",
            }

        # 10. Build feature entries with category-specific explanations
        all_contributions: dict[str, float] = {}
        feature_entries: list[dict[str, Any]] = []

        for i, name in enumerate(self.feature_names):
            contrib = float(sample_shap[i])
            val = float(X[0, i])
            clean_val: int | float = int(val) if val.is_integer() else round(val, 4)
            contrib_rounded = round(contrib, 4)
            all_contributions[name] = contrib_rounded

            if effective_target_class == CLASS_MALWARE:
                if contrib > 0:
                    direction = "positive"
                    impact = "increases_malware_risk"
                elif contrib < 0:
                    direction = "negative"
                    impact = "decreases_malware_risk"
                else:
                    direction = "neutral"
                    impact = "neutral"
            elif effective_target_class == CLASS_LEGITIMATE:
                if contrib > 0:
                    direction = "positive"
                    impact = "supports_legitimate"
                elif contrib < 0:
                    direction = "negative"
                    impact = "reduces_legitimate_confidence"
                else:
                    direction = "neutral"
                    impact = "neutral"
            else:  # CLASS_PHISHING (1)
                if contrib > 0:
                    direction = "positive"
                    impact = "increases_phishing_risk"
                elif contrib < 0:
                    direction = "negative"
                    impact = "decreases_phishing_risk"
                else:
                    direction = "neutral"
                    impact = "neutral"

            explanation = _generate_feature_explanation(
                name,
                clean_val,
                contrib,
                target_class=effective_target_class,
            )

            feature_entries.append({
                "feature_name": name,
                "feature_value": clean_val,
                "contribution": contrib_rounded,
                "direction": direction,
                "impact": impact,
                "explanation": explanation,
                "_abs_contrib": abs(contrib),
            })

        # Rank features descending by absolute magnitude of contribution
        feature_entries.sort(key=lambda item: item["_abs_contrib"], reverse=True)

        # Select top K
        top_k_features: list[dict[str, Any]] = []
        for item in feature_entries[:top_k]:
            cleaned = dict(item)
            del cleaned["_abs_contrib"]
            top_k_features.append(cleaned)

        return {
            "status": "explained",
            "prediction": pred,
            "threat_category": threat_category,
            "target_class": effective_target_class,
            "target_class_name": target_class_name,
            "risk_score": risk_score,
            "class_probabilities": class_probabilities,
            "base_value": base_value,
            "top_features": top_k_features,
            "all_contributions": all_contributions,
            "error": None,
        }

    def explain_url(
        self,
        url: str,
        prediction: int | None = _UNSET,
        target_class: int | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> dict[str, Any]:
        """Convenience method to extract features from a URL and explain prediction."""
        norm_url = canonical_normalize_url(url) or url
        features = extract_url_features(norm_url)
        return self.explain(
            features=features,
            prediction=prediction,
            target_class=target_class,
            top_k=top_k,
        )


# ── Standalone Functional API ─────────────────────────────────────

def explain_prediction(
    model: Any | None,
    features: dict[str, Any] | list | tuple | np.ndarray,
    prediction: int | None = _UNSET,
    target_class: int | None = None,
    feature_names: list[str] | None = None,
    class_mapping: dict[int, str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Explain a prediction on the given features using SHAP."""
    explainer = PhishGuardExplainer(
        model=model,
        feature_names=feature_names,
        class_mapping=class_mapping,
    )
    return explainer.explain(
        features=features,
        prediction=prediction,
        target_class=target_class,
        top_k=top_k,
    )
