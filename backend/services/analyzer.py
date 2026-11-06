"""
PhishGuard – URL Analyzer Service

Handles feature extraction, tri-model inference orchestration (URL, visual, multimodal),
decision arbitration, multiclass SHAP explainability, and SQLite persistence.

Key Guarantees:
- Predicts 3 classes: 0 = legitimate, 1 = phishing, 2 = malware_distribution.
- Never allows binary multimodal predictions to overwrite or demote malware_distribution.
- Routes to multimodal model (51 features) when a valid screenshot is available.
- Falls back to standalone URL model (preserving all 3 classes) when screenshots fail or are missing.
- Never invents synthetic predictions or risk scores when models are unavailable.
- Records rich shadow-mode telemetry without intercepting browser navigation.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np
from sqlalchemy.orm import Session

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database.crud import create_analysis
from backend.database.models import Analysis
from backend.services.decision_arbiter import DecisionArbiter
from ml.explainability.explainer import (
    CLASS_LEGITIMATE,
    CLASS_MALWARE,
    CLASS_MAP,
    CLASS_PHISHING,
    PhishGuardExplainer,
    load_class_mapping,
)
from ml.features.feature_fusion import FUSED_FEATURE_NAMES
from ml.features.url_features import FEATURE_NAMES, extract_url_features
from ml.features.visual_features import VISUAL_FEATURE_NAMES
from ml.training.dataset_pipeline import canonical_normalize_url

logger = logging.getLogger(__name__)

# Default model artifact paths
DEFAULT_URL_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "url_model.joblib")
DEFAULT_VISUAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_model.joblib")
DEFAULT_MULTIMODAL_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "multimodal_model.joblib")

# Default metadata paths
DEFAULT_URL_FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "feature_names.json")
DEFAULT_VISUAL_FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_feature_names.json")
DEFAULT_MULTIMODAL_FEATURE_NAMES_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "multimodal_feature_names.json")

DEFAULT_URL_CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "class_mapping.json")
DEFAULT_VISUAL_CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "visual_class_mapping.json")
DEFAULT_MULTIMODAL_CLASS_MAPPING_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "multimodal_class_mapping.json")


def get_url_model_path() -> str:
    """Return URL model file path from environment or default."""
    return os.getenv("PHISHGUARD_URL_MODEL_PATH", os.getenv("PHISHGUARD_MODEL_PATH", DEFAULT_URL_MODEL_PATH))


def get_model_path() -> str:
    """Backward-compatible alias for URL model path."""
    return get_url_model_path()


def get_visual_model_path() -> str:
    """Return visual model file path from environment or default."""
    return os.getenv("PHISHGUARD_VISUAL_MODEL_PATH", DEFAULT_VISUAL_MODEL_PATH)


def get_multimodal_model_path() -> str:
    """Return multimodal model file path from environment or default."""
    return os.getenv("PHISHGUARD_MULTIMODAL_MODEL_PATH", DEFAULT_MULTIMODAL_MODEL_PATH)


class ModelManager:
    """Manages thread-safe loading, caching, and compatibility checks for all three models."""

    def __init__(self) -> None:
        # URL Model State
        self._url_model: Any | None = None
        self._url_explainer: PhishGuardExplainer | None = None
        self._url_class_mapping: dict[int, str] = dict(CLASS_MAP)
        self._url_loaded_path: str | None = None
        self._url_loaded_mtime: float | None = None

        # Visual Model State
        self._visual_model: Any | None = None
        self._visual_class_mapping: dict[int, str] = {0: "unverified_candidate", 1: "phishing"}
        self._visual_loaded_path: str | None = None
        self._visual_loaded_mtime: float | None = None

        # Multimodal Model State
        self._multimodal_model: Any | None = None
        self._multimodal_class_mapping: dict[int, str] = {0: "unverified_candidate", 1: "phishing"}
        self._multimodal_loaded_path: str | None = None
        self._multimodal_loaded_mtime: float | None = None

    # ── URL Model Accessors ──────────────────────────────────────────

    def get_url_model_and_explainer(
        self,
        model_path: str | None = None,
    ) -> tuple[Any | None, PhishGuardExplainer | None, dict[int, str]]:
        """Return cached URL model and explainer, reloading if path or mtime changed."""
        current_path = model_path or get_url_model_path()

        if not os.path.isfile(current_path):
            if self._url_loaded_path != current_path:
                self._url_model = None
                self._url_explainer = None
                self._url_class_mapping = dict(CLASS_MAP)
                self._url_loaded_path = current_path
                self._url_loaded_mtime = None
            return None, None, self._url_class_mapping

        try:
            mtime = os.path.getmtime(current_path)
        except OSError:
            mtime = None

        if (
            self._url_model is not None
            and self._url_loaded_path == current_path
            and self._url_loaded_mtime == mtime
        ):
            return self._url_model, self._url_explainer, self._url_class_mapping

        return self.load_url_model(current_path)

    def get_model_and_explainer(
        self,
        model_path: str | None = None,
    ) -> tuple[Any | None, PhishGuardExplainer | None, dict[int, str]]:
        """Backward-compatible accessor for URL model and explainer."""
        return self.get_url_model_and_explainer(model_path)

    def load_url_model(
        self,
        model_path: str | None = None,
    ) -> tuple[Any | None, PhishGuardExplainer | None, dict[int, str]]:
        """Load and validate URL model, metadata, and SHAP explainer."""
        path = model_path or get_url_model_path()
        if not os.path.isfile(path):
            logger.info("URL model file not found at %s. Running in model_unavailable mode.", path)
            self._url_model = None
            self._url_explainer = None
            self._url_class_mapping = dict(CLASS_MAP)
            self._url_loaded_path = path
            self._url_loaded_mtime = None
            return None, None, self._url_class_mapping

        try:
            clf = joblib.load(path)
            if not hasattr(clf, "predict"):
                logger.warning("Loaded URL object at %s has no predict method; incompatible model.", path)
                self._url_model = None
                self._url_explainer = None
                self._url_loaded_path = path
                self._url_loaded_mtime = None
                return None, None, self._url_class_mapping

            if hasattr(clf, "n_features_in_") and clf.n_features_in_ != len(FEATURE_NAMES):
                logger.warning(
                    "URL model at %s expects %d features; requires exactly %d features.",
                    path,
                    clf.n_features_in_,
                    len(FEATURE_NAMES),
                )
                self._url_model = None
                self._url_explainer = None
                self._url_loaded_path = path
                self._url_loaded_mtime = None
                return None, None, self._url_class_mapping

            # Load class mapping
            class_mapping = dict(CLASS_MAP)
            try:
                disk_mapping = load_class_mapping(DEFAULT_URL_CLASS_MAPPING_PATH)
                if disk_mapping:
                    class_mapping.update(disk_mapping)
            except Exception:
                pass

            if hasattr(clf, "classes_"):
                classes = list(clf.classes_)
                if len(classes) == 2 and 2 not in classes:
                    class_mapping = {0: "legitimate", 1: "phishing"}

            # Initialize SHAP explainer safely
            try:
                explainer = PhishGuardExplainer(model=clf, class_mapping=class_mapping)
            except Exception as exc:
                logger.warning("Failed to initialize PhishGuardExplainer: %s", exc)
                explainer = None

            self._url_model = clf
            self._url_explainer = explainer
            self._url_class_mapping = class_mapping
            self._url_loaded_path = path
            try:
                self._url_loaded_mtime = os.path.getmtime(path)
            except OSError:
                self._url_loaded_mtime = None

            logger.info("Successfully loaded PhishGuard URL model and explainer from %s", path)
            return self._url_model, self._url_explainer, self._url_class_mapping

        except Exception as exc:
            logger.warning("Failed to load URL model from %s: %s", path, exc)
            self._url_model = None
            self._url_explainer = None
            self._url_loaded_path = path
            self._url_loaded_mtime = None
            return None, None, self._url_class_mapping

    # ── Visual Model Accessors ───────────────────────────────────────

    def get_visual_model(
        self,
        model_path: str | None = None,
    ) -> tuple[Any | None, dict[int, str]]:
        """Return cached visual model and class mapping, reloading if changed."""
        current_path = model_path or get_visual_model_path()

        if not os.path.isfile(current_path):
            self._visual_model = None
            self._visual_loaded_path = current_path
            self._visual_loaded_mtime = None
            return None, self._visual_class_mapping

        try:
            mtime = os.path.getmtime(current_path)
        except OSError:
            mtime = None

        if (
            self._visual_model is not None
            and self._visual_loaded_path == current_path
            and self._visual_loaded_mtime == mtime
        ):
            return self._visual_model, self._visual_class_mapping

        return self.load_visual_model(current_path)

    def load_visual_model(
        self,
        model_path: str | None = None,
    ) -> tuple[Any | None, dict[int, str]]:
        """Load and validate standalone visual model."""
        path = model_path or get_visual_model_path()
        if not os.path.isfile(path):
            logger.info("Visual model file not found at %s.", path)
            self._visual_model = None
            self._visual_loaded_path = path
            self._visual_loaded_mtime = None
            return None, self._visual_class_mapping

        try:
            clf = joblib.load(path)
            if not hasattr(clf, "predict"):
                logger.warning("Loaded visual object has no predict method: %s", path)
                self._visual_model = None
                return None, self._visual_class_mapping

            if hasattr(clf, "n_features_in_") and clf.n_features_in_ != len(VISUAL_FEATURE_NAMES):
                logger.warning(
                    "Visual model expects %d features; requires %d features.",
                    clf.n_features_in_,
                    len(VISUAL_FEATURE_NAMES),
                )
                self._visual_model = None
                return None, self._visual_class_mapping

            # Load class mapping
            cmap = {0: "unverified_candidate", 1: "phishing"}
            if os.path.isfile(DEFAULT_VISUAL_CLASS_MAPPING_PATH):
                try:
                    with open(DEFAULT_VISUAL_CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                        cmap = {int(k): str(v) for k, v in raw.items()}
                except Exception:
                    pass

            self._visual_model = clf
            self._visual_class_mapping = cmap
            self._visual_loaded_path = path
            try:
                self._visual_loaded_mtime = os.path.getmtime(path)
            except OSError:
                self._visual_loaded_mtime = None

            logger.info("Successfully loaded visual model from %s", path)
            return self._visual_model, self._visual_class_mapping

        except Exception as exc:
            logger.warning("Failed to load visual model from %s: %s", path, exc)
            self._visual_model = None
            return None, self._visual_class_mapping

    # ── Multimodal Model Accessors ───────────────────────────────────

    def get_multimodal_model(
        self,
        model_path: str | None = None,
    ) -> tuple[Any | None, dict[int, str]]:
        """Return cached multimodal model and class mapping, reloading if changed."""
        current_path = model_path or get_multimodal_model_path()

        if not os.path.isfile(current_path):
            self._multimodal_model = None
            self._multimodal_loaded_path = current_path
            self._multimodal_loaded_mtime = None
            return None, self._multimodal_class_mapping

        try:
            mtime = os.path.getmtime(current_path)
        except OSError:
            mtime = None

        if (
            self._multimodal_model is not None
            and self._multimodal_loaded_path == current_path
            and self._multimodal_loaded_mtime == mtime
        ):
            return self._multimodal_model, self._multimodal_class_mapping

        return self.load_multimodal_model(current_path)

    def load_multimodal_model(
        self,
        model_path: str | None = None,
    ) -> tuple[Any | None, dict[int, str]]:
        """Load and validate multimodal model."""
        path = model_path or get_multimodal_model_path()
        if not os.path.isfile(path):
            logger.info("Multimodal model file not found at %s.", path)
            self._multimodal_model = None
            self._multimodal_loaded_path = path
            self._multimodal_loaded_mtime = None
            return None, self._multimodal_class_mapping

        try:
            clf = joblib.load(path)
            if not hasattr(clf, "predict"):
                logger.warning("Loaded multimodal object has no predict method: %s", path)
                self._multimodal_model = None
                return None, self._multimodal_class_mapping

            if hasattr(clf, "n_features_in_") and clf.n_features_in_ != len(FUSED_FEATURE_NAMES):
                logger.warning(
                    "Multimodal model expects %d features; requires exactly %d features.",
                    clf.n_features_in_,
                    len(FUSED_FEATURE_NAMES),
                )
                self._multimodal_model = None
                return None, self._multimodal_class_mapping

            # Load class mapping
            cmap = {0: "unverified_candidate", 1: "phishing"}
            if os.path.isfile(DEFAULT_MULTIMODAL_CLASS_MAPPING_PATH):
                try:
                    with open(DEFAULT_MULTIMODAL_CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                        cmap = {int(k): str(v) for k, v in raw.items()}
                except Exception:
                    pass

            self._multimodal_model = clf
            self._multimodal_class_mapping = cmap
            self._multimodal_loaded_path = path
            try:
                self._multimodal_loaded_mtime = os.path.getmtime(path)
            except OSError:
                self._multimodal_loaded_mtime = None

            logger.info("Successfully loaded multimodal model from %s", path)
            return self._multimodal_model, self._multimodal_class_mapping

        except Exception as exc:
            logger.warning("Failed to load multimodal model from %s: %s", path, exc)
            self._multimodal_model = None
            return None, self._multimodal_class_mapping

    # ── Batch Preload ────────────────────────────────────────────────

    def load_all(
        self,
        url_path: str | None = None,
        visual_path: str | None = None,
        multimodal_path: str | None = None,
    ) -> None:
        """Preload all three models during application startup."""
        self.load_url_model(url_path)
        self.load_visual_model(visual_path)
        self.load_multimodal_model(multimodal_path)


# Global singleton instance
_model_manager = ModelManager()


def init_analyzer(model_path: str | None = None) -> None:
    """Preload models and explainer during application startup."""
    _model_manager.load_all(url_path=model_path)


def load_model(model_path: str | None = None) -> Any | None:
    """Convenience accessor to get loaded URL model (backward compatible)."""
    model, _, _ = _model_manager.get_model_and_explainer(model_path)
    return model


def get_model_manager() -> ModelManager:
    """Return the global ModelManager singleton."""
    return _model_manager


# Global decision arbiter
_arbiter = DecisionArbiter()


def analyze_url(
    url: str,
    db: Session,
    model_path: str | None = None,
    screenshot_base64: str | None = None,
    pre_extracted_visual_features: dict[str, Any] | None = None,
) -> Analysis:
    """Extract features, arbitrate multi-model inference, generate SHAP explanations, and persist.

    Enforces the Phase 3 Decision Arbiter rules:
    - Preempts binary multimodal model if URL model detects malware_distribution (threshold >= 0.50).
    - Routes to full multimodal model (51 features) when valid visual features are present.
    - Falls back to standalone URL model (preserving 3 classes) when screenshots fail or are missing.
    - Records comprehensive shadow-mode telemetry inside explanation JSON.
    - Strictly non-blocking: does not cancel or redirect browser navigation.

    Returns the persisted Analysis record.
    """
    # 1. Canonical normalization
    norm_url = canonical_normalize_url(url) or url.strip()

    # 2. Extract exactly the 20 canonical URL features
    url_features = extract_url_features(norm_url)

    # 3. Retrieve loaded models and explainers
    url_clf, explainer, url_cmap = _model_manager.get_url_model_and_explainer(model_path)
    vis_clf, vis_cmap = _model_manager.get_visual_model()
    multi_clf, multi_cmap = _model_manager.get_multimodal_model()

    # 4. Execute Hierarchical Decision Arbitration
    arb_result = _arbiter.arbitrate(
        url=norm_url,
        url_features=url_features,
        url_model=url_clf,
        url_class_mapping=url_cmap,
        visual_model=vis_clf,
        visual_class_mapping=vis_cmap,
        multimodal_model=multi_clf,
        multimodal_class_mapping=multi_cmap,
        screenshot_base64=screenshot_base64,
        pre_extracted_visual_features=pre_extracted_visual_features,
        is_shadow_mode=True,
    )

    # 5. Generate SHAP explanation safely for URL features
    top_features = []
    explanation_payload: dict[str, Any] | None = None
    if explainer is not None and arb_result.prediction is not None:
        try:
            expl_res = explainer.explain(
                features=url_features,
                prediction=arb_result.prediction,
                target_class=arb_result.prediction,
            )
            top_features = expl_res.get("top_features", [])
            explanation_payload = expl_res
        except Exception as exc:
            logger.warning("SHAP explanation failed for URL %s: %s", url, exc)
            explanation_payload = {"status": "error", "error": str(exc)}

    # Ensure explanation payload structure and embed multi-model telemetry
    if explanation_payload is None:
        explanation_payload = {
            "status": "completed" if arb_result.prediction is not None else "unavailable",
            "top_features": top_features,
        }
    explanation_payload["multi_model_telemetry"] = arb_result.to_telemetry_dict()

    # 6. Save analysis in SQLite (URL model metadata, visual features, telemetry)
    record = create_analysis(
        db=db,
        url=url,
        prediction=arb_result.prediction,
        risk_score=arb_result.risk_score,
        analysis_status=arb_result.analysis_status,
        threat_category=arb_result.threat_category,
        class_probabilities=arb_result.class_probabilities,
        top_features=top_features,
        explanation=explanation_payload,
        visual_status=arb_result.visual_status,
        visual_features=arb_result.visual_features,
    )

    # 7. Attach runtime explainability, telemetry, and visual attributes for FastAPI serialization
    record.threat_category = arb_result.threat_category
    record.class_probabilities = arb_result.class_probabilities
    record.probabilities = arb_result.class_probabilities
    record.top_features = top_features
    record.explanation = explanation_payload
    record.visual_status = arb_result.visual_status
    record.visual_features = arb_result.visual_features
    record.visual_message = arb_result.visual_message

    # Multi-model structured response fields
    record.url_model_result = arb_result.url_model_result
    record.visual_model_result = arb_result.visual_model_result
    record.multimodal_model_result = arb_result.multimodal_model_result
    record.model_routing = arb_result.model_routing
    record.is_shadow_mode = arb_result.is_shadow_mode
    record.fallback_reason = arb_result.fallback_reason
    record.confidence = arb_result.confidence

    return record
