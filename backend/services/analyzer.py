"""
PhishGuard – URL Analyzer Service

Handles feature extraction, model inference, multiclass SHAP explainability,
and SQLite persistence.
Supports predictions:
    0 = legitimate
    1 = phishing
    2 = malware_distribution
Never invents predictions or risk scores when model is unavailable.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import joblib
import numpy as np
from sqlalchemy.orm import Session

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database.crud import create_analysis
from backend.database.models import Analysis
from ml.explainability.explainer import (
    CLASS_LEGITIMATE,
    CLASS_MALWARE,
    CLASS_MAP,
    CLASS_PHISHING,
    PhishGuardExplainer,
    load_class_mapping,
)
from ml.features.url_features import FEATURE_NAMES, extract_url_features
from ml.training.dataset_pipeline import canonical_normalize_url

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "models", "url_model.joblib")


def get_model_path() -> str:
    """Return model file path from environment or default."""
    return os.getenv("PHISHGUARD_MODEL_PATH", DEFAULT_MODEL_PATH)


class ModelManager:
    """Manages thread-safe loading, caching, and compatibility checks for model and explainer."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._explainer: PhishGuardExplainer | None = None
        self._class_mapping: dict[int, str] = dict(CLASS_MAP)
        self._loaded_path: str | None = None
        self._loaded_mtime: float | None = None

    def get_model_and_explainer(
        self,
        model_path: str | None = None,
    ) -> tuple[Any | None, PhishGuardExplainer | None, dict[int, str]]:
        """Return cached model and explainer, or reload if path or mtime changed."""
        current_path = model_path or get_model_path()

        if not os.path.isfile(current_path):
            if self._loaded_path != current_path:
                self._model = None
                self._explainer = None
                self._class_mapping = dict(CLASS_MAP)
                self._loaded_path = current_path
                self._loaded_mtime = None
            return None, None, self._class_mapping

        try:
            mtime = os.path.getmtime(current_path)
        except OSError:
            mtime = None

        if (
            self._model is not None
            and self._loaded_path == current_path
            and self._loaded_mtime == mtime
        ):
            return self._model, self._explainer, self._class_mapping

        return self.load(current_path)

    def load(
        self,
        model_path: str | None = None,
    ) -> tuple[Any | None, PhishGuardExplainer | None, dict[int, str]]:
        """Load and validate model and SHAP explainer from specified or default path."""
        path = model_path or get_model_path()
        if not os.path.isfile(path):
            logger.info("Model file not found at %s. Running in model_unavailable mode.", path)
            self._model = None
            self._explainer = None
            self._class_mapping = dict(CLASS_MAP)
            self._loaded_path = path
            self._loaded_mtime = None
            return None, None, self._class_mapping

        try:
            clf = joblib.load(path)
            # Compatibility check 1: Must be a predictor
            if not hasattr(clf, "predict"):
                logger.warning("Loaded object at %s has no predict method; incompatible model.", path)
                self._model = None
                self._explainer = None
                self._loaded_path = path
                self._loaded_mtime = None
                return None, None, self._class_mapping

            # Compatibility check 2: Feature count must equal 20
            if hasattr(clf, "n_features_in_") and clf.n_features_in_ != len(FEATURE_NAMES):
                logger.warning(
                    "Model at %s expects %d features; PhishGuard requires exactly %d features.",
                    path,
                    clf.n_features_in_,
                    len(FEATURE_NAMES),
                )
                self._model = None
                self._explainer = None
                self._loaded_path = path
                self._loaded_mtime = None
                return None, None, self._class_mapping

            # Load class mapping
            class_mapping = dict(CLASS_MAP)
            try:
                disk_mapping = load_class_mapping()
                if disk_mapping:
                    class_mapping.update(disk_mapping)
            except Exception:
                pass

            # Adapt for binary models if detected
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

            self._model = clf
            self._explainer = explainer
            self._class_mapping = class_mapping
            self._loaded_path = path
            try:
                self._loaded_mtime = os.path.getmtime(path)
            except OSError:
                self._loaded_mtime = None

            logger.info("Successfully loaded PhishGuard model and explainer from %s", path)
            return self._model, self._explainer, self._class_mapping

        except Exception as exc:
            logger.warning("Failed to load model from %s: %s", path, exc)
            self._model = None
            self._explainer = None
            self._loaded_path = path
            self._loaded_mtime = None
            return None, None, self._class_mapping


# Global singleton instance
_model_manager = ModelManager()


def init_analyzer(model_path: str | None = None) -> None:
    """Preload model and explainer during application startup."""
    _model_manager.load(model_path)


def load_model(model_path: str | None = None) -> Any | None:
    """Convenience accessor to get loaded model (backward compatible)."""
    model, _, _ = _model_manager.get_model_and_explainer(model_path)
    return model


def get_model_manager() -> ModelManager:
    """Return the global ModelManager singleton."""
    return _model_manager


def analyze_url(
    url: str,
    db: Session,
    model_path: str | None = None,
) -> Analysis:
    """Extract features, run inference and SHAP explanation, and persist result.

    Returns the persisted Analysis record with populated prediction and explainability attributes.
    """
    # 1. Canonical normalization (guarantees training/inference feature extraction parity)
    norm_url = canonical_normalize_url(url) or url.strip()

    # Extract exactly the 20 canonical URL features
    features = extract_url_features(norm_url)
    feature_vector = np.array(
        [[features[name] for name in FEATURE_NAMES]],
        dtype=np.float64,
    )

    # 2. Retrieve loaded model and explainer
    clf, explainer, class_mapping = _model_manager.get_model_and_explainer(model_path)

    if clf is not None:
        pred = int(clf.predict(feature_vector)[0])
        threat_category = class_mapping.get(
            pred,
            "legitimate" if pred == 0 else ("phishing" if pred == 1 else "malware_distribution"),
        )

        # Calculate class probabilities
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(feature_vector)[0]
            class_probs = {
                "legitimate": round(float(proba[0]), 4),
                "phishing": round(float(proba[1]), 4) if len(proba) > 1 else 0.0,
                "malware_distribution": round(float(proba[2]), 4) if len(proba) > 2 else 0.0,
            }
            if len(proba) >= 3:
                # Overall threat risk: 1.0 - P(legitimate)
                risk_score = round(float(1.0 - proba[0]), 4)
            elif len(proba) == 2:
                risk_score = round(float(proba[1]), 4)
            else:
                risk_score = round(float(proba[0]), 4)
        else:
            class_probs = {
                "legitimate": 1.0 if pred == 0 else 0.0,
                "phishing": 1.0 if pred == 1 else 0.0,
                "malware_distribution": 1.0 if pred == 2 else 0.0,
            }
            risk_score = 0.0 if pred == 0 else 1.0

        # Generate SHAP explanation safely
        top_features = []
        explanation_payload = None
        if explainer is not None:
            try:
                expl_res = explainer.explain(
                    features=features,
                    prediction=pred,
                    target_class=pred,
                )
                top_features = expl_res.get("top_features", [])
                explanation_payload = expl_res
            except Exception as exc:
                logger.warning("SHAP explanation failed for URL %s: %s", url, exc)
                explanation_payload = {"status": "error", "error": str(exc)}

        prediction = pred
        analysis_status = "completed"
    else:
        # Never invent predictions or risk scores when model is unavailable
        prediction = None
        threat_category = None
        risk_score = None
        class_probs = None
        top_features = None
        explanation_payload = None
        analysis_status = "model_unavailable"

    # 3. Save analysis in SQLite
    record = create_analysis(
        db=db,
        url=url,
        prediction=prediction,
        risk_score=risk_score,
        analysis_status=analysis_status,
        threat_category=threat_category,
        class_probabilities=class_probs,
        top_features=top_features,
        explanation=explanation_payload,
    )

    # 4. Attach runtime explainability attributes to record for FastAPI serialization
    record.threat_category = threat_category
    record.class_probabilities = class_probs
    record.probabilities = class_probs
    record.top_features = top_features
    record.explanation = explanation_payload

    return record
