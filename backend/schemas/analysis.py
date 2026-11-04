"""
PhishGuard – URL Analysis Pydantic Schemas
"""

from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class AnalyzeRequest(BaseModel):
    """Request payload for analyzing a URL."""

    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        if not v or not isinstance(v, str):
            raise ValueError("URL must be a non-empty string.")
        v = v.strip()
        parsed = urlparse(v)
        if parsed.scheme.lower() not in ("http", "https"):
            raise ValueError("URL must use HTTP or HTTPS protocol.")
        if not parsed.netloc:
            raise ValueError("URL must include a valid domain or host.")
        return v


class FeatureContribution(BaseModel):
    """Individual feature contribution explanation from SHAP."""

    feature_name: str
    feature_value: float | int | str
    contribution: float
    direction: str
    impact: str
    explanation: str

    model_config = ConfigDict(from_attributes=True)


class ExplanationDetails(BaseModel):
    """Detailed explainability metadata."""

    status: str
    base_value: Optional[float] = None
    target_class: Optional[int] = None
    target_class_name: Optional[str] = None
    top_features: list[FeatureContribution] = []
    all_contributions: Optional[dict[str, float]] = None
    error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AnalysisResponse(BaseModel):
    """Response payload representing an analysis record."""

    id: int
    url: str
    prediction: Optional[int | str] = None
    threat_category: Optional[str] = None
    risk_score: Optional[float] = None
    class_probabilities: Optional[dict[str, float]] = None
    probabilities: Optional[dict[str, float]] = None
    top_features: Optional[list[FeatureContribution]] = None
    explanation: Optional[dict[str, Any]] = None
    analysis_status: str
    analyzed_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def populate_threat_category_and_probabilities(cls, data: Any) -> Any:
        class_map = {0: "legitimate", 1: "phishing", 2: "malware_distribution"}

        if isinstance(data, dict):
            pred = data.get("prediction")
            cat = data.get("threat_category")
            if cat is None and pred is not None:
                try:
                    data["threat_category"] = class_map.get(int(pred))
                except (ValueError, TypeError):
                    pass
            # Synchronize probabilities & class_probabilities
            if "class_probabilities" in data and "probabilities" not in data:
                data["probabilities"] = data["class_probabilities"]
            elif "probabilities" in data and "class_probabilities" not in data:
                data["class_probabilities"] = data["probabilities"]
            return data

        # ORM model or other object
        pred = getattr(data, "prediction", None)
        cat = getattr(data, "threat_category", None)
        if cat is None and pred is not None:
            try:
                setattr(data, "threat_category", class_map.get(int(pred)))
            except (ValueError, TypeError, AttributeError):
                pass

        probs = getattr(data, "class_probabilities", None)
        alt_probs = getattr(data, "probabilities", None)
        if probs is not None and alt_probs is None:
            try:
                setattr(data, "probabilities", probs)
            except AttributeError:
                pass
        elif alt_probs is not None and probs is None:
            try:
                setattr(data, "class_probabilities", alt_probs)
            except AttributeError:
                pass

        return data
