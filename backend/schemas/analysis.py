"""
PhishGuard – URL Analysis Pydantic Schemas
"""

from datetime import datetime
import base64
from typing import Any, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

MAX_SCREENSHOT_BASE64_LENGTH = 15 * 1024 * 1024  # 15 MB base64 characters
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB raw decoded bytes


class AnalyzeRequest(BaseModel):
    """Request payload for analyzing a URL and optional webpage screenshot."""

    url: str
    screenshot_base64: Optional[str] = None

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

    @field_validator("screenshot_base64")
    @classmethod
    def validate_screenshot(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("Screenshot must be a base64-encoded string.")

        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Screenshot payload cannot be empty.")

        if len(trimmed) > MAX_SCREENSHOT_BASE64_LENGTH:
            raise ValueError(
                f"Screenshot base64 payload exceeds maximum permitted length of {MAX_SCREENSHOT_BASE64_LENGTH} characters (15MB)."
            )

        payload_data = trimmed
        if payload_data.startswith("data:"):
            if ";base64," in payload_data:
                header, payload_data = payload_data.split(";base64,", 1)
                mime = header[5:].strip().lower()  # strip 'data:'
                if not mime.startswith("image/"):
                    raise ValueError(f"Invalid image MIME type: '{mime}'. Only image formats are supported.")
            else:
                raise ValueError("Data URL must include ';base64,' encoding specifier.")

        # Sanitize any embedded whitespace/newlines before b64decode
        clean_b64 = "".join(payload_data.split())
        try:
            image_bytes = base64.b64decode(clean_b64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 encoding: {exc}") from exc

        if not image_bytes:
            raise ValueError("Decoded screenshot payload is empty.")

        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"Decoded screenshot exceeds maximum permitted size of {MAX_IMAGE_BYTES} bytes (10MB)."
            )

        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            decoded_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise ValueError(f"Failed to process screenshot image data: {exc}") from exc

        if decoded_img is None:
            raise ValueError("Screenshot data could not be decoded as a valid image.")

        h, w = decoded_img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Decoded screenshot has invalid dimensions (0x0).")

        return trimmed


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
    confidence: Optional[float] = None
    class_probabilities: Optional[dict[str, float]] = None
    probabilities: Optional[dict[str, float]] = None
    top_features: Optional[list[FeatureContribution]] = None
    explanation: Optional[dict[str, Any]] = None
    analysis_status: str
    analyzed_at: datetime
    visual_status: Optional[str] = "not_provided"
    visual_features: Optional[dict[str, Any]] = None
    visual_message: Optional[str] = None
    url_model_result: Optional[dict[str, Any]] = None
    visual_model_result: Optional[dict[str, Any]] = None
    multimodal_model_result: Optional[dict[str, Any]] = None
    model_routing: Optional[str] = "url_model_fallback"
    is_shadow_mode: Optional[bool] = True
    fallback_reason: Optional[str] = None

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

            if "visual_status" not in data or data.get("visual_status") is None:
                data["visual_status"] = "not_provided"

            # Extract multi-model telemetry from explanation if present
            expl = data.get("explanation")
            if isinstance(expl, dict) and "multi_model_telemetry" in expl:
                telem = expl["multi_model_telemetry"]
                if "url_model_result" not in data:
                    data["url_model_result"] = telem.get("url_model_result")
                if "visual_model_result" not in data:
                    data["visual_model_result"] = telem.get("visual_model_result")
                if "multimodal_model_result" not in data:
                    data["multimodal_model_result"] = telem.get("multimodal_model_result")
                if "model_routing" not in data:
                    data["model_routing"] = telem.get("model_routing", "url_model_fallback")
                if "is_shadow_mode" not in data:
                    data["is_shadow_mode"] = telem.get("is_shadow_mode", True)
                if "fallback_reason" not in data:
                    data["fallback_reason"] = telem.get("fallback_reason")
                if "confidence" not in data:
                    data["confidence"] = telem.get("confidence", data.get("risk_score"))

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

        if getattr(data, "visual_status", None) is None:
            try:
                setattr(data, "visual_status", "not_provided")
            except AttributeError:
                pass

        return data
