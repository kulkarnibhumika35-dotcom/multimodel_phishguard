"""
PhishGuard – SQLAlchemy Models
"""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Analysis(Base):
    """Analysis model for storing URL scan results."""

    __tablename__ = "analysis"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    url = Column(String, nullable=False, index=True)
    prediction = Column(Integer, nullable=True)
    threat_category = Column(String, nullable=True)
    risk_score = Column(Float, nullable=True)
    class_probabilities = Column(JSON, nullable=True)
    top_features = Column(JSON, nullable=True)
    explanation = Column(JSON, nullable=True)
    analysis_status = Column(String, nullable=False, default="completed")
    analyzed_at = Column(
        DateTime,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    visual_status = Column(String, nullable=True, default="not_provided")
    visual_features = Column(JSON, nullable=True)

    # Transient runtime attributes (not persisted in SQLite table)
    @property
    def visual_message(self) -> str | None:
        return getattr(self, "_visual_message", None)

    @visual_message.setter
    def visual_message(self, value: str | None) -> None:
        self._visual_message = value

    @property
    def url_model_result(self) -> dict | None:
        if hasattr(self, "_url_model_result"):
            return self._url_model_result
        if isinstance(self.explanation, dict) and "multi_model_telemetry" in self.explanation:
            return self.explanation["multi_model_telemetry"].get("url_model_result")
        return None

    @url_model_result.setter
    def url_model_result(self, value: dict | None) -> None:
        self._url_model_result = value

    @property
    def visual_model_result(self) -> dict | None:
        if hasattr(self, "_visual_model_result"):
            return self._visual_model_result
        if isinstance(self.explanation, dict) and "multi_model_telemetry" in self.explanation:
            return self.explanation["multi_model_telemetry"].get("visual_model_result")
        return None

    @visual_model_result.setter
    def visual_model_result(self, value: dict | None) -> None:
        self._visual_model_result = value

    @property
    def multimodal_model_result(self) -> dict | None:
        if hasattr(self, "_multimodal_model_result"):
            return self._multimodal_model_result
        if isinstance(self.explanation, dict) and "multi_model_telemetry" in self.explanation:
            return self.explanation["multi_model_telemetry"].get("multimodal_model_result")
        return None

    @multimodal_model_result.setter
    def multimodal_model_result(self, value: dict | None) -> None:
        self._multimodal_model_result = value

    @property
    def model_routing(self) -> str | None:
        if hasattr(self, "_model_routing"):
            return self._model_routing
        if isinstance(self.explanation, dict) and "multi_model_telemetry" in self.explanation:
            return self.explanation["multi_model_telemetry"].get("model_routing")
        return "url_model_fallback"

    @model_routing.setter
    def model_routing(self, value: str | None) -> None:
        self._model_routing = value

    @property
    def is_shadow_mode(self) -> bool:
        if hasattr(self, "_is_shadow_mode"):
            return self._is_shadow_mode
        if isinstance(self.explanation, dict) and "multi_model_telemetry" in self.explanation:
            return self.explanation["multi_model_telemetry"].get("is_shadow_mode", True)
        return True

    @is_shadow_mode.setter
    def is_shadow_mode(self, value: bool) -> None:
        self._is_shadow_mode = value

    @property
    def fallback_reason(self) -> str | None:
        if hasattr(self, "_fallback_reason"):
            return self._fallback_reason
        if isinstance(self.explanation, dict) and "multi_model_telemetry" in self.explanation:
            return self.explanation["multi_model_telemetry"].get("fallback_reason")
        return None

    @fallback_reason.setter
    def fallback_reason(self, value: str | None) -> None:
        self._fallback_reason = value

    @property
    def confidence(self) -> float | None:
        if hasattr(self, "_confidence"):
            return self._confidence
        if isinstance(self.explanation, dict) and "multi_model_telemetry" in self.explanation:
            return self.explanation["multi_model_telemetry"].get("confidence", self.risk_score)
        return self.risk_score

    @confidence.setter
    def confidence(self, value: float | None) -> None:
        self._confidence = value

    @property
    def probabilities(self) -> dict | None:
        """Alias for class_probabilities for schema compatibility."""
        return self.class_probabilities

    @probabilities.setter
    def probabilities(self, value: dict | None) -> None:
        self.class_probabilities = value

    def __repr__(self) -> str:
        return (
            f"<Analysis(id={self.id}, url='{self.url}', "
            f"prediction={self.prediction}, threat_category='{self.threat_category}', "
            f"risk_score={self.risk_score}, status='{self.analysis_status}')>"
        )
