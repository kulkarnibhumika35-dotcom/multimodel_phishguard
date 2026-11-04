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
