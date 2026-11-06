"""
PhishGuard – Database CRUD Operations
"""

from datetime import datetime, timezone
from typing import Any
from sqlalchemy.orm import Session

from backend.database.models import Analysis


def create_analysis(
    db: Session,
    url: str,
    prediction: int | str | None = None,
    risk_score: float | None = None,
    analysis_status: str = "completed",
    analyzed_at: datetime | None = None,
    threat_category: str | None = None,
    class_probabilities: dict[str, float] | None = None,
    top_features: list[dict[str, Any]] | None = None,
    explanation: dict[str, Any] | None = None,
    visual_status: str | None = "not_provided",
    visual_features: dict[str, Any] | None = None,
) -> Analysis:
    """Create and persist a new Analysis record with probabilities, explainability, and visual extraction."""
    record = Analysis(
        url=url,
        prediction=prediction,
        threat_category=threat_category,
        risk_score=risk_score,
        class_probabilities=class_probabilities,
        top_features=top_features,
        explanation=explanation,
        analysis_status=analysis_status,
        analyzed_at=analyzed_at or datetime.now(timezone.utc),
        visual_status=visual_status or "not_provided",
        visual_features=visual_features,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_analysis_by_id(db: Session, analysis_id: int) -> Analysis | None:
    """Retrieve an Analysis record by primary key ID."""
    return db.query(Analysis).filter(Analysis.id == analysis_id).first()


# Alias for convenience
get_analysis = get_analysis_by_id


def list_analyses(
    db: Session,
    skip: int = 0,
    limit: int = 100,
) -> list[Analysis]:
    """List Analysis records ordered newest first."""
    return (
        db.query(Analysis)
        .order_by(Analysis.analyzed_at.desc(), Analysis.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


# Alias for convenience
list_records = list_analyses
