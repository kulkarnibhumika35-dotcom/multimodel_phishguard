"""
PhishGuard – URL Analysis API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database.crud import get_analysis_by_id, list_analyses
from backend.database.session import get_db
from backend.schemas.analysis import AnalysisResponse, AnalyzeRequest
from backend.services.analyzer import analyze_url

router = APIRouter(tags=["Analysis"])


@router.post("/analyze", response_model=AnalysisResponse, status_code=200)
async def analyze_endpoint(
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
):
    """Analyze a URL for phishing threats.

    Extracts features, evaluates the model if available,
    and records the result in SQLite.
    """
    record = analyze_url(url=payload.url, db=db)
    return record


@router.get("/analyses", response_model=list[AnalysisResponse])
async def list_analyses_endpoint(
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    db: Session = Depends(get_db),
):
    """List historical analysis records ordered newest first."""
    records = list_analyses(db=db, skip=offset, limit=limit)
    return records


@router.get("/analyses/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_endpoint(
    analysis_id: int,
    db: Session = Depends(get_db),
):
    """Get an analysis record by its ID. Returns 404 if not found."""
    record = get_analysis_by_id(db=db, analysis_id=analysis_id)
    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Analysis with ID {analysis_id} not found.",
        )
    return record

