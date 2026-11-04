"""
PhishGuard – API route definitions.
"""

from fastapi import APIRouter

from backend.schemas.common import HealthResponse, RootResponse

router = APIRouter()


@router.get("/", response_model=RootResponse)
async def root():
    """Return project info and API status."""
    return RootResponse(
        project="PhishGuard",
        status="online",
        version="1.0.0",
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Lightweight health probe."""
    return HealthResponse(
        status="healthy",
        service="PhishGuard API",
    )

