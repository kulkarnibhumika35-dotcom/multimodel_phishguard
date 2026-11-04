"""
PhishGuard – Pydantic response schemas.
"""

from pydantic import BaseModel


class RootResponse(BaseModel):
    """GET / response."""
    project: str
    status: str
    version: str


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str
    service: str

