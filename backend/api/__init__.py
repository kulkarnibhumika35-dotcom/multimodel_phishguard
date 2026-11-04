"""
PhishGuard – API Package
"""

from backend.api.analyze import router as analyze_router
from backend.api.routes import router as base_router

__all__ = ["analyze_router", "base_router"]
