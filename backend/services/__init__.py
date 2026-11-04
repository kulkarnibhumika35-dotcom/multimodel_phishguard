"""
PhishGuard – Services Package
"""

from backend.services.analyzer import analyze_url, load_model

__all__ = ["analyze_url", "load_model"]

