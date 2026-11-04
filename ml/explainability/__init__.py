"""
PhishGuard – Explainable AI Module

Provides model interpretability using SHAP for phishing URL predictions.
Identifies key contributing features and distinguishes positive vs negative
contributions towards phishing risk.
"""

from ml.explainability.explainer import (
    EXPLAINABLE_FEATURE_NAMES,
    PhishGuardExplainer,
    explain_prediction,
)

__all__ = [
    "EXPLAINABLE_FEATURE_NAMES",
    "PhishGuardExplainer",
    "explain_prediction",
]

