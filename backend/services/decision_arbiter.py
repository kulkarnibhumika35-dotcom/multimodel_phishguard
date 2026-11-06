"""
PhishGuard – Decision Arbiter Service

Arbitrates classification decisions across URL, visual, and multimodal models
with strict malware preemption, graceful screenshot fallback, and shadow-mode telemetry.

Key Guarantees:
- Never overrides malware_distribution detected by the URL model.
- Derives risk score directly from raw ensemble model probabilities (uncalibrated leaf vote fractions); never uses arbitrary static clamping.
- Uses the multimodal model (51 features, has_visual_features=1.0) when a valid screenshot is available.
- Falls back to the standalone URL model (preserving all 3 classes) when screenshot capture fails or is missing.
- Strictly non-blocking: records shadow-mode telemetry without intercepting browser navigation.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import logging
from typing import Any, Mapping

import cv2
import numpy as np

from ml.features.feature_fusion import FUSED_FEATURE_NAMES, fuse_features
from ml.features.url_features import FEATURE_NAMES as URL_FEATURE_NAMES
from ml.features.visual_features import VISUAL_FEATURE_NAMES, extract_visual_features

logger = logging.getLogger(__name__)

# Default approved threshold for malware preemption
DEFAULT_MALWARE_THRESHOLD = 0.50


@dataclass
class ArbiterResult:
    """Encapsulates the complete decision arbitration outcome and multi-model telemetry."""

    prediction: int | None
    threat_category: str | None
    risk_score: float | None
    confidence: float | None
    class_probabilities: dict[str, float] | None
    url_model_result: dict[str, Any] | None
    visual_model_result: dict[str, Any] | None
    multimodal_model_result: dict[str, Any] | None
    model_routing: str
    visual_status: str
    visual_features: dict[str, Any] | None
    visual_message: str | None
    fallback_reason: str | None
    is_shadow_mode: bool = True
    ambiguity_flag: bool = False
    decision_margin: float | None = None
    analysis_status: str = "completed"

    def to_telemetry_dict(self) -> dict[str, Any]:
        """Convert multi-model telemetry to a JSON-serializable dictionary."""
        return {
            "url_model_result": self.url_model_result,
            "visual_model_result": self.visual_model_result,
            "multimodal_model_result": self.multimodal_model_result,
            "model_routing": self.model_routing,
            "visual_status": self.visual_status,
            "is_shadow_mode": self.is_shadow_mode,
            "fallback_reason": self.fallback_reason,
            "confidence": self.confidence,
            "ambiguity_flag": self.ambiguity_flag,
            "decision_margin": self.decision_margin,
        }


class DecisionArbiter:
    """Hierarchical decision arbiter managing tri-model inference and fallback."""

    def __init__(self, malware_threshold: float = DEFAULT_MALWARE_THRESHOLD) -> None:
        self.malware_threshold = float(malware_threshold)

    def arbitrate(
        self,
        url: str,
        url_features: Mapping[str, float | int],
        url_model: Any | None,
        url_class_mapping: dict[int, str] | None = None,
        visual_model: Any | None = None,
        visual_class_mapping: dict[int, str] | None = None,
        multimodal_model: Any | None = None,
        multimodal_class_mapping: dict[int, str] | None = None,
        screenshot_base64: str | None = None,
        pre_extracted_visual_features: dict[str, Any] | None = None,
        is_shadow_mode: bool = True,
    ) -> ArbiterResult:
        """Arbitrate between models following the approved Phase 3 hierarchy.

        Parameters
        ----------
        url : str
            Normalized target URL.
        url_features : Mapping[str, float | int]
            Extracted 20 URL lexical features.
        url_model : Any | None
            Loaded standalone URL Random Forest model (3 classes).
        url_class_mapping : dict[int, str] | None
            Class mapping for URL model (default: {0: legit, 1: phish, 2: malware}).
        visual_model : Any | None
            Loaded standalone Visual model (30 features, 2 classes).
        visual_class_mapping : dict[int, str] | None
            Class mapping for Visual model.
        multimodal_model : Any | None
            Loaded Multimodal model (51 features, 2 classes).
        multimodal_class_mapping : dict[int, str] | None
            Class mapping for Multimodal model.
        screenshot_base64 : str | None
            Optional raw base64-encoded screenshot string.
        pre_extracted_visual_features : dict[str, Any] | None
            Optional pre-extracted visual feature dictionary (for testing or pipeline bypass).
        is_shadow_mode : bool
            Whether shadow-mode telemetry logging is active (default True).

        Returns
        -------
        ArbiterResult
            Full arbitration decision with multi-model telemetry and fallback status.
        """
        # ── 1. Evaluate Standalone URL Model ─────────────────────────
        url_result = self._evaluate_url_model(
            url_features=url_features,
            url_model=url_model,
            class_mapping=url_class_mapping,
        )

        # ── 2. Process Screenshot / Visual Features ───────────────────
        vis_features, vis_status, vis_msg = self._process_visual_input(
            screenshot_base64=screenshot_base64,
            pre_extracted=pre_extracted_visual_features,
        )

        # ── 3. Primary Model Availability Guard ───────────────────────
        # If URL model is unavailable, the pipeline cannot classify threats or verify malware
        if url_result["status"] != "completed":
            return ArbiterResult(
                prediction=None,
                threat_category=None,
                risk_score=None,
                confidence=None,
                class_probabilities=None,
                url_model_result=url_result,
                visual_model_result={"status": "skipped", "reason": "URL model unavailable"},
                multimodal_model_result={"status": "skipped", "reason": "URL model unavailable"},
                model_routing="model_unavailable",
                visual_status=vis_status,
                visual_features=vis_features,
                visual_message=vis_msg,
                fallback_reason="Primary URL model is unavailable.",
                is_shadow_mode=is_shadow_mode,
                analysis_status="model_unavailable",
            )

        # ── 3. Malware Safeguard Check ───────────────────────────────
        # Check if URL model detected malware distribution
        is_malware = False
        if url_result["status"] == "completed":
            p_mal = url_result["probabilities"].get("malware_distribution", 0.0)
            if url_result["prediction"] == 2 or p_mal >= self.malware_threshold:
                is_malware = True

        if is_malware:
            # Lock decision to malware_distribution (preempt binary multimodal model)
            p_legit = url_result["probabilities"].get("legitimate", 0.0)
            # Raw threat risk score: 1.0 - P(legit) = P(phish) + P(malware)
            # Note: Random Forest predict_proba outputs are raw ensemble vote fractions (raw_model_probability),
            # not empirically calibrated probabilities (e.g. via Platt scaling or isotonic regression).
            raw_threat_risk = round(float(1.0 - p_legit), 4)
            p_malware = url_result["probabilities"].get("malware_distribution", 0.0)

            # Record multimodal status as overridden in shadow telemetry if screenshot was extracted
            multi_result = None
            vis_result = None
            if vis_status == "extracted" and multimodal_model is not None:
                multi_result = {
                    "status": "overridden",
                    "reason": "Multimodal classification preempted by URL malware detection.",
                }
            elif vis_status != "extracted":
                multi_result = {"status": "skipped", "reason": f"Screenshot unavailable ({vis_status})"}

            return ArbiterResult(
                prediction=2,
                threat_category="malware_distribution",
                risk_score=raw_threat_risk,
                confidence=round(float(p_malware), 4),
                class_probabilities=dict(url_result["probabilities"]),
                url_model_result=url_result,
                visual_model_result=vis_result,
                multimodal_model_result=multi_result,
                model_routing="url_malware_preempted",
                visual_status=vis_status,
                visual_features=vis_features,
                visual_message=vis_msg,
                fallback_reason="Malware distribution detected by URL model; multimodal classification preempted.",
                is_shadow_mode=is_shadow_mode,
                analysis_status="completed",
            )

        # ── 4. Malware Negative: Choose Route (Multimodal vs Fallback)
        if vis_status == "extracted" and multimodal_model is not None:
            # Valid visual features and multimodal model loaded: evaluate full multimodal model
            return self._arbitrate_multimodal_full(
                url_features=url_features,
                visual_features=vis_features,
                multimodal_model=multimodal_model,
                visual_model=visual_model,
                url_result=url_result,
                vis_status=vis_status,
                vis_msg=vis_msg,
                is_shadow_mode=is_shadow_mode,
            )

        # Fallback to URL model when screenshot is unavailable, capture failed, or multimodal model missing
        return self._arbitrate_url_fallback(
            url_result=url_result,
            multimodal_model=multimodal_model,
            vis_status=vis_status,
            vis_features=vis_features,
            vis_msg=vis_msg,
            is_shadow_mode=is_shadow_mode,
        )

    # ── Internal Evaluation Helpers ──────────────────────────────────

    def _evaluate_url_model(
        self,
        url_features: Mapping[str, float | int],
        url_model: Any | None,
        class_mapping: dict[int, str] | None,
    ) -> dict[str, Any]:
        """Evaluate the 20-feature URL classifier."""
        if url_model is None:
            return {"status": "unavailable", "prediction": None, "probabilities": None}

        try:
            X_url = np.array(
                [[float(url_features[name]) for name in URL_FEATURE_NAMES]],
                dtype=np.float64,
            )
            pred = int(url_model.predict(X_url)[0])

            if hasattr(url_model, "predict_proba"):
                proba = url_model.predict_proba(X_url)[0]
                probs = {
                    "legitimate": round(float(proba[0]), 4),
                    "phishing": round(float(proba[1]), 4) if len(proba) > 1 else 0.0,
                    "malware_distribution": round(float(proba[2]), 4) if len(proba) > 2 else 0.0,
                }
            else:
                probs = {
                    "legitimate": 1.0 if pred == 0 else 0.0,
                    "phishing": 1.0 if pred == 1 else 0.0,
                    "malware_distribution": 1.0 if pred == 2 else 0.0,
                }

            cmap = class_mapping or {0: "legitimate", 1: "phishing", 2: "malware_distribution"}
            threat_cat = cmap.get(pred, "legitimate" if pred == 0 else ("phishing" if pred == 1 else "malware_distribution"))

            return {
                "status": "completed",
                "prediction": pred,
                "threat_category": threat_cat,
                "probabilities": probs,
            }
        except Exception as exc:
            logger.warning("Error evaluating URL model: %s", exc)
            return {"status": "error", "error": str(exc), "prediction": None, "probabilities": None}

    def _process_visual_input(
        self,
        screenshot_base64: str | None,
        pre_extracted: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, str, str | None]:
        """Decode screenshot and extract 30 visual features."""
        if pre_extracted is not None:
            if pre_extracted.get("width", 0) > 0 and pre_extracted.get("height", 0) > 0:
                return (
                    pre_extracted,
                    "extracted",
                    "Visual features provided directly (30 features).",
                )
            return None, "invalid_image", "Pre-extracted visual features have invalid dimensions."

        if not screenshot_base64 or not str(screenshot_base64).strip():
            return None, "not_provided", None

        try:
            payload_data = screenshot_base64.strip()
            if payload_data.startswith("data:") and ";base64," in payload_data:
                _, payload_data = payload_data.split(";base64,", 1)
            clean_b64 = "".join(payload_data.split())
            raw_bytes = base64.b64decode(clean_b64, validate=True)

            if not raw_bytes:
                return None, "invalid_image", "Decoded screenshot payload is empty."

            # Verify image decode
            nparr = np.frombuffer(raw_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
                return None, "invalid_image", "Failed to decode screenshot as valid image."

            extracted = extract_visual_features(raw_bytes)
            if extracted and extracted.get("width", 0) > 0 and extracted.get("height", 0) > 0:
                return (
                    extracted,
                    "extracted",
                    "Visual features extracted successfully (30 features).",
                )
            return None, "extraction_failed", "Failed to compute visual features from decoded image."

        except Exception as exc:
            logger.warning("Visual input processing exception: %s", exc)
            return None, "invalid_image", f"Error processing screenshot payload: {exc}"

    def _arbitrate_multimodal_full(
        self,
        url_features: Mapping[str, float | int],
        visual_features: dict[str, Any],
        multimodal_model: Any,
        visual_model: Any | None,
        url_result: dict[str, Any],
        vis_status: str,
        vis_msg: str | None,
        is_shadow_mode: bool,
    ) -> ArbiterResult:
        """Execute multimodal full-mode classification and supplementary visual model."""
        try:
            # 1. Build and validate 51-feature fused vector
            fused_dict = fuse_features(url_features, visual_features)
            X_fused = np.array(
                [[float(fused_dict[name]) for name in FUSED_FEATURE_NAMES]],
                dtype=np.float64,
            )

            if X_fused.shape != (1, 51):
                raise ValueError(f"Fused vector dimension mismatch: expected (1, 51), got {X_fused.shape}")
            if fused_dict.get("has_visual_features") != 1.0:
                raise ValueError("Multimodal full mode requires has_visual_features == 1.0")

            # 2. Evaluate Multimodal Model
            pred_multi = int(multimodal_model.predict(X_fused)[0])
            if hasattr(multimodal_model, "predict_proba"):
                proba_multi = multimodal_model.predict_proba(X_fused)[0]
                p_multi_phish = round(float(proba_multi[1]), 4)
                p_multi_cand = round(float(proba_multi[0]), 4)
            else:
                p_multi_phish = 1.0 if pred_multi == 1 else 0.0
                p_multi_cand = 1.0 if pred_multi == 0 else 0.0

            final_threat = "phishing" if pred_multi == 1 else "unverified_candidate"
            margin = round(abs(p_multi_phish - 0.50), 4)
            ambiguity = bool(0.35 <= p_multi_phish <= 0.65)

            # Preserve URL model's malware probability in synchronized output
            p_mal = 0.0
            if url_result["status"] == "completed":
                p_mal = url_result["probabilities"].get("malware_distribution", 0.0)

            class_probs = {
                "unverified_candidate": p_multi_cand,
                "legitimate": p_multi_cand,  # Additive alias for backward compatibility
                "phishing": p_multi_phish,
                "malware_distribution": round(float(p_mal), 4),
            }

            multi_result = {
                "status": "completed",
                "prediction": pred_multi,
                "threat_category": final_threat,
                "probabilities": {
                    "unverified_candidate": p_multi_cand,
                    "phishing": p_multi_phish,
                },
                "decision_margin": margin,
                "ambiguous": ambiguity,
            }

            # 3. Evaluate Supplementary Standalone Visual Model
            vis_result = self._evaluate_visual_model(visual_features, visual_model)

            return ArbiterResult(
                prediction=pred_multi,
                threat_category=final_threat,
                risk_score=p_multi_phish,
                confidence=p_multi_phish if pred_multi == 1 else p_multi_cand,
                class_probabilities=class_probs,
                url_model_result=url_result,
                visual_model_result=vis_result,
                multimodal_model_result=multi_result,
                model_routing="multimodal_full",
                visual_status=vis_status,
                visual_features=visual_features,
                visual_message=vis_msg,
                fallback_reason=None,
                is_shadow_mode=is_shadow_mode,
                ambiguity_flag=ambiguity,
                decision_margin=margin,
                analysis_status="completed",
            )

        except Exception as exc:
            logger.warning("Error during multimodal inference: %s. Falling back to URL model.", exc)
            return self._arbitrate_url_fallback(
                url_result=url_result,
                multimodal_model=multimodal_model,
                vis_status="extraction_failed",
                vis_features=None,
                vis_msg=f"Multimodal inference error: {exc}",
                is_shadow_mode=is_shadow_mode,
                explicit_reason=f"Multimodal inference error: {exc}; fell back to standalone URL model.",
            )

    def _evaluate_visual_model(
        self,
        visual_features: dict[str, Any],
        visual_model: Any | None,
    ) -> dict[str, Any] | None:
        """Evaluate the 30-feature standalone visual classifier for telemetry."""
        if visual_model is None:
            return {"status": "unavailable"}

        try:
            X_vis = np.array(
                [[float(visual_features[name]) for name in VISUAL_FEATURE_NAMES]],
                dtype=np.float64,
            )
            pred_vis = int(visual_model.predict(X_vis)[0])
            if hasattr(visual_model, "predict_proba"):
                proba_vis = visual_model.predict_proba(X_vis)[0]
                p_vis_phish = round(float(proba_vis[1]), 4)
                p_vis_cand = round(float(proba_vis[0]), 4)
            else:
                p_vis_phish = 1.0 if pred_vis == 1 else 0.0
                p_vis_cand = 1.0 if pred_vis == 0 else 0.0

            return {
                "status": "completed",
                "prediction": pred_vis,
                "threat_category": "phishing" if pred_vis == 1 else "unverified_candidate",
                "probabilities": {
                    "unverified_candidate": p_vis_cand,
                    "phishing": p_vis_phish,
                },
            }
        except Exception as exc:
            logger.warning("Error evaluating visual model: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _arbitrate_url_fallback(
        self,
        url_result: dict[str, Any],
        multimodal_model: Any | None,
        vis_status: str,
        vis_features: dict[str, Any] | None,
        vis_msg: str | None,
        is_shadow_mode: bool,
        explicit_reason: str | None = None,
    ) -> ArbiterResult:
        """Fallback to the standalone URL model when screenshots or multimodal models are unavailable."""
        if url_result["status"] == "completed":
            pred = url_result["prediction"]
            threat = url_result["threat_category"]
            probs = dict(url_result["probabilities"])
            # Risk score: threat mass (1.0 - P(legit))
            p_legit = probs.get("legitimate", 0.0)
            risk = round(float(1.0 - p_legit), 4)

            reason = explicit_reason
            if not reason:
                if vis_status == "not_provided":
                    reason = "Screenshot not provided; evaluated with standalone URL model."
                elif vis_status in ("invalid_image", "extraction_failed"):
                    reason = f"Screenshot capture/extraction failed ({vis_status}); fell back to standalone URL model."
                elif multimodal_model is None:
                    reason = "Multimodal model unavailable; fell back to standalone URL model."
                else:
                    reason = "Evaluated with standalone URL model."

            return ArbiterResult(
                prediction=pred,
                threat_category=threat,
                risk_score=risk,
                confidence=round(float(probs.get(threat, risk)), 4),
                class_probabilities=probs,
                url_model_result=url_result,
                visual_model_result={"status": "skipped", "reason": reason},
                multimodal_model_result={"status": "skipped", "reason": reason},
                model_routing="url_model_fallback",
                visual_status=vis_status,
                visual_features=vis_features,
                visual_message=vis_msg,
                fallback_reason=reason,
                is_shadow_mode=is_shadow_mode,
                analysis_status="completed",
            )

        # All models unavailable
        reason = explicit_reason or "All classification models are unavailable."
        return ArbiterResult(
            prediction=None,
            threat_category=None,
            risk_score=None,
            confidence=None,
            class_probabilities=None,
            url_model_result=url_result,
            visual_model_result={"status": "unavailable"},
            multimodal_model_result={"status": "unavailable"},
            model_routing="model_unavailable",
            visual_status=vis_status,
            visual_features=vis_features,
            visual_message=vis_msg,
            fallback_reason=reason,
            is_shadow_mode=is_shadow_mode,
            analysis_status="model_unavailable",
        )

