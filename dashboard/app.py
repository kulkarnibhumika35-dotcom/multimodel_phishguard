"""
PhishGuard – Security Analytics Dashboard

A modern Streamlit dashboard that connects to the PhishGuard FastAPI backend
and SQLite database to display analysis history, four-class statistics,
distribution charts, filters, detailed multi-model telemetry, TreeSHAP explainability,
and live URL analysis.

Launch:
    streamlit run dashboard/app.py

Theme: Dark navy (#0b1120) with cyan (#00e5ff) accents matching the
Chrome extension design.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st

# ── Configuration ───────────────────────────────────────────────────

BACKEND_URL = os.getenv("PHISHGUARD_BACKEND_URL", "http://127.0.0.1:8000")
DEFAULT_LIMIT = 500
REQUEST_TIMEOUT = 10  # seconds


# ── URL Input Validation ───────────────────────────────────────────

def validate_url_input(url: str | None) -> tuple[bool, str]:
    """Validate that input URL is non-empty, uses HTTP or HTTPS, and has a valid host.

    Returns:
        (True, cleaned_url) if valid.
        (False, error_message) if invalid.
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty."

    cleaned = url.strip()
    if not cleaned:
        return False, "URL cannot be empty."

    try:
        parsed = urlparse(cleaned)
    except Exception as exc:
        return False, f"Invalid URL syntax: {exc}"

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return False, "Only HTTP and HTTPS URLs can be analyzed."

    if not parsed.netloc:
        return False, "URL must include a valid domain name or host."

    return True, cleaned


# ── API Client ──────────────────────────────────────────────────────

def analyze_new_url(
    url: str,
    backend_url: str = BACKEND_URL,
    timeout: int = REQUEST_TIMEOUT,
    screenshot_base64: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Submit a URL to the backend POST /analyze endpoint.

    Returns:
        (response_data, None) on success.
        (None, error_message) on failure.
    """
    is_valid, err_or_cleaned = validate_url_input(url)
    if not is_valid:
        return None, err_or_cleaned

    cleaned_url = err_or_cleaned
    endpoint = f"{backend_url.rstrip('/')}/analyze"

    payload: dict[str, Any] = {"url": cleaned_url}
    if screenshot_base64:
        payload["screenshot_base64"] = screenshot_base64

    try:
        resp = requests.post(
            endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
        if resp.status_code == 422:
            detail = resp.json().get("detail", "Invalid URL format.")
            if isinstance(detail, list) and len(detail) > 0:
                detail = detail[0].get("msg", str(detail))
            return None, f"Backend validation rejected URL: {detail}"

        resp.raise_for_status()
        return resp.json(), None

    except requests.ConnectionError:
        return None, (
            f"Cannot connect to the PhishGuard backend at {backend_url}. "
            "Please ensure the server is running."
        )
    except requests.Timeout:
        return None, f"Analysis timed out after {timeout} seconds. Please try again."
    except requests.HTTPError as exc:
        status_code = getattr(exc.response, "status_code", "error")
        return None, f"Backend error: HTTP {status_code}"
    except Exception as exc:
        return None, f"Unexpected error during analysis: {exc}"


def fetch_analyses(
    backend_url: str = BACKEND_URL,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch analysis records from the PhishGuard backend.

    Returns an empty list on connection errors; never raises.
    """
    try:
        resp = requests.get(
            f"{backend_url.rstrip('/')}/analyses",
            params={"limit": limit, "offset": offset},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error(
            "⚠️ Cannot connect to the PhishGuard backend at "
            f"**{backend_url}**. Make sure the server is running:\n\n"
            "```\npython -m uvicorn backend.main:app --host 127.0.0.1 --port 8000\n```"
        )
        return []
    except requests.Timeout:
        st.error("⚠️ Backend request timed out. Please try again.")
        return []
    except requests.HTTPError as exc:
        st.error(f"⚠️ Backend returned an error: {exc.response.status_code}")
        return []
    except Exception as exc:
        st.error(f"⚠️ Unexpected error: {exc}")
        return []


def fetch_single_analysis(
    analysis_id: int,
    backend_url: str = BACKEND_URL,
) -> dict[str, Any] | None:
    """Fetch a single analysis record by ID from the PhishGuard backend."""
    try:
        resp = requests.get(
            f"{backend_url.rstrip('/')}/analyses/{analysis_id}",
            timeout=REQUEST_TIMEOUT,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ── Statistics Computation (Four-Class Semantics) ───────────────────

def compute_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute four-class dashboard statistics from analysis records.

    Categories:
    - 0 (Legitimate): Verified benign baseline URL
    - 0 (Unverified Candidate): Multimodal candidate from Tranco dataset
    - 1 (Phishing): High phishing threat
    - 2 (Malware Distribution): Malware distribution or C2 host
    - model_unavailable / null: Unavailable

    CRITICAL INVARIANT:
    Unverified candidate records are NEVER counted as confirmed legitimate.
    """
    total = len(records)
    phishing = 0
    legitimate = 0
    unverified_candidate = 0
    malware = 0
    unavailable = 0

    risk_scores: list[float] = []
    screenshot_success_count = 0
    multimodal_count = 0
    url_fallback_count = 0

    for rec in records:
        status = str(rec.get("analysis_status") or "").strip().lower()
        prediction = rec.get("prediction")
        threat_cat = str(rec.get("threat_category") or "").strip().lower()
        routing = str(rec.get("model_routing") or "").strip().lower()
        vis_status = str(rec.get("visual_status") or "").strip().lower()
        risk = rec.get("risk_score")

        # Categorization: threat_category is authoritative
        if status == "model_unavailable" or (prediction is None and not threat_cat):
            unavailable += 1
        elif prediction in (2, "2") or threat_cat in ("malware_distribution", "malware", "2"):
            malware += 1
        elif prediction in (1, "1") or threat_cat in ("phishing", "1"):
            phishing += 1
        elif threat_cat in ("unverified_candidate", "candidate") or (prediction in (0, "0") and routing == "multimodal_full"):
            unverified_candidate += 1
        elif prediction in (0, "0") or threat_cat in ("legitimate", "0"):
            legitimate += 1
        else:
            unavailable += 1

        # Pipeline & Operational telemetry
        if risk is not None and status != "model_unavailable":
            try:
                risk_scores.append(float(risk))
            except (ValueError, TypeError):
                pass

        if vis_status == "extracted":
            screenshot_success_count += 1

        if routing == "multimodal_full":
            multimodal_count += 1
        elif routing in ("url_model_fallback", "url_malware_preempted") or not routing:
            url_fallback_count += 1

    avg_risk_score = round(sum(risk_scores) / len(risk_scores), 4) if risk_scores else 0.0
    screenshot_success_rate = round((screenshot_success_count / total) * 100.0, 1) if total > 0 else 0.0

    return {
        "total": total,
        "phishing": phishing,
        "legitimate": legitimate,
        "unverified_candidate": unverified_candidate,
        "malware": malware,
        "unavailable": unavailable,
        "threat_total": phishing + malware,
        "avg_risk_score": avg_risk_score,
        "screenshot_success_rate": screenshot_success_rate,
        "multimodal_count": multimodal_count,
        "url_fallback_count": url_fallback_count,
    }


# ── Label Formatting Helpers ───────────────────────────────────────

def _format_category(rec: dict[str, Any]) -> str:
    """Map a record to its canonical category name."""
    status = str(rec.get("analysis_status") or "").strip().lower()
    pred = rec.get("prediction")
    cat = str(rec.get("threat_category") or "").strip().lower()
    routing = str(rec.get("model_routing") or "").strip().lower()

    if status == "model_unavailable" or (pred is None and not cat):
        return "Unavailable"
    if cat in ("malware_distribution", "malware", "2") or pred in (2, "2"):
        return "Malware Distribution"
    if cat in ("phishing", "1") or pred in (1, "1"):
        return "Phishing"
    if cat in ("unverified_candidate", "candidate") or (pred in (0, "0") and routing == "multimodal_full"):
        return "Unverified Candidate"
    if cat in ("legitimate", "0") or pred in (0, "0"):
        return "Legitimate"
    return "Unknown"


def _prediction_label(
    prediction: Any,
    status: str = "",
    threat_category: str | None = None,
    routing: str | None = None,
) -> str:
    """Convert raw prediction value to a display label.

    Supports four classes:
    - Legitimate: ✅ Legitimate
    - Unverified Candidate: 🔍 Unverified Candidate
    - Phishing: 🚨 Phishing
    - Malware Distribution: ☣️ Malware Distribution
    - model_unavailable / None: ⏳ Unavailable
    """
    cat = str(threat_category or "").strip().lower()
    route = str(routing or "").strip().lower()

    if status == "model_unavailable" or (prediction is None and not cat):
        return "⏳ Unavailable"
    if prediction in (2, "2") or cat in ("malware_distribution", "malware", "2"):
        return "☣️ Malware Distribution"
    if prediction in (1, "1") or cat in ("phishing", "1"):
        return "🚨 Phishing"
    if cat in ("unverified_candidate", "candidate") or (prediction in (0, "0") and route == "multimodal_full"):
        return "🔍 Unverified Candidate"
    if prediction in (0, "0") or cat in ("legitimate", "0"):
        return "✅ Legitimate"
    return "❓ Unknown"


def _threat_category_label(
    threat_category: Any,
    prediction: Any = None,
    status: str = "",
    routing: str | None = None,
) -> str:
    """Return clean human-readable threat category."""
    cat = str(threat_category or "").strip().lower()
    route = str(routing or "").strip().lower()

    if status == "model_unavailable" or (prediction is None and not cat):
        return "Unavailable"
    if cat in ("malware_distribution", "malware", "2") or prediction in (2, "2"):
        return "Malware Distribution"
    if cat in ("phishing", "1") or prediction in (1, "1"):
        return "Phishing"
    if cat in ("unverified_candidate", "candidate") or (prediction in (0, "0") and route == "multimodal_full"):
        return "Unverified Candidate"
    if cat in ("legitimate", "0") or prediction in (0, "0"):
        return "Legitimate"
    return "Unknown"


def _format_routing_label(routing: str | None) -> str:
    """Format model routing pipeline into a human-readable badge."""
    r = str(routing or "").strip().lower()
    if r == "multimodal_full":
        return "Multimodal Full (51 Feat)"
    if r == "url_model_fallback":
        return "URL Fallback (20 Feat)"
    if r == "url_malware_preempted":
        return "Malware Preempted"
    if r == "model_unavailable":
        return "Model Unavailable"
    if not r:
        return "URL Fallback (Legacy)"
    return r.replace("_", " ").title()


def _format_visual_status_label(status: str | None) -> str:
    """Format screenshot extraction status for display."""
    s = str(status or "").strip().lower()
    if s == "extracted":
        return "Captured (30 Feat)"
    if s == "not_provided":
        return "Not Provided"
    if s == "capture_failed":
        return "Capture Failed"
    if s == "invalid_image":
        return "Invalid Image"
    if s == "extraction_failed":
        return "Extraction Failed"
    if not s:
        return "Not Provided"
    return s.replace("_", " ").title()


def _risk_display(risk_score: Any, status: str) -> str:
    """Format risk score for display."""
    if status == "model_unavailable" or risk_score is None:
        return "—"
    try:
        return f"{float(risk_score):.1%}"
    except (ValueError, TypeError):
        return "—"


def _format_timestamp(ts: str | None) -> str:
    """Parse and format ISO 8601 timestamp for display."""
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)


# ── Filtering Helper ───────────────────────────────────────────────

def filter_records(
    records: list[dict[str, Any]],
    query: str = "",
    category: str = "All",
    status: str = "All",
    routing: str = "All",
) -> list[dict[str, Any]]:
    """Filter records by URL substring, threat category, analysis status, and routing."""
    filtered = records

    # 1. URL search substring
    if query and query.strip():
        q_lower = query.strip().lower()
        filtered = [r for r in filtered if q_lower in (r.get("url") or "").lower()]

    # 2. Threat category filter
    if category and category != "All":
        target = category.strip().lower()
        filtered = [
            r for r in filtered
            if _format_category(r).lower() == target
            or (target == "malware" and _format_category(r).lower() == "malware distribution")
            or (target in ("candidate", "unverified") and _format_category(r).lower() == "unverified candidate")
        ]

    # 3. Status filter
    if status and status != "All":
        st_lower = status.strip().lower()
        filtered = [r for r in filtered if (r.get("analysis_status") or "").lower() == st_lower]

    # 4. Routing filter
    if routing and routing != "All":
        r_target = routing.strip().lower()
        filtered = [
            r for r in filtered
            if (r.get("model_routing") or "url_model_fallback").lower() == r_target
            or _format_routing_label(r.get("model_routing")).lower() == r_target
        ]

    return filtered


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert API records to a display-friendly DataFrame with 4-class and multimodal details."""
    columns = [
        "ID",
        "URL",
        "Prediction",
        "Threat Category",
        "Risk Score",
        "Model Routing",
        "Screenshot Status",
        "Shadow Mode",
        "Status",
        "Analyzed At",
    ]
    if not records:
        return pd.DataFrame(columns=columns)

    rows = []
    for rec in records:
        status = rec.get("analysis_status", "")
        pred = rec.get("prediction")
        cat = rec.get("threat_category")
        routing = rec.get("model_routing")
        vis_status = rec.get("visual_status")
        is_shadow = rec.get("is_shadow_mode", True)

        rows.append({
            "ID": rec.get("id", ""),
            "URL": rec.get("url", ""),
            "Prediction": _prediction_label(pred, status, cat, routing),
            "Threat Category": _threat_category_label(cat, pred, status, routing),
            "Risk Score": _risk_display(rec.get("risk_score"), status),
            "Model Routing": _format_routing_label(routing),
            "Screenshot Status": _format_visual_status_label(vis_status),
            "Shadow Mode": "Active 🛡️" if is_shadow else "Disabled",
            "Status": status,
            "Analyzed At": _format_timestamp(rec.get("analyzed_at")),
        })
    return pd.DataFrame(rows)


# ── Chart Data Preparation Helpers ─────────────────────────────────

def prepare_four_class_distribution(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepare DataFrame for four-class threat distribution bar chart."""
    stats = compute_statistics(records)
    categories = [
        "Legitimate",
        "Unverified Candidate",
        "Phishing",
        "Malware Distribution",
    ]
    counts = [
        stats["legitimate"],
        stats["unverified_candidate"],
        stats["phishing"],
        stats["malware"],
    ]
    if stats["unavailable"] > 0:
        categories.append("Unavailable")
        counts.append(stats["unavailable"])

    return pd.DataFrame({"Category": categories, "Count": counts})


def prepare_risk_score_distribution(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepare DataFrame for risk score histogram / tier distribution."""
    tiers = [
        "0.0 - 0.2 (Minimal)",
        "0.2 - 0.4 (Low)",
        "0.4 - 0.6 (Medium)",
        "0.6 - 0.8 (High)",
        "0.8 - 1.0 (Critical)",
    ]
    counts = [0, 0, 0, 0, 0]

    for rec in records:
        if rec.get("analysis_status") == "model_unavailable":
            continue
        score = rec.get("risk_score")
        if score is None:
            continue
        try:
            s = float(score)
            if s < 0.2:
                counts[0] += 1
            elif s < 0.4:
                counts[1] += 1
            elif s < 0.6:
                counts[2] += 1
            elif s < 0.8:
                counts[3] += 1
            else:
                counts[4] += 1
        except (ValueError, TypeError):
            continue

    return pd.DataFrame({"Risk Tier": tiers, "Count": counts})


def prepare_daily_activity(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepare DataFrame for daily analysis activity timeline."""
    dates: list[str] = []
    for rec in records:
        ts = rec.get("analyzed_at", "")
        if ts:
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                dates.append(dt.strftime("%Y-%m-%d"))
            except Exception:
                pass

    if not dates:
        return pd.DataFrame(columns=["Date", "Analyses"])

    series = pd.Series(dates).value_counts().sort_index()
    return pd.DataFrame({"Date": series.index, "Analyses": series.values})


def prepare_model_routing_distribution(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepare DataFrame for model routing pipeline distribution."""
    routes = {
        "Multimodal Full (51 Feat)": 0,
        "URL Fallback (20 Feat)": 0,
        "Malware Preempted": 0,
        "Model Unavailable": 0,
    }
    for rec in records:
        status = rec.get("analysis_status")
        routing = rec.get("model_routing")
        if status == "model_unavailable" or routing == "model_unavailable":
            routes["Model Unavailable"] += 1
        elif routing == "multimodal_full":
            routes["Multimodal Full (51 Feat)"] += 1
        elif routing == "url_malware_preempted":
            routes["Malware Preempted"] += 1
        else:
            routes["URL Fallback (20 Feat)"] += 1

    return pd.DataFrame({"Routing Path": list(routes.keys()), "Count": list(routes.values())})


def prepare_screenshot_availability(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Prepare DataFrame for screenshot availability breakdown."""
    avail = {
        "Captured (Extracted)": 0,
        "Not Provided": 0,
        "Capture Failed": 0,
        "Invalid Image": 0,
    }
    for rec in records:
        s = str(rec.get("visual_status") or "").strip().lower()
        if s == "extracted":
            avail["Captured (Extracted)"] += 1
        elif s == "capture_failed":
            avail["Capture Failed"] += 1
        elif s in ("invalid_image", "extraction_failed"):
            avail["Invalid Image"] += 1
        else:
            avail["Not Provided"] += 1

    return pd.DataFrame({"Screenshot Status": list(avail.keys()), "Count": list(avail.values())})


def prepare_model_disagreement_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute sub-model agreement vs disagreement metrics."""
    total_multi = 0
    agreed = 0
    disagreed = 0
    preempted = 0
    breakdown = {
        "URL Legit vs Multi Phish": 0,
        "URL Phish vs Multi Candidate": 0,
        "Malware Preemption": 0,
    }

    for rec in records:
        routing = rec.get("model_routing")
        url_res = rec.get("url_model_result")
        multi_res = rec.get("multimodal_model_result")

        # Telemetry may also be nested in rec["explanation"]["multi_model_telemetry"]
        if not url_res and isinstance(rec.get("explanation"), dict):
            telem = rec["explanation"].get("multi_model_telemetry", {})
            url_res = telem.get("url_model_result")
            multi_res = telem.get("multimodal_model_result")

        if routing == "url_malware_preempted":
            total_multi += 1
            preempted += 1
            disagreed += 1
            breakdown["Malware Preemption"] += 1
        elif routing == "multimodal_full" and url_res and multi_res:
            total_multi += 1
            url_pred = url_res.get("prediction")
            multi_pred = multi_res.get("prediction")

            if url_pred is not None and multi_pred is not None:
                if (url_pred == 0 and multi_pred == 0) or (url_pred == 1 and multi_pred == 1):
                    agreed += 1
                else:
                    disagreed += 1
                    if url_pred == 0 and multi_pred == 1:
                        breakdown["URL Legit vs Multi Phish"] += 1
                    elif url_pred == 1 and multi_pred == 0:
                        breakdown["URL Phish vs Multi Candidate"] += 1

    disagreement_rate = round((disagreed / total_multi) * 100.0, 1) if total_multi > 0 else 0.0
    agreement_rate = round((agreed / total_multi) * 100.0, 1) if total_multi > 0 else 100.0

    breakdown_df = pd.DataFrame({
        "Disagreement Type": list(breakdown.keys()),
        "Count": list(breakdown.values()),
    })

    return {
        "total_multi_analyses": total_multi,
        "agreed_count": agreed,
        "disagreed_count": disagreed,
        "preempted_count": preempted,
        "disagreement_rate": disagreement_rate,
        "agreement_rate": agreement_rate,
        "breakdown_df": breakdown_df,
    }


# ── UI Theme Injection ─────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    /* ── Global Dark Navy Theme ────────────────────────── */
    .stApp {
        background-color: #0b1120 !important;
        color: #e0e6f0 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0d1526 !important;
    }

    /* ── Metric Cards ─────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #111b2e;
        border: 1px solid #1c2d45;
        border-radius: 10px;
        padding: 14px 18px;
        transition: border-color 0.3s;
    }
    [data-testid="stMetric"]:hover {
        border-color: #00e5ff44;
    }
    [data-testid="stMetricValue"] {
        color: #00e5ff !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] {
        color: #5a7a96 !important;
        text-transform: uppercase;
        font-weight: 600 !important;
        letter-spacing: 0.8px;
    }

    /* ── Header Gradient ──────────────────────────────── */
    .dashboard-title {
        background: linear-gradient(135deg, #00e5ff, #00b0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0;
    }
    .dashboard-tagline {
        text-align: center;
        color: #7b8fa8;
        font-size: 0.9rem;
        letter-spacing: 0.5px;
        margin-top: 2px;
        margin-bottom: 12px;
    }

    /* ── Shadow-Mode Advisory Banner ──────────────────── */
    .shadow-advisory-banner {
        background: #0c1c2e;
        border: 1px solid #19466e;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        color: #7dd3fc;
        font-size: 0.92rem;
        line-height: 1.4;
    }

    /* ── Status Badge Styles ──────────────────────────── */
    .badge-safe       { color: #00e676; font-weight: 700; }
    .badge-candidate  { color: #38bdf8; font-weight: 700; }
    .badge-danger     { color: #ff4060; font-weight: 700; }
    .badge-malware    { color: #d500f9; font-weight: 700; }
    .badge-warn       { color: #ffab00; font-weight: 700; }
    .badge-neutral    { color: #7b8fa8; font-weight: 600; }

    /* ── Feature Cards ────────────────────────────────── */
    .feature-card {
        background: #111b2e;
        border: 1px solid #1c2d45;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
    }
    .feature-name {
        color: #00e5ff;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .feature-explanation {
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 4px;
        margin-bottom: 0;
    }

    /* ── Sub-Model Detail Box ─────────────────────────── */
    .submodel-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .submodel-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #8be9fd;
        margin-bottom: 4px;
    }

    /* ── Dataframe Styling ────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1c2d45;
        border-radius: 8px;
    }

    /* ── Divider ──────────────────────────────────────── */
    hr {
        border-color: #1c2d45 !important;
    }

    /* ── Sidebar Inputs ───────────────────────────────── */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #111b2e !important;
        border: 1px solid #1c2d45 !important;
        color: #e0e6f0 !important;
    }
</style>
"""


# ── Detailed Result Card Renderer ───────────────────────────────────

def _render_latest_analysis_card(data: dict[str, Any]) -> None:
    """Render the detailed result card for a newly analyzed URL."""
    url = data.get("url", "")
    pred = data.get("prediction")
    cat = str(data.get("threat_category") or "").strip().lower()
    routing = str(data.get("model_routing") or "").strip().lower()
    status = data.get("analysis_status", "completed")
    risk_score = data.get("risk_score")

    # Determine visual state and colors
    if status == "model_unavailable" or (pred is None and not cat):
        status_title = "⏳ Model Unavailable"
        header_color = "#ffab00"
        summary_msg = "Model could not be loaded. No prediction or probabilities were generated."
    elif pred in (2, "2") or cat in ("malware_distribution", "malware", "2") or routing == "url_malware_preempted":
        status_title = "☣️ Malware Distribution Detected"
        header_color = "#d500f9"
        summary_msg = (
            "CRITICAL: Direct IP hosting, executable payloads, or C2 patterns detected by URL classifier. "
            "Multimodal classification preempted for safety."
        )
    elif pred in (1, "1") or cat in ("phishing", "1"):
        status_title = "🚨 Phishing Detected"
        header_color = "#ff4060"
        summary_msg = "HIGH RISK: Deceptive domain, credential harvesting indicators, or suspicious structure detected."
    elif cat in ("unverified_candidate", "candidate") or (pred in (0, "0") and routing == "multimodal_full"):
        status_title = "🔍 Unverified Candidate Domain"
        header_color = "#38bdf8"
        summary_msg = (
            "CANDIDATE: Low phishing indicators detected by multimodal AI. "
            "Note: This domain is an unverified candidate and has NOT been independently verified as safe."
        )
    else:
        status_title = "✅ Legitimate URL (Verified Safe)"
        header_color = "#00e676"
        summary_msg = "SAFE: Standard domain, reputable structure, HTTPS enabled, verified against certified benign baseline."

    st.markdown(
        f"""
        <div style="background:#111b2e; border:1px solid #1c2d45; border-left:5px solid {header_color}; border-radius:8px; padding:16px; margin-top:10px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.15rem; font-weight:700; color:{header_color};">{status_title}</span>
                <span style="font-size:0.9rem; color:#7b8fa8;">Risk Score: <strong style="color:{header_color}; font-size:1.05rem;">{_risk_display(risk_score, status)}</strong></span>
            </div>
            <p style="color:#e0e6f0; font-family:monospace; margin-top:8px; margin-bottom:4px; word-break:break-all;"><strong>URL:</strong> {url}</p>
            <p style="color:#94a3b8; font-size:0.88rem; margin-bottom:4px;">{summary_msg}</p>
            <div style="display:flex; gap:16px; font-size:0.82rem; color:#64748b; margin-top:6px;">
                <span>Routing: <strong style="color:#8be9fd;">{_format_routing_label(data.get('model_routing'))}</strong></span>
                <span>Visual Status: <strong style="color:#8be9fd;">{_format_visual_status_label(data.get('visual_status'))}</strong></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probabilities
    probs = data.get("class_probabilities") or data.get("probabilities")
    if probs and isinstance(probs, dict):
        is_candidate_eval = (
            cat in ("unverified_candidate", "candidate")
            or "unverified_candidate" in probs
            or routing == "multimodal_full"
        )
        legit_label = "🔍 Candidate" if is_candidate_eval else "✅ Legitimate"
        legit_val = float(probs.get("unverified_candidate") if is_candidate_eval and "unverified_candidate" in probs else probs.get("legitimate", 0.0))
        phish_val = float(probs.get("phishing", 0.0))
        malware_val = float(probs.get("malware_distribution", probs.get("malware", 0.0)))

        st.markdown("##### 📈 Class Probability Breakdown")
        pcol1, pcol2, pcol3 = st.columns(3)
        with pcol1:
            st.metric(legit_label, f"{legit_val:.1%}")
            st.progress(min(max(legit_val, 0.0), 1.0))
        with pcol2:
            st.metric("🚨 Phishing", f"{phish_val:.1%}")
            st.progress(min(max(phish_val, 0.0), 1.0))
        with pcol3:
            st.metric("☣️ Malware Distribution", f"{malware_val:.1%}")
            st.progress(min(max(malware_val, 0.0), 1.0))

    # SHAP Top Features & Explanation
    top_feats = data.get("top_features")
    expl_data = data.get("explanation")
    if isinstance(expl_data, dict) and not top_feats:
        top_feats = expl_data.get("top_features")

    if top_feats and isinstance(top_feats, list):
        st.markdown("##### 🧬 Top Contributing Indicators (TreeSHAP Attributions)")
        for feat in top_feats:
            fn = feat.get("feature_name", "feature")
            fv = feat.get("feature_value", "")
            contrib = float(feat.get("contribution", 0.0))
            impact = str(feat.get("impact", "")).lower()
            explanation = feat.get("explanation", "")

            sign = "+" if contrib >= 0 else ""
            if cat in ("unverified_candidate", "candidate"):
                if contrib >= 0:
                    polarity_badge = "🔵 Candidate Indicator"
                    badge_style = "color:#38bdf8;"
                else:
                    polarity_badge = "🔴 Phishing Risk Factor"
                    badge_style = "color:#ff5252;"
            elif "safe" in impact or "legitimate" in impact or (contrib < 0 and pred == 0):
                polarity_badge = "🟢 Safe / Mitigating"
                badge_style = "color:#00e676;"
            elif "malware" in impact or pred == 2:
                polarity_badge = "🟣 Malware Factor"
                badge_style = "color:#d500f9;"
            else:
                polarity_badge = "🔴 Phishing Risk Factor"
                badge_style = "color:#ff5252;"

            st.markdown(
                f"""
                <div class="feature-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="feature-name">{fn} = {fv}</span>
                        <span style="font-size:0.85rem; font-weight:600; {badge_style}">
                            {polarity_badge} ({sign}{contrib:.3f})
                        </span>
                    </div>
                    <p class="feature-explanation">{explanation}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    elif status == "completed":
        st.caption("ℹ️ *No SHAP explanation features returned for this analysis.*")

    if st.button("✕ Dismiss Result", key="dismiss_latest_analysis_btn"):
        st.session_state["latest_analysis"] = None
        st.rerun()


# ── Dashboard Layout ────────────────────────────────────────────────

def main() -> None:
    """Render the PhishGuard Analytics Dashboard."""

    # Page config
    st.set_page_config(
        page_title="PhishGuard Dashboard",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # ── Sidebar ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        backend_url = st.text_input(
            "Backend URL",
            value=BACKEND_URL,
            help="PhishGuard FastAPI backend URL",
        )
        fetch_limit = st.number_input(
            "Max records to fetch",
            min_value=10,
            max_value=5000,
            value=DEFAULT_LIMIT,
            step=50,
        )

        st.markdown("---")
        st.markdown("### 🔍 Filters & Search")
        search_query = st.text_input(
            "Search URLs",
            placeholder="Enter URL or keyword...",
            help="Filter analysis history by URL substring",
        )

        category_filter = st.selectbox(
            "Threat Category",
            options=["All", "Legitimate", "Unverified Candidate", "Phishing", "Malware Distribution", "Unavailable"],
            index=0,
            help="Filter by classified threat category (4-class system)",
        )

        routing_filter = st.selectbox(
            "Model Routing",
            options=["All", "multimodal_full", "url_model_fallback", "url_malware_preempted", "model_unavailable"],
            index=0,
            help="Filter by decision pipeline routing",
        )

        status_filter = st.selectbox(
            "Analysis Status",
            options=["All", "completed", "model_unavailable"],
            index=0,
            help="Filter by execution status",
        )

        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    # ── Header ─────────────────────────────────────────────────
    st.markdown('<h1 class="dashboard-title">🛡️ PhishGuard Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-tagline">Multimodal AI Threat Detection & Explainability Analytics</p>', unsafe_allow_html=True)

    # ── PART 6: Persistent Shadow Mode Advisory Banner ─────────
    st.markdown(
        """
        <div class="shadow-advisory-banner">
            <span style="font-size:1.3rem;">🛡️</span>
            <span><strong>PhishGuard is operating in shadow mode.</strong> Predictions are informational and do not block browser navigation.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 1. Dedicated "Analyze New URL" Section ─────────────────
    st.markdown("### 🌐 Analyze New URL")
    st.markdown(
        '<p style="color:#7b8fa8; font-size:0.88rem; margin-top:-6px; margin-bottom:12px;">'
        'Scan any live HTTP or HTTPS webpage to inspect multi-model threat classifications, '
        'three-model probabilities, and TreeSHAP indicators.'
        '</p>',
        unsafe_allow_html=True,
    )

    with st.container():
        input_col, btn_col = st.columns([5, 1])
        with input_col:
            new_url_input = st.text_input(
                "Enter URL to analyze:",
                placeholder="https://example.com/login",
                key="analyze_new_url_input",
                label_visibility="collapsed",
            )
        with btn_col:
            analyze_clicked = st.button("🚀 Analyze URL", type="primary", use_container_width=True)

        if analyze_clicked:
            if not new_url_input or not new_url_input.strip():
                st.warning("⚠️ Please enter a URL to analyze.")
            else:
                with st.spinner("Extracting features, arbitrating models, and generating SHAP..."):
                    result_data, err_msg = analyze_new_url(
                        url=new_url_input,
                        backend_url=backend_url,
                    )
                if err_msg:
                    st.error(f"❌ {err_msg}")
                elif result_data:
                    st.session_state["latest_analysis"] = result_data
                    st.success("✅ Analysis completed and saved to database!")
                    st.rerun()

    # Display Latest Analysis Result Card if present
    latest = st.session_state.get("latest_analysis")
    if latest:
        _render_latest_analysis_card(latest)

    st.markdown("---")

    # ── Fetch Data ─────────────────────────────────────────────
    records = fetch_analyses(backend_url=backend_url, limit=int(fetch_limit))

    if not records:
        st.info("🛡️ No analysis records found. Enter a URL above and click **Analyze URL** to populate data.")
        return

    # ── Apply Filters ──────────────────────────────────────────
    filtered_records = filter_records(
        records,
        query=search_query,
        category=category_filter,
        status=status_filter,
        routing=routing_filter,
    )

    # ── PART 2: Dashboard Overview & Statistics Cards ──────────
    stats = compute_statistics(records)

    st.markdown("### 📊 Operational Overview")

    # Row 1: Four-Class Classification Counts
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Analyzed", value=stats["total"])
    with col2:
        st.metric(label="✅ Legitimate", value=stats["legitimate"], help="Verified safe domains from certified baseline.")
    with col3:
        st.metric(label="🔍 Unverified Candidate", value=stats["unverified_candidate"], help="Candidates with low phishing signs; NOT verified safe.")
    with col4:
        st.metric(label="🚨 Phishing", value=stats["phishing"], help="High-confidence phishing threats.")
    with col5:
        st.metric(label="☣️ Malware", value=stats["malware"], help="Malware distribution or C2 hosts.")

    # Row 2: Operational & Pipeline Metrics
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        st.metric(label="🛡️ Total Threats", value=stats["threat_total"], help="Total malicious records (Phishing + Malware).")
    with pcol2:
        st.metric(label="📈 Avg Risk Score", value=f"{stats['avg_risk_score']:.1%}", help="Average raw model risk score across analyses.")
    with pcol3:
        st.metric(label="📸 Screenshot Rate", value=f"{stats['screenshot_success_rate']:.1f}%", help="Percentage of analyses with successful screenshot extraction.")
    with pcol4:
        st.metric(
            label="🔀 Multimodal vs Fallback",
            value=f"{stats['multimodal_count']} / {stats['url_fallback_count']}",
            help="Multimodal Full analyses vs URL Fallback analyses.",
        )

    # Identifying Experimental Models Notice
    st.caption(
        "ℹ️ *Experimental Models Notice: Multimodal fusion (51 canonical features) and visual screenshot feature extraction "
        "are experimental shadow-mode capabilities. Unverified candidate domains have NOT been independently verified as safe.*"
    )
    if stats["unavailable"] > 0:
        st.caption(f"⏳ *{stats['unavailable']} record(s) recorded under model_unavailable status.*")

    st.markdown("---")

    # ── PART 3: Visual Analytics (6 Charts) ─────────────────────
    st.markdown("### 📈 Threat Distribution & Multimodal Pipeline Analytics")

    # Row 1: Four-Class Distribution & Risk Score Histogram
    chart_row1_col1, chart_row1_col2 = st.columns(2)
    with chart_row1_col1:
        st.markdown("##### 1. Four-Class Threat Distribution")
        dist_df = prepare_four_class_distribution(records)
        if dist_df["Count"].sum() > 0:
            st.bar_chart(dist_df.set_index("Category"), color="#00e5ff")
        else:
            st.info("No categorized data available for chart.")

    with chart_row1_col2:
        st.markdown("##### 2. Risk Score Tier Distribution")
        risk_df = prepare_risk_score_distribution(records)
        if risk_df["Count"].sum() > 0:
            st.bar_chart(risk_df.set_index("Risk Tier"), color="#ff4060")
        else:
            st.info("No risk score data available for chart.")

    # Row 2: Daily Activity & Model Routing Distribution
    chart_row2_col1, chart_row2_col2 = st.columns(2)
    with chart_row2_col1:
        st.markdown("##### 3. Daily Analysis Activity Timeline")
        timeline_df = prepare_daily_activity(records)
        if not timeline_df.empty:
            st.line_chart(timeline_df.set_index("Date"), color="#00e5ff")
        else:
            st.info("No timeline activity recorded yet.")

    with chart_row2_col2:
        st.markdown("##### 4. Model Routing Distribution")
        routing_df = prepare_model_routing_distribution(records)
        if routing_df["Count"].sum() > 0:
            st.bar_chart(routing_df.set_index("Routing Path"), color="#38bdf8")
        else:
            st.info("No routing data available.")

    # Row 3: Screenshot Availability & Model Disagreement Breakdown
    chart_row3_col1, chart_row3_col2 = st.columns(2)
    with chart_row3_col1:
        st.markdown("##### 5. Screenshot Availability Breakdown")
        vis_df = prepare_screenshot_availability(records)
        if vis_df["Count"].sum() > 0:
            st.bar_chart(vis_df.set_index("Screenshot Status"), color="#a78bfa")
        else:
            st.info("No screenshot telemetry available.")

    with chart_row3_col2:
        st.markdown("##### 6. Sub-Model Disagreement Statistics")
        disagree_stats = prepare_model_disagreement_statistics(records)
        dcol1, dcol2, dcol3 = st.columns(3)
        with dcol1:
            st.metric("Multi Evaluated", disagree_stats["total_multi_analyses"])
        with dcol2:
            st.metric("Agreement Rate", f"{disagree_stats['agreement_rate']:.1f}%")
        with dcol3:
            st.metric("Disagreements", disagree_stats["disagreed_count"])

        if disagree_stats["disagreed_count"] > 0:
            st.bar_chart(disagree_stats["breakdown_df"].set_index("Disagreement Type"), color="#f59e0b")
        else:
            st.caption("✅ *Zero sub-model disagreements observed across multimodal records.*")

    st.markdown("---")

    # ── PART 5: Detailed Analysis Inspection Station ────────────
    st.markdown("### 🔎 Detailed Analysis Inspection")
    if filtered_records:
        options = {
            f"ID #{r.get('id')} - {r.get('url')} [{_prediction_label(r.get('prediction'), r.get('analysis_status', ''), r.get('threat_category'), r.get('model_routing'))}]": r.get("id")
            for r in filtered_records
        }
        selected_label = st.selectbox(
            "Select an analysis record to inspect multi-model telemetry and SHAP attributions:",
            options=list(options.keys()),
            index=0,
        )
        selected_id = options.get(selected_label)
        selected_rec = next((r for r in filtered_records if r.get("id") == selected_id), None)

        if selected_rec:
            # Main record summary cards
            dcol1, dcol2, dcol3, dcol4 = st.columns([3, 2, 2, 2])
            with dcol1:
                st.markdown(f"**URL:** `{selected_rec.get('url')}`")
                st.caption(f"Scanned: {_format_timestamp(selected_rec.get('analyzed_at'))}")
            with dcol2:
                cat_text = _threat_category_label(
                    selected_rec.get("threat_category"),
                    selected_rec.get("prediction"),
                    selected_rec.get("analysis_status"),
                    selected_rec.get("model_routing"),
                )
                st.metric("Threat Category", cat_text)
            with dcol3:
                st.metric("Risk Score", _risk_display(selected_rec.get("risk_score"), selected_rec.get("analysis_status")))
            with dcol4:
                st.metric("Model Routing", _format_routing_label(selected_rec.get("model_routing")))

            # Second row info badges
            rcol1, rcol2, rcol3 = st.columns(3)
            with rcol1:
                st.markdown(f"**Screenshot Status:** `{_format_visual_status_label(selected_rec.get('visual_status'))}`")
            with rcol2:
                st.markdown(f"**Execution Status:** `{selected_rec.get('analysis_status', 'completed')}`")
            with rcol3:
                st.markdown("**Shadow Mode:** `Active 🛡️`")

            # Fallback or preemption advisory
            fallback_reason = selected_rec.get("fallback_reason")
            if fallback_reason:
                st.info(f"ℹ️ **Routing Reason / Fallback:** {fallback_reason}")

            if selected_rec.get("model_routing") == "url_malware_preempted":
                st.warning("☣️ **Malware Safeguard:** URL model detected malware distribution. Multimodal classification preempted.")

            # Multi-Model Architecture Breakdown
            st.markdown("##### 🔬 Multi-Model Architecture Telemetry")

            # Extract sub-model results
            telem = (
                (selected_rec.get("explanation") or {}).get("multi_model_telemetry")
                if isinstance(selected_rec.get("explanation"), dict)
                else {}
            )
            url_res = selected_rec.get("url_model_result") or telem.get("url_model_result")
            vis_res = selected_rec.get("visual_model_result") or telem.get("visual_model_result")
            multi_res = selected_rec.get("multimodal_model_result") or telem.get("multimodal_model_result")

            mcol1, mcol2, mcol3 = st.columns(3)

            # Sub-Model 1: URL Model (Tri-Class)
            with mcol1:
                st.markdown(
                    """
                    <div class="submodel-card">
                        <div class="submodel-title">🌐 Standalone URL Model</div>
                    """,
                    unsafe_allow_html=True,
                )
                if url_res and isinstance(url_res, dict):
                    st.caption(f"Status: **{url_res.get('status', 'completed').title()}** (20 Feat)")
                    url_pred_label = _threat_category_label(url_res.get("threat_category"), url_res.get("prediction"))
                    st.markdown(f"Prediction: **{url_pred_label}**")
                    uprobs = url_res.get("probabilities") or {}
                    if uprobs:
                        p_leg = float(uprobs.get("legitimate", 0.0))
                        p_phi = float(uprobs.get("phishing", 0.0))
                        p_mal = float(uprobs.get("malware_distribution", 0.0))
                        st.markdown(f"- Legit: `{p_leg:.1%}`")
                        st.markdown(f"- Phish: `{p_phi:.1%}`")
                        st.markdown(f"- Malware: `{p_mal:.1%}`")
                elif selected_rec.get("class_probabilities"):
                    # Historical fallback
                    cprobs = selected_rec.get("class_probabilities", {})
                    st.caption("Status: **Evaluated (Historical)**")
                    st.markdown(f"- Legit: `{float(cprobs.get('legitimate', 0.0)):.1%}`")
                    st.markdown(f"- Phish: `{float(cprobs.get('phishing', 0.0)):.1%}`")
                    st.markdown(f"- Malware: `{float(cprobs.get('malware_distribution', 0.0)):.1%}`")
                else:
                    st.caption("Status: *Telemetry unavailable*")
                st.markdown("</div>", unsafe_allow_html=True)

            # Sub-Model 2: Visual Screenshot Model (Auxiliary)
            with mcol2:
                st.markdown(
                    """
                    <div class="submodel-card">
                        <div class="submodel-title">📸 Visual Screenshot Model</div>
                    """,
                    unsafe_allow_html=True,
                )
                if vis_res and isinstance(vis_res, dict):
                    v_status = vis_res.get("status", "skipped").title()
                    st.caption(f"Status: **{v_status}** (30 Feat)")
                    if vis_res.get("status") == "completed":
                        v_pred_label = _threat_category_label(vis_res.get("threat_category"), vis_res.get("prediction"))
                        st.markdown(f"Prediction: **{v_pred_label}**")
                        vprobs = vis_res.get("probabilities") or {}
                        if vprobs:
                            p_cand = float(vprobs.get("unverified_candidate", 0.0))
                            p_phi = float(vprobs.get("phishing", 0.0))
                            st.markdown(f"- Candidate: `{p_cand:.1%}`")
                            st.markdown(f"- Phish: `{p_phi:.1%}`")
                    else:
                        st.caption(f"Reason: *{vis_res.get('reason', 'Skipped')}*")
                else:
                    st.caption("Status: *Skipped (Screenshot not provided)*")
                st.markdown("</div>", unsafe_allow_html=True)

            # Sub-Model 3: Multimodal Model (Primary Arbiter)
            with mcol3:
                st.markdown(
                    """
                    <div class="submodel-card">
                        <div class="submodel-title">🧠 Multimodal Fusion Model</div>
                    """,
                    unsafe_allow_html=True,
                )
                if multi_res and isinstance(multi_res, dict):
                    m_status = multi_res.get("status", "completed").title()
                    st.caption(f"Status: **{m_status}** (51 Feat)")
                    if multi_res.get("status") == "completed":
                        m_pred_label = _threat_category_label(multi_res.get("threat_category"), multi_res.get("prediction"), routing="multimodal_full")
                        st.markdown(f"Prediction: **{m_pred_label}**")
                        mprobs = multi_res.get("probabilities") or {}
                        if mprobs:
                            p_cand = float(mprobs.get("unverified_candidate", 0.0))
                            p_phi = float(mprobs.get("phishing", 0.0))
                            st.markdown(f"- Candidate: `{p_cand:.1%}`")
                            st.markdown(f"- Phish: `{p_phi:.1%}`")
                    elif multi_res.get("status") == "overridden":
                        st.markdown("Prediction: **Preempted by Malware Guard**")
                        st.caption(f"Reason: *{multi_res.get('reason', '')}*")
                    else:
                        st.caption(f"Reason: *{multi_res.get('reason', 'Skipped')}*")
                else:
                    st.caption("Status: *Skipped (URL fallback path)*")
                st.markdown("</div>", unsafe_allow_html=True)

            # Expandable Screenshot Visual Features (30 Feat)
            vis_features = selected_rec.get("visual_features")
            if vis_features and isinstance(vis_features, dict):
                with st.expander("🖼️ View Extracted Screenshot Features (30 Features)"):
                    vf_cols = st.columns(4)
                    with vf_cols[0]:
                        st.metric("Width", vis_features.get("width", "—"))
                        st.metric("Aspect Ratio", vis_features.get("aspect_ratio", "—"))
                    with vf_cols[1]:
                        st.metric("Height", vis_features.get("height", "—"))
                        st.metric("Mean Brightness", vis_features.get("mean_brightness", "—"))
                    with vf_cols[2]:
                        st.metric("Image Entropy", vis_features.get("image_entropy", "—"))
                        st.metric("Contrast (Std)", vis_features.get("std_brightness", "—"))
                    with vf_cols[3]:
                        st.metric("Edge Density", vis_features.get("edge_density", "—"))
                        st.metric("Dominant Hue", vis_features.get("dominant_color_h", "—"))

            # SHAP Top Features & Explanation
            top_feats = selected_rec.get("top_features")
            expl_data = selected_rec.get("explanation")
            if isinstance(expl_data, dict) and not top_feats:
                top_feats = expl_data.get("top_features")

            if top_feats and isinstance(top_feats, list):
                st.markdown("##### 🧬 Top Contributing Indicators (TreeSHAP Attributions)")
                for feat in top_feats:
                    fn = feat.get("feature_name", "feature")
                    fv = feat.get("feature_value", "")
                    contrib = float(feat.get("contribution", 0.0))
                    impact = str(feat.get("impact", "")).lower()
                    explanation = feat.get("explanation", "")

                    sign = "+" if contrib >= 0 else ""
                    current_cat = str(selected_rec.get("threat_category") or "").lower()
                    current_pred = selected_rec.get("prediction")

                    if current_cat in ("unverified_candidate", "candidate"):
                        if contrib >= 0:
                            polarity_badge = "🔵 Candidate Indicator"
                            badge_style = "color:#38bdf8;"
                        else:
                            polarity_badge = "🔴 Phishing Risk Factor"
                            badge_style = "color:#ff5252;"
                    elif "safe" in impact or "legitimate" in impact or (contrib < 0 and current_pred == 0):
                        polarity_badge = "🟢 Safe / Mitigating"
                        badge_style = "color:#00e676;"
                    elif "malware" in impact or current_pred == 2:
                        polarity_badge = "🟣 Malware Factor"
                        badge_style = "color:#d500f9;"
                    else:
                        polarity_badge = "🔴 Phishing Risk Factor"
                        badge_style = "color:#ff5252;"

                    st.markdown(
                        f"""
                        <div class="feature-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="feature-name">{fn} = {fv}</span>
                                <span style="font-size:0.85rem; font-weight:600; {badge_style}">
                                    {polarity_badge} ({sign}{contrib:.3f})
                                </span>
                            </div>
                            <p class="feature-explanation">{explanation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("ℹ️ *No SHAP explanation features recorded for this analysis.*")
    else:
        st.info("No matching records for detailed inspection.")

    st.markdown("---")

    # ── PART 4: Analysis History Table ──────────────────────────
    st.markdown(
        f"### 📋 Recent Analyses"
        + (f"  —  *{len(filtered_records)} of {len(records)} records*"
           if len(filtered_records) != len(records)
           else f"  —  *{len(records)} records*")
    )

    df = records_to_dataframe(filtered_records)
    if not df.empty:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "URL": st.column_config.TextColumn("URL", width="large"),
                "Prediction": st.column_config.TextColumn("Prediction", width="medium"),
                "Threat Category": st.column_config.TextColumn("Threat Category", width="medium"),
                "Risk Score": st.column_config.TextColumn("Risk Score", width="small"),
                "Model Routing": st.column_config.TextColumn("Model Routing", width="medium"),
                "Screenshot Status": st.column_config.TextColumn("Screenshot Status", width="medium"),
                "Shadow Mode": st.column_config.TextColumn("Shadow Mode", width="small"),
                "Status": st.column_config.TextColumn("Status", width="small"),
                "Analyzed At": st.column_config.TextColumn("Analyzed At", width="medium"),
            },
        )
    else:
        st.info("No records matching the selected filters.")

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#3a4f66; font-size:0.8rem;">'
        'PhishGuard Dashboard · Connected to '
        f'<span style="color:#00e5ff">{backend_url}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
