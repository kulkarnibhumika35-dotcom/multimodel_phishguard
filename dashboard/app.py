"""
PhishGuard – Security Analytics Dashboard

A modern Streamlit dashboard that connects to the PhishGuard FastAPI backend
and SQLite database to display analysis history, three-class statistics,
distribution charts, filters, detailed SHAP explainability, and live URL analysis.

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

    try:
        resp = requests.post(
            endpoint,
            json={"url": cleaned_url},
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


# ── Statistics Computation ──────────────────────────────────────────

def compute_statistics(records: list[dict[str, Any]]) -> dict[str, int]:
    """Compute three-class dashboard statistics from analysis records.

    Categories:
    - 0 = Legitimate
    - 1 = Phishing
    - 2 = Malware Distribution
    - model_unavailable / null = Unavailable

    IMPORTANT: model_unavailable records are NEVER counted as legitimate,
    phishing, or malware. Only completed records with valid prediction values
    are counted.
    """
    total = len(records)
    phishing = 0
    legitimate = 0
    malware = 0
    unavailable = 0

    for rec in records:
        status = rec.get("analysis_status", "")
        prediction = rec.get("prediction")
        threat_cat = str(rec.get("threat_category") or "").strip().lower()

        if status == "model_unavailable" or (prediction is None and not threat_cat):
            unavailable += 1
        elif prediction == 2 or prediction == "2" or threat_cat in ("malware_distribution", "malware", "2"):
            malware += 1
        elif prediction == 1 or prediction == "1" or threat_cat in ("phishing", "1"):
            phishing += 1
        elif prediction == 0 or prediction == "0" or threat_cat in ("legitimate", "0"):
            legitimate += 1
        else:
            # Unknown prediction value – treat as unavailable
            unavailable += 1

    return {
        "total": total,
        "phishing": phishing,
        "legitimate": legitimate,
        "malware": malware,
        "unavailable": unavailable,
        "threat_total": phishing + malware,
    }


# ── Filtering Helper ───────────────────────────────────────────────

def _format_category(rec: dict[str, Any]) -> str:
    """Map a record to its canonical category name."""
    status = rec.get("analysis_status", "")
    pred = rec.get("prediction")
    cat = str(rec.get("threat_category") or "").strip().lower()

    if status == "model_unavailable" or (pred is None and not cat):
        return "Unavailable"
    if pred == 2 or pred == "2" or cat in ("malware_distribution", "malware", "2"):
        return "Malware Distribution"
    if pred == 1 or pred == "1" or cat in ("phishing", "1"):
        return "Phishing"
    if pred == 0 or pred == "0" or cat in ("legitimate", "0"):
        return "Legitimate"
    return "Unknown"


def filter_records(
    records: list[dict[str, Any]],
    query: str = "",
    category: str = "All",
    status: str = "All",
) -> list[dict[str, Any]]:
    """Filter records by URL substring, threat category, and analysis status."""
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
        ]

    # 3. Status filter
    if status and status != "All":
        st_lower = status.strip().lower()
        filtered = [r for r in filtered if (r.get("analysis_status") or "").lower() == st_lower]

    return filtered


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
        margin-bottom: 20px;
    }

    /* ── Status Badge Styles ──────────────────────────── */
    .badge-safe       { color: #00e676; font-weight: 700; }
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


# ── Helper: Format Records for Display ──────────────────────────────

def _prediction_label(prediction: Any, status: str = "", threat_category: str | None = None) -> str:
    """Convert raw prediction value to a display label.

    Supports three classes:
    - 0: Legitimate
    - 1: Phishing
    - 2: Malware Distribution
    - model_unavailable / None: Unavailable
    """
    cat = str(threat_category or "").strip().lower()

    if status == "model_unavailable" or (prediction is None and not cat):
        return "⏳ Unavailable"
    if prediction == 2 or prediction == "2" or cat in ("malware_distribution", "malware", "2"):
        return "☣️ Malware Distribution"
    if prediction == 1 or prediction == "1" or cat in ("phishing", "1"):
        return "🚨 Phishing"
    if prediction == 0 or prediction == "0" or cat in ("legitimate", "0"):
        return "✅ Legitimate"
    return "❓ Unknown"


def _threat_category_label(threat_category: Any, prediction: Any = None, status: str = "") -> str:
    """Return clean human-readable threat category."""
    cat = str(threat_category or "").strip().lower()
    if status == "model_unavailable" or (prediction is None and not cat):
        return "Unavailable"
    if cat in ("malware_distribution", "malware", "2") or prediction in (2, "2"):
        return "Malware Distribution"
    if cat in ("phishing", "1") or prediction in (1, "1"):
        return "Phishing"
    if cat in ("legitimate", "0") or prediction in (0, "0"):
        return "Legitimate"
    return "Unknown"


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
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return str(ts)


def records_to_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert API records to a display-friendly DataFrame with 3-class details."""
    if not records:
        return pd.DataFrame(columns=[
            "ID",
            "URL",
            "Prediction",
            "Threat Category",
            "Risk Score",
            "Status",
            "Analyzed At",
        ])

    rows = []
    for rec in records:
        status = rec.get("analysis_status", "")
        pred = rec.get("prediction")
        cat = rec.get("threat_category")
        rows.append({
            "ID": rec.get("id", ""),
            "URL": rec.get("url", ""),
            "Prediction": _prediction_label(pred, status, cat),
            "Threat Category": _threat_category_label(cat, pred, status),
            "Risk Score": _risk_display(rec.get("risk_score"), status),
            "Status": status,
            "Analyzed At": _format_timestamp(rec.get("analyzed_at")),
        })
    return pd.DataFrame(rows)


# ── Detailed Result Card Renderer ───────────────────────────────────

def _render_latest_analysis_card(data: dict[str, Any]) -> None:
    """Render the detailed result card for a newly analyzed URL."""
    url = data.get("url", "")
    pred = data.get("prediction")
    cat = str(data.get("threat_category") or "").strip().lower()
    status = data.get("analysis_status", "completed")
    risk_score = data.get("risk_score")

    # Determine visual state and colors
    if status == "model_unavailable" or (pred is None and not cat):
        status_title = "⏳ Model Unavailable"
        header_color = "#ffab00"
        summary_msg = "Model could not be loaded. No prediction or probabilities were generated."
    elif pred == 2 or pred == "2" or cat in ("malware_distribution", "malware", "2"):
        status_title = "☣️ Malware Distribution Detected"
        header_color = "#d500f9"
        summary_msg = "CRITICAL: Direct IP hosting, executable payloads, or C2 patterns detected. Avoid downloading files."
    elif pred == 1 or pred == "1" or cat in ("phishing", "1"):
        status_title = "🚨 Phishing Detected"
        header_color = "#ff4060"
        summary_msg = "HIGH RISK: Deceptive domain, credential harvesting keywords, or suspicious structure detected."
    else:
        status_title = "✅ Legitimate URL (Safe)"
        header_color = "#00e676"
        summary_msg = "SAFE: Standard domain, reputable structure, HTTPS enabled, and low threat indicators."

    st.markdown(
        f"""
        <div style="background:#111b2e; border:1px solid #1c2d45; border-left:5px solid {header_color}; border-radius:8px; padding:16px; margin-top:10px; margin-bottom:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.15rem; font-weight:700; color:{header_color};">{status_title}</span>
                <span style="font-size:0.9rem; color:#7b8fa8;">Risk Score: <strong style="color:{header_color}; font-size:1.05rem;">{_risk_display(risk_score, status)}</strong></span>
            </div>
            <p style="color:#e0e6f0; font-family:monospace; margin-top:8px; margin-bottom:4px; word-break:break-all;"><strong>URL:</strong> {url}</p>
            <p style="color:#94a3b8; font-size:0.88rem; margin-bottom:0;">{summary_msg}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probabilities
    probs = data.get("class_probabilities") or data.get("probabilities")
    if probs and isinstance(probs, dict):
        st.markdown("##### 📈 Three-Class Probability Distribution")
        pcol1, pcol2, pcol3 = st.columns(3)
        legit_p = float(probs.get("legitimate", 0.0))
        phish_p = float(probs.get("phishing", 0.0))
        malware_p = float(probs.get("malware_distribution", 0.0))

        with pcol1:
            st.metric("✅ Legitimate", f"{legit_p:.1%}")
            st.progress(min(max(legit_p, 0.0), 1.0))
        with pcol2:
            st.metric("🚨 Phishing", f"{phish_p:.1%}")
            st.progress(min(max(phish_p, 0.0), 1.0))
        with pcol3:
            st.metric("☣️ Malware Distribution", f"{malware_p:.1%}")
            st.progress(min(max(malware_p, 0.0), 1.0))

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
            if contrib < 0 or "safe" in impact or "legitimate" in impact or "decreases" in impact:
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
            options=["All", "Legitimate", "Phishing", "Malware Distribution", "Unavailable"],
            index=0,
            help="Filter by classified threat category",
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
    st.markdown('<p class="dashboard-tagline">AI-Powered Threat Detection & Explainability Analytics</p>', unsafe_allow_html=True)

    # ── 1. Dedicated "Analyze New URL" Section ─────────────────
    st.markdown("### 🌐 Analyze New URL")
    st.markdown(
        '<p style="color:#7b8fa8; font-size:0.88rem; margin-top:-6px; margin-bottom:12px;">'
        'Scan any live HTTP or HTTPS webpage to inspect threat classifications, '
        'three-class probabilities, and TreeSHAP indicators.'
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
                with st.spinner("Extracting 20 features, evaluating model, and computing SHAP..."):
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
    )

    # ── Statistics Cards ───────────────────────────────────────
    stats = compute_statistics(records)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Total Analyzed", value=stats["total"])
    with col2:
        st.metric(label="✅ Legitimate", value=stats["legitimate"])
    with col3:
        st.metric(label="🚨 Phishing", value=stats["phishing"])
    with col4:
        st.metric(label="☣️ Malware", value=stats["malware"])
    with col5:
        st.metric(label="🛡️ Total Threats", value=stats["threat_total"])

    if stats["unavailable"] > 0:
        st.caption(f"⏳ *{stats['unavailable']} record(s) recorded under model_unavailable status.*")

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────
    st.markdown("### 📊 Threat Distribution & Trends")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### Three-Class Threat Distribution")
        chart_data = {
            "Category": ["Legitimate", "Phishing", "Malware Distribution"],
            "Count": [stats["legitimate"], stats["phishing"], stats["malware"]],
        }
        if stats["unavailable"] > 0:
            chart_data["Category"].append("Unavailable")
            chart_data["Count"].append(stats["unavailable"])

        dist_df = pd.DataFrame(chart_data)
        if dist_df["Count"].sum() > 0:
            st.bar_chart(dist_df.set_index("Category"), color="#00e5ff")
        else:
            st.info("No categorised data available for chart.")

    with chart_col2:
        st.markdown("##### Analysis Activity Timeline")
        dates = []
        for rec in records:
            ts = rec.get("analyzed_at", "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    dates.append(dt.date())
                except Exception:
                    pass

        if dates:
            date_counts = pd.Series(dates).value_counts().sort_index()
            timeline_df = pd.DataFrame({
                "Date": date_counts.index,
                "Analyses": date_counts.values,
            }).set_index("Date")
            st.line_chart(timeline_df, color="#00e5ff")
        else:
            st.info("No timeline data available yet.")

    st.markdown("---")

    # ── Detailed Analysis View (Inspection Station) ────────────
    st.markdown("### 🔎 Detailed Analysis Inspection")
    if filtered_records:
        options = {
            f"ID #{r.get('id')} - {r.get('url')} [{_prediction_label(r.get('prediction'), r.get('analysis_status', ''), r.get('threat_category'))}]": r.get("id")
            for r in filtered_records
        }
        selected_label = st.selectbox(
            "Select an analysis record to inspect probabilities and SHAP explanations:",
            options=list(options.keys()),
            index=0,
        )
        selected_id = options.get(selected_label)
        selected_rec = next((r for r in filtered_records if r.get("id") == selected_id), None)

        if selected_rec:
            # Detailed metrics cards
            dcol1, dcol2, dcol3, dcol4 = st.columns([3, 2, 2, 2])
            with dcol1:
                st.markdown(f"**URL:** `{selected_rec.get('url')}`")
                st.caption(f"Scanned: {_format_timestamp(selected_rec.get('analyzed_at'))}")
            with dcol2:
                cat_text = _threat_category_label(selected_rec.get("threat_category"), selected_rec.get("prediction"), selected_rec.get("analysis_status"))
                st.metric("Threat Category", cat_text)
            with dcol3:
                st.metric("Risk Score", _risk_display(selected_rec.get("risk_score"), selected_rec.get("analysis_status")))
            with dcol4:
                st.metric("Status", selected_rec.get("analysis_status", "completed"))

            # Three-Class Probabilities
            probs = selected_rec.get("class_probabilities") or selected_rec.get("probabilities")
            if probs and isinstance(probs, dict):
                st.markdown("##### 📈 Three-Class Probabilities")
                pcol1, pcol2, pcol3 = st.columns(3)
                legit_p = float(probs.get("legitimate", 0.0))
                phish_p = float(probs.get("phishing", 0.0))
                malware_p = float(probs.get("malware_distribution", 0.0))

                with pcol1:
                    st.metric("✅ Legitimate", f"{legit_p:.1%}")
                    st.progress(min(max(legit_p, 0.0), 1.0))
                with pcol2:
                    st.metric("🚨 Phishing", f"{phish_p:.1%}")
                    st.progress(min(max(phish_p, 0.0), 1.0))
                with pcol3:
                    st.metric("☣️ Malware Distribution", f"{malware_p:.1%}")
                    st.progress(min(max(malware_p, 0.0), 1.0))
            else:
                st.caption("ℹ️ *No probability distribution recorded for this analysis.*")

            # SHAP Top Features & Explanation
            top_feats = selected_rec.get("top_features")
            expl_data = selected_rec.get("explanation")
            if isinstance(expl_data, dict) and not top_feats:
                top_feats = expl_data.get("top_features")

            if top_feats and isinstance(top_feats, list):
                st.markdown("##### 🧬 Top Contributing Indicators (SHAP Attributions)")
                for feat in top_feats:
                    fn = feat.get("feature_name", "feature")
                    fv = feat.get("feature_value", "")
                    contrib = float(feat.get("contribution", 0.0))
                    impact = feat.get("impact", "")
                    explanation = feat.get("explanation", "")

                    sign = "+" if contrib >= 0 else ""
                    polarity_badge = "🟢 Safe / Mitigating" if (contrib < 0 or "safe" in impact or "legitimate" in impact) else "🔴 Risk Factor"

                    st.markdown(
                        f"""
                        <div class="feature-card">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span class="feature-name">{fn} = {fv}</span>
                                <span style="font-size:0.85rem; font-weight:600;">
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

    # ── Analysis History Table ──────────────────────────────────
    st.markdown(
        f"### 📋 Analysis History"
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
                "Status": st.column_config.TextColumn("Status", width="medium"),
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
