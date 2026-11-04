"""
PhishGuard – URL Feature Extraction

Extracts 20 numerical features from a URL for phishing detection.
Uses only Python standard libraries. No network requests are made.
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse

# ── Constants ───────────────────────────────────────────────────

SUSPICIOUS_TLDS = frozenset({
    "zip", "top", "xyz", "buzz", "tk", "ml", "ga", "cf", "gq",
    "work", "click", "link", "surf", "rest", "fit", "cam",
})

SUSPICIOUS_KEYWORDS = frozenset({
    "login", "signin", "verify", "account", "update", "secure",
    "banking", "confirm", "password", "credential", "suspend",
    "paypal", "ebay", "apple", "microsoft", "netflix",
    "free", "lucky", "winner", "prize", "reward",
})

_IP_RE = re.compile(
    r"(?:^|\.)(?:\d{1,3}\.){3}\d{1,3}(?:$|:)"          # IPv4
    r"|"
    r"\[?[0-9a-fA-F:]{2,39}\]?"                          # IPv6 (simplified)
)

# ── Feature names (canonical order) ─────────────────────────────

FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at",
    "num_question_marks",
    "num_ampersands",
    "num_equals",
    "num_percent",
    "num_digits",
    "num_subdomains",
    "is_https",
    "has_ip_address",
    "has_suspicious_tld",
    "has_suspicious_keyword",
    "path_length",
    "query_length",
    "url_entropy",
]


# ── Public API ──────────────────────────────────────────────────

def extract_url_features(url: str) -> dict[str, int | float]:
    """Extract 20 numerical features from *url*.

    Returns a dictionary whose keys match ``FEATURE_NAMES``.
    Malformed or empty URLs are handled gracefully – every value
    defaults to ``0`` (or ``0.0`` for entropy).
    """
    if not url or not isinstance(url, str):
        return _empty_features()

    url = url.strip()
    if not url:
        return _empty_features()

    parsed = _safe_parse(url)
    hostname = (parsed.hostname or "").lower()
    url_lower = url.lower()

    return {
        "url_length":             len(url),
        "hostname_length":        len(hostname),
        "num_dots":               url.count("."),
        "num_hyphens":            url.count("-"),
        "num_underscores":        url.count("_"),
        "num_slashes":            url.count("/"),
        "num_at":                 url.count("@"),
        "num_question_marks":     url.count("?"),
        "num_ampersands":         url.count("&"),
        "num_equals":             url.count("="),
        "num_percent":            url.count("%"),
        "num_digits":             sum(c.isdigit() for c in url),
        "num_subdomains":         _count_subdomains(hostname),
        "is_https":               int(parsed.scheme.lower() == "https"),
        "has_ip_address":         int(bool(_IP_RE.search(hostname))),
        "has_suspicious_tld":     int(_has_suspicious_tld(hostname)),
        "has_suspicious_keyword": int(_has_suspicious_keyword(url_lower)),
        "path_length":            len(parsed.path),
        "query_length":           len(parsed.query),
        "url_entropy":            round(_shannon_entropy(url), 4),
    }


# ── Internal helpers ────────────────────────────────────────────

def _empty_features() -> dict[str, int | float]:
    return {name: 0.0 if name == "url_entropy" else 0 for name in FEATURE_NAMES}


def _safe_parse(url: str):
    """Parse *url*, prepending a scheme when missing so urlparse works."""
    if "://" not in url and not url.startswith("//"):
        url = "http://" + url
    return urlparse(url)


def _count_subdomains(hostname: str) -> int:
    """Return the number of sub-domain levels (e.g. ``a.b.example.com`` → 2)."""
    if not hostname:
        return 0
    parts = hostname.split(".")
    # domain.tld → 0 subdomains; sub.domain.tld → 1; etc.
    return max(0, len(parts) - 2)


def _has_suspicious_tld(hostname: str) -> bool:
    if not hostname:
        return False
    tld = hostname.rsplit(".", 1)[-1]
    return tld in SUSPICIOUS_TLDS


def _has_suspicious_keyword(url_lower: str) -> bool:
    return any(kw in url_lower for kw in SUSPICIOUS_KEYWORDS)


def _shannon_entropy(text: str) -> float:
    """Compute the Shannon entropy (bits) of *text*."""
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )

