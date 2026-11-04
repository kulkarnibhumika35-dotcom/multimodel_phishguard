# PhishGuard – Dataset Quality, Overlap & Provenance Audit

**Audit Date**: 2026-11-04  
**Scope**: Ingestion audit of the three raw candidate datasets:
1. `datasets/raw/phishing.csv` (PhishTank verified online feed)
2. `datasets/raw/tranco.csv` (Tranco Top 1M Sites List)
3. `datasets/raw/kaggle_urls.csv` (Multi-source feed: URLhaus + Tranco + OpenPhish)

---

## 1. Executive Summary

| Dataset | Total Rows | Missing URLs / Fields | Duplicates Dropped | Retained Candidates | Verified Clean |
|---|---|---|---|---|---|
| **PhishTank** (`phishing.csv`) | 50,175 | 0 | 2 | **50,173** | ✅ 100% Verified Online Phishing |
| **Tranco** (`tranco.csv`) | 1,000,000 | 0 | 0 | **1,000,000** | ⚠️ Unlabeled Candidates (Never auto-labeled 0) |
| **Kaggle URLs** (`kaggle_urls.csv`) | 48,812 | 0 | 0 | **48,812** | ⚠️ Multi-source with domain overlap & conflicts |

All raw files were preserved without modification. Independent provenance manifests (`.meta.json`) with SHA-256 integrity checksums were generated for each source.

---

## 2. Cross-Source Overlap & Conflicting Label Analysis

A cross-comparison between **PhishTank** (50,173 unique verified phishing URLs) and **Kaggle URLs** (48,812 multi-source records) revealed critical findings:

### A. Overlapping URLs

* **Exact URL Overlaps**: 3 URLs
* **Normalized URL Overlaps** *(lowercase, trailing slash normalization)*: 4 URLs

| Normalized URL | PhishTank Label | Kaggle Source | Kaggle Label | Status | Impact |
|---|---|---|---|---|---|
| `https://gemincxologin.gitbook.io/us` | **1** (Phishing) | OpenPhish | **1** | **Agreement** | True Positive confirmed across feeds |
| `https://robl.vercel.app` | **1** (Phishing) | OpenPhish | **1** | **Agreement** | True Positive confirmed across feeds |
| `https://kakaku.com` | **1** (Phishing) | Tranco | **0** | 🚨 **CONFLICT** | Same domain labeled legitimate in Tranco, phishing in PhishTank |
| `https://pokeapi.co` | **1** (Phishing) | Tranco | **0** | 🚨 **CONFLICT** | Same domain labeled legitimate in Tranco, phishing in PhishTank |

### B. Root Cause of Label Conflicts

1. **Tranco Popularity vs. Security State**:
   - `pokeapi.co` and `kakaku.com` are high-traffic legitimate platforms listed in Tranco.
   - When a popular domain is compromised, hosts user-submitted malicious landing pages, or is submitted to PhishTank as a confirmed threat, labeling the root domain as `0` (legitimate) while simultaneously treating the URL as `1` (phishing) produces contradictory ground truth.
2. **Policy Enforcement**:
   - **Never automatically treat Tranco candidates as verified legitimate URLs**.
   - Before merging into training data, Tranco URLs must undergo domain conflict screening and threat intelligence cross-referencing.

---

## 3. URL Normalization Consistency

An audit of URL structural representations across sources identified distinct normalization profiles:

| Source | Default Scheme | Path Representation | Parameter / Query Behavior |
|---|---|---|---|
| **PhishTank** | Both `http://` (32%) and `https://` (68%) | Deep paths (91%), queries (28%) | Subdomains, credential paths, IP addresses, custom ports |
| **Tranco** (`tranco.csv`) | None (raw domain: `google.com`) | None | Bare domains; normalized to `https://{domain}` |
| **Kaggle: Tranco** | `https://` (100%) | None (root only) | Flat domains, no query strings or paths |
| **Kaggle: URLhaus** | `http://` (78%), `https://` (22%) | Specific file paths (97%) | Binary droppers (`.sh`, `.exe`, `.bin`), raw IP addresses, high ports |
| **Kaggle: OpenPhish** | `https://` (69%), `http://` (31%) | Deep credential paths (73%) | Phishing landing paths, spoofed brands |

### Normalization Requirements
- Strip whitespace, lowercase domain/hostname names.
- Strip trailing `/` from root domain paths to avoid duplicate entries (`https://example.com/` vs `https://example.com`).
- Preserve case-sensitive query tokens and paths where tokens may carry entropy or obfuscation indicators.

---

## 4. Threat Type Disambiguation: URLhaus vs PhishTank

The audit confirmed that **URLhaus** and **PhishTank** represent distinct threat paradigms:

```
[Threat Types]
├── Confirmed Phishing (PhishTank, OpenPhish)
│   ├── Target: User credentials, banking, identity theft
│   └── Characteristics: Brand keywords, deceptive subdomains, suspicious TLDs (.sbs, .xyz, .cc)
│
└── Malware Distribution (URLhaus)
    ├── Target: Device compromise, botnet propagation (Mirai, ELF)
    └── Characteristics: Direct IP addresses, unusual ports (e.g. :54956), executable extensions (.bin, .sh, .mips, .exe)
```

**Recommendation**: Retain `threat_type` metadata (`phishing` vs `malware_distribution`). While both map to malicious (`label=1`), keeping threat types distinct ensures the model does not overfit to malware dropper signatures at the expense of credential harvesting detection.

---

## 5. Dataset Leakage & Cross-Split Contamination Risks

A domain-level overlap analysis was performed to identify data leakage risks:

### A. Shared Hostnames Across Sources

* **Shared Hostnames (PhishTank Phishing vs. Kaggle Tranco)**: **67 hostnames**
  - *Link Shorteners*: `bit.ly`, `clck.ru`, `goo.su`, `1drv.ms`
  - *Cloud Hosting / Web Builders*: `forms.gle`, `airtable.com`, `beacons.ai`, `bio.site`, `framer.com`, `coda.io`, `gofile.io`
  - *Major Tech Platforms*: `google.com`
* **Shared Hostnames (Kaggle URLhaus Malware vs. Kaggle Tranco)**: **6 hostnames**
  - `github.com`, `gitlab.com`, `pastebin.com`, `bitbucket.org`, `archive.org`, `pixeldrain.com`
* **Shared Hostnames (PhishTank vs. Full Tranco Top 100k)**: **112 hostnames**

### B. Leakage Vulnerability in Standard Train/Test Splits

If random URL-level train/test splitting (`train_test_split`) is used:
1. `github.com/legitimate_repo` could appear in the train set as `0`, while `github.com/attacker/malware.exe` appears in the test set as `1`.
2. A random forest classifier evaluating `hostname_length` or lexical features could memorize platform names rather than true phishing traits.
3. **Mandatory Mitigation**: When preparing the final combined training dataset, splitting must be performed via **Group-aware splitting** (e.g., grouping by registered/apex domain) to ensure all URLs from a given domain reside solely in either the training or the test fold.

---

## 6. Compatibility with the 20 Canonical URL Features

All 20 PhishGuard lexical and structural URL features were tested on 1,000-sample batches from each raw source:

| Feature Name | PhishTank Compatible | Tranco Compatible | Kaggle Compatible |
|---|:---:|:---:|:---:|
| `url_length` | ✅ | ✅ | ✅ |
| `hostname_length` | ✅ | ✅ | ✅ |
| `num_dots` | ✅ | ✅ | ✅ |
| `num_hyphens` | ✅ | ✅ | ✅ |
| `num_underscores` | ✅ | ✅ | ✅ |
| `num_slashes` | ✅ | ✅ | ✅ |
| `num_at` | ✅ | ✅ | ✅ |
| `num_question_marks` | ✅ | ✅ | ✅ |
| `num_ampersands` | ✅ | ✅ | ✅ |
| `num_equals` | ✅ | ✅ | ✅ |
| `num_percent` | ✅ | ✅ | ✅ |
| `num_digits` | ✅ | ✅ | ✅ |
| `num_subdomains` | ✅ | ✅ | ✅ |
| `is_https` | ✅ | ✅ | ✅ |
| `has_ip_address` | ✅ | ✅ | ✅ |
| `has_suspicious_tld` | ✅ | ✅ | ✅ |
| `has_suspicious_keyword` | ✅ | ✅ | ✅ |
| `path_length` | ✅ | ✅ | ✅ |
| `query_length` | ✅ | ✅ | ✅ |
| `url_entropy` | ✅ | ✅ | ✅ |
| **Total Extraction Errors** | **0 / 1000** | **0 / 1000** | **0 / 1000** |

All tested samples yielded 100% valid numerical outputs with zero `None`, `NaN`, or `Inf` values.

---

## 7. Actionable Recommendations for Step 16

1. **Deduplicate and Resolve Conflicts**:
   - Discard conflicting URLs (`pokeapi.co`, `kakaku.com`) where PhishTank and Tranco disagree, or prioritize the verified PhishTank label.
2. **Exclude Shared Platform Apex Domains**:
   - Shared hosting/shortener domains (`bit.ly`, `github.com`, `forms.gle`) should not be classified purely on their apex domain.
3. **Curate a Balanced Gold Dataset**:
   - Sample 35,000 verified PhishTank phishing URLs + 15,000 URLhaus malware URLs (Total 50,000 malicious).
   - Sample 50,000 verified top Tranco domains confirmed free from malware/phishing reports (Total 50,000 legitimate).
4. **Enforce Domain-Grouped Splitting**:
   - Use domain-grouped cross-validation to guarantee zero data leakage between train and test splits.

