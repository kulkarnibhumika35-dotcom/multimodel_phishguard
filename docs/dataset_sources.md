# PhishGuard – Dataset Sources & Preparation Guide

This document defines the verified sources, download procedures, licensing requirements, and schema specifications for training and evaluating PhishGuard machine learning models.

---

## 1. Verified Dataset Sources

PhishGuard models require high-quality, ground-truth samples of both malicious (phishing) and legitimate (benign) URLs.

### A. Phishing URL Sources

#### 1. PhishTank
* **Provider**: Cisco / OpenDNS Community
* **Website**: [https://phishtank.org](https://phishtank.org)
* **Data Feed**:
  * CSV Feed: `http://data.phishtank.com/data/online-valid.csv`
  * Developer API: `https://data.phishtank.com/data/<API_KEY>/online-valid.csv`
* **Description**: Community-submitted and human-verified active phishing URLs. Records include verification status, submission timestamp, target brand, and active status.
* **Licensing & Terms**:
  * Free for non-commercial and research use under the PhishTank Developer License.
  * Commercial use requires explicit permission from Cisco Systems.
* **Attribution Requirement**:
  > *"Data provided by PhishTank (https://phishtank.org)."*

#### 2. URLhaus (abuse.ch)
* **Provider**: abuse.ch Project (Bern University of Applied Sciences)
* **Website**: [https://urlhaus.abuse.ch](https://urlhaus.abuse.ch)
* **Data Feed**:
  * Online URLs (CSV): `https://urlhaus.abuse.ch/downloads/csv_online/`
  * Recent URLs (CSV): `https://urlhaus.abuse.ch/downloads/csv_recent/`
* **Description**: Comprehensive tracking of malicious URLs actively distributing malware, Trojans, and phishing payloads.
* **Licensing**:
  * **Creative Commons Zero (CC0 1.0 Universal / Public Domain)**.
  * Free for academic, non-commercial, and commercial applications.
* **Attribution Requirement**:
  > *"Data provided by URLhaus (https://urlhaus.abuse.ch) under CC0 1.0 Universal."*

---

### B. Legitimate URL Sources

#### 1. Tranco List
* **Provider**: Tranco Research Consortium (Le Pochat et al.)
* **Website**: [https://tranco-list.eu](https://tranco-list.eu)
* **Data Feed**:
  * Top 1M domains: `https://tranco-list.eu/download/daily/top-1m.csv.zip`
  * Dated research snapshots: Available via permalinks on the Tranco website.
* **Description**: A research-grade top-sites ranking that hardens against malicious manipulation by aggregating Cisco Umbrella, Majestic, Cloudflare Radar, and Chrome UX Report.
* **Licensing**:
  * **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
* **Attribution Requirement**:
  > *"Tranco research list provided by Victor Le Pochat et al., 'Tranco: A Research-Oriented Top Sites Ranking Hardened Against Manipulation', NDSS 2019 (https://tranco-list.eu)."*

---

## 2. Manual Download & Ingestion Workflow

To ensure reproducibility, respect provider rate limits, and prevent unintended network egress, **PhishGuard strictly prohibits automated background downloading during model execution or testing**.

All datasets must be downloaded manually or through approved pipeline jobs and saved into `datasets/raw/`.

### Step-by-Step Instructions

1. **Download Phishing URLs**:
   * Navigate to PhishTank (`https://phishtank.org/developer_info.php`) or URLhaus (`https://urlhaus.abuse.ch/api/`).
   * Download the latest verified online CSV export.
   * Save the raw file to:
     ```text
     datasets/raw/phishtank_raw.csv
     # or
     datasets/raw/urlhaus_raw.csv
     ```

2. **Download Legitimate URLs**:
   * Navigate to `https://tranco-list.eu` and download the latest `top-1m.csv.zip`.
   * Unpack the CSV and format the top entries (e.g. top 50,000 domains) with standard `https://` schemes.
   * Save the raw file to:
     ```text
     datasets/raw/tranco_legitimate_raw.csv
     ```

3. **Construct the Raw Unified CSV**:
   Combine the verified samples into a single dataset at:
   ```text
   datasets/raw/phishguard_raw_dataset.csv
   ```
   accompanied by its provenance manifest:
   ```text
   datasets/raw/phishguard_raw_dataset.meta.json
   ```

---

## 3. Required CSV Specification

Raw datasets provided to `ml/training/prepare_dataset.py` and `ml/training/validate_dataset.py` must adhere to the following schema:

```csv
url,label
https://google.com,0
https://github.com,0
http://192.168.1.100/secure-update/login.php,1
http://account-verification-service.xyz/banking,1
```

### Column Definitions

| Column | Type | Allowed Values | Description |
|---|---|---|---|
| `url` | String | Valid HTTP/HTTPS URL | The full uniform resource locator to inspect. Must not be empty. |
| `label` | Integer / String | `0` or `1` | Binary class assignment: <br>• `0`: Legitimate / Benign<br>• `1`: Phishing / Malicious |

*Alternative acceptable labels before normalisation*:
- `0`: `"legitimate"`, `"benign"`, `"safe"`, `"good"`
- `1`: `"phishing"`, `"malicious"`, `"bad"`, `"spam"`

---

## 4. Dataset Validation Criteria

Before feature extraction or model training, the dataset must pass the PhishGuard validation suite (`ml/training/validate_dataset.py`):

1. **Missing Value Integrity**:
   - Every row must contain a non-empty, non-whitespace `url`.
   - Every row must contain a non-null, non-empty `label`.
2. **Deduplication**:
   - Duplicate URLs must be detected.
   - Any contradictory duplicates (the same URL labeled both 0 and 1) are flagged as critical integrity violations.
3. **Label Correctness**:
   - All labels must normalize cleanly to binary integers `0` or `1`. Unrecognized values are dropped and counted.
4. **Class Balance & Minimum Sample Count**:
   - Training requires at least **20 total samples** and at least **5 samples per class**.
   - Ideal production balance: 40%–60% positive class (phishing). Imbalance exceeding 90/10 triggers a warning.
5. **Scheme Protocol Integrity**:
   - URLs should adhere to standard `http://` or `https://` web protocols.

---

## 5. Dataset Provenance & Collection Metadata

Every raw dataset ingested into PhishGuard should be paired with a metadata manifest (`.meta.json`) recording:

```json
{
  "dataset_name": "phishguard_combined_v1",
  "source_name": "PhishTank + Tranco Top 50k",
  "collection_date": "2026-11-04",
  "collector": "PhishGuard Data Pipeline",
  "license": "PhishTank Developer License / CC BY 4.0",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "total_records": 100000,
  "legitimate_count": 50000,
  "phishing_count": 50000,
  "notes": "Verified online phishing URLs from PhishTank paired with top 50k Tranco domains."
}
```

This manifest preserves source integrity, auditing compliance, and legal attribution.

