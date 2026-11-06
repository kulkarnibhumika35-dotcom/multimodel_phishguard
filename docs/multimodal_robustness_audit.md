# PhishGuard – Multimodal Model Robustness, Data Leakage & Ablation Audit (Phase 2, Step 2.7)

**Audit Date**: 2026-11-05T18:06:01.111452+00:00  
**Audited Model**: `ml/models/multimodal_model.joblib` (SHA-256: `a59c0092c21122cb26c0bf4b0f5a72b0613de28493373d4c299bbd6f17ea1ca3`)  
**Evaluation Partitions**: Held-Out Test Set ($N=296$) vs Training Split ($N=1,190$)  

> [!WARNING]
> **Data Governance Notice & Provenance Limitation**:  
> CRITICAL GOVERNANCE LIMITATION: The non-phishing class comprises Tranco popularity candidates ('unverified_candidate') rather than verified benign ground truth. The model cannot be claimed as production-ready or generalizable to arbitrary real-world traffic without controlled live testing, telemetry monitoring, and human-in-the-loop triage.

---

## 1. Data Leakage Audit Findings

| Audit Check | Standard | Observed Result | Status |
| :--- | :---: | :---: | :---: |
| **Registered Domain Overlap** | 0 domains | **0** (0) | PASS (Zero Leakage) |
| **Exact URL Overlap** | 0 URLs | **0** (0) | PASS (Zero Leakage) |
| **Normalized URL Overlap** | 0 URLs | **0** (0) | PASS (Zero Leakage) |
| **Screenshot SHA-256 Hash Overlap** | 0 hashes | **0** (0) | PASS (Zero Leakage) |
| **Visual Feature Vector Exact Duplicates** | 0 vectors | **0** (0) | PASS (Zero Leakage) |
| **Metadata Columns Excluded from Model Input** | 9 columns excluded | **0 leaked** ([]) | PASS (Isolated) |
| **Target Columns Excluded from Model Input** | Target excluded | **0 leaked** ([]) | PASS (Isolated) |
| **Model Feature Dimension Verification** | Exactly 51 features | **51 features** | PASS (Exact Match) |
| **Collection Sequence Bias (csv_line_num)** | Metadata strictly excluded | **Excluded** ($r_{true}=0.3608$, $r_{prob}=0.3478$) | PASS (No Input Leakage) |

### Collection Sequence & Row Identifier Bias Analysis
- **Raw Zenodo Ordering Correlation**: In the raw Zenodo V2.0 CSV, line numbers exhibit a non-random ordering correlation with the ground truth label ($r = 0.3608$, $p < 0.0001$ on the test split; $r = 0.3828$ on train) because candidate and phishing URLs were appended in sequential batches by the dataset authors.
- **Strict Metadata Exclusion**: `csv_line_num` is categorized under `METADATA_COLUMNS` and is completely excluded from model inputs. The model is trained exclusively on the 51 canonical lexical and visual features.
- **Model Output Correlation**: The observed correlation between `csv_line_num` and the model's predicted probability ($r = 0.3478$) is strictly a downstream consequence of the model correctly predicting the underlying ground truth label ($r \approx 0.3608$), confirming that the model cannot and does not utilize row numbers or file positions for inference.

---

## 2. Feature Ablation Audit (True Ablation vs Standalone Models)

Ablation experiments evaluate the *already-trained* multimodal decision forest when inputs are masked at inference time, compared against independently trained single-modality models:

| Evaluation Configuration | Model Used | Features Available | Accuracy | Precision | Recall | F1-Score | Macro F1 | ROC-AUC | Phishing Caught | False Alarms |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Multimodal Baseline** | `multimodal_model` | All 51 features | **98.65%** | **100.00%** | **97.08%** | **0.9852** | **0.9864** | **0.9987** | 133 / 137 | **0 / 159** |
| **True Ablation: URL-Only** | `multimodal_model` | 20 URL (Visual masked) | 96.62% | 97.04% | 95.62% | 0.9632 | 0.9660 | 0.9944 | 131 / 137 | 4 / 159 |
| **True Ablation: Visual-Only** | `multimodal_model` | 30 Visual (URL masked) | 64.53% | 100.00% | 23.36% | 0.3787 | 0.5652 | 0.6427 | 32 / 137 | 0 / 159 |
| **Standalone URL Model** | `url_model.joblib` | 20 URL features | 59.46% | 53.31% | 100.00% | 0.6954 | 0.5447 | 0.4664 | 137 / 137 | 120 / 159 |
| **Standalone Visual Model** | `visual_model.joblib` | 30 Visual features | 80.07% | 80.95% | 74.45% | 0.7757 | 0.7982 | 0.8491 | 102 / 137 | 24 / 159 |

### Key Ablation Insights:
1. **Graceful Single-Modality Degradation**:
   When visual data is unavailable (e.g. screenshot timeout, headless browser failure), masking visual features with zeros and toggling `has_visual_features=0` retains **96.62% accuracy** and **95.62% phishing recall**. The model does NOT collapse into random guessing.
2. **True Ablation vs Standalone Architecture**:
   The multimodal model was trained jointly; its root splits prioritize lexical features before refining leaves with visual features. Consequently, zeroing URL features at test time leaves precision intact (100%), but drops recall to 23.36%. In contrast, the standalone visual model learned independent visual splits without relying on lexical features, achieving 74.45% recall.

---

## 3. Class Mapping & Governance Audit

| Model Component | Class 0 Label | Class 1 Label | Class 2 Label | Positive Threat Class | Malware Isolated? |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Multimodal Model** | `unverified_candidate` | `phishing` | None (Binary) | Class 1 (`phishing`) | N/A (0 in dataset) |
| **Visual Screenshot Model** | `unverified_candidate` | `phishing` | None (Binary) | Class 1 (`phishing`) | N/A (0 in dataset) |
| **URL Lexical Model** | `legitimate` | `phishing` | `malware_distribution` | Class 1 (`phishing`) | **YES** (Strictly Class 2) |

- **No Malware Bleed**: `malware_distribution` is never silently mapped into `legitimate` or `unverified_candidate`.
- **Candidate Semantics**: Tranco candidates represent top-ranked domain names whose legitimate status is unverified; all metrics strictly measure discrimination against this candidate population.

---

## 4. Robustness Analysis & Error Deep Dive

### A. False Negative Root Cause Analysis (4 Missed Threats)
On the held-out test set, the multimodal model missed exactly 4 verified phishing attacks:

| Line # | Target URL | Registered Domain | Multi P(Phish) | Standalone URL P(Phish) | Standalone Visual P(Phish) | Structural Characteristics |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| `1457` | `http://cw-net.de` | `cw-net.de` | 0.32 | 1.00 | 0.22 | Path length: 0, Slashes: 2, URL len: 16 |
| `14909` | `http://www.my-loyola.com` | `my-loyola.com` | 0.31 | 1.00 | 0.40 | Path length: 0, Slashes: 2, URL len: 24 |
| `17559` | `https://micronicsolutions.com` | `micronicsolutions.com` | 0.30 | 1.00 | 0.90 | Path length: 0, Slashes: 2, URL len: 29 |
| `23481` | `http://demerdashscan.com` | `demerdashscan.com` | 0.30 | 1.00 | 0.59 | Path length: 0, Slashes: 2, URL len: 24 |

> [!IMPORTANT]
> **Common Failure Mode**:
> All 4 false negatives are **apex / bare homepages** (`path_length = 0`, `num_slashes = 2`). Because the vast majority of phishing in Zenodo V2.0 resides on subdirectories or dynamic redirect paths, bare domain phishing mimics the lexical simplicity of genuine homepages, driving tree predictions slightly below the 0.50 decision threshold (P ≈ 0.30 - 0.32).

### B. Prediction Confidence & Decision Margins
- **Mean Margin from Threshold (|P - 0.5|)**: **0.4476** (Highly decisive).
- **Ambiguous Predictions (0.30 ≤ P ≤ 0.70)**: Only **11 out of 296 samples (3.72%)**.

### C. URL Characteristic Breakdown (Phishing Recall on Test Set, N=137)
- **HTTPS Phishing**: 114 samples → **113 caught (99.12% recall)**.
- **HTTP Phishing**: 23 samples → **20 caught (86.96% recall)**.
- **Suspicious TLD Phishing**: 9 samples → **9 caught (100.00% recall)**.
- **Regular TLD Phishing**: 128 samples → **124 caught (96.88% recall)**.

### D. Cross-Modality Discordance Resolution
- **Case 1: URL says Phishing, Visual says Candidate (N=137)**:
  - Ground truth: 102 Candidates, 35 Phishing.
  - Multimodal resolution: Predicted Candidate for 104, Phishing for 33. **(97.8% correct resolution)**.
- **Case 2: URL says Candidate, Visual says Phishing (N=6)**:
  - Ground truth: 6 Candidates, 0 Phishing.
  - Multimodal resolution: Predicted Candidate for all 6. **(100.0% correct resolution)**.

---

## 5. Controlled Integration Readiness Assessment

| Readiness Criterion | Audit Assessment | Status |
| :--- | :--- | :---: |
| **Leakage Isolation** | Zero domain, URL, hash, or feature vector leakage across train/test splits. | READY |
| **Graceful Degradation** | Retains 95.6% recall and 97.0% precision when visual input is unavailable. | READY |
| **Precision Stability** | 0 false alarms on held-out test candidates; eliminates URL model over-flagging. | READY |
| **Controlled Integration Readiness** | **READY for a controlled canary / shadow-mode integration experiment.** | **APPROVED** |

> [!CAUTION]
> **Controlled Experiment Recommendations**:
> 1. Deploy the multimodal model in **shadow mode** alongside the primary URL detector first.
> 2. Maintain an audit log of telemetry comparing decisions where screenshot acquisition times out.
> 3. Do not treat Tranco candidates as production-verified benign traffic until evaluated against enterprise telemetry.
