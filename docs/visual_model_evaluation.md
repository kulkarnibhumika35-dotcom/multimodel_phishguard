# PhishGuard – Visual Classifier Evaluation Report (Phase 2, Step 2.4)

**Evaluation Date**: 2026-11-05T17:15:48.662904+00:00  
**Model Architecture**: `RandomForestClassifier(n_estimators=100, random_state=42)`  
**Input Features**: Exactly 30 canonical visual features (0 metadata fields)  
**Partitioning**: Registered-domain grouped splitting (0% domain overlap)  

> [!WARNING]
> **Data Governance Notice & Provenance Limitation**:  
> CRITICAL LIMITATION: Legitimate class consists of Tranco candidates ('unverified_candidate') rather than manually verified benign ground truth. The visual classifier cannot be claimed as fully validated or deployed as a standalone production oracle without human/hybrid corroboration.  
> All metrics reported below represent classification performance against the candidate dataset baseline.

---

## 1. Executive Performance Summary

| Metric | Score | Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **80.07%** | Overall classification accuracy on held-out domains |
| **Macro F1-Score** | **0.7982** | Unweighted mean of both classes |
| **Phishing F1-Score** | **0.7757** | Primary threat detection capability |
| **Phishing Precision** | **0.8095** | Proportion of visual phishing alarms that are true threats |
| **Phishing Recall** | **0.7445** | Proportion of actual phishing screenshots detected |
| **ROC-AUC Score** | **0.8491** | Area under the Receiver Operating Characteristic curve |
| **5-Fold CV F1 (Train)** | **0.7333 (±0.0356)** | 5-fold stratified cross-validation on training set |

---

## 2. Confusion Matrix (Held-Out Test Set: N = 296)

| Ground Truth \ Prediction | Predicted Candidate (0) | Predicted Phishing (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual Candidate (Tranco)** | **135** (TN) | 24 (FP) | 159 |
| **Actual Phishing (PhishTank)** | 35 (FN) | **102** (TP) | 137 |

---

## 3. Detailed Classification Report

```
                      precision    recall  f1-score   support

unverified_candidate       0.79      0.85      0.82       159
            phishing       0.81      0.74      0.78       137

            accuracy                           0.80       296
           macro avg       0.80      0.80      0.80       296
        weighted avg       0.80      0.80      0.80       296

```

---

## 4. Top 10 Visual Feature Importances

| Rank | Feature Name | Gini Importance |
| :---: | :--- | :---: |
| 1 | `std_brightness` | 0.07741 |
| 2 | `std_blue` | 0.06962 |
| 3 | `color_variance` | 0.06934 |
| 4 | `std_green` | 0.06157 |
| 5 | `image_entropy` | 0.06147 |
| 6 | `std_red` | 0.05612 |
| 7 | `edge_density` | 0.04719 |
| 8 | `dominant_color_3_proportion` | 0.04410 |
| 9 | `dominant_color_1_g` | 0.03256 |
| 10 | `dominant_color_2_r` | 0.03193 |

---

## 5. Artifact Manifest

- Trained Model: `ml/models/visual_model.joblib`
- Feature Names: `ml/models/visual_feature_names.json`
- Class Mapping: `ml/models/visual_class_mapping.json`
- Evaluation JSON: `ml/evaluation/visual_model_eval.json`
- Train Split: `datasets/processed/visual_train.csv`
- Test Split: `datasets/processed/visual_test.csv`
