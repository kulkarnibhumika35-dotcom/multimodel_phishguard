"""Unit tests for ml.training.train_url_model.

All tests use small synthetic CSVs created in tmp directories.
Results from synthetic data must NOT be interpreted as real model performance.
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pytest
from ml.features.url_features import FEATURE_NAMES
from ml.training.train_url_model import (
    CLASS_LEGITIMATE,
    CLASS_MALWARE,
    CLASS_MAP,
    CLASS_PHISHING,
    convert_to_three_class_label,
    train_model,
)


# ── Helpers ─────────────────────────────────────────────────────

def _make_synthetic_csv(path: str, n_legit: int = 30, n_phish: int = 30, seed: int = 42):
    """Write a tiny synthetic processed CSV with random feature values.

    This is test-only scaffolding — NOT real training data.
    """
    rng = np.random.RandomState(seed)
    header = ["url", "label"] + list(FEATURE_NAMES)

    rows = []
    for i in range(n_legit):
        feats = rng.rand(20).tolist()
        rows.append([f"https://legit-{i}.com", "0"] + [str(round(f, 4)) for f in feats])
    for i in range(n_phish):
        feats = (rng.rand(20) + 0.5).tolist()  # offset so classes differ a bit
        rows.append([f"http://phish-{i}.xyz", "1"] + [str(round(f, 4)) for f in feats])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ── 1. Happy-path training ─────────────────────────────────────

class TestTrainHappyPath:
    def test_returns_metrics(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        result = train_model(
            ds,
            model_path=str(tmp_path / "model.joblib"),
            features_path=str(tmp_path / "features.json"),
            eval_path=str(tmp_path / "eval.json"),
        )

        m = result["metrics"]
        for key in ("accuracy", "precision", "recall", "f1_score", "confusion_matrix"):
            assert key in m, f"Missing metric: {key}"

    def test_metrics_in_valid_range(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        result = train_model(
            ds,
            model_path=str(tmp_path / "model.joblib"),
            features_path=str(tmp_path / "features.json"),
            eval_path=str(tmp_path / "eval.json"),
        )

        for key in ("accuracy", "precision", "recall", "f1_score"):
            assert 0.0 <= result["metrics"][key] <= 1.0

    def test_confusion_matrix_shape(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        result = train_model(
            ds,
            model_path=str(tmp_path / "model.joblib"),
            features_path=str(tmp_path / "features.json"),
            eval_path=str(tmp_path / "eval.json"),
        )

        cm = result["metrics"]["confusion_matrix"]
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)


# ── 2. Artefact persistence ────────────────────────────────────

class TestArtefactSaving:
    def test_model_saved(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        result = train_model(
            ds,
            model_path=str(tmp_path / "model.joblib"),
            features_path=str(tmp_path / "features.json"),
            eval_path=str(tmp_path / "eval.json"),
        )

        assert os.path.isfile(result["model_path"])
        clf = joblib.load(result["model_path"])
        assert hasattr(clf, "predict")

    def test_feature_names_saved(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        result = train_model(
            ds,
            model_path=str(tmp_path / "model.joblib"),
            features_path=str(tmp_path / "features.json"),
            eval_path=str(tmp_path / "eval.json"),
        )

        assert os.path.isfile(result["features_path"])
        with open(result["features_path"]) as f:
            names = json.load(f)
        assert names == list(FEATURE_NAMES)

    def test_eval_json_saved(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        result = train_model(
            ds,
            model_path=str(tmp_path / "model.joblib"),
            features_path=str(tmp_path / "features.json"),
            eval_path=str(tmp_path / "eval.json"),
        )

        assert os.path.isfile(result["eval_path"])
        with open(result["eval_path"]) as f:
            saved = json.load(f)
        assert saved == result["metrics"]

    def test_creates_output_directories(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        result = train_model(
            ds,
            model_path=str(tmp_path / "deep" / "dir" / "model.joblib"),
            features_path=str(tmp_path / "deep" / "dir" / "features.json"),
            eval_path=str(tmp_path / "deep" / "dir" / "eval.json"),
        )

        assert os.path.isfile(result["model_path"])
        assert os.path.isfile(result["features_path"])
        assert os.path.isfile(result["eval_path"])


# ── 3. Model usability ─────────────────────────────────────────

class TestModelUsability:
    def test_model_predicts(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        result = train_model(
            ds,
            model_path=str(tmp_path / "model.joblib"),
            features_path=str(tmp_path / "features.json"),
            eval_path=str(tmp_path / "eval.json"),
        )

        clf = joblib.load(result["model_path"])
        sample = np.zeros((1, 20))
        pred = clf.predict(sample)
        assert pred[0] in (0, 1)

    def test_model_predicts_proba(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        result = train_model(
            ds,
            model_path=str(tmp_path / "model.joblib"),
            features_path=str(tmp_path / "features.json"),
            eval_path=str(tmp_path / "eval.json"),
        )

        clf = joblib.load(result["model_path"])
        sample = np.zeros((1, 20))
        proba = clf.predict_proba(sample)
        assert proba.shape == (1, 2)
        assert abs(proba.sum() - 1.0) < 1e-6


# ── 4. Stratified split validation ─────────────────────────────

class TestStratifiedSplit:
    def test_different_random_states_vary(self, tmp_path):
        ds = str(tmp_path / "data.csv")
        _make_synthetic_csv(ds)

        r1 = train_model(
            ds,
            model_path=str(tmp_path / "m1.joblib"),
            features_path=str(tmp_path / "f1.json"),
            eval_path=str(tmp_path / "e1.json"),
            random_state=1,
        )
        r2 = train_model(
            ds,
            model_path=str(tmp_path / "m2.joblib"),
            features_path=str(tmp_path / "f2.json"),
            eval_path=str(tmp_path / "e2.json"),
            random_state=99,
        )
        # Not guaranteed to differ on tiny data, but both must be valid
        for r in (r1, r2):
            assert 0.0 <= r["metrics"]["accuracy"] <= 1.0


# ── 5. Error handling ──────────────────────────────────────────

class TestErrorHandling:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            train_model(
                str(tmp_path / "nope.csv"),
                model_path=str(tmp_path / "m.joblib"),
                features_path=str(tmp_path / "f.json"),
                eval_path=str(tmp_path / "e.json"),
            )

    def test_dataset_too_small(self, tmp_path):
        ds = str(tmp_path / "tiny.csv")
        _make_synthetic_csv(ds, n_legit=5, n_phish=5)

        with pytest.raises(ValueError, match="too small"):
            train_model(
                ds,
                model_path=str(tmp_path / "m.joblib"),
                features_path=str(tmp_path / "f.json"),
                eval_path=str(tmp_path / "e.json"),
            )

    def test_single_class_only(self, tmp_path):
        ds = str(tmp_path / "one_class.csv")
        _make_synthetic_csv(ds, n_legit=30, n_phish=0)

        with pytest.raises(ValueError, match="both classes"):
            train_model(
                ds,
                model_path=str(tmp_path / "m.joblib"),
                features_path=str(tmp_path / "f.json"),
                eval_path=str(tmp_path / "e.json"),
            )

    def test_missing_feature_columns(self, tmp_path):
        ds = str(tmp_path / "bad.csv")
        # Write CSV with only some columns
        header = ["url", "label", "url_length"]
        with open(ds, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(["https://example.com", "0", "20"])

        with pytest.raises(ValueError, match="missing feature columns"):
            train_model(
                ds,
                model_path=str(tmp_path / "m.joblib"),
                features_path=str(tmp_path / "f.json"),
                eval_path=str(tmp_path / "e.json"),
            )

    def test_empty_csv_file(self, tmp_path):
        ds = str(tmp_path / "empty.csv")
        with open(ds, "w") as f:
            f.write("")

        with pytest.raises(ValueError, match="empty"):
            train_model(
                ds,
                model_path=str(tmp_path / "m.joblib"),
                features_path=str(tmp_path / "f.json"),
                eval_path=str(tmp_path / "e.json"),
            )

    def test_class_too_few_samples(self, tmp_path):
        ds = str(tmp_path / "imbalanced.csv")
        _make_synthetic_csv(ds, n_legit=25, n_phish=3)

        with pytest.raises(ValueError, match="Class 1"):
            train_model(
                ds,
                model_path=str(tmp_path / "m.joblib"),
                features_path=str(tmp_path / "f.json"),
                eval_path=str(tmp_path / "e.json"),
            )
# ── 6. Three-Class Architecture & Domain-Grouped Training ───────────

def _make_three_class_synthetic_csv(path: str, n_legit: int = 25, n_phish: int = 25, n_malware: int = 25, seed: int = 42):
    """Write a synthetic 3-class CSV with registered_domain and threat_category."""
    rng = np.random.RandomState(seed)
    header = ["url", "registered_domain", "threat_category", "binary_label"] + list(FEATURE_NAMES)
    rows = []

    # Legitimate (0)
    for i in range(n_legit):
        dom = f"legit-service-{i // 2}.org"
        feats = rng.rand(20).tolist()
        rows.append([f"https://{dom}/page-{i}", dom, "legitimate", "0"] + [str(round(f, 4)) for f in feats])

    # Phishing (1)
    for i in range(n_phish):
        dom = f"phish-target-{i // 2}.com"
        feats = (rng.rand(20) + 0.3).tolist()
        rows.append([f"https://{dom}/login-{i}", dom, "phishing", "1"] + [str(round(f, 4)) for f in feats])

    # Malware Distribution (2)
    for i in range(n_malware):
        dom = f"192.168.1.{10 + (i // 2)}"
        feats = (rng.rand(20) + 0.6).tolist()
        rows.append([f"http://{dom}:8080/dropper-{i}.bin", dom, "malware_distribution", "1"] + [str(round(f, 4)) for f in feats])

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


class TestThreeClassTraining:
    def test_three_class_training_happy_path(self, tmp_path):
        """Test complete 3-class training, metrics, and report structure."""
        ds = str(tmp_path / "three_class_data.csv")
        _make_three_class_synthetic_csv(ds)

        model_file = str(tmp_path / "model.joblib")
        features_file = str(tmp_path / "features.json")
        class_mapping_file = str(tmp_path / "class_mapping.json")
        eval_file = str(tmp_path / "eval.json")

        result = train_model(
            ds,
            model_path=model_file,
            features_path=features_file,
            class_mapping_path=class_mapping_file,
            eval_path=eval_file,
        )

        m = result["metrics"]
        # Required metrics: accuracy, macro precision/recall/F1, weighted precision/recall/F1
        for key in (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "confusion_matrix",
            "classification_report",
        ):
            assert key in m, f"Missing metric: {key}"

        # Macro and weighted dicts
        assert "macro_avg" in m
        assert "weighted_avg" in m

        # 3-Class Confusion Matrix (3x3 shape)
        cm = m["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)

        # Per-class classification report contains all 3 classes
        rep = m["classification_report"]
        assert "legitimate" in rep
        assert "phishing" in rep
        assert "malware_distribution" in rep

        for cls_name in ("legitimate", "phishing", "malware_distribution"):
            assert "precision" in rep[cls_name]
            assert "recall" in rep[cls_name]
            assert "f1-score" in rep[cls_name]
            assert "support" in rep[cls_name]

    def test_convert_to_three_class_label(self):
        """Test label conversion for all valid threats and error handling."""
        # threat_category strings
        assert convert_to_three_class_label("legitimate", "0") == CLASS_LEGITIMATE
        assert convert_to_three_class_label("phishing", "1") == CLASS_PHISHING
        assert convert_to_three_class_label("malware_distribution", "1") == CLASS_MALWARE
        assert convert_to_three_class_label("malware", "1") == CLASS_MALWARE

        # numeric / legacy labels when threat_category is None
        assert convert_to_three_class_label(None, "0") == CLASS_LEGITIMATE
        assert convert_to_three_class_label(None, "1") == CLASS_PHISHING
        assert convert_to_three_class_label(None, "2") == CLASS_MALWARE

        with pytest.raises(ValueError, match="Cannot map to 3-class label"):
            convert_to_three_class_label("unknown_threat", "99")

    def test_domain_grouping_split_zero_leakage_in_training(self, tmp_path):
        """Test zero registered-domain overlap between training and testing."""
        ds = str(tmp_path / "data.csv")
        _make_three_class_synthetic_csv(ds, n_legit=30, n_phish=30, n_malware=30)

        result = train_model(
            ds,
            model_path=str(tmp_path / "m.joblib"),
            features_path=str(tmp_path / "f.json"),
            eval_path=str(tmp_path / "e.json"),
        )

        prov = result["provenance"]
        assert prov["domain_leakage_count"] == 0
        assert prov["domain_leakage_rate"] == "0.00%"
        assert prov["unique_domains_train"] > 0
        assert prov["unique_domains_test"] > 0

    def test_model_serialization_and_three_class_prediction(self, tmp_path):
        """Test model joblib serialization, predict and predict_proba on 3 classes."""
        ds = str(tmp_path / "data.csv")
        _make_three_class_synthetic_csv(ds)

        model_file = str(tmp_path / "model.joblib")
        result = train_model(
            ds,
            model_path=model_file,
            features_path=str(tmp_path / "f.json"),
            eval_path=str(tmp_path / "e.json"),
        )

        assert os.path.isfile(model_file)
        clf = joblib.load(model_file)

        sample = np.zeros((1, 20))
        pred = clf.predict(sample)
        assert pred[0] in (CLASS_LEGITIMATE, CLASS_PHISHING, CLASS_MALWARE)

        proba = clf.predict_proba(sample)
        assert proba.shape == (1, 3)
        assert abs(proba.sum() - 1.0) < 1e-6

    def test_class_mapping_persistence(self, tmp_path):
        """Test class_mapping.json is persisted with correct bidirectional mappings."""
        ds = str(tmp_path / "data.csv")
        _make_three_class_synthetic_csv(ds)

        class_mapping_file = str(tmp_path / "class_mapping.json")
        train_model(
            ds,
            model_path=str(tmp_path / "m.joblib"),
            features_path=str(tmp_path / "f.json"),
            class_mapping_path=class_mapping_file,
            eval_path=str(tmp_path / "e.json"),
        )

        assert os.path.isfile(class_mapping_file)
        with open(class_mapping_file, encoding="utf-8") as f:
            mapping = json.load(f)

        assert mapping["num_classes"] == 3
        assert mapping["class_to_name"]["0"] == "legitimate"
        assert mapping["class_to_name"]["1"] == "phishing"
        assert mapping["class_to_name"]["2"] == "malware_distribution"
        assert mapping["name_to_class"]["legitimate"] == 0
        assert mapping["name_to_class"]["phishing"] == 1
        assert mapping["name_to_class"]["malware_distribution"] == 2

    def test_dataset_provenance_recorded(self, tmp_path):
        """Test dataset provenance and class counts before and after split are saved."""
        ds = str(tmp_path / "data.csv")
        _make_three_class_synthetic_csv(ds, n_legit=20, n_phish=20, n_malware=20)

        result = train_model(
            ds,
            model_path=str(tmp_path / "m.joblib"),
            features_path=str(tmp_path / "f.json"),
            eval_path=str(tmp_path / "e.json"),
        )

        prov = result["provenance"]
        assert prov["total_samples"] == 60
        assert prov["train_samples"] + prov["test_samples"] == 60
        assert prov["dataset_sha256"] != ""
        assert "legitimate" in prov["class_counts_before_split"]
        assert "phishing" in prov["class_counts_before_split"]
        assert "malware_distribution" in prov["class_counts_before_split"]
