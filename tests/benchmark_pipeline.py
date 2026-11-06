"""
PhishGuard – Step 3.5: Performance and Reliability Benchmark Script

Measures:
1. URL analysis latency (feature extraction + URL model predict_proba)
2. Screenshot capture simulation / decoding latency
3. Visual feature extraction latency (cv2 decoding + 30 canonical visual features)
4. Multimodal inference latency (fusion + multimodal model predict_proba)
5. Complete API response latency (end-to-end FastAPI endpoint with SQLite persistence)
"""

import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.database.session import create_db_engine, init_db, get_db
from backend.main import app
from backend.services.analyzer import get_model_manager
from ml.features.feature_fusion import FUSED_FEATURE_NAMES, fuse_features
from ml.features.url_features import FEATURE_NAMES, extract_url_features
from ml.features.visual_features import VISUAL_FEATURE_NAMES, extract_visual_features


def create_sample_screenshot_b64(width=800, height=600) -> str:
    """Create a realistic synthetic webpage screenshot."""
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    # Header bar
    img[:80, :] = [30, 45, 90]
    # Content block
    img[120:400, 50:500] = [255, 255, 255]
    cv2.rectangle(img, (50, 120), (500, 400), (200, 200, 200), 1)
    # Button
    img[320:360, 80:240] = [20, 120, 220]
    # Noise/texture
    noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    success, encoded = cv2.imencode(".png", img)
    assert success
    b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def run_benchmarks(iterations=50):
    print("=" * 70)
    print(f"PhishGuard Step 3.5: Pipeline Latency Benchmarking ({iterations} iterations)")
    print("=" * 70)

    mgr = get_model_manager()
    url_model, _, _ = mgr.get_url_model_and_explainer()
    visual_model, _ = mgr.get_visual_model()
    multi_model, _ = mgr.get_multimodal_model()

    assert url_model is not None, "URL model not loaded!"
    assert visual_model is not None, "Visual model not loaded!"
    assert multi_model is not None, "Multimodal model not loaded!"

    test_url = "https://www.paypal.com/signin"
    phish_url = "http://secure-paypal-login-credential-verification.tk/login"
    malware_url = "http://192.168.1.100:8080/setup.exe"

    screenshot_b64 = create_sample_screenshot_b64(800, 600)

    # 1. URL Analysis Latency (Feature extraction + model predict_proba)
    url_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        feats = extract_url_features(test_url)
        X = np.array([[feats[col] for col in FEATURE_NAMES]])
        _ = url_model.predict_proba(X)
        t1 = time.perf_counter()
        url_times.append((t1 - t0) * 1000.0)

    # 2. Screenshot Base64 Decoding & Ingestion
    decode_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        payload = screenshot_b64.split(",", 1)[-1] if "," in screenshot_b64 else screenshot_b64
        buf = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        decode_times.append((t1 - t0) * 1000.0)

    # 3. Visual Feature Extraction Latency (cv2 decoding + 30 features)
    visual_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        payload = screenshot_b64.split(",", 1)[-1] if "," in screenshot_b64 else screenshot_b64
        buf = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        _ = extract_visual_features(img)
        t1 = time.perf_counter()
        visual_times.append((t1 - t0) * 1000.0)

    # 4. Multimodal Inference Latency (feature fusion + predict_proba)
    url_feats = extract_url_features(test_url)
    payload = screenshot_b64.split(",", 1)[-1] if "," in screenshot_b64 else screenshot_b64
    buf = np.frombuffer(base64.b64decode(payload), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    vis_feats = extract_visual_features(img)

    multi_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fused = fuse_features(url_feats, vis_feats)
        X_multi = np.array([[fused[col] for col in FUSED_FEATURE_NAMES]])
        _ = multi_model.predict_proba(X_multi)
        t1 = time.perf_counter()
        multi_times.append((t1 - t0) * 1000.0)

    # 5. Complete API Response Latency (FastAPI client with SQLite)
    temp_db = "data/benchmark_test.db"
    engine = create_db_engine(temp_db)
    init_db(engine_override=engine)

    from sqlalchemy.orm import sessionmaker
    BenchSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_db():
        db = BenchSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    client = TestClient(app)

    # Warm up client
    _ = client.post("/analyze", json={"url": test_url})

    # Measure URL-only fallback API latency
    api_fallback_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = client.post("/analyze", json={"url": test_url})
        t1 = time.perf_counter()
        assert res.status_code == 200
        api_fallback_times.append((t1 - t0) * 1000.0)

    # Measure Multimodal full API latency
    api_multi_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        res = client.post("/analyze", json={"url": test_url, "screenshot_base64": screenshot_b64})
        t1 = time.perf_counter()
        assert res.status_code == 200
        api_multi_times.append((t1 - t0) * 1000.0)

    app.dependency_overrides.clear()
    engine.dispose()
    if Path(temp_db).exists():
        try:
            Path(temp_db).unlink()
        except OSError:
            pass

    def summarize(name, times):
        times_sorted = sorted(times)
        mean_v = statistics.mean(times)
        median_v = statistics.median(times)
        p90_v = times_sorted[int(len(times_sorted) * 0.90)]
        p99_v = times_sorted[min(int(len(times_sorted) * 0.99), len(times_sorted) - 1)]
        min_v = times_sorted[0]
        max_v = times_sorted[-1]
        std_v = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"\n[{name}]")
        print(f"  Mean:   {mean_v:7.2f} ms ± {std_v:5.2f} ms")
        print(f"  Median: {median_v:7.2f} ms")
        print(f"  P90:    {p90_v:7.2f} ms")
        print(f"  P99:    {p99_v:7.2f} ms")
        print(f"  Min/Max:{min_v:7.2f} ms / {max_v:7.2f} ms")
        return {
            "mean": round(mean_v, 2),
            "median": round(median_v, 2),
            "p90": round(p90_v, 2),
            "p99": round(p99_v, 2),
            "min": round(min_v, 2),
            "max": round(max_v, 2),
            "std": round(std_v, 2),
        }

    results = {
        "url_analysis": summarize("1. URL Analysis (Extraction + RF)", url_times),
        "screenshot_decoding": summarize("2. Screenshot Base64 Ingestion & Decode", decode_times),
        "visual_feature_extraction": summarize("3. Visual Feature Extraction (30 Features)", visual_times),
        "multimodal_inference": summarize("4. Multimodal RF Inference (51 Features)", multi_times),
        "api_fallback_roundtrip": summarize("5. Full API Round-Trip (URL Fallback + DB)", api_fallback_times),
        "api_multimodal_roundtrip": summarize("6. Full API Round-Trip (Multimodal + DB)", api_multi_times),
    }

    out_file = Path("docs/benchmark_latency_metrics.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics written to {out_file}")


if __name__ == "__main__":
    run_benchmarks(iterations=50)
