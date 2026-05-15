import json
import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import EvidentlyDriftDetector, build_reference_df, _extract_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURES = ("brightness", "contrast", "blur")


def _make_reference(n: int = 100, brightness_range=(100, 160)) -> pd.DataFrame:
    """Reference DataFrame with controlled brightness distribution."""
    rng = np.random.default_rng(0)
    lo, hi = brightness_range
    return pd.DataFrame({
        "brightness": rng.uniform(lo, hi, n),
        "contrast":   rng.uniform(40, 80, n),
        "blur":       rng.uniform(200, 400, n),
    })


def _uniform_image(value: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _detector_from_df(ref_df: pd.DataFrame, tmp_path, buffer_size: int = 30) -> EvidentlyDriftDetector:
    path = str(tmp_path / "drift_reference.parquet")
    ref_df.to_parquet(path, index=False)
    return EvidentlyDriftDetector(path, buffer_size=buffer_size)


# ---------------------------------------------------------------------------
# _extract_features
# ---------------------------------------------------------------------------

def test_extract_features_returns_all_keys():
    img = _uniform_image(128)
    feats = _extract_features(img)
    assert set(feats.keys()) == {"brightness", "contrast", "blur"}
    assert all(isinstance(v, float) for v in feats.values())


def test_extract_features_brightness_range():
    assert _extract_features(_uniform_image(0))["brightness"] == pytest.approx(0.0)
    assert _extract_features(_uniform_image(255))["brightness"] == pytest.approx(255.0)


# ---------------------------------------------------------------------------
# build_reference_df
# ---------------------------------------------------------------------------

def test_build_reference_df_schema(tmp_path):
    from PIL import Image as PILImage
    paths = []
    for i in range(10):
        img = PILImage.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        p = tmp_path / f"img_{i}.jpg"
        img.save(str(p))
        paths.append(str(p))

    df = build_reference_df(paths)
    assert list(df.columns) == list(_FEATURES)
    assert len(df) == 10
    assert df.dtypes["brightness"] == np.float64 or df.dtypes["brightness"] == float


def test_build_reference_df_samples_cap(tmp_path):
    from PIL import Image as PILImage
    paths = []
    for i in range(20):
        img = PILImage.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        p = tmp_path / f"img_{i}.jpg"
        img.save(str(p))
        paths.append(str(p))

    df = build_reference_df(paths, samples=7)
    assert len(df) == 7


# ---------------------------------------------------------------------------
# EvidentlyDriftDetector — buffer filling
# ---------------------------------------------------------------------------

def test_no_result_before_buffer_full(tmp_path):
    """Detector must return drifted=False while buffer has < 5 observations."""
    detector = _detector_from_df(_make_reference(), tmp_path, buffer_size=30)
    for _ in range(4):
        result = detector.check(_uniform_image(128))
    assert result["drifted"] is False
    assert result["report"] == {}


# ---------------------------------------------------------------------------
# EvidentlyDriftDetector — drift detection
# ---------------------------------------------------------------------------

def test_no_drift_same_distribution(tmp_path):
    """Images drawn from the same distribution as reference must not be flagged."""
    rng = np.random.default_rng(0)

    # Build reference from random images with mid-range brightness
    ref_rows = []
    for _ in range(200):
        img = (rng.uniform(100, 160, (64, 64, 3))).astype(np.uint8)
        ref_rows.append(_extract_features(img))
    ref = pd.DataFrame(ref_rows)

    detector = _detector_from_df(ref, tmp_path, buffer_size=10)

    # Feed 11 images from the same distribution (random, same brightness range)
    rng2 = np.random.default_rng(1)
    result = None
    for _ in range(11):
        img = (rng2.uniform(100, 160, (64, 64, 3))).astype(np.uint8)
        result = detector.check(img)

    assert result["drifted"] is False


def test_drift_detected_very_dark_images(tmp_path):
    """Images far outside the reference distribution must trigger drift."""
    ref = _make_reference(200, brightness_range=(100, 160))
    detector = _detector_from_df(ref, tmp_path, buffer_size=10)

    # Feed 10 very dark images — far from reference brightness of 100–160
    for _ in range(11):
        result = detector.check(_uniform_image(5))

    assert result["drifted"] is True
    assert result["report"]["drifted_columns"] >= 1


# ---------------------------------------------------------------------------
# runtime._log (unchanged logic, still worth verifying)
# ---------------------------------------------------------------------------

def test_log_written(tmp_path, monkeypatch):
    import src.models.runtime as rt_module
    from unittest.mock import MagicMock

    log_path = str(tmp_path / "predictions.jsonl")
    monkeypatch.setattr(rt_module, "_LOG_PATH", log_path)

    runtime = rt_module.PlantDiseaseRuntime(MagicMock())
    runtime._log({"disease": "Apple___Apple_scab", "confidence": 95.0, "top3": []})
    runtime._log({"disease": "Healthy", "confidence": 40.0, "top3": []})

    lines = (tmp_path / "predictions.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["disease"] == "Apple___Apple_scab"
    assert json.loads(lines[1])["confidence"] == 40.0
