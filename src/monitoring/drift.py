import os
from collections import deque

import numpy as np
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

_FEATURES = ("brightness", "contrast", "blur")


class EvidentlyDriftDetector:
    """
    Drift detector backed by Evidently AI.

    Maintains a rolling buffer of recent image feature vectors and compares
    them against the training-set reference DataFrame using Evidently's
    DataDriftPreset (Z-test / KS-test per column).

    Reference data is loaded from a parquet file produced by compute_baseline.py.
    """

    def __init__(self, reference_path: str, buffer_size: int = 30):
        self._reference = pd.read_parquet(reference_path)[list(_FEATURES)]
        self._buffer: deque = deque(maxlen=buffer_size)
        self._last_report: dict = {}

    # ------------------------------------------------------------------
    # Public API (same signature as the previous DriftDetector.check())
    # ------------------------------------------------------------------

    def check(self, image: np.ndarray) -> dict:
        """
        Args:
            image: uint8 RGB array of shape (H, W, 3).
        Returns:
            {"drifted": bool, "report": {Evidently summary dict}}
        Drift is evaluated once the buffer has at least 5 observations;
        returns drifted=False with an empty report until then.
        """
        self._buffer.append(_extract_features(image))

        if len(self._buffer) < 5:
            return {"drifted": False, "report": {}}

        current = pd.DataFrame(list(self._buffer), columns=list(_FEATURES))
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=self._reference, current_data=current)
        result = report.as_dict()["metrics"][0]["result"]

        self._last_report = result
        return {
            "drifted": bool(result["dataset_drift"]),
            "report": {
                "drifted_columns": result["number_of_drifted_columns"],
                "total_columns": result["number_of_columns"],
                "drift_share": result["share_of_drifted_columns"],
            },
        }

    def last_report_dict(self) -> dict:
        return self._last_report


# ------------------------------------------------------------------
# Feature extraction (used by detector and compute_baseline.py)
# ------------------------------------------------------------------

def _extract_features(image: np.ndarray) -> dict:
    """Extract brightness, contrast, blur scalars from a uint8 RGB array."""
    gray = np.mean(image, axis=2).astype(np.float32)
    lap = (
        np.roll(gray, -1, 0) + np.roll(gray, 1, 0) +
        np.roll(gray, -1, 1) + np.roll(gray, 1, 1) -
        4 * gray
    )
    return {
        "brightness": float(np.mean(image)),
        "contrast":   float(np.std(image)),
        "blur":       float(np.var(lap)),
    }


def build_reference_df(image_paths: list, samples: int = None) -> pd.DataFrame:
    """
    Build a reference DataFrame from a list of image paths.
    Called by compute_baseline.py — kept here so tests can import it directly.
    """
    if samples is not None:
        rng = np.random.default_rng(42)
        image_paths = rng.choice(
            image_paths, size=min(samples, len(image_paths)), replace=False
        ).tolist()

    rows = []
    for path in image_paths:
        img = _load_rgb(path)
        if img is not None:
            rows.append(_extract_features(img))

    return pd.DataFrame(rows, columns=list(_FEATURES))


def _load_rgb(path: str):
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        return None
