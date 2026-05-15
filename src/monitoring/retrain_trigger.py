"""
Retraining trigger: reads the rolling prediction log, evaluates drift-alert rate
and mean confidence, and calls train.train() if thresholds are exceeded.

Usage:
    python src/monitoring/retrain_trigger.py
    python src/monitoring/retrain_trigger.py --dry-run
    python src/monitoring/retrain_trigger.py --log logs/predictions.jsonl --window 200
"""
import argparse
import json
import os
import sys

import mlflow
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _load_window(log_path: str, window: int) -> list:
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return [json.loads(l) for l in lines[-window:]]


def _evaluate(records: list, confidence_threshold: float, drift_alert_rate_threshold: float) -> dict:
    if not records:
        return {"triggered": False, "reason": "no prediction logs found", "n": 0}

    n = len(records)
    drifted = sum(1 for r in records if r.get("drift", {}).get("drifted", False))
    low_conf = sum(1 for r in records if r.get("confidence", 100) / 100 < confidence_threshold)

    drift_rate = drifted / n
    low_conf_rate = low_conf / n

    reasons = []
    if drift_rate > drift_alert_rate_threshold:
        reasons.append(f"drift_alert_rate={drift_rate:.2%} > {drift_alert_rate_threshold:.2%}")
    if low_conf_rate > drift_alert_rate_threshold:
        reasons.append(f"low_confidence_rate={low_conf_rate:.2%} > {drift_alert_rate_threshold:.2%}")

    return {
        "triggered": bool(reasons),
        "reason": "; ".join(reasons) if reasons else "metrics within bounds",
        "n": n,
        "drift_alert_rate": drift_rate,
        "low_confidence_rate": low_conf_rate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="logs/predictions.jsonl")
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Evaluate thresholds but do not retrain")
    args = parser.parse_args()

    with open("configs/params.yml") as f:
        params = yaml.safe_load(f)

    drift_cfg = params.get("drift", {})
    window = args.window or drift_cfg.get("window_size", 100)
    conf_thr = drift_cfg.get("confidence_threshold", 0.70)
    rate_thr = drift_cfg.get("drift_alert_rate_threshold", 0.20)

    records = _load_window(args.log, window)
    result = _evaluate(records, conf_thr, rate_thr)

    print(f"Evaluated {result['n']} predictions (window={window})")
    print(f"  drift_alert_rate : {result.get('drift_alert_rate', 'n/a')}")
    print(f"  low_conf_rate    : {result.get('low_confidence_rate', 'n/a')}")
    print(f"  triggered        : {result['triggered']}")
    print(f"  reason           : {result['reason']}")

    if not result["triggered"]:
        print("No retraining needed.")
        return

    if args.dry_run:
        print("[dry-run] Would trigger retraining — skipping.")
        return

    print("Triggering retraining...")
    # Reuse train.train() directly — no code duplication
    from src.train import train  # noqa: PLC0415

    mlflow.set_tracking_uri(
        params.get("mlflow", {}).get(
            "tracking_uri",
            "https://dagshub.com/lorenzodifolco00/agritech.mlflow/"
        )
    )
    mlflow.set_experiment("Plant-Disease-Classification")

    with mlflow.start_run(run_name="retrain-triggered") as run:
        mlflow.log_param("trigger_reason", result["reason"])
        mlflow.log_param("trigger_window", result["n"])
        mlflow.log_metric("drift_alert_rate", result.get("drift_alert_rate", 0))
        mlflow.log_metric("low_confidence_rate", result.get("low_confidence_rate", 0))

        # Log the last Evidently drift report as an artifact for traceability
        last_record = _load_window(args.log, 1)
        if last_record and "drift" in last_record[0]:
            mlflow.log_dict(last_record[0]["drift"], "evidently_drift_report.json")

        print(f"MLflow run: {run.info.run_id}")

    train()
    print("Retraining complete.")


if __name__ == "__main__":
    main()
