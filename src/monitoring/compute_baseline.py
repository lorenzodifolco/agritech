"""
One-off script: build the Evidently reference dataset from the training set
and write models/drift_reference.parquet.  Run once after data is available via DVC.

Usage:
    python src/monitoring/compute_baseline.py
    python src/monitoring/compute_baseline.py --train-dir data/raw/train --samples 200
    python src/monitoring/compute_baseline.py --output models/drift_reference.parquet
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data.dataset import PlantDiseaseDataset          # reuse path enumeration
from src.monitoring.drift import build_reference_df        # reuse feature extraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=None)
    parser.add_argument("--samples", type=int, default=None,
                        help="Max images per class (default: read from params.yml)")
    parser.add_argument("--output", default="models/drift_reference.parquet")
    args = parser.parse_args()

    with open("configs/params.yml") as f:
        params = yaml.safe_load(f)

    train_dir = args.train_dir or params["data"]["train_dir"]
    samples_per_class = args.samples or params.get("drift", {}).get("baseline_samples_per_class", 100)
    output_path = args.output

    print(f"Loading paths from: {train_dir}")
    dataset = PlantDiseaseDataset(data_dir=train_dir, transform=None)
    print(f"Total images found: {len(dataset.image_paths)}")

    # Sample up to N per class
    from collections import defaultdict
    by_class: dict = defaultdict(list)
    for path, label in zip(dataset.image_paths, dataset.labels):
        by_class[label].append(path)

    sampled = []
    for paths in by_class.values():
        sampled.extend(paths[:samples_per_class])
    print(f"Sampled {len(sampled)} images ({samples_per_class} per class max)")

    df = build_reference_df(sampled)
    print(f"Reference DataFrame shape: {df.shape}")
    print(df.describe())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Reference dataset written to {output_path}")


if __name__ == "__main__":
    main()
