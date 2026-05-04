import json
import os

TRAIN_DIR = "data/raw/train"
OUTPUT_PATH = "src/models/class_names.json"


def save_class_names():
    class_names = sorted(os.listdir(TRAIN_DIR))
    with open(OUTPUT_PATH, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"Saved {len(class_names)} class names to {OUTPUT_PATH}")


if __name__ == "__main__":
    save_class_names()
