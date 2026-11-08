"""Compute per-channel normalization statistics from the train split only."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from ml_product_infrastructure.settings import load_env


def calculate_stats(root: Path) -> dict[str, list[float]]:
    """Calculate population mean and standard deviation in the [0, 1] range."""
    annotation = json.loads((root / "annotations" / "train.json").read_text(encoding="utf-8"))
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    for record in annotation["images"]:
        with Image.open(root / record["file_name"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
        channel_sum += pixels.sum(axis=(0, 1))
        channel_square_sum += np.square(pixels).sum(axis=(0, 1))
        pixel_count += pixels.shape[0] * pixels.shape[1]
    if pixel_count == 0:
        raise ValueError("Training split is empty")
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_square_sum / pixel_count - np.square(mean), 0.0)
    result = {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}
    (root / "stats.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.getenv("DATA_DIR", "data") + "/interim/cifar10")
    args = parser.parse_args()
    print(json.dumps(calculate_stats(Path(args.root)), indent=2))


if __name__ == "__main__":
    main()
