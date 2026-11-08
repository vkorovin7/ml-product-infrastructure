"""Create a deterministic stratified train/validation split from train.json."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from ml_product_infrastructure.settings import load_env


def stratified_split(
    root: Path, validation_size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Write train.json and val.json while leaving image files untouched."""
    source_annotation = root / "source" / "train.json"
    annotation_path = (
        source_annotation if source_annotation.exists() else root / "annotations" / "train.json"
    )
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    records = data["images"]
    if not 0 < validation_size < len(records):
        raise ValueError("validation_size must be between 1 and the number of train images - 1")

    by_label: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_label[int(record["label"])].append(record)
    labels = sorted(by_label)
    base, remainder = divmod(validation_size, len(labels))
    validation: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []

    for position, label in enumerate(labels):
        class_records = sorted(by_label[label], key=lambda item: item["image_id"])
        random.Random(seed + label).shuffle(class_records)
        count = base + (1 if position < remainder else 0)
        validation.extend(class_records[:count])
        train.extend(class_records[count:])

    for split, split_records in (("train", train), ("val", validation)):
        split_records.sort(key=lambda item: item["image_id"])
        target = root / "annotations" / f"{split}.json"
        target.write_text(
            json.dumps({"classes": data["classes"], "images": split_records}, indent=2) + "\n",
            encoding="utf-8",
        )
    return train, validation


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.getenv("DATA_DIR", "data") + "/interim/cifar10")
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train, validation = stratified_split(Path(args.root), args.validation_size, args.seed)
    print(f"Wrote {len(train)} train and {len(validation)} validation records")


if __name__ == "__main__":
    main()
