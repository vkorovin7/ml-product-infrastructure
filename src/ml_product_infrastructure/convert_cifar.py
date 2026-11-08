"""Convert the trusted official CIFAR-10 pickle archive into PNGs and JSON labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ml_product_infrastructure.settings import load_env


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_archive(archive_path: Path, output_dir: Path) -> None:
    """Decode official CIFAR batches and write images plus train/test annotations."""
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    source_dir = output_dir / "source"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    classes: list[str] = []
    train_index = 0
    test_index = 0

    with tarfile.open(archive_path, "r:gz") as archive:
        members = sorted(
            (member for member in archive.getmembers() if member.isfile()),
            key=lambda member: member.name,
        )
        metadata_member = next(
            member for member in members if member.name.endswith("batches.meta")
        )
        metadata_stream = archive.extractfile(metadata_member)
        if metadata_stream is None:
            raise ValueError("CIFAR archive is missing batches.meta")
        metadata = pickle.load(metadata_stream, encoding="bytes")
        classes = [name.decode("utf-8") for name in metadata[b"label_names"]]

        batch_members = [
            member
            for member in members
            if "/data_batch_" in member.name or member.name.endswith("test_batch")
        ]
        for member in batch_members:
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unable to read archive member {member.name}")
            batch = pickle.load(stream, encoding="bytes")
            data = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
            data = data.transpose(0, 2, 3, 1)
            labels = batch.get(b"labels", batch.get(b"fine_labels"))
            if labels is None:
                raise ValueError(f"No labels in archive member {member.name}")
            is_test = member.name.endswith("test_batch")
            split = "test" if is_test else "train"
            for pixels, label_value in zip(data, labels, strict=True):
                image_index = test_index if is_test else train_index
                image_id = f"{split}_{image_index:05d}"
                image_path = images_dir / f"{image_id}.png"
                Image.fromarray(pixels, mode="RGB").save(image_path, format="PNG")
                label = int(label_value)
                split_records[split].append(
                    {
                        "image_id": image_id,
                        "file_name": f"images/{image_id}.png",
                        "label": label,
                        "label_name": classes[label],
                        "sha256": _sha256(image_path),
                    }
                )
                if is_test:
                    test_index += 1
                else:
                    train_index += 1

    for split, records in split_records.items():
        target_dir = source_dir if split == "train" else annotations_dir
        target = target_dir / f"{split}.json"
        target.write_text(
            json.dumps({"classes": classes, "images": records}, ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", default=os.getenv("DATA_DIR", "data") + "/raw/cifar-10-python.tar.gz"
    )
    parser.add_argument("--output-dir", default=os.getenv("DATA_DIR", "data") + "/interim/cifar10")
    args = parser.parse_args()
    convert_archive(Path(args.archive), Path(args.output_dir))


if __name__ == "__main__":
    main()
