"""Create a manifest and publish one immutable processed dataset version to S3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ml_product_infrastructure.settings import load_env
from ml_product_infrastructure.storage import bucket_name, s3_client, upload_tree


def _file_metadata(path: Path, root: Path) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "file": path.relative_to(root).as_posix(),
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def build_manifest(root: Path, validation_size: int, seed: int) -> dict[str, Any]:
    """Create the version manifest from generated annotations and current Git revision."""
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        annotation = json.loads(
            (root / "annotations" / f"{split}.json").read_text(encoding="utf-8")
        )
        counts[split] = len(annotation["images"])
    files = [
        _file_metadata(root / "annotations" / f"{split}.json", root)
        for split in ("train", "val", "test")
    ]
    files.append(_file_metadata(root / "stats.json", root))
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "dataset": "cifar10",
        "split_counts": counts,
        "metadata_files": files,
        "processing": {"validation_size": validation_size, "seed": seed},
        "git_commit": commit,
    }


def _prefix_exists(client: Any, bucket: str, prefix: str) -> bool:
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return bool(response.get("Contents"))


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.getenv("DATA_DIR", "data") + "/interim/cifar10")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not re.fullmatch(r"v[1-9][0-9]*", args.version):
        parser.error("--version must look like v1, v2, and so on")

    root = Path(args.root)
    manifest = build_manifest(root, args.validation_size, args.seed)
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    bucket = bucket_name()
    prefix = f"processed/cifar10/{args.version}/"
    client = s3_client()
    if _prefix_exists(client, bucket, prefix):
        raise FileExistsError(
            f"s3://{bucket}/{prefix} already contains objects; choose a new --version"
        )
    upload_tree(client, bucket, root, prefix, excluded_dirs={"source"})
    print(f"Published immutable version to s3://{bucket}/{prefix}")


if __name__ == "__main__":
    main()
