"""Download the immutable raw CIFAR archive from S3 to the local data/raw folder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ml_product_infrastructure.settings import load_env
from ml_product_infrastructure.storage import bucket_name, s3_client


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-key", default="raw/cifar-10-python.tar.gz")
    parser.add_argument(
        "--output", default=os.getenv("DATA_DIR", "data") + "/raw/cifar-10-python.tar.gz"
    )
    args = parser.parse_args()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    s3_client().download_file(bucket_name(), args.object_key, str(destination))
    print(f"Downloaded s3://{bucket_name()}/{args.object_key} to {destination}")


if __name__ == "__main__":
    main()
