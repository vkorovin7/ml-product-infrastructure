"""Download the official CIFAR-10 archive and store it unchanged in S3."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

from ml_product_infrastructure.settings import load_env
from ml_product_infrastructure.storage import bucket_name, object_exists, s3_client


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.getenv(
            "CIFAR10_URL", "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        ),
    )
    parser.add_argument("--object-key", default="raw/cifar-10-python.tar.gz")
    args = parser.parse_args()

    client = s3_client()
    bucket = bucket_name()
    if object_exists(client, bucket, args.object_key):
        print(f"S3 object s3://{bucket}/{args.object_key} already exists; leaving it unchanged")
        return

    with tempfile.NamedTemporaryFile(prefix="cifar10-", suffix=".tar.gz", delete=False) as temp:
        archive_path = Path(temp.name)
    try:
        with urlopen(args.url, timeout=60) as response, archive_path.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        with archive_path.open("rb") as archive:
            client.put_object(Bucket=bucket, Key=args.object_key, Body=archive)
        print(f"Uploaded unchanged archive to s3://{bucket}/{args.object_key}")
    finally:
        archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
