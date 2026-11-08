"""S3 client and immutable object helpers."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()


def s3_client() -> Any:
    """Build an S3 client from environment configuration."""
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError("Set S3_ENDPOINT_URL, S3_ACCESS_KEY, and S3_SECRET_KEY in .env")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(s3={"addressing_style": "path"}),
    )


def bucket_name() -> str:
    """Return the configured bucket name."""
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise RuntimeError("Set S3_BUCKET in .env")
    return bucket


def object_exists(client: Any, bucket: str, key: str) -> bool:
    """Check for an object without masking errors other than not-found."""
    try:
        client.head_object(Bucket=bucket, Key=key)
    except Exception as error:
        response = getattr(error, "response", {})
        code = response.get("Error", {}).get("Code")
        status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return False
        raise
    return True


def upload_tree(
    client: Any, bucket: str, root: Path, prefix: str, excluded_dirs: set[str] | None = None
) -> None:
    """Upload a directory tree concurrently beneath an object prefix."""
    excluded = excluded_dirs or set()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not excluded.intersection(path.relative_to(root).parts)
    )

    def upload(path: Path) -> None:
        key = f"{prefix.rstrip('/')}/{path.relative_to(root).as_posix()}"
        with path.open("rb") as content:
            client.put_object(Bucket=bucket, Key=key, Body=content)

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(upload, files))
