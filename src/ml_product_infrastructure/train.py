"""Train and evaluate one reproducible CIFAR-10 experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from ml_product_infrastructure.settings import load_env
from ml_product_infrastructure.storage import bucket_name, s3_client


class CIFARJsonDataset:
    """Read PNG samples from a processed dataset and normalize them by train stats."""

    def __init__(
        self, root: Path, records: list[dict[str, Any]], mean: list[float], std: list[float]
    ) -> None:
        self.root = root
        self.records = records
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(self.root / record["file_name"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).to(torch.float32).div_(255.0)
        return (tensor - self.mean) / self.std, int(record["label"])


def _download_object(client: Any, bucket: str, key: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    response = client.get_object(Bucket=bucket, Key=key)
    with temporary.open("wb") as output:
        for chunk in iter(lambda: response["Body"].read(1024 * 1024), b""):
            output.write(chunk)
    temporary.replace(target)


def _ensure_dataset(config: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    data_config = config["data"]
    s3_path = str(data_config["s3_path"]).strip("/")
    root = Path(os.getenv("DATA_DIR", "data")) / s3_path
    bucket = bucket_name()
    client = s3_client()
    prefix = s3_path + "/"
    metadata = (
        "annotations/train.json",
        "annotations/val.json",
        "annotations/test.json",
        "stats.json",
    )
    for name in metadata:
        target = root / name
        if not target.is_file():
            _download_object(client, bucket, prefix + name, target)

    annotations = {
        split: json.loads((root / "annotations" / f"{split}.json").read_text(encoding="utf-8"))
        for split in ("train", "val", "test")
    }
    train_records = annotations["train"]["images"]
    max_train = data_config.get("max_train_samples")
    if max_train is not None:
        train_records = train_records[: int(max_train)]
    records = train_records + annotations["val"]["images"] + annotations["test"]["images"]
    pending = {
        record["file_name"] for record in records if not (root / record["file_name"]).is_file()
    }
    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = [
            executor.submit(_download_object, client, bucket, prefix + name, root / name)
            for name in sorted(pending)
        ]
        for index, future in enumerate(futures, start=1):
            future.result()
            if index % 1000 == 0 or index == len(futures):
                print(f"Downloaded {index}/{len(futures)} CIFAR images", flush=True)

    stats = json.loads((root / "stats.json").read_text(encoding="utf-8"))
    return root, {**annotations, "train_records": train_records}, stats


def _build_model(model_type: str, params: dict[str, Any]) -> nn.Module:
    if model_type == "linear":
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 10))
    if model_type == "simple_cnn":
        channels = [int(value) for value in params.get("channels", [16, 32])]
        if len(channels) not in {2, 3}:
            raise ValueError("simple_cnn requires two or three channel widths")
        layers: list[nn.Module] = []
        in_channels = 3
        for width in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            in_channels = width
        layers.extend([nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(in_channels, 10)])
        return nn.Sequential(*layers)
    raise ValueError(f"Unsupported model type: {model_type}")


def _scores(labels: list[int], predictions: list[int]) -> tuple[float, float, list[list[int]]]:
    matrix = [[0 for _ in range(10)] for _ in range(10)]
    for expected, predicted in zip(labels, predictions, strict=True):
        matrix[expected][predicted] += 1
    accuracy = sum(matrix[index][index] for index in range(10)) / max(len(labels), 1)
    f1_values: list[float] = []
    for label in range(10):
        true_positive = matrix[label][label]
        false_positive = sum(matrix[row][label] for row in range(10)) - true_positive
        false_negative = sum(matrix[label]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1_values.append(2 * true_positive / denominator if denominator else 0.0)
    return accuracy, sum(f1_values) / 10, matrix


def _evaluate(
    model: nn.Module, loader: DataLoader[tuple[torch.Tensor, int]], criterion: nn.Module
) -> tuple[float, float, float, list[list[int]]]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images)
            total_loss += float(criterion(logits, targets).item()) * len(targets)
            labels.extend(int(value) for value in targets.tolist())
            predictions.extend(int(value) for value in logits.argmax(dim=1).tolist())
    accuracy, macro_f1, matrix = _scores(labels, predictions)
    return total_loss / max(len(labels), 1), accuracy, macro_f1, matrix


def _write_confusion_matrix(matrix: list[list[int]], destination: Path) -> None:
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["true/pred", *range(10)])
        for label, row in enumerate(matrix):
            writer.writerow([label, *row])


def _git_commit() -> str:
    explicit = os.getenv("GIT_COMMIT")
    if explicit:
        return explicit
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def _config_params(config: dict[str, Any]) -> dict[str, str | int | float | bool]:
    params: dict[str, str | int | float | bool] = {
        "data.s3_path": str(config["data"]["s3_path"]),
        "seed": int(config["seed"]),
        "model.type": str(config["model"]["type"]),
        "train.epochs": int(config["train"]["epochs"]),
        "train.batch_size": int(config["train"]["batch_size"]),
        "train.lr": float(config["train"]["lr"]),
    }
    max_samples = config["data"].get("max_train_samples")
    if max_samples is not None:
        params["data.max_train_samples"] = int(max_samples)
    for key, value in config["model"].get("params", {}).items():
        if isinstance(value, str | int | float | bool):
            params[f"model.params.{key}"] = value
        else:
            params[f"model.params.{key}"] = json.dumps(value)
    return params


def run_training(config: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    """Execute, track, and publish one run; test data is evaluated only after model selection."""
    load_env()
    torch.set_num_threads(int(config["train"].get("num_threads", 2)))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    root, annotations, stats = _ensure_dataset(config)
    train_records = annotations["train_records"]
    val_records = annotations["val"]["images"]
    test_records = annotations["test"]["images"]
    mean = [float(value) for value in stats["mean"]]
    std = [float(value) for value in stats["std"]]
    train_data = CIFARJsonDataset(root, train_records, mean, std)
    val_data = CIFARJsonDataset(root, val_records, mean, std)
    test_data = CIFARJsonDataset(root, test_records, mean, std)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(config["train"]["batch_size"])
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    model = _build_model(str(config["model"]["type"]), config["model"].get("params", {}))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["train"]["lr"]))
    criterion = nn.CrossEntropyLoss()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(str(config["experiment_name"]))
    chosen_run_name = run_name or str(config.get("run_name", "run"))
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="cifar10-run-") as temp_name:
        work = Path(temp_name)
        model_path = work / "model.pt"
        confusion_path = work / "confusion_matrix_val.csv"
        config_path = work / "config.yaml"
        metrics_path = work / "metrics.json"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        with mlflow.start_run(run_name=chosen_run_name):
            mlflow.log_params(_config_params(config))
            mlflow.log_param("data.version", str(config["data"]["s3_path"]))
            mlflow.log_param("git.commit", _git_commit())
            mlflow.log_dict(config, "config.yaml")
            for epoch in range(1, int(config["train"]["epochs"]) + 1):
                model.train()
                for images, targets in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(images), targets)
                    loss.backward()
                    optimizer.step()
                val_loss, val_accuracy, val_f1, matrix = _evaluate(model, val_loader, criterion)
                history.append(
                    {
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_accuracy": val_accuracy,
                        "val_macro_f1": val_f1,
                    }
                )
                mlflow.log_metrics(
                    {
                        "val.loss": val_loss,
                        "val.accuracy": val_accuracy,
                        "val.macro_f1": val_f1,
                    },
                    step=epoch,
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), model_path)
                    _write_confusion_matrix(matrix, confusion_path)
                print(
                    f"{chosen_run_name}: epoch {epoch} val_loss={val_loss:.4f} "
                    f"val_accuracy={val_accuracy:.4f} val_macro_f1={val_f1:.4f}",
                    flush=True,
                )

            training_seconds = time.perf_counter() - started
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            test_loss, test_accuracy, test_f1, _ = _evaluate(model, test_loader, criterion)
            result: dict[str, Any] = {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_history": history,
                "training_seconds": training_seconds,
                "test": {
                    "loss": test_loss,
                    "accuracy": test_accuracy,
                    "macro_f1": test_f1,
                },
            }
            metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            mlflow.log_metric("best.epoch", best_epoch)
            mlflow.log_metric("training.seconds", training_seconds)
            mlflow.log_metrics(
                {
                    "test.loss": test_loss,
                    "test.accuracy": test_accuracy,
                    "test.macro_f1": test_f1,
                }
            )
            mlflow.log_artifact(str(confusion_path), artifact_path="validation")
            mlflow.log_artifact(str(model_path), artifact_path="best-model")
            mlflow.log_artifact(str(metrics_path), artifact_path="results")

        model_prefix = f"models/{config['experiment_name']}/{chosen_run_name}/"
        client = s3_client()
        bucket = bucket_name()
        for path, name in (
            (model_path, "model.pt"),
            (config_path, "config.yaml"),
            (metrics_path, "metrics.json"),
        ):
            content = path.read_bytes()
            client.put_object(
                Bucket=bucket,
                Key=model_prefix + name,
                Body=content,
                Metadata={"sha256": hashlib.sha256(content).hexdigest()},
            )
    print(
        f"Completed {chosen_run_name}: best_epoch={best_epoch} "
        f"best_val_loss={best_val_loss:.4f} test_accuracy={test_accuracy:.4f}",
        flush=True,
    )
    return result


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_training(config)


if __name__ == "__main__":
    main()
