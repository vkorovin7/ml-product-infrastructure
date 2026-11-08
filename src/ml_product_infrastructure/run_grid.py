"""Expand a YAML hyperparameter grid and train every combination in one MLflow experiment."""

from __future__ import annotations

import argparse
import copy
import itertools
import re
from pathlib import Path
from typing import Any

import yaml

from ml_product_infrastructure.settings import load_env
from ml_product_infrastructure.train import run_training


def _set_dotted_value(config: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    target = config
    for part in parts[:-1]:
        next_value = target.get(part)
        if not isinstance(next_value, dict):
            raise ValueError(f"Grid path {path!r} does not point into a mapping")
        target = next_value
    if parts[-1] not in target:
        raise ValueError(f"Grid path {path!r} does not exist in the base config")
    target[parts[-1]] = value


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def expand_grid(grid_path: Path) -> list[dict[str, Any]]:
    grid_config = yaml.safe_load(grid_path.read_text(encoding="utf-8"))
    base_path = (grid_path.parent / grid_config["base"]).resolve()
    base_config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    axes = grid_config["grid"]
    if not axes or any(not values for values in axes.values()):
        raise ValueError("Grid must define at least one non-empty parameter list")
    keys = list(axes)
    experiment_name = str(grid_config.get("experiment_name", base_config["experiment_name"]))
    variants: list[dict[str, Any]] = []
    for values in itertools.product(*(axes[key] for key in keys)):
        config = copy.deepcopy(base_config)
        config["experiment_name"] = experiment_name
        tokens: list[str] = [str(config["model"]["type"])]
        for key, value in zip(keys, values, strict=True):
            _set_dotted_value(config, key, value)
            leaf = key.split(".")[-1]
            tokens.append(f"{leaf}{_format_value(value)}")
        tokens.append(f"seed{config['seed']}")
        config["run_name"] = re.sub(r"[^A-Za-z0-9_.-]+", "-", "-".join(tokens))
        variants.append(config)
    return variants


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    variants = expand_grid(args.config)
    print(f"Running {len(variants)} experiments in {variants[0]['experiment_name']}", flush=True)
    for index, config in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] {config['run_name']}", flush=True)
        run_training(config)


if __name__ == "__main__":
    main()
