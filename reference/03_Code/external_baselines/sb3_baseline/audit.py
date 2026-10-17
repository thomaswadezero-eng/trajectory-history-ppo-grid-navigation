from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import gymnasium
import numpy
import sb3_contrib
import stable_baselines3
import torch

from .core import IntegrityError, read_json, sha256_file


EXPECTED_VERSIONS = {
    "python": "3.10.11",
    "torch": "2.8.0+cu128",
    "gymnasium": "1.3.0",
    "stable_baselines3": "2.9.0",
    "sb3_contrib": "2.9.0",
}


def environment_identity() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "sb3_contrib": sb3_contrib.__version__,
        "numpy": numpy.__version__,
        "cloudpickle": importlib.metadata.version("cloudpickle"),
    }


def verify_environment(device: str) -> dict[str, Any]:
    identity = environment_identity()
    for key, expected in EXPECTED_VERSIONS.items():
        if identity[key] != expected:
            raise IntegrityError(f"environment version mismatch: {key}={identity[key]} expected={expected}")
    if device.startswith("cuda") and not bool(identity["cuda_available"]):
        raise IntegrityError("CUDA device requested but torch.cuda.is_available() is false")
    return identity


def verify_frozen_files(project_root: Path, config: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, record in config["inputs"].items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        path = (project_root / str(record["path"])).resolve()
        if not path.is_file():
            raise IntegrityError(f"missing frozen input: {path}")
        actual = sha256_file(path)
        if actual != str(record["sha256"]).upper():
            raise IntegrityError(f"frozen input SHA mismatch: {path}; actual={actual}")
        result[name] = actual
    return result


def source_manifest(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path.resolve()): sha256_file(path) for path in sorted(paths)}


def load_config(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if payload.get("status") != "AUTHORIZED_FOR_MANUAL_EXECUTION_ONLY":
        raise IntegrityError("external baseline config is not authorized for manual execution")
    if payload["reward"].get("dynamic_reward") is not False or payload["reward"].get("astar") is not False:
        raise IntegrityError("external baseline must disable Dynamic Reward and A*")
    return payload
