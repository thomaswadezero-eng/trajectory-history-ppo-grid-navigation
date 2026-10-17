from __future__ import annotations

import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from common.errors import IntegrityError, ProtocolError
from common.io_utils import manifest_digest, read_json, sha256_file


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = FRAMEWORK_ROOT.parents[1]
PROTOCOL_PATH = FRAMEWORK_ROOT / "comparison_protocol_v1.json"
FRAMEWORK_MANIFEST_PATH = FRAMEWORK_ROOT / "framework_hash_manifest.json"
METHODS = ("DDQN", "D3QN", "A2C")
OFFICIAL_SEEDS = (42, 1004, 6010)


@dataclass(frozen=True)
class ProtocolBundle:
    protocol: Dict[str, Any]
    config: Dict[str, Any]
    protocol_sha256: str
    config_sha256: str
    framework_sha256: str


def method_config_path(method: str) -> Path:
    if method not in METHODS:
        raise ProtocolError(f"unsupported method: {method}")
    return FRAMEWORK_ROOT / "configs" / f"{method.lower()}_v1.json"


def set_seed(seed: int) -> None:
    if seed < 0 or seed >= 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_project_path(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ProtocolError(f"project-relative path escapes project: {relative}") from exc
    return path


def verify_framework_manifest() -> str:
    manifest = read_json(FRAMEWORK_MANIFEST_PATH)
    if manifest.get("schema_version") != "comparison_baselines.framework_manifest.v1":
        raise ProtocolError("unsupported framework manifest schema")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise ProtocolError("framework manifest has no files")
    for record in records:
        path = _resolve_project_path(str(record["path"]))
        if not path.is_file():
            raise IntegrityError(f"missing framework file: {path}")
        actual = sha256_file(path)
        if actual != str(record["sha256"]).upper():
            raise IntegrityError(f"framework file SHA mismatch: {path}")
    actual_aggregate = manifest_digest(records)
    if actual_aggregate != str(manifest.get("aggregate_sha256", "")).upper():
        raise IntegrityError("framework aggregate SHA mismatch")
    return actual_aggregate


def load_bundle(method: str) -> ProtocolBundle:
    protocol = read_json(PROTOCOL_PATH)
    config_path = method_config_path(method)
    config = read_json(config_path)
    protocol_sha = sha256_file(PROTOCOL_PATH)
    config_sha = sha256_file(config_path)
    if protocol.get("schema_version") != "comparison_baselines.protocol.v1":
        raise ProtocolError("unsupported comparison protocol schema")
    if protocol.get("status") != "FROZEN_BEFORE_FORMAL_RUNS":
        raise ProtocolError("comparison protocol is not frozen")
    if tuple(protocol.get("official_seeds", [])) != OFFICIAL_SEEDS:
        raise ProtocolError("official seeds mismatch")
    if config.get("schema_version") != "comparison_baselines.method_config.v1":
        raise ProtocolError("unsupported method configuration schema")
    if config.get("status") != "FROZEN_BEFORE_FORMAL_RUNS" or config.get("method") != method:
        raise ProtocolError("method configuration identity mismatch")
    expected_hash = str(protocol["method_configs"][method]["sha256"]).upper()
    if config_sha != expected_hash:
        raise IntegrityError(f"frozen {method} configuration SHA mismatch")
    if int(protocol["training"]["interaction_budget"]) != 900000:
        raise ProtocolError("interaction budget mismatch")
    if int(protocol["training"]["validation_interval_interactions"]) != 100000:
        raise ProtocolError("validation interval mismatch")
    if protocol["innovations"]["baseline_dynamic_reward"] is not False:
        raise ProtocolError("Dynamic Reward must be disabled for baselines")
    if protocol["innovations"]["baseline_astar_planning_prior"] is not False:
        raise ProtocolError("A* planning prior must be disabled for baselines")
    if protocol["innovations"]["baseline_curriculum"] is not False:
        raise ProtocolError("curriculum must be disabled for baselines")
    framework_sha = verify_framework_manifest()
    return ProtocolBundle(protocol, config, protocol_sha, config_sha, framework_sha)


def verify_v7_authority(protocol: Mapping[str, Any]) -> Dict[str, str]:
    verified: Dict[str, str] = {}
    for name, record in protocol["v7_authority"]["files"].items():
        path = _resolve_project_path(str(record["path"]))
        actual = sha256_file(path)
        if actual != str(record["sha256"]).upper():
            raise IntegrityError(f"V7 authority SHA mismatch for {name}: {path}")
        verified[name] = actual
    return verified


def _load_task_file(record: Mapping[str, Any]) -> List[Dict[str, Any]]:
    path = _resolve_project_path(str(record["path"]))
    actual = sha256_file(path)
    if actual != str(record["sha256"]).upper():
        raise IntegrityError(f"task file SHA mismatch: {path}")
    payload = read_json(path)
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise IntegrityError(f"task file lacks a tasks list: {path}")
    return [dict(task) for task in tasks]


def validate_task_splits(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    split_spec: Mapping[str, Any],
    max_steps: int,
) -> Dict[str, Any]:
    all_ids: List[str] = []
    map_sets: Dict[str, set[int]] = {}
    summary: Dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        tasks = list(groups[split])
        expected = split_spec[split]
        expected_maps = {int(x) for x in expected["map_ids"]}
        ids = [str(task["task_id"]) for task in tasks]
        maps = {int(task["map_id"]) for task in tasks}
        if len(tasks) != int(expected["task_count"]):
            raise IntegrityError(f"task count mismatch for {split}")
        if len(ids) != len(set(ids)):
            raise IntegrityError(f"duplicate task IDs in {split}")
        if maps != expected_maps:
            raise IntegrityError(f"map membership mismatch for {split}")
        if any(str(task["split"]) != split for task in tasks):
            raise IntegrityError(f"split label mismatch for {split}")
        if any(int(task["max_steps"]) != max_steps for task in tasks):
            raise IntegrityError(f"max_steps mismatch for {split}")
        if any(int(task["shortest_path_length"]) <= 0 for task in tasks):
            raise IntegrityError(f"invalid shortest path length in {split}")
        map_sets[split] = maps
        all_ids.extend(ids)
        summary[split] = {
            "task_count": len(tasks),
            "map_count": len(maps),
            "map_ids": sorted(maps),
            "unique_task_ids": len(set(ids)),
        }
    if len(all_ids) != len(set(all_ids)):
        raise IntegrityError("task IDs overlap across splits")
    if map_sets["train"] & map_sets["validation"]:
        raise IntegrityError("train and validation maps overlap")
    if map_sets["train"] & map_sets["test"]:
        raise IntegrityError("train and test maps overlap")
    if map_sets["validation"] & map_sets["test"]:
        raise IntegrityError("validation and test maps overlap")
    summary["task_id_overlap_count"] = 0
    summary["map_overlap_count"] = 0
    return summary


def load_and_verify_tasks(protocol: Mapping[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    groups = {
        split: _load_task_file(protocol["tasks"][split])
        for split in ("train", "validation", "test")
    }
    summary = validate_task_splits(
        groups,
        protocol["splits"],
        int(protocol["environment"]["max_steps"]),
    )
    return groups, summary


def cspace_manifest(groups: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    unique: Dict[str, Path] = {}
    for tasks in groups.values():
        for task in tasks:
            path = Path(str(task["cspace_file"])).resolve()
            unique[str(path).lower()] = path
    records = []
    for path in sorted(unique.values(), key=lambda item: str(item).lower()):
        if not path.is_file():
            raise IntegrityError(f"missing cspace file: {path}")
        records.append({"path": str(path), "sha256": sha256_file(path)})
    return {
        "hash_algorithm": "SHA-256",
        "file_count": len(records),
        "aggregate_sha256": manifest_digest(records),
        "files": records,
    }


def runtime_identity(device: torch.device) -> Dict[str, Any]:
    cuda_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "device_name": cuda_name,
    }

