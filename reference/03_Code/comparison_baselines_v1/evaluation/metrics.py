from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from common.errors import IntegrityError


def episode_record(
    task: Mapping[str, Any],
    trajectory: Sequence[Sequence[int]],
    steps: int,
    success: bool,
    collision: bool,
    timeout: bool,
) -> Dict[str, Any]:
    outcomes = int(bool(success)) + int(bool(collision)) + int(bool(timeout))
    if outcomes != 1:
        raise IntegrityError("episode outcome must be exactly one of success/collision/timeout")
    path_length = sum(tuple(trajectory[i]) != tuple(trajectory[i - 1]) for i in range(1, len(trajectory)))
    optimal = int(task["shortest_path_length"])
    if optimal <= 0:
        raise IntegrityError("shortest path length must be positive")
    denominator = max(optimal, path_length)
    spl = int(bool(success)) * optimal / float(denominator)
    efficiency = optimal / float(denominator) if success else None
    return {
        "task_id": str(task["task_id"]),
        "map_id": int(task["map_id"]),
        "success": int(bool(success)),
        "collision": int(bool(collision)),
        "timeout": int(bool(timeout)),
        "SPL": float(spl),
        "path_length": int(path_length),
        "steps": int(steps),
        "optimal_path_length": optimal,
        "path_efficiency": efficiency,
    }


def aggregate_episode_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        raise IntegrityError("cannot aggregate an empty evaluation")
    count = len(records)
    success_count = sum(int(row["success"]) for row in records)
    collision_count = sum(int(row["collision"]) for row in records)
    timeout_count = sum(int(row["timeout"]) for row in records)
    if success_count + collision_count + timeout_count != count:
        raise IntegrityError("evaluation outcomes are not exhaustive")
    efficiencies = [float(row["path_efficiency"]) for row in records if row["path_efficiency"] is not None]
    return {
        "episode_count": count,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "success_rate": success_count / float(count),
        "collision_rate": collision_count / float(count),
        "timeout_rate": timeout_count / float(count),
        "average_steps": sum(int(row["steps"]) for row in records) / float(count),
        "average_path_length": sum(int(row["path_length"]) for row in records) / float(count),
        "spl": float(np.mean([float(row["SPL"]) for row in records])),
        "path_efficiency": float(np.mean(efficiencies)) if efficiencies else 0.0,
        "path_efficiency_success_count": len(efficiencies),
    }


def per_map_aggregates(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[int(row["map_id"])].append(row)
    rows = []
    for map_id in sorted(grouped):
        summary = aggregate_episode_records(grouped[map_id])
        rows.append({"map_id": map_id, **summary})
    return rows


def checkpoint_key(metrics: Mapping[str, Any], interaction: int) -> Tuple[float, ...]:
    return (
        -float(metrics["success_rate"]),
        -float(metrics["spl"]),
        float(metrics["timeout_rate"]),
        float(metrics["average_steps"]),
        float(interaction),
    )

