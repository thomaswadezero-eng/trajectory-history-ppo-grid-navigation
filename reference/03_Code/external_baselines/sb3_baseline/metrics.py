from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    task_id: str
    map_id: int
    success: int
    collision: int
    timeout: int
    steps: int
    path_length: int
    optimal_path_length: int
    spl: float
    path_efficiency: float

    @classmethod
    def from_outcome(
        cls,
        *,
        episode: int,
        task_id: str,
        map_id: int,
        success: bool,
        collision: bool,
        timeout: bool,
        steps: int,
        path_length: int,
        optimal_path_length: int,
    ) -> "EpisodeMetrics":
        success_i = int(bool(success))
        collision_i = int(bool(collision))
        timeout_i = int(bool(timeout))
        if success_i + collision_i + timeout_i != 1:
            raise ValueError("episode outcomes must be mutually exclusive and exhaustive")
        denominator = float(max(int(optimal_path_length), int(path_length)))
        spl = success_i * int(optimal_path_length) / denominator
        efficiency = int(optimal_path_length) / denominator if success_i else 0.0
        return cls(
            episode=int(episode),
            task_id=str(task_id),
            map_id=int(map_id),
            success=success_i,
            collision=collision_i,
            timeout=timeout_i,
            steps=int(steps),
            path_length=int(path_length),
            optimal_path_length=int(optimal_path_length),
            spl=float(spl),
            path_efficiency=float(efficiency),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def aggregate_episode_metrics(rows: Iterable[EpisodeMetrics]) -> dict[str, Any]:
    episodes = list(rows)
    if not episodes:
        raise ValueError("cannot aggregate zero episodes")
    count = len(episodes)
    success_count = sum(row.success for row in episodes)
    collision_count = sum(row.collision for row in episodes)
    timeout_count = sum(row.timeout for row in episodes)
    if success_count + collision_count + timeout_count != count:
        raise ValueError("aggregate outcomes are not exhaustive")
    successful_efficiencies = [row.path_efficiency for row in episodes if row.success]
    return {
        "episode_count": count,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "success_rate": success_count / count,
        "collision_rate": collision_count / count,
        "timeout_rate": timeout_count / count,
        "average_steps": sum(row.steps for row in episodes) / count,
        "average_path_length": sum(row.path_length for row in episodes) / count,
        "spl": float(np.mean([row.spl for row in episodes])),
        "path_efficiency": float(np.mean(successful_efficiencies)) if successful_efficiencies else 0.0,
        "success_rate_fraction": f"{success_count}/{count}",
        "collision_rate_fraction": f"{collision_count}/{count}",
        "timeout_rate_fraction": f"{timeout_count}/{count}",
    }


def checkpoint_key(metrics: Mapping[str, Any], interaction: int) -> tuple[float, ...]:
    return (
        -float(metrics["success_rate"]),
        -float(metrics["spl"]),
        float(metrics["timeout_rate"]),
        float(metrics["average_steps"]),
        float(interaction),
    )
