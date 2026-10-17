from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from .core import BarnNavigationEnv, MapStore, TaskRecord, write_json_new
from .metrics import EpisodeMetrics, aggregate_episode_metrics


def model_parameter_sha256(model: MaskablePPO) -> str:
    digest = hashlib.sha256()
    state = model.policy.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def evaluate_tasks(
    *,
    model: MaskablePPO,
    tasks: Sequence[TaskRecord],
    split: str,
    seed: int,
    output_dir: Path,
    checkpoint_interaction: int,
    record_trajectories: bool,
    map_store: MapStore | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    trajectory_dir = output_dir / "trajectories"
    if record_trajectories:
        trajectory_dir.mkdir(parents=False, exist_ok=False)
    store = map_store or MapStore(30)
    env = BarnNavigationEnv(tasks, split, seed, map_provider=store.get, training_sampling=False)
    before_sha = model_parameter_sha256(model)
    rows: list[EpisodeMetrics] = []
    started = time.perf_counter()
    csv_path = output_dir / "evaluation_metrics.csv"
    with csv_path.open("x", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "record_type",
            "split",
            "checkpoint_interaction",
            "episode",
            "task_id",
            "map_id",
            "success",
            "collision",
            "timeout",
            "SPL",
            "path_efficiency",
            "path_length",
            "steps",
            "optimal_path_length",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "average_steps",
            "average_path_length",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode, task in enumerate(tasks, 1):
            observation, _ = env.reset(options={"task": task})
            trajectory = [list(env.builder.position)]
            states: list[dict[str, Any]] = [
                {
                    "step": 0,
                    "position": list(env.builder.position),
                    "history_state": observation["history"].tolist(),
                }
            ]
            terminated = truncated = False
            final: dict[str, Any] = {}
            with torch.no_grad():
                while not (terminated or truncated):
                    mask = env.action_masks()
                    action, _ = model.predict(
                        observation,
                        action_masks=mask,
                        deterministic=True,
                    )
                    states[-1]["action_mask"] = mask.astype(int).tolist()
                    states[-1]["selected_action"] = int(action)
                    observation, _, terminated, truncated, final = env.step(int(action))
                    trajectory.append(list(env.builder.position))
                    states.append(
                        {
                            "step": env.builder.step_count,
                            "position": list(env.builder.position),
                            "history_state": observation["history"].tolist(),
                        }
                    )
            row = EpisodeMetrics.from_outcome(
                episode=episode,
                task_id=task.task_id,
                map_id=task.map_id,
                success=bool(final["success"]),
                collision=bool(final["collision"]),
                timeout=bool(final["timeout"]),
                steps=env.builder.step_count,
                path_length=int(final["path_length"]),
                optimal_path_length=task.shortest_path_length,
            )
            rows.append(row)
            writer.writerow(
                {
                    "record_type": "episode",
                    "split": split,
                    "checkpoint_interaction": checkpoint_interaction,
                    "episode": row.episode,
                    "task_id": row.task_id,
                    "map_id": row.map_id,
                    "success": row.success,
                    "collision": row.collision,
                    "timeout": row.timeout,
                    "SPL": row.spl,
                    "path_efficiency": row.path_efficiency if row.success else "",
                    "path_length": row.path_length,
                    "steps": row.steps,
                    "optimal_path_length": row.optimal_path_length,
                    "success_rate": "",
                    "collision_rate": "",
                    "timeout_rate": "",
                    "average_steps": "",
                    "average_path_length": "",
                }
            )
            if record_trajectories:
                grid = env.grid
                assert grid is not None
                write_json_new(
                    trajectory_dir / f"episode_{episode:04d}.json",
                    {
                        "episode": episode,
                        "task_id": task.task_id,
                        "map_id": task.map_id,
                        "start_position": list(task.start),
                        "goal_position": list(task.goal),
                        "obstacle_coordinates": np.argwhere(grid == 1).tolist(),
                        "state_trajectory": states,
                        "trajectory_coordinates": trajectory,
                        **row.to_dict(),
                        "astar_runtime_calls": 0,
                        "dynamic_reward": False,
                    },
                )
        summary = aggregate_episode_metrics(rows)
        writer.writerow(
            {
                "record_type": "summary",
                "split": split,
                "checkpoint_interaction": checkpoint_interaction,
                "episode": "",
                "task_id": "",
                "map_id": "",
                "success": summary["success_count"],
                "collision": summary["collision_count"],
                "timeout": summary["timeout_count"],
                "SPL": summary["spl"],
                "path_efficiency": summary["path_efficiency"],
                "path_length": "",
                "steps": "",
                "optimal_path_length": "",
                "success_rate": summary["success_rate"],
                "collision_rate": summary["collision_rate"],
                "timeout_rate": summary["timeout_rate"],
                "average_steps": summary["average_steps"],
                "average_path_length": summary["average_path_length"],
            }
        )
    after_sha = model_parameter_sha256(model)
    if before_sha != after_sha:
        raise RuntimeError(f"model parameters changed during {split} evaluation")
    summary.update(
        {
            "split": split,
            "seed": int(seed),
            "checkpoint_interaction": int(checkpoint_interaction),
            "deterministic": True,
            "astar_runtime_calls": 0,
            "dynamic_reward": False,
            "parameter_updates": 0,
            "parameter_sha256_before": before_sha,
            "parameter_sha256_after": after_sha,
            "inference_seconds": time.perf_counter() - started,
        }
    )
    write_json_new(output_dir / "evaluation_summary.json", summary)
    return summary
