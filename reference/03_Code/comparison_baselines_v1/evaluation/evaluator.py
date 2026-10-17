from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch

from common.errors import IntegrityError
from common.io_utils import write_csv, write_json
from environment.navigation import MapStore, NavigationEnvironment
from evaluation.metrics import aggregate_episode_records, episode_record, per_map_aggregates


EPISODE_FIELDS = [
    "episode",
    "task_id",
    "map_id",
    "success",
    "collision",
    "timeout",
    "SPL",
    "path_length",
    "steps",
    "optimal_path_length",
    "path_efficiency",
]


class Evaluator:
    def __init__(self, protocol: Mapping[str, Any], maps: MapStore, method: str, seed: int):
        self.protocol = protocol
        self.maps = maps
        self.method = method
        self.seed = int(seed)
        self.runtime_astar_calls = 0
        self.parameter_updates = 0

    def evaluate(
        self,
        agent,
        tasks: Sequence[Mapping[str, Any]],
        split: str,
        interaction: int,
        output_directory: Path,
        record_trajectories: bool,
    ) -> Dict[str, Any]:
        output_directory.mkdir(parents=True, exist_ok=False)
        trajectory_directory: Optional[Path] = None
        if record_trajectories:
            trajectory_directory = output_directory / "trajectories"
            trajectory_directory.mkdir()
        environment = NavigationEnvironment(self.protocol, self.maps)
        previous_mode = agent.set_evaluation_mode(True)
        records = []
        started = time.perf_counter()
        try:
            with torch.no_grad():
                for episode, task in enumerate(tasks, 1):
                    observation = environment.reset(task)
                    trajectory = [list(environment.builder.position)]
                    state_trajectory = [
                        {
                            "step": 0,
                            "position": list(environment.builder.position),
                            "history_state": observation.history.tolist(),
                        }
                    ]
                    done = False
                    final: Dict[str, Any] = {}
                    while not done:
                        mask = environment.action_mask()
                        action = int(agent.deterministic_action(observation, mask))
                        if not bool(mask[action]):
                            raise IntegrityError("deterministic evaluation selected an illegal action")
                        state_trajectory[-1]["action_mask"] = mask.astype(int).tolist()
                        state_trajectory[-1]["selected_action"] = action
                        observation, _, done, final = environment.step(action)
                        trajectory.append(list(environment.builder.position))
                        state_trajectory.append(
                            {
                                "step": environment.builder.step_count,
                                "position": list(environment.builder.position),
                                "history_state": observation.history.tolist(),
                            }
                        )
                    row = episode_record(
                        task,
                        trajectory,
                        environment.builder.step_count,
                        bool(final["success"]),
                        bool(final["collision"]),
                        bool(final["timeout"]),
                    )
                    row = {"episode": episode, **row}
                    records.append(row)
                    if trajectory_directory is not None:
                        if environment.grid is None:
                            raise IntegrityError("map missing during trajectory serialization")
                        write_json(
                            trajectory_directory / f"episode_{episode:04d}.json",
                            {
                                **row,
                                "method": self.method,
                                "seed": self.seed,
                                "split": split,
                                "checkpoint_interaction": int(interaction),
                                "map_size": [int(x) for x in environment.grid.shape],
                                "obstacle_coordinates": np.argwhere(environment.grid == 1).tolist(),
                                "start_position": list(task["start"]),
                                "goal_position": list(task["goal"]),
                                "state_trajectory": state_trajectory,
                                "trajectory_coordinates": trajectory,
                                "astar_runtime_calls": 0,
                                "dynamic_reward_enabled": False,
                                "parameter_updates": 0,
                            },
                        )
        finally:
            agent.restore_mode(previous_mode)
        summary = aggregate_episode_records(records)
        summary.update(
            {
                "method": self.method,
                "seed": self.seed,
                "split": split,
                "checkpoint_interaction": int(interaction),
                "inference_seconds": time.perf_counter() - started,
                "astar_runtime_calls": 0,
                "dynamic_reward_enabled": False,
                "parameter_updates": 0,
                "deterministic_masked_policy": True,
            }
        )
        write_csv(output_directory / "evaluation_metrics.csv", records, EPISODE_FIELDS)
        map_rows = per_map_aggregates(records)
        write_csv(output_directory / "per_map_metrics.csv", map_rows, list(map_rows[0].keys()))
        write_json(output_directory / "evaluation_summary.json", summary)
        return summary

