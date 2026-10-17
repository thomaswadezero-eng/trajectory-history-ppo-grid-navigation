from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class ProtocolError(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: object, *, must_not_exist: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if must_not_exist and path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if must_not_exist and path.exists():
        temporary.unlink()
        raise FileExistsError(path)
    os.replace(temporary, path)


def append_csv(path: Path, row: Mapping[str, Any], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow(dict(row))


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


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    split: str
    map_id: int
    cspace_file: str
    start: tuple[int, int]
    goal: tuple[int, int]
    max_steps: int
    shortest_path_length: int

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "TaskRecord":
        return cls(
            task_id=str(row["task_id"]),
            split=str(row["split"]),
            map_id=int(row["map_id"]),
            cspace_file=str(row["cspace_file"]),
            start=(int(row["start"][0]), int(row["start"][1])),
            goal=(int(row["goal"][0]), int(row["goal"][1])),
            max_steps=int(row["max_steps"]),
            shortest_path_length=int(row["shortest_path_length"]),
        )


def load_frozen_tasks(
    config: Mapping[str, Any], project_root: Path, splits: Sequence[str]
) -> dict[str, list[TaskRecord]]:
    requested = tuple(str(item) for item in splits)
    if len(requested) != len(set(requested)) or any(
        item not in ("train", "validation", "test") for item in requested
    ):
        raise ProtocolError(f"invalid requested splits: {requested}")
    input_keys = {
        "train": "training_tasks",
        "validation": "validation_tasks",
        "test": "test_tasks",
    }
    outputs: dict[str, list[TaskRecord]] = {}
    all_ids: set[str] = set()
    all_maps: dict[str, set[int]] = {}
    for split in requested:
        spec = config["inputs"][input_keys[split]]
        path = (project_root / str(spec["path"])).resolve()
        if not path.is_file() or sha256_file(path) != str(spec["sha256"]).upper():
            raise IntegrityError(f"frozen {split} task file identity mismatch: {path}")
        rows = read_json(path).get("tasks")
        if not isinstance(rows, list) or len(rows) != int(spec["count"]):
            raise IntegrityError(f"invalid {split} task count")
        tasks = [TaskRecord.from_mapping(row) for row in rows]
        if any(task.split != split for task in tasks):
            raise IntegrityError(f"{split} task label mismatch")
        ids = {task.task_id for task in tasks}
        if len(ids) != len(tasks) or ids & all_ids:
            raise IntegrityError(f"duplicate or overlapping task IDs in {split}")
        maps = {task.map_id for task in tasks}
        if any(maps & previous for previous in all_maps.values()):
            raise IntegrityError(f"map split overlap involving {split}")
        if any(task.max_steps != int(config["environment"]["max_steps"]) for task in tasks):
            raise IntegrityError(f"episode horizon mismatch in {split}")
        if any(task.shortest_path_length <= 0 for task in tasks):
            raise IntegrityError(f"invalid shortest-path metadata in {split}")
        outputs[split] = tasks
        all_ids.update(ids)
        all_maps[split] = maps
    return outputs


class MapStore:
    def __init__(self, size: int):
        self.size = int(size)
        self.cache: dict[str, np.ndarray] = {}

    def get(self, task: TaskRecord) -> np.ndarray:
        path = Path(task.cspace_file).resolve()
        key = str(path).lower()
        if key not in self.cache:
            if path.suffix.lower() == ".npy":
                raw = np.load(path, allow_pickle=False)
            elif path.suffix.lower() == ".csv":
                raw = np.loadtxt(path, delimiter=",")
            else:
                raise IntegrityError(f"unsupported map format: {path}")
            grid = np.asarray(raw)
            if grid.shape != (self.size, self.size):
                raise IntegrityError(f"map shape mismatch: {path} {grid.shape}")
            if not np.all(np.logical_or(grid == 0, grid == 1)):
                raise IntegrityError(f"map is not binary: {path}")
            self.cache[key] = grid.astype(np.uint8, copy=True)
        return self.cache[key].copy()


@dataclass(frozen=True)
class Observation:
    spatial: np.ndarray
    history: np.ndarray


class StateBuilder:
    """Exact frozen V7 5x30x30 spatial and 10-D history construction."""

    def __init__(self, environment_spec: Mapping[str, Any], state_spec: Mapping[str, Any]):
        self.size = int(environment_spec["map_size"])
        self.visit_clip = int(state_spec["visit_count_clip"])
        self.trace_decay = float(state_spec["recent_trace_decay"])
        self.no_progress_clip = int(state_spec["no_progress_counter_clip"])
        self.progress_epsilon = float(state_spec["minimum_progress_epsilon"])
        self.reset_storage()

    def reset_storage(self) -> None:
        self.grid: Optional[np.ndarray] = None
        self.position = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 0
        self.step_count = 0
        self.no_progress = 0
        self.previous_action = -1
        self.visits = np.zeros((self.size, self.size), dtype=np.uint16)
        self.trace = np.zeros((self.size, self.size), dtype=np.float32)
        self.positions: list[tuple[int, int]] = []

    def reset(
        self, grid: np.ndarray, start: Sequence[int], goal: Sequence[int], max_steps: int
    ) -> Observation:
        self.grid = np.asarray(grid, dtype=np.uint8).copy()
        self.position = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.no_progress = 0
        self.previous_action = -1
        self.visits.fill(0)
        self.trace.fill(0.0)
        self.visits[self.position] = 1
        self.trace[self.position] = 1.0
        self.positions = [self.position]
        return self.build()

    @property
    def distance_scale(self) -> float:
        return math.hypot(self.size - 1, self.size - 1)

    def distance(self, position: tuple[int, int]) -> float:
        return math.hypot(position[0] - self.goal[0], position[1] - self.goal[1]) / self.distance_scale

    def update(self, action: int, position: tuple[int, int], progress: float) -> None:
        self.step_count += 1
        self.trace *= self.trace_decay
        self.position = position
        self.visits[position] += 1
        self.trace[position] = 1.0
        self.no_progress = 0 if progress > self.progress_epsilon else self.no_progress + 1
        self.previous_action = int(action)
        self.positions.append(position)

    def build(self) -> Observation:
        if self.grid is None:
            raise RuntimeError("state is not initialized")
        spatial = np.zeros((5, self.size, self.size), dtype=np.float32)
        spatial[0] = self.grid
        spatial[1, self.position[0], self.position[1]] = 1.0
        spatial[2, self.goal[0], self.goal[1]] = 1.0
        spatial[3] = np.minimum(self.visits, self.visit_clip) / float(self.visit_clip)
        spatial[4] = np.clip(self.trace, 0.0, 1.0)
        history = np.zeros(10, dtype=np.float32)
        history[0] = min(self.step_count / float(max(1, self.max_steps)), 1.0)
        history[1] = min(self.no_progress / float(self.no_progress_clip), 1.0)
        if self.previous_action >= 0:
            history[2 + self.previous_action] = 1.0
        for action, (dr, dc) in enumerate(MOVE_DELTAS):
            row, col = self.position[0] + dr, self.position[1] + dc
            count = self.visit_clip if not (0 <= row < self.size and 0 <= col < self.size) else self.visits[row, col]
            history[6 + action] = min(count, self.visit_clip) / float(self.visit_clip)
        return Observation(spatial=spatial, history=history)


class BarnEnvironment:
    """V7-equivalent Base-Reward-only environment; intentionally has no planner hook."""

    def __init__(self, config: Mapping[str, Any], maps: MapStore):
        self.environment_spec = config["environment"]
        self.reward_spec = config["reward"]["base"]
        self.builder = StateBuilder(self.environment_spec, config["state"])
        self.maps = maps
        self.grid: Optional[np.ndarray] = None
        self.task: Optional[TaskRecord] = None
        self.done = True

    def reset(self, task: TaskRecord) -> Observation:
        self.task = task
        self.grid = self.maps.get(task)
        if self.grid[task.start] != 0 or self.grid[task.goal] != 0 or task.start == task.goal:
            raise IntegrityError(f"invalid frozen task geometry: {task.task_id}")
        self.done = False
        return self.builder.reset(self.grid, task.start, task.goal, task.max_steps)

    def action_mask(self) -> np.ndarray:
        if self.grid is None or self.done:
            raise RuntimeError("action_mask called outside active episode")
        row, col = self.builder.position
        mask = np.zeros(4, dtype=np.bool_)
        for action, (dr, dc) in enumerate(MOVE_DELTAS):
            nr, nc = row + dr, col + dc
            mask[action] = 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1] and self.grid[nr, nc] == 0
        if not mask.any():
            raise IntegrityError("reachable task produced an all-invalid action mask")
        return mask

    def step(self, action: int) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.done or self.grid is None or self.task is None:
            raise RuntimeError("step called outside active episode")
        action = int(action)
        mask = self.action_mask()
        old = self.builder.position
        valid = 0 <= action < 4 and bool(mask[action])
        collision = not valid
        if valid:
            dr, dc = MOVE_DELTAS[action]
            new = (old[0] + dr, old[1] + dc)
        else:
            new = old
        progress = self.builder.distance(old) - self.builder.distance(new)
        self.builder.update(action, new, progress)
        success = new == self.builder.goal
        timeout = self.builder.step_count >= self.builder.max_steps and not success and not collision
        self.done = bool(success or collision or timeout)
        reward = float(self.reward_spec["step"])
        reward += float(self.reward_spec["goal"]) if success else 0.0
        reward += float(self.reward_spec["collision"]) if collision else 0.0
        reward += float(self.reward_spec["timeout"]) if timeout else 0.0
        info = {
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
            "valid_action": bool(valid),
            "position": new,
            "path_length": sum(
                self.builder.positions[index] != self.builder.positions[index - 1]
                for index in range(1, len(self.builder.positions))
            ),
            "dynamic_reward_enabled": False,
            "astar_runtime_calls": 0,
        }
        return self.builder.build(), reward, self.done, info


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
    path_efficiency: Optional[float]
    episode_reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def episode_metrics(
    episode: int,
    task: TaskRecord,
    success: bool,
    collision: bool,
    timeout: bool,
    steps: int,
    path_length: int,
    episode_reward: float,
) -> EpisodeMetrics:
    outcomes = int(success) + int(collision) + int(timeout)
    if outcomes != 1:
        raise IntegrityError("episode outcomes must be mutually exclusive and exhaustive")
    denominator = max(task.shortest_path_length, int(path_length))
    spl = int(success) * task.shortest_path_length / float(denominator)
    efficiency = task.shortest_path_length / float(denominator) if success else None
    return EpisodeMetrics(
        episode=int(episode), task_id=task.task_id, map_id=task.map_id,
        success=int(success), collision=int(collision), timeout=int(timeout), steps=int(steps),
        path_length=int(path_length), optimal_path_length=task.shortest_path_length,
        spl=float(spl), path_efficiency=efficiency, episode_reward=float(episode_reward),
    )


def aggregate_metrics(rows: Iterable[EpisodeMetrics]) -> dict[str, Any]:
    episodes = list(rows)
    if not episodes:
        raise IntegrityError("cannot aggregate zero evaluation episodes")
    count = len(episodes)
    successes = sum(row.success for row in episodes)
    collisions = sum(row.collision for row in episodes)
    timeouts = sum(row.timeout for row in episodes)
    if successes + collisions + timeouts != count:
        raise IntegrityError("aggregate outcomes are not exhaustive")
    efficiencies = [float(row.path_efficiency) for row in episodes if row.path_efficiency is not None]
    return {
        "episode_count": count,
        "success_count": successes,
        "collision_count": collisions,
        "timeout_count": timeouts,
        "success_rate": successes / float(count),
        "collision_rate": collisions / float(count),
        "timeout_rate": timeouts / float(count),
        "success_rate_fraction": f"{successes}/{count}",
        "collision_rate_fraction": f"{collisions}/{count}",
        "timeout_rate_fraction": f"{timeouts}/{count}",
        "average_steps": float(np.mean([row.steps for row in episodes])),
        "average_path_length": float(np.mean([row.path_length for row in episodes])),
        "spl": float(np.mean([row.spl for row in episodes])),
        "path_efficiency": float(np.mean(efficiencies)) if efficiencies else 0.0,
        "path_efficiency_success_count": len(efficiencies),
        "average_episode_reward": float(np.mean([row.episode_reward for row in episodes])),
    }


def checkpoint_key(metrics: Mapping[str, Any], interaction: int) -> tuple[float, ...]:
    return (
        -float(metrics["success_rate"]),
        -float(metrics["spl"]),
        float(metrics["timeout_rate"]),
        float(metrics["average_steps"]),
        float(interaction),
    )
