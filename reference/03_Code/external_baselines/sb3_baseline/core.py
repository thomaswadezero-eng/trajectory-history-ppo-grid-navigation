from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces


MOVE_DELTAS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


class IntegrityError(RuntimeError):
    """Raised when a frozen input or protocol invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise IntegrityError(f"JSON root must be an object: {path}")
    return payload


def write_json_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_json_replace(path: Path, payload: object) -> None:
    """Replace only a run-local status/pointer file using an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    split: str
    map_id: int
    grid_file: str
    cspace_file: str
    start: tuple[int, int]
    goal: tuple[int, int]
    max_steps: int
    shortest_path_length: int
    task_generation_seed: int

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "TaskRecord":
        return cls(
            task_id=str(row["task_id"]),
            split=str(row["split"]),
            map_id=int(row["map_id"]),
            grid_file=str(row["grid_file"]),
            cspace_file=str(row["cspace_file"]),
            start=(int(row["start"][0]), int(row["start"][1])),
            goal=(int(row["goal"][0]), int(row["goal"][1])),
            max_steps=int(row["max_steps"]),
            shortest_path_length=int(row["shortest_path_length"]),
            task_generation_seed=int(row["task_generation_seed"]),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["start"] = list(self.start)
        result["goal"] = list(self.goal)
        return result


def load_frozen_tasks(
    path: Path,
    expected_sha256: str,
    expected_split: str,
    expected_count: int,
    expected_map_ids: set[int],
    max_steps: int = 180,
) -> list[TaskRecord]:
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256.upper():
        raise IntegrityError(f"task SHA mismatch: {path}; actual={actual_sha}")
    payload = read_json(path)
    rows = payload.get("tasks")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise IntegrityError(f"task count mismatch for {expected_split}: {path}")
    tasks = [TaskRecord.from_mapping(row) for row in rows]
    if any(task.split != expected_split for task in tasks):
        raise IntegrityError(f"split label mismatch for {expected_split}")
    if {task.map_id for task in tasks} != expected_map_ids:
        raise IntegrityError(f"map membership mismatch for {expected_split}")
    if any(task.max_steps != max_steps for task in tasks):
        raise IntegrityError(f"max_steps mismatch for {expected_split}")
    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise IntegrityError(f"duplicate task id in {expected_split}")
    return tasks


class MapStore:
    def __init__(self, size: int = 30):
        self.size = int(size)
        self._cache: dict[str, np.ndarray] = {}

    def get(self, task: TaskRecord) -> np.ndarray:
        path = Path(task.cspace_file).resolve()
        key = str(path)
        if key not in self._cache:
            if not path.is_file():
                raise IntegrityError(f"missing cspace file: {path}")
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
            self._cache[key] = grid.astype(np.uint8, copy=True)
        return self._cache[key].copy()


def cspace_aggregate_sha256(tasks: Iterable[TaskRecord]) -> tuple[str, list[dict[str, Any]]]:
    unique = {(task.map_id, str(Path(task.cspace_file).resolve())) for task in tasks}
    records: list[dict[str, Any]] = []
    lines: list[str] = []
    for map_id, path_text in sorted(unique):
        value = sha256_file(Path(path_text))
        records.append({"map_id": map_id, "path": path_text, "sha256": value})
        lines.append(f"{map_id}|{path_text}|{value}")
    canonical = ("\n".join(lines) + "\n").encode("utf-8")
    return hashlib.sha256(canonical).hexdigest().upper(), records


@dataclass
class StateBuilder:
    size: int = 30
    visit_clip: int = 10
    trace_decay: float = 0.85
    no_progress_clip: int = 20
    progress_epsilon: float = 1e-6

    def reset(self, grid: np.ndarray, start: Sequence[int], goal: Sequence[int], max_steps: int) -> dict[str, np.ndarray]:
        self.grid = np.asarray(grid, dtype=np.uint8).copy()
        self.position = (int(start[0]), int(start[1]))
        self.goal = (int(goal[0]), int(goal[1]))
        self.max_steps = int(max_steps)
        self.step_count = 0
        self.no_progress = 0
        self.previous_action = -1
        self.visits = np.zeros((self.size, self.size), dtype=np.uint16)
        self.trace = np.zeros((self.size, self.size), dtype=np.float32)
        self.visits[self.position] = 1
        self.trace[self.position] = 1.0
        self.positions: list[tuple[int, int]] = [self.position]
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

    def build(self) -> dict[str, np.ndarray]:
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
        return {"spatial": spatial, "history": history}


class BarnNavigationEnv(gym.Env[dict[str, np.ndarray], int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        tasks: Sequence[TaskRecord],
        split: str,
        official_seed: int,
        map_provider: Callable[[TaskRecord], np.ndarray] | None = None,
        training_sampling: bool = False,
        size: int = 30,
        task_seed_offset: int = 104729,
    ):
        super().__init__()
        if not tasks:
            raise ValueError("tasks must not be empty")
        self.tasks = list(tasks)
        self.split = str(split)
        self.official_seed = int(official_seed)
        self.training_sampling = bool(training_sampling)
        self.task_rng = np.random.RandomState((self.official_seed + task_seed_offset) % (2**32))
        self.map_store = MapStore(size)
        self.map_provider = map_provider or self.map_store.get
        self.builder = StateBuilder(size=size)
        self.observation_space = spaces.Dict(
            {
                "spatial": spaces.Box(0.0, 1.0, shape=(5, size, size), dtype=np.float32),
                "history": spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(4)
        self.current_task: TaskRecord | None = None
        self.grid: np.ndarray | None = None
        self.done = True
        self.episode_index = 0

    def _choose_task(self, options: Mapping[str, Any] | None) -> TaskRecord:
        if options is not None and "task" in options:
            task = options["task"]
            return task if isinstance(task, TaskRecord) else TaskRecord.from_mapping(task)
        if self.training_sampling:
            return self.tasks[int(self.task_rng.randint(0, len(self.tasks)))]
        index = self.episode_index % len(self.tasks)
        return self.tasks[index]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        task = self._choose_task(options)
        if task.split != self.split:
            raise IntegrityError(f"task split mismatch: expected={self.split}, actual={task.split}")
        grid = np.asarray(self.map_provider(task), dtype=np.uint8)
        if grid.shape != (self.builder.size, self.builder.size):
            raise IntegrityError(f"adapter grid shape mismatch: {grid.shape}")
        if grid[task.start] != 0 or grid[task.goal] != 0:
            raise IntegrityError(f"start or goal is occupied: {task.task_id}")
        self.current_task = task
        self.grid = grid.copy()
        self.done = False
        self.episode_index += 1
        observation = self.builder.reset(self.grid, task.start, task.goal, task.max_steps)
        return observation, self._base_info()

    def _base_info(self) -> dict[str, Any]:
        if self.current_task is None:
            return {}
        return {
            "task_id": self.current_task.task_id,
            "map_id": self.current_task.map_id,
            "split": self.current_task.split,
            "shortest_path_length": self.current_task.shortest_path_length,
        }

    def action_masks(self) -> np.ndarray:
        if self.grid is None or self.done:
            raise RuntimeError("action_masks called outside an active episode")
        row, col = self.builder.position
        mask = np.zeros(4, dtype=np.bool_)
        for action, (dr, dc) in enumerate(MOVE_DELTAS):
            nr, nc = row + dr, col + dc
            mask[action] = 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1] and self.grid[nr, nc] == 0
        if not mask.any():
            raise IntegrityError("reachable task produced an all-invalid action mask")
        return mask

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.done or self.grid is None or self.current_task is None:
            raise RuntimeError("step called outside an active episode")
        action = int(action)
        mask = self.action_masks()
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
        terminated = bool(success or collision or timeout)
        truncated = False
        self.done = terminated
        reward = -0.002
        if success:
            reward += 1.0
        if collision:
            reward -= 0.2
        if timeout:
            reward -= 0.2
        info = {
            **self._base_info(),
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
            "valid_action": bool(valid),
            "position": list(new),
            "step_count": self.builder.step_count,
            "path_length": sum(
                self.builder.positions[index] != self.builder.positions[index - 1]
                for index in range(1, len(self.builder.positions))
            ),
            "outcome": "success" if success else "collision" if collision else "timeout" if timeout else "running",
            "dynamic_reward": False,
            "astar_runtime_calls": 0,
        }
        return self.builder.build(), float(reward), terminated, truncated, info

    def render(self) -> None:
        return None
