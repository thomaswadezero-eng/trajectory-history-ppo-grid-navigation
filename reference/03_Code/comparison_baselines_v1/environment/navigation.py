from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from common.errors import IntegrityError


MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Observation:
    spatial: np.ndarray
    history: np.ndarray


@dataclass(frozen=True)
class BaseRewardRecord:
    goal_reward: float
    collision_penalty: float
    timeout_penalty: float
    step_penalty: float

    @property
    def total_reward(self) -> float:
        return self.goal_reward + self.collision_penalty + self.timeout_penalty + self.step_penalty


class MapStore:
    def __init__(self, size: int):
        self.size = int(size)
        self.cache: Dict[str, np.ndarray] = {}

    def get(self, task: Mapping[str, Any]) -> np.ndarray:
        path = Path(str(task["cspace_file"])).resolve()
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


class StateBuilder:
    """Exact baseline-side reproduction of the frozen V7 observation builder."""

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
        self.positions: List[Tuple[int, int]] = []

    def reset(
        self,
        grid: np.ndarray,
        start: Sequence[int],
        goal: Sequence[int],
        max_steps: int,
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

    def distance(self, position: Tuple[int, int]) -> float:
        return math.hypot(position[0] - self.goal[0], position[1] - self.goal[1]) / self.distance_scale

    def update(self, action: int, position: Tuple[int, int], progress: float) -> None:
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


class NavigationEnvironment:
    """V7-equivalent four-neighbour environment with base reward only.

    This class intentionally has no Dynamic Reward or A* entry point.
    """

    def __init__(self, protocol: Mapping[str, Any], maps: MapStore):
        self.protocol = protocol
        self.maps = maps
        self.environment_spec = protocol["environment"]
        self.reward_spec = protocol["base_reward"]
        self.builder = StateBuilder(self.environment_spec, protocol["state"])
        self.grid: Optional[np.ndarray] = None
        self.task: Optional[Mapping[str, Any]] = None
        self.done = True

    def reset(self, task: Mapping[str, Any]) -> Observation:
        self.task = task
        self.grid = self.maps.get(task)
        start = tuple(int(x) for x in task["start"])
        goal = tuple(int(x) for x in task["goal"])
        if self.grid[start] != 0 or self.grid[goal] != 0 or start == goal:
            raise IntegrityError(f"invalid frozen task geometry: {task['task_id']}")
        self.done = False
        return self.builder.reset(self.grid, task["start"], task["goal"], int(task["max_steps"]))

    def action_mask(self) -> np.ndarray:
        if self.grid is None:
            raise RuntimeError("environment not initialized")
        row, col = self.builder.position
        mask = np.zeros(4, dtype=np.bool_)
        for action, (dr, dc) in enumerate(MOVE_DELTAS):
            nr, nc = row + dr, col + dc
            mask[action] = (
                0 <= nr < self.grid.shape[0]
                and 0 <= nc < self.grid.shape[1]
                and self.grid[nr, nc] == 0
            )
        if not mask.any():
            raise IntegrityError("reachable task produced an all-invalid action mask")
        return mask

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done or self.grid is None:
            raise RuntimeError("step outside active episode")
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
        reward_record = BaseRewardRecord(
            goal_reward=float(self.reward_spec["goal"]) if success else 0.0,
            collision_penalty=float(self.reward_spec["collision"]) if collision else 0.0,
            timeout_penalty=float(self.reward_spec["timeout"]) if timeout else 0.0,
            step_penalty=float(self.reward_spec["step"]),
        )
        info = {
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
            "valid_action": bool(valid),
            "position": new,
            "reward": reward_record,
            "dynamic_reward_enabled": False,
            "astar_calls": 0,
        }
        return self.builder.build(), reward_record.total_reward, self.done, info

