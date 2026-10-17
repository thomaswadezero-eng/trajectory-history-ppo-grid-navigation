#!/usr/bin/env python
"""Independent two-factor PPO experiment with training-only A* planning priors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


SCRIPT_PATH = Path(__file__).resolve()
SOURCE_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SOURCE_DIR.parent.parent
DEFAULT_CONFIG = SOURCE_DIR / "experiment_config.json"
DEFAULT_PROTOCOL = SOURCE_DIR / "experiment_protocol.json"
CONFIG_HASH_PATTERN = re.compile(
    rb'("config_sha256"\s*:\s*")([0-9A-Fa-f]{64})(")'
)
PROTOCOL_HASH_PATTERN = re.compile(
    rb'("protocol_sha256"\s*:\s*")([0-9A-Fa-f]{64})(")'
)
MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CONFIGURATIONS = {
    "PPO": (False, False),
    "PPO_Astar": (True, False),
    "PPO_Reward": (False, True),
    "FULL": (True, True),
}


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


def verified_self_hash(path: Path) -> str:
    raw = path.read_bytes()
    matches = list(CONFIG_HASH_PATTERN.finditer(raw))
    if len(matches) != 1:
        raise ProtocolError("configuration must contain exactly one config_sha256")
    match = matches[0]
    stored = match.group(2).decode("ascii").upper()
    zeroed = raw[: match.start(2)] + b"0" * 64 + raw[match.end(2) :]
    calculated = hashlib.sha256(zeroed).hexdigest().upper()
    if stored != calculated:
        raise ProtocolError(
            f"configuration SHA mismatch: stored={stored}, calculated={calculated}"
        )
    return stored


def verified_protocol_hash(path: Path) -> str:
    raw = path.read_bytes()
    matches = list(PROTOCOL_HASH_PATTERN.finditer(raw))
    if len(matches) != 1:
        raise ProtocolError("protocol must contain exactly one protocol_sha256")
    match = matches[0]
    stored = match.group(2).decode("ascii").upper()
    zeroed = raw[: match.start(2)] + b"0" * 64 + raw[match.end(2) :]
    calculated = hashlib.sha256(zeroed).hexdigest().upper()
    if stored != calculated:
        raise ProtocolError(
            f"protocol SHA mismatch: stored={stored}, calculated={calculated}"
        )
    return stored


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class FrozenConfig:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.sha256 = verified_self_hash(self.path)
        self.data = read_json(self.path)

    def get(self, dotted: str) -> Any:
        value: Any = self.data
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ProtocolError(f"missing configuration field: {dotted}")
            value = value[part]
        return value

    def integer(self, dotted: str) -> int:
        value = self.get(dotted)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProtocolError(f"expected integer field: {dotted}")
        return int(value)

    def number(self, dotted: str) -> float:
        value = self.get(dotted)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(f"expected numeric field: {dotted}")
        return float(value)


class FrozenProtocol:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.sha256 = verified_protocol_hash(self.path)
        self.data = read_json(self.path)

    def get(self, dotted: str) -> Any:
        value: Any = self.data
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ProtocolError(f"missing protocol field: {dotted}")
            value = value[part]
        return value


def validate_config(config: FrozenConfig, protocol: FrozenProtocol) -> None:
    if config.get("schema_version") != "final_two_factor_experiment.v7":
        raise ProtocolError("unsupported final experiment schema")
    if config.get("status") != "FROZEN_AND_AUTHORIZED":
        raise ProtocolError("configuration is not frozen")
    if config.get("formal_execution_allowed") is not True:
        raise ProtocolError("formal execution is not authorized")
    if config.get("authorized_program") != (
        "03_Code/final_two_factor_experiment_v7/final_two_factor_experiment_v7.py"
    ):
        raise ProtocolError("authorized program path mismatch")
    if str(config.get("authorized_program_sha256")).upper() != sha256_file(SCRIPT_PATH):
        raise ProtocolError("authorized program SHA mismatch")
    declared = config.get("configurations")
    expected = {
        name: {"astar_supervision": flags[0], "dynamic_reward": flags[1]}
        for name, flags in CONFIGURATIONS.items()
    }
    if declared != expected:
        raise ProtocolError("two-factor configuration matrix mismatch")
    if list(config.get("official_seeds")) != [42, 1004, 6010]:
        raise ProtocolError("official seeds mismatch")
    if config.integer("training.interaction_budget") != 900000:
        raise ProtocolError("training budget mismatch")
    if config.integer("environment.action_count") != 4:
        raise ProtocolError("action count mismatch")
    if tuple(tuple(x) for x in config.get("environment.move_deltas")) != MOVE_DELTAS:
        raise ProtocolError("move ordering mismatch")
    expected_network = {
        "spatial_input_shape": [5, 30, 30],
        "convolutions": [
            {"in_channels": 5, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
            {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1},
            {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1},
        ],
        "adaptive_pool": [4, 4],
        "spatial_projection": [1024, 256],
        "history_mlp": [10, 64, 64],
        "fusion": [320, 256],
        "actor_head": [256, 128, 4],
        "critic_head": [256, 128, 1],
        "planning_head": [256, 128, 4],
        "planning_head_instantiated_for_all_configurations": True,
        "activation": "ReLU",
        "fusion_normalization": "LayerNorm",
        "weight_initialization": "pytorch_default",
        "recurrent_layers": False,
        "batch_normalization": False,
        "dropout": False,
    }
    if config.get("network") != expected_network:
        raise ProtocolError("network definition mismatch")
    if protocol.get("schema_version") != "final_two_factor_experiment.v7.protocol":
        raise ProtocolError("unsupported protocol schema")
    if protocol.get("status") != "FROZEN_AND_AUTHORIZED":
        raise ProtocolError("protocol is not frozen")
    if protocol.get("formal_execution_allowed") is not True:
        raise ProtocolError("protocol execution is not authorized")
    if protocol.get("authorized_program") != config.get("authorized_program"):
        raise ProtocolError("protocol program path mismatch")
    if str(protocol.get("authorized_program_sha256")).upper() != sha256_file(SCRIPT_PATH):
        raise ProtocolError("protocol program SHA mismatch")
    if protocol.get("configuration.path") != (
        "03_Code/final_two_factor_experiment_v7/experiment_config.json"
    ):
        raise ProtocolError("protocol configuration path mismatch")
    if str(protocol.get("configuration.self_field_zeroed_sha256")).upper() != config.sha256:
        raise ProtocolError("protocol configuration SHA mismatch")
    if protocol.get("configurations") != expected:
        raise ProtocolError("protocol matrix mismatch")
    if protocol.get("inputs") != config.get("inputs"):
        raise ProtocolError("protocol input identity mismatch")
    if config.get("astar.integration") != "soft_planning_auxiliary_loss":
        raise ProtocolError("unsupported A* integration")
    if config.get("astar.action_imitation_loss_enabled") is not False:
        raise ProtocolError("A* action imitation must be disabled")
    if config.get("astar.modifies_reward") is not False or config.get("astar.modifies_ppo_advantage") is not False:
        raise ProtocolError("A* must not modify reward or PPO advantage")
    if config.get("astar.anchor_rehearsal_enabled") is not False:
        raise ProtocolError("A* replay must be disabled")
    if config.number("astar.soft_target_temperature") <= 0.0:
        raise ProtocolError("A* target temperature must be positive")
    if not 0.0 <= config.number("astar.astar_distill_weight") <= 1.0:
        raise ProtocolError("A* distillation weight must be in [0,1]")
    if config.number("astar.planning_loss_weight") < 0.0:
        raise ProtocolError("A* component weights must be nonnegative")
    if not 0.0 < config.number("astar.maximum_auxiliary_weight") <= 1.0:
        raise ProtocolError("invalid A* maximum auxiliary weight")
    for schedule_name in ("astar.distillation_schedule", "astar.astar_guidance_schedule"):
        constant_fraction = config.number(f"{schedule_name}.constant_until_fraction")
        zero_fraction = config.number(f"{schedule_name}.zero_after_fraction")
        if not 0.0 <= constant_fraction < zero_fraction <= 1.0:
            raise ProtocolError(f"invalid A* interaction schedule fractions: {schedule_name}")
        if config.integer(f"{schedule_name}.reference_interactions") != config.integer("training.interaction_budget"):
            raise ProtocolError(f"A* schedule reference must equal the shared interaction budget: {schedule_name}")
    if config.number("astar.distillation_schedule.zero_after_fraction") >= config.number("astar.astar_guidance_schedule.zero_after_fraction"):
        raise ProtocolError("A* distillation must end before the planning-cost auxiliary loss")
    if config.integer("astar.cache_capacity") <= 0:
        raise ProtocolError("A* cache capacity must be positive")


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


def verify_inputs(config: FrozenConfig) -> Dict[str, str]:
    actual: Dict[str, str] = {}
    for name, record in config.get("inputs").items():
        path = (PROJECT_ROOT / str(record["path"])).resolve()
        if not path.is_file():
            raise IntegrityError(f"missing frozen input: {path}")
        value = sha256_file(path)
        if value != str(record["sha256"]).upper():
            raise IntegrityError(f"frozen input SHA mismatch: {path}")
        actual[name] = value
    return actual


def load_tasks(config: FrozenConfig) -> Tuple[List[Dict[str, Any]], ...]:
    split_manifest = read_json(SOURCE_DIR / "map_split.json")
    task_manifest = read_json(SOURCE_DIR / "task_split.json")
    allowed = {
        split: {int(x) for x in split_manifest[f"{split}_map_ids"]}
        for split in ("train", "validation", "test")
    }
    outputs: List[List[Dict[str, Any]]] = []
    for split in ("train", "validation", "test"):
        entry = task_manifest["splits"][split]
        path = (PROJECT_ROOT / str(entry["path"])).resolve()
        if sha256_file(path) != str(entry["sha256"]).upper():
            raise IntegrityError(f"task manifest SHA mismatch for {split}")
        payload = read_json(path)
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != int(entry["task_count"]):
            raise IntegrityError(f"invalid task count for {split}")
        records = [dict(task) for task in tasks]
        if any(str(task["split"]) != split for task in records):
            raise IntegrityError(f"split label mismatch for {split}")
        if {int(task["map_id"]) for task in records} != allowed[split]:
            raise IntegrityError(f"map membership mismatch for {split}")
        if any(int(task["max_steps"]) != config.integer("environment.max_steps") for task in records):
            raise IntegrityError(f"max_steps mismatch for {split}")
        ids = [str(task["task_id"]) for task in records]
        if len(ids) != len(set(ids)):
            raise IntegrityError(f"duplicate task IDs in {split}")
        outputs.append(records)
    all_ids = [str(task["task_id"]) for group in outputs for task in group]
    if len(all_ids) != len(set(all_ids)):
        raise IntegrityError("task IDs overlap across splits")
    if allowed["train"] & allowed["validation"] or allowed["train"] & allowed["test"] or allowed["validation"] & allowed["test"]:
        raise IntegrityError("map splits overlap")
    return tuple(outputs)  # type: ignore[return-value]


class MapStore:
    def __init__(self, size: int):
        self.size = int(size)
        self.cache: Dict[str, np.ndarray] = {}

    def get(self, task: Mapping[str, Any]) -> np.ndarray:
        path = Path(str(task["cspace_file"])).resolve()
        key = str(path)
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
    def __init__(self, config: FrozenConfig):
        self.size = config.integer("environment.map_size")
        self.visit_clip = config.integer("state.visit_count_clip")
        self.trace_decay = config.number("state.recent_trace_decay")
        self.no_progress_clip = config.integer("state.no_progress_counter_clip")
        self.progress_epsilon = config.number("state.minimum_progress_epsilon")
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

    def reset(self, grid: np.ndarray, start: Sequence[int], goal: Sequence[int], max_steps: int) -> Observation:
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
        return Observation(spatial, history)


class ActorCritic(nn.Module):
    def __init__(self, config: FrozenConfig):
        super().__init__()
        network = config.get("network")
        conv = network["convolutions"]
        self.cnn = nn.Sequential(
            *[
                layer
                for item in conv
                for layer in (
                    nn.Conv2d(
                        int(item["in_channels"]),
                        int(item["out_channels"]),
                        int(item["kernel_size"]),
                        int(item["stride"]),
                        int(item["padding"]),
                    ),
                    nn.ReLU(),
                )
            ],
            nn.AdaptiveAvgPool2d(tuple(int(x) for x in network["adaptive_pool"])),
            nn.Flatten(),
            nn.Linear(*[int(x) for x in network["spatial_projection"]]),
            nn.LayerNorm(int(network["spatial_projection"][-1])),
            nn.ReLU(),
        )
        history = [int(x) for x in network["history_mlp"]]
        fusion = [int(x) for x in network["fusion"]]
        actor = [int(x) for x in network["actor_head"]]
        critic = [int(x) for x in network["critic_head"]]
        planning = [int(x) for x in network["planning_head"]]
        self.history = nn.Sequential(nn.Linear(history[0], history[1]), nn.ReLU(), nn.Linear(history[1], history[2]), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(fusion[0], fusion[1]), nn.LayerNorm(fusion[1]), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(actor[0], actor[1]), nn.ReLU(), nn.Linear(actor[1], actor[2]))
        self.critic = nn.Sequential(nn.Linear(critic[0], critic[1]), nn.ReLU(), nn.Linear(critic[1], critic[2]))
        self.planning = nn.Sequential(nn.Linear(planning[0], planning[1]), nn.ReLU(), nn.Linear(planning[1], planning[2]))

    def encode(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat((self.cnn(spatial), self.history(history)), dim=-1))

    def forward(self, spatial: torch.Tensor, history: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(spatial, history)
        return self.actor(features), self.critic(features).squeeze(-1)

    def planning_forward(self, spatial: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.planning(self.encode(spatial, history))

    @staticmethod
    def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        masks = masks.bool()
        if torch.any(~masks.any(dim=-1)):
            raise IntegrityError("all actions are invalid")
        return logits.masked_fill(~masks, torch.finfo(logits.dtype).min)


@dataclass
class RewardRecord:
    goal_reward: float = 0.0
    distance_reward: float = 0.0
    collision_penalty: float = 0.0
    step_penalty: float = 0.0
    timeout_penalty: float = 0.0
    revisit_penalty: float = 0.0
    oscillation_penalty: float = 0.0
    stagnation_penalty: float = 0.0
    escape_reward: float = 0.0

    @property
    def other_reward_terms(self) -> float:
        return self.timeout_penalty + self.revisit_penalty + self.oscillation_penalty + self.stagnation_penalty + self.escape_reward

    @property
    def total_reward(self) -> float:
        return self.goal_reward + self.distance_reward + self.collision_penalty + self.step_penalty + self.other_reward_terms


class NavigationEnvironment:
    def __init__(self, config: FrozenConfig, maps: MapStore, reward_enabled: bool):
        self.config = config
        self.maps = maps
        self.reward_enabled = bool(reward_enabled)
        self.builder = StateBuilder(config)
        self.grid: Optional[np.ndarray] = None
        self.task: Optional[Mapping[str, Any]] = None
        self.done = True

    def reset(self, task: Mapping[str, Any]) -> Observation:
        self.task = task
        self.grid = self.maps.get(task)
        self.done = False
        return self.builder.reset(self.grid, task["start"], task["goal"], int(task["max_steps"]))

    def action_mask(self) -> np.ndarray:
        if self.grid is None:
            raise RuntimeError("environment not initialized")
        row, col = self.builder.position
        mask = np.zeros(4, dtype=np.bool_)
        for action, (dr, dc) in enumerate(MOVE_DELTAS):
            nr, nc = row + dr, col + dc
            mask[action] = 0 <= nr < self.grid.shape[0] and 0 <= nc < self.grid.shape[1] and self.grid[nr, nc] == 0
        if not mask.any():
            raise IntegrityError("reachable task produced an all-invalid action mask")
        return mask

    def step(self, action: int) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.done or self.grid is None:
            raise RuntimeError("step outside active episode")
        mask = self.action_mask()
        old = self.builder.position
        prior_visits = self.builder.visits.copy()
        prior_positions = tuple(self.builder.positions)
        prior_no_progress = self.builder.no_progress
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
        self.done = success or collision or timeout
        record = RewardRecord(
            goal_reward=self.config.number("reward.base.goal") if success else 0.0,
            collision_penalty=self.config.number("reward.base.collision") if collision else 0.0,
            step_penalty=self.config.number("reward.base.step"),
            timeout_penalty=self.config.number("reward.base.timeout") if timeout else 0.0,
        )
        if self.reward_enabled:
            reward = self.config.get("reward.dynamic")
            delta = float(np.clip(progress, -float(reward["distance_progress_clip"]), float(reward["distance_progress_clip"])))
            raw_positive = float(reward["distance_weight"]) * max(delta, 0.0)
            negative = float(reward["distance_weight"]) * min(delta, 0.0)
            visits = int(prior_visits[new])
            stagnation = min(prior_no_progress / float(self.config.integer("state.no_progress_counter_clip")), 1.0)
            oscillation = int(any(len(prior_positions) >= n and new == prior_positions[-n] for n in reward["oscillation_lookback_steps"]))
            escape = int(prior_no_progress >= int(reward["escape_threshold"]) and visits == 0 and delta > float(reward["escape_minimum_progress"]) and oscillation == 0)
            escape_value = raw_positive if escape else 0.0
            record.distance_reward = negative + raw_positive - escape_value
            intensity = min(visits, self.config.integer("state.visit_count_clip")) / float(self.config.integer("state.visit_count_clip"))
            record.revisit_penalty = -float(reward["revisit_weight"]) * (1.0 + stagnation) * intensity
            record.oscillation_penalty = -float(reward["oscillation_weight"]) * (1.0 + stagnation) * oscillation
            record.stagnation_penalty = -float(reward["stagnation_weight"]) * stagnation
            record.escape_reward = escape_value
            if abs(raw_positive - max(record.distance_reward - negative, 0.0) - escape_value) > float(reward["accounting_tolerance"]):
                raise IntegrityError("dynamic reward accounting mismatch")
        info = {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "valid_action": valid,
            "reward": record,
            "position": new,
        }
        return self.builder.build(), record.total_reward, self.done, info


def astar_path(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    queue: List[Tuple[int, int, Tuple[int, int]]] = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0, start)]
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    cost = {start: 0}
    while queue:
        _, value, current = heapq.heappop(queue)
        if value != cost.get(current):
            continue
        if current == goal:
            result: List[Tuple[int, int]] = []
            cursor: Optional[Tuple[int, int]] = current
            while cursor is not None:
                result.append(cursor)
                cursor = parent[cursor]
            return list(reversed(result))
        for dr, dc in MOVE_DELTAS:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < grid.shape[0] and 0 <= nxt[1] < grid.shape[1] and grid[nxt] == 0):
                continue
            candidate = value + 1
            if candidate < cost.get(nxt, sys.maxsize):
                cost[nxt] = candidate
                parent[nxt] = current
                heuristic = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                heapq.heappush(queue, (candidate + heuristic, candidate, nxt))
    return None


@dataclass(frozen=True)
class PlanningTarget:
    soft_target: np.ndarray
    normalized_costs: np.ndarray
    legal_mask: np.ndarray
    distillation_weight: float
    planning_weight: float
    target_entropy: float
    interaction_count: int


class AStarPlanningSupervisor:
    """Produces training-only soft planning targets without choosing actions."""

    def __init__(self, config: FrozenConfig, rng: np.random.RandomState):
        self.config = config
        self.rng = rng
        self.cache: OrderedDict[Tuple[str, Tuple[int, int], Tuple[int, int]], np.ndarray] = OrderedDict()
        self.capacity = config.integer("astar.cache_capacity")
        self.planner_calls = 0
        self.cache_lookups = 0
        self.training_transitions = 0
        self.query_attempts = 0
        self.targets_generated = 0

    @property
    def query_frequency(self) -> float:
        return self.query_attempts / float(max(1, self.training_transitions))

    def _schedule_scale(self, interaction_count: int, schedule_name: str) -> float:
        reference = self.config.integer(f"astar.{schedule_name}.reference_interactions")
        progress = float(np.clip(interaction_count / float(reference), 0.0, 1.0))
        constant_until = self.config.number(f"astar.{schedule_name}.constant_until_fraction")
        zero_after = self.config.number(f"astar.{schedule_name}.zero_after_fraction")
        if progress < constant_until:
            return 1.0
        if progress < zero_after:
            fraction = (progress - constant_until) / (zero_after - constant_until)
            return 0.5 * (1.0 + math.cos(math.pi * fraction))
        return 0.0

    def schedule(self, interaction_count: int) -> Tuple[float, float, float]:
        distillation_scale = self._schedule_scale(interaction_count, "distillation_schedule")
        planning_scale = self._schedule_scale(interaction_count, "astar_guidance_schedule")
        maximum = self.config.number("astar.maximum_auxiliary_weight")
        distillation_weight = maximum * self.config.number("astar.astar_distill_weight") * distillation_scale
        planning_weight = maximum * self.config.number("astar.planning_loss_weight") * planning_scale
        query_probability = max(distillation_scale, planning_scale)
        return query_probability, distillation_weight, planning_weight

    def _successor_costs(
        self,
        grid: np.ndarray,
        map_key: str,
        position: Tuple[int, int],
        goal: Tuple[int, int],
        action_mask: np.ndarray,
        max_steps: int,
    ) -> np.ndarray:
        key = (map_key, position, goal)
        self.cache_lookups += 1
        cached = self.cache.pop(key, None)
        if cached is not None:
            self.cache[key] = cached
            return cached.copy()
        cap = self.config.number("astar.unreachable_successor_cost_cap")
        costs = np.full(len(MOVE_DELTAS), cap, dtype=np.float32)
        for action, (dr, dc) in enumerate(MOVE_DELTAS):
            if not bool(action_mask[action]):
                continue
            successor = (position[0] + dr, position[1] + dc)
            self.planner_calls += 1
            path = astar_path(grid, successor, goal)
            if path is not None:
                costs[action] = min((len(path) - 1) / float(max_steps), cap)
        self.cache[key] = costs.copy()
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return costs

    def collect_target(
        self,
        grid: np.ndarray,
        map_key: str,
        position: Tuple[int, int],
        goal: Tuple[int, int],
        action_mask: np.ndarray,
        max_steps: int,
        interaction_count: int,
    ) -> Tuple[Optional[PlanningTarget], Dict[str, Any]]:
        self.training_transitions += 1
        query_probability, distillation_weight, planning_weight = self.schedule(interaction_count)
        audit: Dict[str, Any] = {
            "interaction_count": int(interaction_count),
            "query_probability": query_probability,
            "distill_weight": distillation_weight,
            "planning_weight": planning_weight,
            "auxiliary_weight": max(distillation_weight, planning_weight),
            "query_performed": False,
            "target_available": False,
        }
        if query_probability <= 0.0 or self.rng.random_sample() > query_probability:
            return None, audit
        self.query_attempts += 1
        costs = self._successor_costs(grid, map_key, position, goal, action_mask, max_steps)
        legal = np.asarray(action_mask, dtype=np.bool_)
        if not legal.any():
            raise IntegrityError("A* planning target has no legal action")
        temperature = self.config.number("astar.soft_target_temperature")
        logits = np.full(len(MOVE_DELTAS), -np.inf, dtype=np.float64)
        logits[legal] = -costs[legal].astype(np.float64) / temperature
        maximum = float(np.max(logits[legal]))
        weights = np.zeros(len(MOVE_DELTAS), dtype=np.float64)
        weights[legal] = np.exp(logits[legal] - maximum)
        weights /= float(weights.sum())
        entropy = -float(np.sum(weights[legal] * np.log(weights[legal] + np.finfo(np.float64).eps)))
        target = PlanningTarget(
            soft_target=weights.astype(np.float32),
            normalized_costs=costs.astype(np.float32),
            legal_mask=legal.copy(),
            distillation_weight=distillation_weight,
            planning_weight=planning_weight,
            target_entropy=entropy,
            interaction_count=int(interaction_count),
        )
        self.targets_generated += 1
        audit.update({
            "query_performed": True,
            "target_available": True,
            "target_entropy": entropy,
            "target_max_probability": float(weights.max()),
            "target_min_legal_probability": float(weights[legal].min()),
            "normalized_cost_mean": float(costs[legal].mean()),
        })
        return target, audit

    def auxiliary_losses(
        self,
        model: ActorCritic,
        spatial: torch.Tensor,
        history: torch.Tensor,
        action_masks: torch.Tensor,
        targets: Sequence[PlanningTarget],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_soft = torch.as_tensor(np.stack([target.soft_target for target in targets]), dtype=spatial.dtype, device=spatial.device)
        target_costs = torch.as_tensor(np.stack([target.normalized_costs for target in targets]), dtype=spatial.dtype, device=spatial.device)
        target_legal = torch.as_tensor(np.stack([target.legal_mask for target in targets]), dtype=torch.bool, device=spatial.device)
        distillation_weights = torch.as_tensor([target.distillation_weight for target in targets], dtype=spatial.dtype, device=spatial.device)
        planning_weights = torch.as_tensor([target.planning_weight for target in targets], dtype=spatial.dtype, device=spatial.device)
        actor_logits, _ = model(spatial, history)
        masked_logits = model.masked_logits(actor_logits, action_masks)
        distillation_per_sample = -(target_soft * F.log_softmax(masked_logits, dim=-1)).sum(dim=-1)
        distillation_loss = distillation_per_sample.mean()
        weighted_distillation_loss = (distillation_weights * distillation_per_sample).mean()
        predicted_costs = model.planning_forward(spatial, history)
        per_action = F.huber_loss(
            predicted_costs[target_legal],
            target_costs[target_legal],
            delta=self.config.number("astar.planning_huber_delta"),
            reduction="none",
        )
        legal_counts = target_legal.sum(dim=-1)
        if torch.any(legal_counts <= 0):
            raise IntegrityError("A* planning target has no legal action")
        sample_indices = torch.arange(len(targets), device=spatial.device).unsqueeze(1).expand_as(target_legal)[target_legal]
        planning_per_sample = torch.zeros(len(targets), dtype=predicted_costs.dtype, device=spatial.device)
        planning_per_sample.scatter_add_(0, sample_indices, per_action)
        planning_per_sample = planning_per_sample / legal_counts.to(dtype=predicted_costs.dtype)
        planning_loss = planning_per_sample.mean()
        weighted_planning_loss = (planning_weights * planning_per_sample).mean()
        return distillation_loss, planning_loss, weighted_distillation_loss, weighted_planning_loss


class RolloutBuffer:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def add(self, **row: Any) -> None:
        self.rows.append(row)

    def clear(self) -> None:
        self.rows.clear()

    def __len__(self) -> int:
        return len(self.rows)


class AStarUsageTracker:
    def __init__(self):
        self.episodes_with_usage = 0
        self.successes_with_usage = 0
        self.episodes_without_usage = 0
        self.successes_without_usage = 0

    def record_episode(self, success: bool, used_astar: bool) -> None:
        if used_astar:
            self.episodes_with_usage += 1
            self.successes_with_usage += int(success)
        else:
            self.episodes_without_usage += 1
            self.successes_without_usage += int(success)

    @property
    def success_rate_with_usage(self) -> float:
        return self.successes_with_usage / float(max(1, self.episodes_with_usage))

    @property
    def success_rate_without_usage(self) -> float:
        return self.successes_without_usage / float(max(1, self.episodes_without_usage))


class PPOAgent:
    def __init__(self, config: FrozenConfig, model: ActorCritic, device: torch.device):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.number("ppo.learning_rate"),
            weight_decay=config.number("ppo.weight_decay"),
            betas=tuple(float(x) for x in config.get("ppo.adam_betas")),
            eps=config.number("ppo.adam_eps"),
        )
        self.update_count = 0

    def tensors(self, observation: Observation, mask: np.ndarray) -> Tuple[torch.Tensor, ...]:
        spatial = torch.as_tensor(observation.spatial[None], dtype=torch.float32, device=self.device)
        history = torch.as_tensor(observation.history[None], dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(mask[None], dtype=torch.bool, device=self.device)
        return spatial, history, masks

    def act(self, observation: Observation, mask: np.ndarray, deterministic: bool) -> Tuple[int, float, float, np.ndarray]:
        spatial, history, masks = self.tensors(observation, mask)
        with torch.no_grad():
            logits, values = self.model(spatial, history)
            masked = self.model.masked_logits(logits, masks)
            distribution = Categorical(logits=masked)
            action = torch.argmax(masked, dim=-1) if deterministic else distribution.sample()
            log_probability = distribution.log_prob(action)
        return int(action.item()), float(log_probability.item()), float(values.item()), masked[0].cpu().numpy()

    def value(self, observation: Observation) -> float:
        spatial = torch.as_tensor(observation.spatial[None], dtype=torch.float32, device=self.device)
        history = torch.as_tensor(observation.history[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, value = self.model(spatial, history)
        return float(value.item())

    def update(self, buffer: RolloutBuffer, supervisor: Optional[AStarPlanningSupervisor]) -> Dict[str, float]:
        if not buffer.rows:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "ppo_loss": 0.0, "distillation_loss": 0.0, "planning_loss": 0.0, "weighted_distillation_loss": 0.0, "weighted_planning_loss": 0.0, "astar_auxiliary_loss": 0.0, "astar_loss_ratio": 0.0, "total_loss": 0.0, "effective_astar_samples": 0.0}
        spatial = torch.as_tensor(np.stack([r["spatial"] for r in buffer.rows]), dtype=torch.float32, device=self.device)
        history = torch.as_tensor(np.stack([r["history"] for r in buffer.rows]), dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(np.stack([r["mask"] for r in buffer.rows]), dtype=torch.bool, device=self.device)
        actions = torch.as_tensor([r["action"] for r in buffer.rows], dtype=torch.long, device=self.device)
        old_log_probs = torch.as_tensor([r["log_prob"] for r in buffer.rows], dtype=torch.float32, device=self.device)
        rewards = np.asarray([r["reward"] for r in buffer.rows], dtype=np.float32)
        dones = np.asarray([r["done"] for r in buffer.rows], dtype=np.float32)
        values = np.asarray([r["value"] for r in buffer.rows], dtype=np.float32)
        next_values = np.asarray([r["next_value"] for r in buffer.rows], dtype=np.float32)
        gamma = self.config.number("ppo.gamma")
        gae_lambda = self.config.number("ppo.gae_lambda")
        advantages = np.zeros_like(rewards)
        accumulator = 0.0
        for index in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[index]
            delta = rewards[index] + gamma * next_values[index] * nonterminal - values[index]
            accumulator = delta + gamma * gae_lambda * nonterminal * accumulator
            advantages[index] = accumulator
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + self.config.number("ppo.advantage_normalization_epsilon")
        )
        advantages = np.clip(advantages, -self.config.number("ppo.advantage_clip"), self.config.number("ppo.advantage_clip"))
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        clip = self.config.number("ppo.clip_epsilon")
        entropy_coefficient = max(
            self.config.number("ppo.entropy_minimum"),
            self.config.number("ppo.entropy_initial") * self.config.number("ppo.entropy_decay") ** self.update_count,
        )
        planning_targets = [row.get("planning_target") for row in buffer.rows]
        if supervisor is None and any(target is not None for target in planning_targets):
            raise IntegrityError("A* target exists while the A* factor is disabled")
        totals = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "ppo_loss": 0.0, "distillation_loss": 0.0, "planning_loss": 0.0, "weighted_distillation_loss": 0.0, "weighted_planning_loss": 0.0, "astar_auxiliary_loss": 0.0, "astar_loss_ratio": 0.0, "total_loss": 0.0, "effective_astar_samples": 0.0}
        epochs = self.config.integer("ppo.epochs")
        for _ in range(epochs):
            logits, predicted_values = self.model(spatial, history)
            masked = self.model.masked_logits(logits, masks)
            distribution = Categorical(logits=masked)
            log_probs = distribution.log_prob(actions)
            ratio = torch.exp(log_probs - old_log_probs)
            surrogate_a = ratio * advantages_t
            surrogate_b = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages_t
            actor_loss = -torch.min(surrogate_a, surrogate_b).mean()
            critic_loss = F.mse_loss(predicted_values, returns_t)
            entropy = distribution.entropy().mean()
            ppo_loss = actor_loss + self.config.number("ppo.value_coefficient") * critic_loss - entropy_coefficient * entropy
            distillation_loss = torch.zeros((), dtype=ppo_loss.dtype, device=self.device)
            planning_loss = torch.zeros((), dtype=ppo_loss.dtype, device=self.device)
            weighted_distillation_loss = torch.zeros((), dtype=ppo_loss.dtype, device=self.device)
            weighted_planning_loss = torch.zeros((), dtype=ppo_loss.dtype, device=self.device)
            astar_auxiliary_loss = torch.zeros((), dtype=ppo_loss.dtype, device=self.device)
            selected_indices = [index for index, target in enumerate(planning_targets) if target is not None]
            if selected_indices:
                if supervisor is None:
                    raise IntegrityError("A* planning targets require an enabled supervisor")
                indices = torch.as_tensor(selected_indices, dtype=torch.long, device=self.device)
                selected_targets = [planning_targets[index] for index in selected_indices]
                distillation_loss, planning_loss, weighted_distillation_loss, weighted_planning_loss = supervisor.auxiliary_losses(
                    self.model,
                    spatial[indices],
                    history[indices],
                    masks[indices],
                    selected_targets,
                )
                astar_auxiliary_loss = weighted_distillation_loss + weighted_planning_loss
            total = ppo_loss + astar_auxiliary_loss
            loss_ratio = abs(float(astar_auxiliary_loss.detach().item())) / (
                abs(float(ppo_loss.detach().item()))
                + abs(float(astar_auxiliary_loss.detach().item()))
                + self.config.number("astar.loss_ratio_epsilon")
            )
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.number("ppo.max_gradient_norm"))
            self.optimizer.step()
            totals = {
                "actor_loss": float(actor_loss.item()),
                "critic_loss": float(critic_loss.item()),
                "entropy": float(entropy.item()),
                "ppo_loss": float(ppo_loss.item()),
                "distillation_loss": float(distillation_loss.item()),
                "planning_loss": float(planning_loss.item()),
                "weighted_distillation_loss": float(weighted_distillation_loss.item()),
                "weighted_planning_loss": float(weighted_planning_loss.item()),
                "astar_auxiliary_loss": float(astar_auxiliary_loss.item()),
                "astar_loss_ratio": loss_ratio,
                "total_loss": float(total.item()),
                "effective_astar_samples": float(len(selected_indices)),
            }
        self.update_count += 1
        buffer.clear()
        return totals


class CsvSink:
    def __init__(self, path: Path, fields: Sequence[str]):
        self.handle = path.open("w", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=list(fields))
        self.writer.writeheader()
        self.count = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow(row)
        self.count += 1
        if self.count % 1000 == 0:
            self.handle.flush()

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


class RunLogger:
    def __init__(self, run: Path):
        self.console = (run / "run_log.txt").open("w", encoding="utf-8")
        self.training = CsvSink(run / "training_metrics.csv", ["episode", "interaction_count", "reward", "success", "success_rate", "collision", "timeout", "truncated", "steps", "actor_loss", "critic_loss", "entropy", "ppo_loss", "distillation_loss", "planning_loss", "weighted_distillation_loss", "weighted_planning_loss", "astar_auxiliary_loss", "astar_loss_ratio", "effective_astar_samples", "total_loss"])
        self.reward = CsvSink(run / "reward_components.csv", ["episode", "step", "interaction_count", "goal_reward", "distance_reward", "collision_penalty", "step_penalty", "timeout_penalty", "revisit_penalty", "oscillation_penalty", "stagnation_penalty", "escape_reward", "other_reward_terms", "total_reward"])
        self.astar = CsvSink(run / "astar_metrics.csv", ["event_type", "episode", "step", "interaction_count", "query_probability", "query_performed", "target_available", "auxiliary_weight", "distill_weight", "planning_weight", "target_entropy", "target_max_probability", "target_min_legal_probability", "normalized_cost_mean", "training_astar_calls", "astar_query_frequency", "targets_generated", "distillation_loss", "planning_loss", "weighted_distillation_loss", "weighted_planning_loss", "astar_auxiliary_loss", "astar_loss_ratio", "effective_astar_samples", "episode_astar_used", "success_rate_with_astar_usage", "success_rate_without_astar_usage"])
        self.astar_schedule = CsvSink(run / "astar_loss_weight_schedule.csv", ["interaction_count", "query_probability", "distill_weight", "planning_weight"])
        self.astar_guidance = CsvSink(run / "astar_guidance_schedule.csv", ["interaction_step", "distill_weight", "planning_weight", "astar_active"])
        self.evaluation = CsvSink(run / "evaluation_metrics.csv", ["record_type", "method", "seed", "split", "checkpoint_interaction", "episode", "task_id", "map_id", "success", "collision", "timeout", "SPL", "path_length", "steps", "optimal_path_length", "success_rate", "collision_rate", "timeout_rate", "average_steps", "average_path_length", "path_efficiency"])

    def log_astar_episode(self, episode: int, step: int, interaction: int, used_astar: bool, supervisor: Optional[AStarPlanningSupervisor], tracker: AStarUsageTracker) -> None:
        self.astar.write({
            "event_type": "episode_summary",
            "episode": episode,
            "step": step,
            "interaction_count": interaction,
            "query_probability": "",
            "query_performed": "",
            "target_available": "",
            "auxiliary_weight": "",
            "distill_weight": "",
            "planning_weight": "",
            "target_entropy": "",
            "target_max_probability": "",
            "target_min_legal_probability": "",
            "normalized_cost_mean": "",
            "training_astar_calls": 0 if supervisor is None else supervisor.planner_calls,
            "astar_query_frequency": 0.0 if supervisor is None else supervisor.query_frequency,
            "targets_generated": 0 if supervisor is None else supervisor.targets_generated,
            "distillation_loss": "",
            "planning_loss": "",
            "weighted_distillation_loss": "",
            "weighted_planning_loss": "",
            "astar_auxiliary_loss": "",
            "astar_loss_ratio": "",
            "effective_astar_samples": "",
            "episode_astar_used": int(used_astar),
            "success_rate_with_astar_usage": tracker.success_rate_with_usage,
            "success_rate_without_astar_usage": tracker.success_rate_without_usage,
        })

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        self.console.write(line + "\n")
        self.console.flush()

    def close(self) -> None:
        self.training.close()
        self.reward.close()
        self.astar.close()
        self.astar_schedule.close()
        self.astar_guidance.close()
        self.evaluation.close()
        self.console.close()


class Evaluator:
    def __init__(self, config: FrozenConfig, model: ActorCritic, device: torch.device, maps: MapStore, logger: RunLogger, method: str, seed: int):
        self.config = config
        self.model = model
        self.device = device
        self.maps = maps
        self.logger = logger
        self.method = method
        self.seed = seed
        self.runtime_astar_calls = 0

    def evaluate(self, tasks: Sequence[Mapping[str, Any]], split: str, interaction: int, trajectory_dir: Optional[Path] = None) -> Dict[str, Any]:
        environment = NavigationEnvironment(self.config, self.maps, reward_enabled=False)
        agent = PPOAgent.__new__(PPOAgent)
        agent.config, agent.model, agent.device = self.config, self.model, self.device
        original_mode = self.model.training
        self.model.eval()
        success_count = collision_count = timeout_count = total_steps = total_path_length = 0
        spl_values: List[float] = []
        efficiencies: List[float] = []
        started = time.perf_counter()
        with torch.no_grad():
            for episode, task in enumerate(tasks, 1):
                observation = environment.reset(task)
                trajectory = [list(environment.builder.position)]
                state_trajectory: List[Dict[str, Any]] = [{"step": 0, "position": list(environment.builder.position), "history_state": observation.history.tolist()}]
                done = False
                final: Dict[str, Any] = {}
                while not done:
                    mask = environment.action_mask()
                    action, _, _, _ = agent.act(observation, mask, deterministic=True)
                    state_trajectory[-1]["action_mask"] = mask.astype(int).tolist()
                    state_trajectory[-1]["selected_action"] = int(action)
                    observation, _, done, final = environment.step(action)
                    trajectory.append(list(environment.builder.position))
                    state_trajectory.append({"step": environment.builder.step_count, "position": list(environment.builder.position), "history_state": observation.history.tolist()})
                success = int(bool(final["success"]))
                collision = int(bool(final["collision"]))
                timeout = int(bool(final["timeout"]))
                steps = environment.builder.step_count
                path_length = sum(trajectory[i] != trajectory[i - 1] for i in range(1, len(trajectory)))
                optimal = int(task["shortest_path_length"])
                spl = success * optimal / float(max(optimal, path_length))
                efficiency = optimal / float(max(optimal, path_length)) if success else 0.0
                success_count += success
                collision_count += collision
                timeout_count += timeout
                total_steps += steps
                total_path_length += path_length
                spl_values.append(spl)
                if success:
                    efficiencies.append(efficiency)
                self.logger.evaluation.write({"record_type": "episode", "method": self.method, "seed": self.seed, "split": split, "checkpoint_interaction": interaction, "episode": episode, "task_id": task["task_id"], "map_id": task["map_id"], "success": success, "collision": collision, "timeout": timeout, "SPL": spl, "path_length": path_length, "steps": steps, "optimal_path_length": optimal, "success_rate": "", "collision_rate": "", "timeout_rate": "", "average_steps": "", "average_path_length": "", "path_efficiency": ""})
                if trajectory_dir is not None:
                    grid = environment.grid
                    if grid is None:
                        raise IntegrityError("map missing during trajectory serialization")
                    obstacles = np.argwhere(grid == 1).tolist()
                    write_json(trajectory_dir / f"episode_{episode:04d}.json", {"episode": episode, "task_id": task["task_id"], "map_id": int(task["map_id"]), "map_size": [int(grid.shape[0]), int(grid.shape[1])], "obstacle_coordinates": obstacles, "start_position": list(task["start"]), "goal_position": list(task["goal"]), "success": bool(success), "collision": bool(collision), "timeout": bool(timeout), "state_trajectory": state_trajectory, "trajectory_coordinates": trajectory, "path_length": path_length, "steps": steps, "optimal_path_length": optimal, "SPL": spl, "astar_runtime_calls": 0})
        self.model.train(original_mode)
        count = len(tasks)
        if success_count + collision_count + timeout_count != count:
            raise IntegrityError("evaluation outcomes are not exhaustive")
        summary = {"episode_count": count, "success_count": success_count, "collision_count": collision_count, "timeout_count": timeout_count, "success_rate": success_count / count, "collision_rate": collision_count / count, "timeout_rate": timeout_count / count, "average_steps": total_steps / count, "average_path_length": total_path_length / count, "spl": float(np.mean(spl_values)), "path_efficiency": float(np.mean(efficiencies)) if efficiencies else 0.0, "inference_seconds": time.perf_counter() - started, "astar_runtime_calls": 0, "parameter_updates": 0}
        self.logger.evaluation.write({"record_type": "summary", "method": self.method, "seed": self.seed, "split": split, "checkpoint_interaction": interaction, "episode": "", "task_id": "", "map_id": "", "success": success_count, "collision": collision_count, "timeout": timeout_count, "SPL": summary["spl"], "path_length": "", "steps": "", "optimal_path_length": "", "success_rate": summary["success_rate"], "collision_rate": summary["collision_rate"], "timeout_rate": summary["timeout_rate"], "average_steps": summary["average_steps"], "average_path_length": summary["average_path_length"], "path_efficiency": summary["path_efficiency"]})
        return summary


def checkpoint_key(metrics: Mapping[str, Any], interaction: int) -> Tuple[float, ...]:
    return (-float(metrics["success_rate"]), -float(metrics["spl"]), float(metrics["timeout_rate"]), float(metrics["average_steps"]), float(interaction))


def save_checkpoint(path: Path, agent: PPOAgent, interaction: int, config: FrozenConfig, protocol: FrozenProtocol, program_sha: str, method: str, seed: int, validation: Mapping[str, Any]) -> None:
    payload = {"schema_version": "final_two_factor_checkpoint.v7", "program_sha256": program_sha, "config_sha256": config.sha256, "protocol_sha256": protocol.sha256, "method": method, "seed": seed, "interaction_count": interaction, "model_state_dict": agent.model.state_dict(), "optimizer_state_dict": agent.optimizer.state_dict(), "ppo_update_count": agent.update_count, "validation_metrics": dict(validation), "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(), "torch_cpu_random_state": torch.get_rng_state(), "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model: ActorCritic, device: torch.device, config: FrozenConfig, protocol: FrozenProtocol, program_sha: str, method: str, seed: int) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("program_sha256") != program_sha or payload.get("config_sha256") != config.sha256 or payload.get("protocol_sha256") != protocol.sha256 or payload.get("method") != method or int(payload.get("seed", -1)) != seed:
        raise IntegrityError("checkpoint identity mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def ensure_archive(config: FrozenConfig, protocol: FrozenProtocol, root: Path) -> None:
    archive = root / "protocol"
    archive.mkdir(parents=True, exist_ok=True)
    for source in (config.path, protocol.path, SOURCE_DIR / "map_split.json", SOURCE_DIR / "task_split.json", SOURCE_DIR / "hash_manifest.json"):
        destination = archive / source.name
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise IntegrityError(f"protocol archive conflict: {destination}")
        else:
            shutil.copy2(source, destination)
    (root / "analysis_ready").mkdir(parents=True, exist_ok=True)


def create_run_directory(root: Path, method: str, seed: int) -> Path:
    run = root / method / f"seed_{seed}" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "models").mkdir()
    (run / "trajectories").mkdir()
    return run


def run_experiment(config: FrozenConfig, protocol: FrozenProtocol, method: str, seed: int, device: torch.device) -> Path:
    if method not in CONFIGURATIONS or seed not in config.get("official_seeds"):
        raise ProtocolError("method or seed is not authorized")
    astar_enabled, reward_enabled = CONFIGURATIONS[method]
    program_sha = sha256_file(SCRIPT_PATH)
    input_hashes = verify_inputs(config)
    train_tasks, validation_tasks, test_tasks = load_tasks(config)
    root = Path(str(config.get("output_root"))).resolve()
    ensure_archive(config, protocol, root)
    run = create_run_directory(root, method, seed)
    logger = RunLogger(run)
    manifest = {"schema_version": "final_two_factor_run.v7", "status": "running", "method": method, "seed": seed, "program_sha256": program_sha, "config_sha256": config.sha256, "protocol_sha256": protocol.sha256, "input_sha256": input_hashes, "run_directory": str(run), "training_interaction_budget": config.integer("training.interaction_budget"), "executed_interaction_count": 0, "training_astar_calls": 0, "astar_training_query_attempts": 0, "astar_training_targets_generated": 0, "validation_astar_runtime_calls": 0, "test_astar_runtime_calls": 0, "selected_checkpoint": None}
    write_json(run / "run_manifest.json", manifest)
    try:
        set_seed(seed)
        maps = MapStore(config.integer("environment.map_size"))
        model = ActorCritic(config)
        agent = PPOAgent(config, model, device)
        task_rng = np.random.RandomState(
            (seed + config.integer("random_streams.training_task_seed_offset")) % (2**32)
        )
        supervisor = (
            AStarPlanningSupervisor(
                config,
                np.random.RandomState(
                    (seed + config.integer("random_streams.astar_query_seed_offset")) % (2**32)
                ),
            )
            if astar_enabled
            else None
        )
        initial_schedule = (0.0, 0.0, 0.0) if supervisor is None else supervisor.schedule(0)
        logger.astar_schedule.write({"interaction_count": 0, "query_probability": initial_schedule[0], "distill_weight": initial_schedule[1], "planning_weight": initial_schedule[2]})
        logger.astar_guidance.write({"interaction_step": 0, "distill_weight": initial_schedule[1], "planning_weight": initial_schedule[2], "astar_active": int(initial_schedule[0] > 0.0)})
        astar_tracker = AStarUsageTracker()
        environment = NavigationEnvironment(config, maps, reward_enabled)
        evaluator = Evaluator(config, model, device, maps, logger, method, seed)
        buffer = RolloutBuffer()
        interaction = 0
        episode = 0
        latest_losses = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "ppo_loss": 0.0, "distillation_loss": 0.0, "planning_loss": 0.0, "weighted_distillation_loss": 0.0, "weighted_planning_loss": 0.0, "astar_auxiliary_loss": 0.0, "astar_loss_ratio": 0.0, "effective_astar_samples": 0.0, "total_loss": 0.0}
        running_successes = 0
        validation_interval = config.integer("training.validation_interval_interactions")
        next_validation = validation_interval
        best_key: Optional[Tuple[float, ...]] = None
        best_path: Optional[Path] = None
        budget = config.integer("training.interaction_budget")
        while interaction < budget:
            episode += 1
            task = train_tasks[int(task_rng.randint(0, len(train_tasks)))]
            observation = environment.reset(task)
            done = False
            episode_reward = 0.0
            episode_astar_used = False
            while not done and interaction < budget:
                mask = environment.action_mask()
                action, log_probability, value, logits = agent.act(observation, mask, deterministic=False)
                planning_target: Optional[PlanningTarget] = None
                astar_record: Dict[str, Any] = {"interaction_count": interaction, "query_probability": 0.0, "auxiliary_weight": 0.0, "distill_weight": 0.0, "planning_weight": 0.0, "query_performed": False, "target_available": False}
                if supervisor is not None and environment.grid is not None:
                    planning_target, astar_record = supervisor.collect_target(
                        environment.grid,
                        str(task["cspace_file"]),
                        environment.builder.position,
                        environment.builder.goal,
                        mask,
                        environment.builder.max_steps,
                        interaction,
                    )
                old_observation = observation
                if planning_target is not None:
                    episode_astar_used = True
                observation, environment_reward, done, info = environment.step(action)
                reward = environment_reward
                interaction += 1
                episode_reward += reward
                next_value = 0.0 if done else agent.value(observation)
                buffer.add(spatial=old_observation.spatial, history=old_observation.history, mask=mask, action=action, log_prob=log_probability, reward=reward, done=done, value=value, next_value=next_value, planning_target=planning_target)
                record: RewardRecord = info["reward"]
                logger.reward.write({"episode": episode, "step": environment.builder.step_count, "interaction_count": interaction, "goal_reward": record.goal_reward, "distance_reward": record.distance_reward, "collision_penalty": record.collision_penalty, "step_penalty": record.step_penalty, "timeout_penalty": record.timeout_penalty, "revisit_penalty": record.revisit_penalty, "oscillation_penalty": record.oscillation_penalty, "stagnation_penalty": record.stagnation_penalty, "escape_reward": record.escape_reward, "other_reward_terms": record.other_reward_terms, "total_reward": record.total_reward})
                logger.astar.write({"event_type": "target_collection", "episode": episode, "step": environment.builder.step_count, "interaction_count": interaction, "query_probability": astar_record["query_probability"], "query_performed": int(bool(astar_record["query_performed"])), "target_available": int(bool(astar_record["target_available"])), "auxiliary_weight": astar_record["auxiliary_weight"], "distill_weight": astar_record["distill_weight"], "planning_weight": astar_record["planning_weight"], "target_entropy": astar_record.get("target_entropy", ""), "target_max_probability": astar_record.get("target_max_probability", ""), "target_min_legal_probability": astar_record.get("target_min_legal_probability", ""), "normalized_cost_mean": astar_record.get("normalized_cost_mean", ""), "training_astar_calls": 0 if supervisor is None else supervisor.planner_calls, "astar_query_frequency": 0.0 if supervisor is None else supervisor.query_frequency, "targets_generated": 0 if supervisor is None else supervisor.targets_generated, "distillation_loss": "", "planning_loss": "", "weighted_distillation_loss": "", "weighted_planning_loss": "", "astar_auxiliary_loss": "", "astar_loss_ratio": "", "effective_astar_samples": "", "episode_astar_used": int(episode_astar_used), "success_rate_with_astar_usage": astar_tracker.success_rate_with_usage, "success_rate_without_astar_usage": astar_tracker.success_rate_without_usage})
                if len(buffer) >= config.integer("ppo.rollout_size"):
                    latest_losses = agent.update(buffer, supervisor)
                    update_schedule = (0.0, 0.0, 0.0) if supervisor is None else supervisor.schedule(interaction)
                    logger.astar_schedule.write({"interaction_count": interaction, "query_probability": update_schedule[0], "distill_weight": update_schedule[1], "planning_weight": update_schedule[2]})
                    logger.astar_guidance.write({"interaction_step": interaction, "distill_weight": update_schedule[1], "planning_weight": update_schedule[2], "astar_active": int(update_schedule[0] > 0.0)})
                    logger.astar.write({"event_type": "ppo_update", "episode": episode, "step": environment.builder.step_count, "interaction_count": interaction, "query_probability": update_schedule[0], "query_performed": "", "target_available": "", "auxiliary_weight": max(update_schedule[1], update_schedule[2]), "distill_weight": update_schedule[1], "planning_weight": update_schedule[2], "target_entropy": "", "target_max_probability": "", "target_min_legal_probability": "", "normalized_cost_mean": "", "training_astar_calls": 0 if supervisor is None else supervisor.planner_calls, "astar_query_frequency": 0.0 if supervisor is None else supervisor.query_frequency, "targets_generated": 0 if supervisor is None else supervisor.targets_generated, "distillation_loss": latest_losses["distillation_loss"], "planning_loss": latest_losses["planning_loss"], "weighted_distillation_loss": latest_losses["weighted_distillation_loss"], "weighted_planning_loss": latest_losses["weighted_planning_loss"], "astar_auxiliary_loss": latest_losses["astar_auxiliary_loss"], "astar_loss_ratio": latest_losses["astar_loss_ratio"], "effective_astar_samples": latest_losses["effective_astar_samples"], "episode_astar_used": int(episode_astar_used), "success_rate_with_astar_usage": astar_tracker.success_rate_with_usage, "success_rate_without_astar_usage": astar_tracker.success_rate_without_usage})
                if interaction >= next_validation and interaction < budget:
                    metrics = evaluator.evaluate(validation_tasks, "validation", interaction)
                    checkpoint = run / "models" / f"checkpoint_interaction{interaction}.pth"
                    save_checkpoint(checkpoint, agent, interaction, config, protocol, program_sha, method, seed, metrics)
                    key = checkpoint_key(metrics, interaction)
                    if best_key is None or key < best_key:
                        best_key, best_path = key, checkpoint
                        write_json(run / "models" / "best_checkpoint.json", {"checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "interaction_count": interaction, "validation_metrics": metrics})
                    logger.log(f"validation interaction={interaction} success={metrics['success_rate']:.4f} SPL={metrics['spl']:.4f}")
                    next_validation += validation_interval
            episode_success = bool(info["success"]) if done else False
            running_successes += int(episode_success)
            astar_tracker.record_episode(episode_success, episode_astar_used)
            logger.log_astar_episode(episode, environment.builder.step_count, interaction, episode_astar_used, supervisor, astar_tracker)
            logger.training.write({"episode": episode, "interaction_count": interaction, "reward": episode_reward, "success": int(episode_success), "success_rate": running_successes / float(episode), "collision": int(bool(info["collision"])) if done else 0, "timeout": int(bool(info["timeout"])) if done else 0, "truncated": int(not done), "steps": environment.builder.step_count, **latest_losses})
        if len(buffer):
            latest_losses = agent.update(buffer, supervisor)
            final_schedule = (0.0, 0.0, 0.0) if supervisor is None else supervisor.schedule(interaction)
            logger.astar_schedule.write({"interaction_count": interaction, "query_probability": final_schedule[0], "distill_weight": final_schedule[1], "planning_weight": final_schedule[2]})
            logger.astar_guidance.write({"interaction_step": interaction, "distill_weight": final_schedule[1], "planning_weight": final_schedule[2], "astar_active": int(final_schedule[0] > 0.0)})
        metrics = evaluator.evaluate(validation_tasks, "validation", budget)
        checkpoint = run / "models" / f"checkpoint_interaction{budget}.pth"
        save_checkpoint(checkpoint, agent, budget, config, protocol, program_sha, method, seed, metrics)
        key = checkpoint_key(metrics, budget)
        if best_key is None or key < best_key:
            best_key, best_path = key, checkpoint
            write_json(run / "models" / "best_checkpoint.json", {"checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "interaction_count": budget, "validation_metrics": metrics})
        if best_path is None:
            raise IntegrityError("no validation-selected checkpoint")
        selected = load_checkpoint(best_path, model, device, config, protocol, program_sha, method, seed)
        test_metrics = evaluator.evaluate(test_tasks, "test", int(selected["interaction_count"]), run / "trajectories")
        if evaluator.runtime_astar_calls != 0 or int(test_metrics["astar_runtime_calls"]) != 0:
            raise IntegrityError("evaluation attempted to use A*")
        manifest.update({"status": "completed", "executed_interaction_count": interaction, "ppo_update_count": agent.update_count, "training_astar_calls": 0 if supervisor is None else supervisor.planner_calls, "astar_training_cache_lookups": 0 if supervisor is None else supervisor.cache_lookups, "astar_training_query_attempts": 0 if supervisor is None else supervisor.query_attempts, "astar_training_targets_generated": 0 if supervisor is None else supervisor.targets_generated, "astar_training_query_frequency": 0.0 if supervisor is None else supervisor.query_frequency, "success_rate_with_astar_usage": astar_tracker.success_rate_with_usage, "success_rate_without_astar_usage": astar_tracker.success_rate_without_usage, "validation_astar_runtime_calls": 0, "test_astar_runtime_calls": 0, "selected_checkpoint": str(best_path), "selected_checkpoint_sha256": sha256_file(best_path), "selected_checkpoint_interaction": int(selected["interaction_count"]), "test_metrics": test_metrics})
        write_json(run / "run_manifest.json", manifest)
        result = {"schema_version": "final_two_factor_result.v7", "method": method, "seed": seed, "program_sha256": program_sha, "config_sha256": config.sha256, "protocol_sha256": protocol.sha256, "input_sha256": input_hashes, "training_interactions": interaction, "training_astar_calls": 0 if supervisor is None else supervisor.planner_calls, "astar_training_query_attempts": 0 if supervisor is None else supervisor.query_attempts, "astar_training_targets_generated": 0 if supervisor is None else supervisor.targets_generated, "astar_training_query_frequency": 0.0 if supervisor is None else supervisor.query_frequency, "success_rate_with_astar_usage": astar_tracker.success_rate_with_usage, "success_rate_without_astar_usage": astar_tracker.success_rate_without_usage, "validation_astar_runtime_calls": 0, "test_astar_runtime_calls": 0, "selected_checkpoint": str(best_path), "selected_checkpoint_sha256": sha256_file(best_path), "test": test_metrics}
        result_path = run / f"formal_result_{method}_seed{seed}.json"
        write_json(result_path, result)
        write_json(root / "analysis_ready" / f"{method}_seed{seed}_{run.name}.json", result)
        logger.log(f"completed method={method} seed={seed} success={test_metrics['success_rate']:.4f} SPL={test_metrics['spl']:.4f}")
        return run
    except Exception:
        manifest["status"] = "failed"
        write_json(run / "run_manifest.json", manifest)
        raise
    finally:
        logger.close()


def summarize(config: FrozenConfig, protocol: FrozenProtocol) -> Path:
    root = Path(str(config.get("output_root"))).resolve()
    records: List[Dict[str, Any]] = []
    for method in CONFIGURATIONS:
        for seed in config.get("official_seeds"):
            matches = list((root / method / f"seed_{seed}").glob("run_*/formal_result_*.json"))
            if len(matches) != 1:
                raise IntegrityError(f"expected one result for {method}/seed_{seed}, found {matches}")
            records.append(read_json(matches[0]))
    metrics = ["success_rate", "spl", "path_efficiency", "timeout_rate", "collision_rate", "average_steps"]
    rows: List[Dict[str, Any]] = []
    for method in CONFIGURATIONS:
        subset = [record for record in records if record["method"] == method]
        row: Dict[str, Any] = {"method": method, "seed_count": len(subset)}
        for metric in metrics:
            values = np.asarray([float(record["test"][metric]) for record in subset], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    output = root / "analysis_ready" / "aggregate_mean_std.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "analysis_ready" / "aggregate_mean_std.json", {"schema_version": "final_two_factor_aggregate.v7", "config_sha256": config.sha256, "protocol_sha256": protocol.sha256, "official_seeds": config.get("official_seeds"), "rows": rows})
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final two-factor PPO navigation experiment v7 with early training-only A* planning guidance")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--method", choices=tuple(CONFIGURATIONS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FrozenConfig(args.config)
    protocol = FrozenProtocol(args.protocol)
    validate_config(config, protocol)
    hashes = verify_inputs(config)
    load_tasks(config)
    if args.audit_only:
        print(json.dumps({"config_sha256": config.sha256, "protocol_sha256": protocol.sha256, "program_sha256": sha256_file(SCRIPT_PATH), "input_sha256": hashes, "formal_execution_allowed": True}, indent=2))
        return
    if args.summarize:
        print(summarize(config, protocol))
        return
    if args.method is None or args.seed is None or args.device is None:
        raise ProtocolError("--method, --seed, and --device are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable")
    print(run_experiment(config, protocol, args.method, args.seed, torch.device(args.device)))


if __name__ == "__main__":
    main()
