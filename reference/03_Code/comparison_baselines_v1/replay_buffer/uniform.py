from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from common.errors import IntegrityError
from environment.navigation import Observation


class UniformReplayBuffer:
    """Disk-backed, uniform, one-step replay with exact float32 observations.

    A single observation array stores current and successor observations using
    the standard memory-efficient circular-buffer layout. The current insertion
    position is excluded from sampling after wrap-around because its successor
    slot has been overwritten.
    """

    def __init__(
        self,
        directory: Path,
        capacity: int,
        spatial_shape=(5, 30, 30),
        history_size: int = 10,
    ):
        if capacity < 3:
            raise ValueError("replay capacity must be at least 3")
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.capacity = int(capacity)
        self.slots = self.capacity + 1
        self.spatial_shape = tuple(int(x) for x in spatial_shape)
        self.history_size = int(history_size)
        self.position = 0
        self.full = False
        self.total_added = 0
        self.spatial = np.memmap(
            self.directory / "spatial.float32.dat",
            mode="w+",
            dtype=np.float32,
            shape=(self.slots, *self.spatial_shape),
        )
        self.history = np.memmap(
            self.directory / "history.float32.dat",
            mode="w+",
            dtype=np.float32,
            shape=(self.slots, self.history_size),
        )
        self.actions = np.memmap(self.directory / "actions.int64.dat", mode="w+", dtype=np.int64, shape=(self.slots,))
        self.rewards = np.memmap(self.directory / "rewards.float32.dat", mode="w+", dtype=np.float32, shape=(self.slots,))
        self.dones = np.memmap(self.directory / "dones.uint8.dat", mode="w+", dtype=np.uint8, shape=(self.slots,))
        self.next_masks = np.memmap(self.directory / "next_masks.uint8.dat", mode="w+", dtype=np.uint8, shape=(self.slots, 4))

    def __len__(self) -> int:
        return self.capacity if self.full else self.position

    def _validate_observation(self, observation: Observation) -> None:
        if observation.spatial.shape != self.spatial_shape or observation.spatial.dtype != np.float32:
            raise IntegrityError("replay spatial observation identity mismatch")
        if observation.history.shape != (self.history_size,) or observation.history.dtype != np.float32:
            raise IntegrityError("replay history observation identity mismatch")

    def add(
        self,
        observation: Observation,
        action: int,
        reward: float,
        done: bool,
        next_observation: Observation,
        next_mask: np.ndarray,
    ) -> None:
        self._validate_observation(observation)
        self._validate_observation(next_observation)
        mask = np.asarray(next_mask, dtype=np.bool_)
        if mask.shape != (4,):
            raise IntegrityError("next action mask must have shape [4]")
        if not done and not mask.any():
            raise IntegrityError("nonterminal replay transition has no legal next action")
        if not 0 <= int(action) < 4:
            raise IntegrityError("replay action is outside the four-action space")
        index = self.position
        next_index = (index + 1) % self.slots
        self.spatial[index] = observation.spatial
        self.history[index] = observation.history
        self.spatial[next_index] = next_observation.spatial
        self.history[next_index] = next_observation.history
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = int(bool(done))
        self.next_masks[index] = mask.astype(np.uint8)
        self.position = next_index
        self.full = self.full or self.position == 0
        self.total_added += 1

    def _sample_indices(self, batch_size: int, rng: np.random.RandomState) -> np.ndarray:
        size = len(self)
        if size < batch_size:
            raise IntegrityError(f"replay has {size} samples but batch size is {batch_size}")
        if not self.full:
            return rng.randint(0, self.position, size=batch_size)
        raw = rng.randint(0, self.capacity, size=batch_size)
        return raw + (raw >= self.position)

    def sample(self, batch_size: int, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
        indices = self._sample_indices(int(batch_size), rng)
        next_indices = (indices + 1) % self.slots
        return {
            "spatial": np.asarray(self.spatial[indices]).copy(),
            "history": np.asarray(self.history[indices]).copy(),
            "actions": np.asarray(self.actions[indices]).copy(),
            "rewards": np.asarray(self.rewards[indices]).copy(),
            "dones": np.asarray(self.dones[indices], dtype=np.float32).copy(),
            "next_spatial": np.asarray(self.spatial[next_indices]).copy(),
            "next_history": np.asarray(self.history[next_indices]).copy(),
            "next_masks": np.asarray(self.next_masks[indices], dtype=np.bool_).copy(),
        }

    @property
    def allocated_bytes(self) -> int:
        observation_bytes = self.slots * (int(np.prod(self.spatial_shape)) + self.history_size) * 4
        transition_bytes = self.slots * (8 + 4 + 1 + 4)
        return observation_bytes + transition_bytes

    def flush(self) -> None:
        for array in (self.spatial, self.history, self.actions, self.rewards, self.dones, self.next_masks):
            array.flush()

    def close(self) -> None:
        """Flush and release Windows file handles while preserving replay files."""
        for array in (self.spatial, self.history, self.actions, self.rewards, self.dones, self.next_masks):
            array.flush()
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
