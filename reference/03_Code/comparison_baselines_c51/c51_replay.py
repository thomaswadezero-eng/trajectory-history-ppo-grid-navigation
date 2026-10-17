from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from barn_protocol import IntegrityError, Observation


class C51ReplayBuffer:
    """Uniform disk-backed replay with explicit current/next observations and masks."""

    def __init__(self, directory: Path, capacity: int):
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("replay capacity must be positive")
        self.position = 0
        self.size = 0
        self.total_added = 0
        self.spatial = self._memmap("spatial.float32.dat", np.float32, (self.capacity, 5, 30, 30))
        self.history = self._memmap("history.float32.dat", np.float32, (self.capacity, 10))
        self.next_spatial = self._memmap("next_spatial.float32.dat", np.float32, (self.capacity, 5, 30, 30))
        self.next_history = self._memmap("next_history.float32.dat", np.float32, (self.capacity, 10))
        self.actions = self._memmap("actions.int64.dat", np.int64, (self.capacity,))
        self.rewards = self._memmap("rewards.float32.dat", np.float32, (self.capacity,))
        self.dones = self._memmap("dones.uint8.dat", np.uint8, (self.capacity,))
        self.current_masks = self._memmap("current_masks.uint8.dat", np.uint8, (self.capacity, 4))
        self.next_masks = self._memmap("next_masks.uint8.dat", np.uint8, (self.capacity, 4))

    def _memmap(self, name: str, dtype: np.dtype, shape: tuple[int, ...]) -> np.memmap:
        path = self.directory / name
        if path.exists():
            raise FileExistsError(path)
        return np.memmap(path, mode="w+", dtype=dtype, shape=shape)

    def __len__(self) -> int:
        return self.size

    @staticmethod
    def _validate_observation(observation: Observation) -> None:
        if observation.spatial.shape != (5, 30, 30) or observation.spatial.dtype != np.float32:
            raise IntegrityError("replay spatial observation identity mismatch")
        if observation.history.shape != (10,) or observation.history.dtype != np.float32:
            raise IntegrityError("replay history observation identity mismatch")

    def add(
        self,
        observation: Observation,
        current_mask: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_observation: Observation,
        next_mask: np.ndarray,
    ) -> None:
        self._validate_observation(observation)
        self._validate_observation(next_observation)
        current = np.asarray(current_mask, dtype=np.bool_)
        following = np.asarray(next_mask, dtype=np.bool_)
        if current.shape != (4,) or not current.any() or following.shape != (4,):
            raise IntegrityError("replay masks must be saved four-action masks")
        if not 0 <= int(action) < 4 or not bool(current[int(action)]):
            raise IntegrityError("replay transition action is illegal under the saved current mask")
        if not bool(done) and not following.any():
            raise IntegrityError("nonterminal replay transition has no legal next action")
        index = self.position
        self.spatial[index] = observation.spatial
        self.history[index] = observation.history
        self.next_spatial[index] = next_observation.spatial
        self.next_history[index] = next_observation.history
        self.actions[index] = int(action)
        self.rewards[index] = float(reward)
        self.dones[index] = int(bool(done))
        self.current_masks[index] = current.astype(np.uint8)
        self.next_masks[index] = following.astype(np.uint8)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        self.total_added += 1

    def sample(self, batch_size: int, rng: np.random.RandomState) -> dict[str, np.ndarray]:
        batch_size = int(batch_size)
        if self.size < batch_size:
            raise IntegrityError(f"replay has {self.size} samples but batch size is {batch_size}")
        indices = rng.randint(0, self.size, size=batch_size) if self.size < self.capacity else rng.randint(0, self.capacity, size=batch_size)
        return {
            "spatial": np.asarray(self.spatial[indices]).copy(),
            "history": np.asarray(self.history[indices]).copy(),
            "actions": np.asarray(self.actions[indices]).copy(),
            "rewards": np.asarray(self.rewards[indices]).copy(),
            "dones": np.asarray(self.dones[indices], dtype=np.float32).copy(),
            "next_spatial": np.asarray(self.next_spatial[indices]).copy(),
            "next_history": np.asarray(self.next_history[indices]).copy(),
            "current_masks": np.asarray(self.current_masks[indices], dtype=np.bool_).copy(),
            "next_masks": np.asarray(self.next_masks[indices], dtype=np.bool_).copy(),
        }

    @property
    def allocated_bytes(self) -> int:
        arrays = (
            self.spatial, self.history, self.next_spatial, self.next_history,
            self.actions, self.rewards, self.dones, self.current_masks, self.next_masks,
        )
        return int(sum(array.nbytes for array in arrays))

    def manifest(self) -> Mapping[str, object]:
        return {
            "schema_version": "c51_replay.v1",
            "type": "uniform_disk_backed_explicit_transition_storage",
            "capacity": self.capacity,
            "allocated_bytes": self.allocated_bytes,
            "spatial_shape": [5, 30, 30],
            "history_size": 10,
            "stores_current_observation": True,
            "stores_next_observation": True,
            "stores_current_action_mask": True,
            "stores_next_action_mask": True,
            "prioritized_replay": False,
            "n_step_returns": False,
        }

    def flush(self) -> None:
        for array in (
            self.spatial, self.history, self.next_spatial, self.next_history,
            self.actions, self.rewards, self.dones, self.current_masks, self.next_masks,
        ):
            array.flush()

    def close(self) -> None:
        for array in (
            self.spatial, self.history, self.next_spatial, self.next_history,
            self.actions, self.rewards, self.dones, self.current_masks, self.next_masks,
        ):
            array.flush()
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
