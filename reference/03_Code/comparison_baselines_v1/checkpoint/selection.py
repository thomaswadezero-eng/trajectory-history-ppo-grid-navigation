from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from common.errors import IntegrityError
from common.io_utils import sha256_file, write_json
from evaluation.metrics import checkpoint_key


def save_checkpoint(
    path: Path,
    agent,
    interaction: int,
    identity: Mapping[str, Any],
    validation_metrics: Mapping[str, Any],
) -> str:
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "schema_version": "comparison_baselines.checkpoint.v1",
        **dict(identity),
        "interaction_count": int(interaction),
        "agent_state": agent.checkpoint_state(),
        "validation_metrics": dict(validation_metrics),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


class CheckpointSelector:
    def __init__(self, models_directory: Path):
        self.models_directory = models_directory
        self.best_key = None
        self.best_path: Optional[Path] = None
        self.history = []

    def consider(
        self,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        interaction: int,
        metrics: Mapping[str, Any],
    ) -> bool:
        key = checkpoint_key(metrics, interaction)
        selected = self.best_key is None or key < self.best_key
        row = {
            "interaction_count": int(interaction),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "validation_metrics": dict(metrics),
            "ranking_key": list(key),
            "became_best": bool(selected),
        }
        self.history.append(row)
        if selected:
            self.best_key = key
            self.best_path = checkpoint_path
            write_json(self.models_directory / "best_checkpoint.json", row)
        write_json(self.models_directory / "checkpoint_history.json", self.history)
        return selected


def load_selected_checkpoint(
    path: Path,
    agent,
    device: torch.device,
    identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("schema_version") != "comparison_baselines.checkpoint.v1":
        raise IntegrityError("checkpoint schema mismatch")
    for key, expected in identity.items():
        if payload.get(key) != expected:
            raise IntegrityError(f"checkpoint identity mismatch: {key}")
    agent.load_checkpoint_state(payload["agent_state"], load_optimizer=False)
    return payload

