from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from stable_baselines3.common.callbacks import BaseCallback

from .core import TaskRecord, read_json, sha256_file, write_json_replace
from .evaluation import evaluate_tasks
from .metrics import checkpoint_key
from .model import entropy_coefficient


class V7ValidationAndBudgetCallback(BaseCallback):
    """Validate at frozen interactions and stop at the exact formal budget."""

    def __init__(
        self,
        *,
        validation_tasks: Sequence[TaskRecord],
        seed: int,
        run_dir: Path,
        manifest_path: Path,
        interaction_budget: int = 900000,
        validation_interval: int = 100000,
        validation_map_store: Any | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.validation_tasks = list(validation_tasks)
        self.seed = int(seed)
        self.run_dir = run_dir
        self.manifest_path = manifest_path
        self.interaction_budget = int(interaction_budget)
        self.validation_map_store = validation_map_store
        self.validation_interactions = list(range(validation_interval, interaction_budget + 1, validation_interval))
        if not self.validation_interactions or self.validation_interactions[-1] != interaction_budget:
            raise ValueError("validation schedule must include the exact interaction budget")
        self.next_validation_index = 0
        self.best_key: tuple[float, ...] | None = None
        self.best_checkpoint: Path | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.validation_history: list[dict[str, Any]] = []

    def _on_training_start(self) -> None:
        self.model.ent_coef = entropy_coefficient(0)

    def _on_rollout_end(self) -> None:
        self.model.ent_coef = entropy_coefficient(int(self.model._n_updates))

    def _validate(self, interaction: int) -> None:
        validation_dir = self.run_dir / "validation" / f"interaction_{interaction:09d}"
        metrics = evaluate_tasks(
            model=self.model,
            tasks=self.validation_tasks,
            split="validation",
            seed=self.seed,
            output_dir=validation_dir,
            checkpoint_interaction=interaction,
            record_trajectories=False,
            map_store=self.validation_map_store,
        )
        checkpoint_path = self.run_dir / "models" / f"checkpoint_interaction_{interaction:09d}.zip"
        if checkpoint_path.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint_path}")
        self.model.save(checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        record = {
            "interaction_count": interaction,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "metrics": metrics,
        }
        self.validation_history.append(record)
        key = checkpoint_key(metrics, interaction)
        if self.best_key is None or key < self.best_key:
            self.best_key = key
            self.best_checkpoint = checkpoint_path
            self.best_metrics = metrics
            write_json_replace(self.run_dir / "models" / "best_checkpoint.json", record)
        write_json_replace(self.run_dir / "validation_history.json", self.validation_history)
        manifest = read_json(self.manifest_path)
        manifest.update(
            {
                "executed_interaction_count": interaction,
                "latest_validation": record,
                "selected_checkpoint": None if self.best_checkpoint is None else str(self.best_checkpoint),
                "selected_checkpoint_sha256": None if self.best_checkpoint is None else sha256_file(self.best_checkpoint),
                "selected_checkpoint_metrics": self.best_metrics,
                "validation_astar_runtime_calls": 0,
                "validation_parameter_updates": 0,
            }
        )
        write_json_replace(self.manifest_path, manifest)

    def _on_step(self) -> bool:
        current = int(self.model.num_timesteps)
        while (
            self.next_validation_index < len(self.validation_interactions)
            and current >= self.validation_interactions[self.next_validation_index]
        ):
            scheduled = self.validation_interactions[self.next_validation_index]
            if current != scheduled:
                raise RuntimeError(f"validation boundary skipped: scheduled={scheduled}, current={current}")
            self._validate(scheduled)
            self.next_validation_index += 1
        if current >= self.interaction_budget:
            if current != self.interaction_budget:
                raise RuntimeError(f"formal interaction budget overshot: {current}")
            return False
        return True

    def _on_training_end(self) -> None:
        if self.next_validation_index != len(self.validation_interactions):
            raise RuntimeError("training ended before all frozen validation points")
        if self.best_checkpoint is None:
            raise RuntimeError("training ended without a validation-selected checkpoint")
