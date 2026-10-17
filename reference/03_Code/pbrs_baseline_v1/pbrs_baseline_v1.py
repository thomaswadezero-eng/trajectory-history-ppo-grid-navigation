#!/usr/bin/env python
"""Strict PPO+PBRS baseline derived from the frozen final two-factor V7 protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
SOURCE_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SOURCE_DIR.parent.parent
BASELINE_PATH = PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7" / "final_two_factor_experiment_v7.py"
BASELINE_CONFIG_PATH = BASELINE_PATH.parent / "experiment_config.json"
EXPECTED_BASELINE_SHA256 = "96093A6B5F024F5D9505ACCEBEAE7592700228BA17587BF0590092A241B7544D"
EXPECTED_BASELINE_CONFIG_FILE_SHA256 = "CF229DE128FF16E29B73308A7E9B2756DE51294671573F067306274EDA8D4166"
DEFAULT_CONFIG = SOURCE_DIR / "pbrs_config_v1.json"
DEFAULT_PROTOCOL = SOURCE_DIR / "pbrs_protocol_v1.json"
HASH_MANIFEST = SOURCE_DIR / "hash_manifest.json"
METHOD = "PPO_PBRS"
OFFICIAL_SEEDS = [42, 1004, 6010, 68882, 19816, 74494, 81399, 31524]
PROGRESS_INTERVAL = 50000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _load_frozen_baseline() -> Any:
    if not BASELINE_PATH.is_file():
        raise RuntimeError(f"missing frozen V7 baseline: {BASELINE_PATH}")
    actual = sha256_file(BASELINE_PATH)
    if actual != EXPECTED_BASELINE_SHA256:
        raise RuntimeError(
            f"frozen V7 baseline SHA mismatch: expected={EXPECTED_BASELINE_SHA256}, actual={actual}"
        )
    spec = importlib.util.spec_from_file_location("frozen_final_two_factor_v7", BASELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen V7 baseline module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v7 = _load_frozen_baseline()
FrozenConfig = v7.FrozenConfig
FrozenProtocol = v7.FrozenProtocol
ProtocolError = v7.ProtocolError
IntegrityError = v7.IntegrityError
Observation = v7.Observation
StateBuilder = v7.StateBuilder
ActorCritic = v7.ActorCritic
PPOAgent = v7.PPOAgent
RolloutBuffer = v7.RolloutBuffer
MapStore = v7.MapStore
Evaluator = v7.Evaluator
CsvSink = v7.CsvSink
MOVE_DELTAS = v7.MOVE_DELTAS


def read_json(path: Path) -> Dict[str, Any]:
    return v7.read_json(path)


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _source_v7_config() -> Dict[str, Any]:
    if sha256_file(BASELINE_CONFIG_PATH) != EXPECTED_BASELINE_CONFIG_FILE_SHA256:
        raise IntegrityError("frozen V7 configuration physical SHA mismatch")
    return read_json(BASELINE_CONFIG_PATH)


def validate_config(config: FrozenConfig, protocol: FrozenProtocol) -> None:
    source = _source_v7_config()
    if config.get("schema_version") != "pbrs_baseline.v1":
        raise ProtocolError("unsupported PBRS configuration schema")
    if config.get("status") != "FROZEN_AND_AUTHORIZED" or config.get("formal_execution_allowed") is not True:
        raise ProtocolError("PBRS configuration is not frozen and authorized")
    expected_program = "03_Code/pbrs_baseline_v1/pbrs_baseline_v1.py"
    if config.get("authorized_program") != expected_program:
        raise ProtocolError("PBRS authorized program path mismatch")
    if str(config.get("authorized_program_sha256")).upper() != sha256_file(SCRIPT_PATH):
        raise ProtocolError("PBRS authorized program SHA mismatch")
    if str(config.get("source_v7.program_sha256")).upper() != EXPECTED_BASELINE_SHA256:
        raise ProtocolError("PBRS source V7 program identity mismatch")
    if str(config.get("source_v7.config_file_sha256")).upper() != EXPECTED_BASELINE_CONFIG_FILE_SHA256:
        raise ProtocolError("PBRS source V7 configuration identity mismatch")
    if list(config.get("official_seeds")) != OFFICIAL_SEEDS:
        raise ProtocolError("PBRS official seed list mismatch")
    if config.get("method") != METHOD:
        raise ProtocolError("PBRS method identity mismatch")
    expected_factors = {"pbrs": True, "dynamic_reward": False, "astar_supervision": False}
    if config.get("factors") != expected_factors:
        raise ProtocolError("PBRS factor isolation mismatch")
    for section in ("random_streams", "environment", "state", "network", "ppo", "astar", "training", "evaluation"):
        if config.get(section) != source[section]:
            raise ProtocolError(f"protected V7 section changed: {section}")
    if config.get("reward.base") != source["reward"]["base"]:
        raise ProtocolError("base reward differs from frozen V7")
    dynamic = config.get("reward.dynamic_disabled")
    expected_dynamic = {
        "progress_weight": 0.0,
        "revisit_weight": 0.0,
        "oscillation_weight": 0.0,
        "stagnation_weight": 0.0,
    }
    if dynamic != expected_dynamic:
        raise ProtocolError("Dynamic Reward must be exactly zero in all four components")
    source_alpha = float(source["reward"]["dynamic"]["distance_weight"])
    if config.number("reward.pbrs.alpha") != source_alpha:
        raise ProtocolError("PBRS alpha must equal frozen V7 Progress distance_weight")
    if config.get("reward.pbrs.alpha_source") != "source_v7.reward.dynamic.distance_weight":
        raise ProtocolError("PBRS alpha provenance mismatch")
    if config.get("reward.pbrs.distance_metric") != source["reward"]["dynamic"]["distance_metric"]:
        raise ProtocolError("PBRS distance metric differs from frozen V7 Progress")
    if config.get("reward.pbrs.distance_normalization") != source["reward"]["dynamic"]["distance_normalization"]:
        raise ProtocolError("PBRS normalization differs from frozen V7 Progress")
    if config.get("reward.pbrs.formula") != "alpha * (gamma * Phi(s_next) - Phi(s_current))":
        raise ProtocolError("PBRS formula declaration mismatch")
    if config.get("reward.pbrs.clipping") is not False:
        raise ProtocolError("PBRS clipping must be disabled")
    if config.get("reward.pbrs.terminal_phi_next") != 0.0:
        raise ProtocolError("terminal Phi(s_next) must be zero")
    if config.integer("training.interaction_budget") != 900000:
        raise ProtocolError("PBRS training budget mismatch")
    if config.integer("training.validation_interval_interactions") != 100000:
        raise ProtocolError("PBRS validation interval mismatch")
    if config.number("ppo.gamma") != float(source["ppo"]["gamma"]):
        raise ProtocolError("PBRS gamma differs from frozen V7 PPO gamma")
    if config.get("output_root") != r"redacted_local_root\revision_pbrs_v1":
        raise ProtocolError("PBRS output root mismatch")
    if protocol.get("schema_version") != "pbrs_baseline.v1.protocol":
        raise ProtocolError("unsupported PBRS protocol schema")
    if protocol.get("status") != "FROZEN_AND_AUTHORIZED" or protocol.get("formal_execution_allowed") is not True:
        raise ProtocolError("PBRS protocol is not frozen and authorized")
    if protocol.get("authorized_program") != expected_program:
        raise ProtocolError("PBRS protocol program path mismatch")
    if str(protocol.get("authorized_program_sha256")).upper() != sha256_file(SCRIPT_PATH):
        raise ProtocolError("PBRS protocol program SHA mismatch")
    if str(protocol.get("configuration.self_field_zeroed_sha256")).upper() != config.sha256:
        raise ProtocolError("PBRS protocol/config identity mismatch")
    if list(protocol.get("official_seeds")) != OFFICIAL_SEEDS:
        raise ProtocolError("PBRS protocol seed list mismatch")
    if protocol.get("inputs") != config.get("inputs"):
        raise ProtocolError("PBRS protocol input identity mismatch")
    if protocol.get("factors") != expected_factors:
        raise ProtocolError("PBRS protocol factor isolation mismatch")


def verify_inputs(config: FrozenConfig) -> Dict[str, str]:
    return v7.verify_inputs(config)


def load_tasks(config: FrozenConfig) -> Tuple[List[Dict[str, Any]], ...]:
    return v7.load_tasks(config)


def potential(position: Tuple[int, int], goal: Tuple[int, int], grid_size: int) -> float:
    """Phi(s) = -Euclidean(position, goal) / grid diagonal; no clipping."""
    diagonal = math.hypot(grid_size - 1, grid_size - 1)
    if diagonal <= 0.0:
        raise IntegrityError("grid diagonal must be positive")
    value = -math.hypot(position[0] - goal[0], position[1] - goal[1]) / diagonal
    if not math.isfinite(value):
        raise IntegrityError("non-finite PBRS potential")
    return value


def pbrs_shaping(
    current_position: Tuple[int, int],
    next_position: Tuple[int, int],
    goal: Tuple[int, int],
    grid_size: int,
    alpha: float,
    gamma: float,
    terminal: bool,
) -> Tuple[float, float, float]:
    """Return alpha*(gamma*Phi(s_next)-Phi(s)); terminal Phi(s_next) is exactly zero."""
    phi_current = potential(current_position, goal, grid_size)
    # The zero convention applies uniformly to success, collision, and timeout terminals.
    phi_next = 0.0 if terminal else potential(next_position, goal, grid_size)
    shaping = alpha * (gamma * phi_next - phi_current)
    if not all(math.isfinite(x) for x in (phi_current, phi_next, shaping)):
        raise IntegrityError("non-finite PBRS reward")
    return shaping, phi_current, phi_next


@dataclass
class PBRSRewardRecord:
    goal_reward: float = 0.0
    collision_penalty: float = 0.0
    step_penalty: float = 0.0
    timeout_penalty: float = 0.0
    pbrs_reward: float = 0.0
    phi_current: float = 0.0
    phi_next: float = 0.0
    progress_reward: float = 0.0
    revisit_penalty: float = 0.0
    oscillation_penalty: float = 0.0
    stagnation_penalty: float = 0.0

    @property
    def base_reward(self) -> float:
        return self.goal_reward + self.collision_penalty + self.step_penalty + self.timeout_penalty

    @property
    def total_reward(self) -> float:
        return self.base_reward + self.pbrs_reward


class PBRSNavigationEnvironment(v7.NavigationEnvironment):
    """Frozen V7 environment transition with Dynamic Reward off and strict PBRS added."""

    def __init__(self, config: FrozenConfig, maps: MapStore):
        super().__init__(config, maps, reward_enabled=False)
        self.alpha = config.number("reward.pbrs.alpha")
        self.gamma = config.number("ppo.gamma")

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
        self.done = success or collision or timeout
        pbrs_reward, phi_current, phi_next = pbrs_shaping(
            old,
            new,
            self.builder.goal,
            self.config.integer("environment.map_size"),
            self.alpha,
            self.gamma,
            self.done,
        )
        record = PBRSRewardRecord(
            goal_reward=self.config.number("reward.base.goal") if success else 0.0,
            collision_penalty=self.config.number("reward.base.collision") if collision else 0.0,
            step_penalty=self.config.number("reward.base.step"),
            timeout_penalty=self.config.number("reward.base.timeout") if timeout else 0.0,
            pbrs_reward=pbrs_reward,
            phi_current=phi_current,
            phi_next=phi_next,
        )
        info = {
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "valid_action": valid,
            "reward": record,
            "position": new,
            "astar_queries": 0,
        }
        return self.builder.build(), record.total_reward, self.done, info


class PBRSRunLogger:
    def __init__(self, run: Path):
        self.console = (run / "run_log.txt").open("x", encoding="utf-8", buffering=1)
        self.training = CsvSink(run / "training_metrics.csv", ["episode", "interaction_count", "reward", "success", "success_rate", "collision", "timeout", "truncated", "steps", "mean_pbrs_reward", "actor_loss", "critic_loss", "entropy", "ppo_loss", "distillation_loss", "planning_loss", "weighted_distillation_loss", "weighted_planning_loss", "astar_auxiliary_loss", "astar_loss_ratio", "effective_astar_samples", "total_loss"])
        self.reward = CsvSink(run / "reward_components.csv", ["episode", "step", "interaction_count", "goal_reward", "collision_penalty", "step_penalty", "timeout_penalty", "base_reward", "pbrs_reward", "phi_current", "phi_next", "alpha", "gamma", "progress_reward", "revisit_penalty", "oscillation_penalty", "stagnation_penalty", "total_reward"])
        self.astar = CsvSink(run / "astar_metrics.csv", ["event_type", "episode", "step", "interaction_count", "training_astar_calls", "astar_query_attempts", "targets_generated", "distillation_loss", "planning_loss", "weighted_distillation_loss", "weighted_planning_loss", "astar_auxiliary_loss", "astar_loss_ratio", "effective_astar_samples"])
        self.astar_schedule = CsvSink(run / "astar_loss_weight_schedule.csv", ["interaction_count", "query_probability", "distill_weight", "planning_weight"])
        self.astar_guidance = CsvSink(run / "astar_guidance_schedule.csv", ["interaction_step", "distill_weight", "planning_weight", "astar_active"])
        self.evaluation = CsvSink(run / "evaluation_metrics.csv", ["record_type", "method", "seed", "split", "checkpoint_interaction", "episode", "task_id", "map_id", "success", "collision", "timeout", "SPL", "path_length", "steps", "optimal_path_length", "success_rate", "collision_rate", "timeout_rate", "average_steps", "average_path_length", "path_efficiency"])

    def log(self, message: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        self.console.write(line + "\n")

    def close(self) -> None:
        self.training.close()
        self.reward.close()
        self.astar.close()
        self.astar_schedule.close()
        self.astar_guidance.close()
        self.evaluation.close()
        self.console.flush()
        self.console.close()


def _model_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest().upper()


def checkpoint_key(metrics: Mapping[str, Any], interaction: int) -> Tuple[float, ...]:
    return v7.checkpoint_key(metrics, interaction)


def save_checkpoint(path: Path, agent: PPOAgent, interaction: int, config: FrozenConfig, protocol: FrozenProtocol, program_sha: str, seed: int, validation: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": "pbrs_baseline_checkpoint.v1",
        "program_sha256": program_sha,
        "config_sha256": config.sha256,
        "protocol_sha256": protocol.sha256,
        "method": METHOD,
        "seed": seed,
        "interaction_count": interaction,
        "model_state_dict": agent.model.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "ppo_update_count": agent.update_count,
        "validation_metrics": dict(validation),
        "python_random_state": v7.random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_random_state": torch.get_rng_state(),
        "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, model: ActorCritic, device: torch.device, config: FrozenConfig, protocol: FrozenProtocol, program_sha: str, seed: int) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    expected = (program_sha, config.sha256, protocol.sha256, METHOD, seed)
    actual = (
        payload.get("program_sha256"),
        payload.get("config_sha256"),
        payload.get("protocol_sha256"),
        payload.get("method"),
        int(payload.get("seed", -1)),
    )
    if actual != expected:
        raise IntegrityError("PBRS checkpoint identity mismatch")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return payload


def ensure_archive(config: FrozenConfig, protocol: FrozenProtocol, root: Path) -> None:
    archive = root / "protocol"
    archive.mkdir(parents=True, exist_ok=True)
    sources = [config.path, protocol.path, HASH_MANIFEST, BASELINE_PATH, BASELINE_CONFIG_PATH, BASELINE_PATH.parent / "map_split.json", BASELINE_PATH.parent / "task_split.json"]
    for source in sources:
        destination = archive / source.name
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise IntegrityError(f"PBRS protocol archive conflict: {destination}")
        else:
            shutil.copy2(source, destination)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _gpu_memory_mb(device: torch.device) -> str:
    if device.type != "cuda":
        return "N/A"
    return f"{torch.cuda.memory_allocated(device) / (1024 ** 2):.1f}"


def _print_validation(logger: PBRSRunLogger, seed: int, interaction: int, metrics: Mapping[str, Any], updated: bool, best_metrics: Mapping[str, Any]) -> None:
    logger.log("=" * 60)
    logger.log(f"[VALIDATION] Seed {seed} | {interaction} / 900000")
    logger.log(f"Success Rate : {float(metrics['success_rate']):.4f}")
    logger.log(f"SPL          : {float(metrics['spl']):.4f}")
    logger.log(f"Path Eff.    : {float(metrics['path_efficiency']):.4f}")
    logger.log(f"Avg. Steps   : {float(metrics['average_steps']):.4f}")
    logger.log(f"Timeout Rate : {float(metrics['timeout_rate']):.4f}")
    logger.log(f"Checkpoint   : {'UPDATED' if updated else 'NOT UPDATED'}")
    logger.log(f"Best so far  : success={float(best_metrics['success_rate']):.4f}, SPL={float(best_metrics['spl']):.4f}")
    if updated:
        logger.log("[CHECKPOINT] New best validation checkpoint saved.")
    logger.log("=" * 60)


def _write_seed_summary_csv(path: Path, result: Mapping[str, Any]) -> None:
    row = {
        "seed": result["seed"],
        "success_rate": result["test"]["success_rate"],
        "spl": result["test"]["spl"],
        "path_efficiency": result["test"]["path_efficiency"],
        "average_steps": result["test"]["average_steps"],
        "timeout_rate": result["test"]["timeout_rate"],
        "selected_checkpoint": result["selected_checkpoint"],
        "training_wall_clock_seconds": result["training_wall_clock_seconds"],
    }
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def run_experiment(config: FrozenConfig, protocol: FrozenProtocol, seed: int, device: torch.device, run: Path) -> Path:
    if seed not in OFFICIAL_SEEDS:
        raise ProtocolError("seed is not authorized for PBRS")
    root = Path(str(config.get("output_root"))).resolve()
    expected_seed_dir = root / f"seed_{seed}"
    try:
        run.resolve().relative_to(expected_seed_dir.resolve())
    except ValueError as exc:
        raise ProtocolError(f"run directory must be below {expected_seed_dir}") from exc
    if run.exists():
        raise FileExistsError(run)
    ensure_archive(config, protocol, root)
    run.mkdir(parents=True, exist_ok=False)
    (run / "models").mkdir()
    (run / "trajectories").mkdir()
    shutil.copy2(config.path, run / config.path.name)
    shutil.copy2(protocol.path, run / protocol.path.name)
    shutil.copy2(HASH_MANIFEST, run / HASH_MANIFEST.name)
    logger = PBRSRunLogger(run)
    program_sha = sha256_file(SCRIPT_PATH)
    input_hashes = verify_inputs(config)
    train_tasks, validation_tasks, test_tasks = load_tasks(config)
    manifest: Dict[str, Any] = {
        "schema_version": "pbrs_baseline_run.v1",
        "status": "running",
        "method": METHOD,
        "seed": seed,
        "program_sha256": program_sha,
        "source_v7_program_sha256": EXPECTED_BASELINE_SHA256,
        "config_sha256": config.sha256,
        "protocol_sha256": protocol.sha256,
        "input_sha256": input_hashes,
        "run_directory": str(run),
        "training_interaction_budget": config.integer("training.interaction_budget"),
        "executed_interaction_count": 0,
        "pbrs_alpha": config.number("reward.pbrs.alpha"),
        "gamma": config.number("ppo.gamma"),
        "pbrs_potential": "Phi(s)=-||position-goal||_2/grid_diagonal",
        "pbrs_terminal_handling": "Phi(s_next)=0 for success, collision, and timeout",
        "pbrs_clipping": False,
        "dynamic_reward": {"progress": 0.0, "revisit": 0.0, "oscillation": 0.0, "stagnation": 0.0},
        "training_astar_calls": 0,
        "astar_training_query_attempts": 0,
        "astar_training_targets_generated": 0,
        "validation_astar_runtime_calls": 0,
        "test_astar_runtime_calls": 0,
        "selected_checkpoint": None,
    }
    write_json_atomic(run / "run_manifest.json", manifest)
    started_total = time.perf_counter()
    try:
        v7.set_seed(seed)
        maps = MapStore(config.integer("environment.map_size"))
        model = ActorCritic(config)
        agent = PPOAgent(config, model, device)
        task_rng = np.random.RandomState((seed + config.integer("random_streams.training_task_seed_offset")) % (2**32))
        environment = PBRSNavigationEnvironment(config, maps)
        evaluator = Evaluator(config, model, device, maps, logger, METHOD, seed)
        buffer = RolloutBuffer()
        logger.astar_schedule.write({"interaction_count": 0, "query_probability": 0.0, "distill_weight": 0.0, "planning_weight": 0.0})
        logger.astar_guidance.write({"interaction_step": 0, "distill_weight": 0.0, "planning_weight": 0.0, "astar_active": 0})
        interaction = 0
        episode = 0
        running_successes = 0
        recent_rewards: deque[float] = deque(maxlen=100)
        recent_successes: deque[int] = deque(maxlen=100)
        progress_pbrs_sum = 0.0
        progress_pbrs_count = 0
        budget = config.integer("training.interaction_budget")
        validation_interval = config.integer("training.validation_interval_interactions")
        next_validation = validation_interval
        next_progress = PROGRESS_INTERVAL
        best_key: Optional[Tuple[float, ...]] = None
        best_path: Optional[Path] = None
        best_metrics: Optional[Mapping[str, Any]] = None
        latest_losses = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "ppo_loss": 0.0, "distillation_loss": 0.0, "planning_loss": 0.0, "weighted_distillation_loss": 0.0, "weighted_planning_loss": 0.0, "astar_auxiliary_loss": 0.0, "astar_loss_ratio": 0.0, "effective_astar_samples": 0.0, "total_loss": 0.0}
        training_started = time.perf_counter()
        logger.log(f"[TRAIN] PBRS seed={seed} budget={budget} device={device}")
        while interaction < budget:
            episode += 1
            task = train_tasks[int(task_rng.randint(0, len(train_tasks)))]
            observation = environment.reset(task)
            done = False
            episode_reward = 0.0
            episode_pbrs_sum = 0.0
            episode_steps = 0
            info: Dict[str, Any] = {"success": False, "collision": False, "timeout": False}
            while not done and interaction < budget:
                mask = environment.action_mask()
                action, log_probability, value, _ = agent.act(observation, mask, deterministic=False)
                old_observation = observation
                observation, reward, done, info = environment.step(action)
                interaction += 1
                episode_steps += 1
                episode_reward += reward
                record: PBRSRewardRecord = info["reward"]
                episode_pbrs_sum += record.pbrs_reward
                progress_pbrs_sum += record.pbrs_reward
                progress_pbrs_count += 1
                next_value = 0.0 if done else agent.value(observation)
                buffer.add(spatial=old_observation.spatial, history=old_observation.history, mask=mask, action=action, log_prob=log_probability, reward=reward, done=done, value=value, next_value=next_value, planning_target=None)
                logger.reward.write({"episode": episode, "step": environment.builder.step_count, "interaction_count": interaction, "goal_reward": record.goal_reward, "collision_penalty": record.collision_penalty, "step_penalty": record.step_penalty, "timeout_penalty": record.timeout_penalty, "base_reward": record.base_reward, "pbrs_reward": record.pbrs_reward, "phi_current": record.phi_current, "phi_next": record.phi_next, "alpha": environment.alpha, "gamma": environment.gamma, "progress_reward": 0.0, "revisit_penalty": 0.0, "oscillation_penalty": 0.0, "stagnation_penalty": 0.0, "total_reward": record.total_reward})
                if len(buffer) >= config.integer("ppo.rollout_size"):
                    latest_losses = agent.update(buffer, None)
                    if any(latest_losses[key] != 0.0 for key in ("distillation_loss", "planning_loss", "weighted_distillation_loss", "weighted_planning_loss", "astar_auxiliary_loss", "effective_astar_samples")):
                        raise IntegrityError("A* auxiliary quantity became nonzero in PBRS")
                    logger.astar_schedule.write({"interaction_count": interaction, "query_probability": 0.0, "distill_weight": 0.0, "planning_weight": 0.0})
                    logger.astar_guidance.write({"interaction_step": interaction, "distill_weight": 0.0, "planning_weight": 0.0, "astar_active": 0})
                    logger.astar.write({"event_type": "ppo_update", "episode": episode, "step": environment.builder.step_count, "interaction_count": interaction, "training_astar_calls": 0, "astar_query_attempts": 0, "targets_generated": 0, "distillation_loss": latest_losses["distillation_loss"], "planning_loss": latest_losses["planning_loss"], "weighted_distillation_loss": latest_losses["weighted_distillation_loss"], "weighted_planning_loss": latest_losses["weighted_planning_loss"], "astar_auxiliary_loss": latest_losses["astar_auxiliary_loss"], "astar_loss_ratio": latest_losses["astar_loss_ratio"], "effective_astar_samples": latest_losses["effective_astar_samples"]})
                if interaction >= next_progress:
                    elapsed = time.perf_counter() - training_started
                    eta = elapsed * (budget - interaction) / float(max(1, interaction))
                    mean_reward = float(np.mean(recent_rewards)) if recent_rewards else episode_reward
                    mean_success = float(np.mean(recent_successes)) if recent_successes else float(bool(info.get("success")))
                    mean_pbrs = progress_pbrs_sum / float(max(1, progress_pbrs_count))
                    logger.log(f"[Seed {seed}] {interaction} / {budget} ({100.0 * interaction / budget:.1f}%)")
                    logger.log(f"Elapsed: {_format_duration(elapsed)} | ETA: {_format_duration(eta)} | Episodes: {episode}")
                    logger.log(f"Recent train reward: {mean_reward:.6f} | Recent success: {mean_success:.4f}")
                    logger.log(f"PBRS mean reward: {mean_pbrs:.8f} | GPU memory: {_gpu_memory_mb(device)} MB | A* queries: 0")
                    progress_pbrs_sum = 0.0
                    progress_pbrs_count = 0
                    while next_progress <= interaction:
                        next_progress += PROGRESS_INTERVAL
                if interaction >= next_validation and interaction < budget:
                    metrics = evaluator.evaluate(validation_tasks, "validation", interaction)
                    checkpoint = run / "models" / f"checkpoint_interaction{interaction}.pth"
                    save_checkpoint(checkpoint, agent, interaction, config, protocol, program_sha, seed, metrics)
                    key = checkpoint_key(metrics, interaction)
                    updated = best_key is None or key < best_key
                    if updated:
                        best_key, best_path, best_metrics = key, checkpoint, metrics
                        write_json_atomic(run / "models" / "best_checkpoint.json", {"checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "interaction_count": interaction, "validation_metrics": metrics})
                    if best_metrics is None:
                        raise IntegrityError("missing best validation metrics")
                    _print_validation(logger, seed, interaction, metrics, updated, best_metrics)
                    while next_validation <= interaction:
                        next_validation += validation_interval
            episode_success = bool(info["success"]) if done else False
            running_successes += int(episode_success)
            recent_rewards.append(episode_reward)
            recent_successes.append(int(episode_success))
            mean_episode_pbrs = episode_pbrs_sum / float(max(1, episode_steps))
            logger.training.write({"episode": episode, "interaction_count": interaction, "reward": episode_reward, "success": int(episode_success), "success_rate": running_successes / float(episode), "collision": int(bool(info["collision"])) if done else 0, "timeout": int(bool(info["timeout"])) if done else 0, "truncated": int(not done), "steps": environment.builder.step_count, "mean_pbrs_reward": mean_episode_pbrs, **latest_losses})
        if len(buffer):
            latest_losses = agent.update(buffer, None)
            if any(latest_losses[key] != 0.0 for key in ("distillation_loss", "planning_loss", "weighted_distillation_loss", "weighted_planning_loss", "astar_auxiliary_loss", "effective_astar_samples")):
                raise IntegrityError("A* auxiliary quantity became nonzero in final PBRS update")
            logger.astar_schedule.write({"interaction_count": interaction, "query_probability": 0.0, "distill_weight": 0.0, "planning_weight": 0.0})
            logger.astar_guidance.write({"interaction_step": interaction, "distill_weight": 0.0, "planning_weight": 0.0, "astar_active": 0})
        training_wall = time.perf_counter() - training_started
        metrics = evaluator.evaluate(validation_tasks, "validation", budget)
        checkpoint = run / "models" / f"checkpoint_interaction{budget}.pth"
        save_checkpoint(checkpoint, agent, budget, config, protocol, program_sha, seed, metrics)
        key = checkpoint_key(metrics, budget)
        updated = best_key is None or key < best_key
        if updated:
            best_key, best_path, best_metrics = key, checkpoint, metrics
            write_json_atomic(run / "models" / "best_checkpoint.json", {"checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "interaction_count": budget, "validation_metrics": metrics})
        if best_path is None or best_metrics is None:
            raise IntegrityError("no validation-selected PBRS checkpoint")
        _print_validation(logger, seed, budget, metrics, updated, best_metrics)
        selected = load_checkpoint(best_path, model, device, config, protocol, program_sha, seed)
        before_test = _model_fingerprint(model)
        test_metrics = evaluator.evaluate(test_tasks, "test", int(selected["interaction_count"]), run / "trajectories")
        after_test = _model_fingerprint(model)
        if before_test != after_test:
            raise IntegrityError("model parameters changed during final test")
        if evaluator.runtime_astar_calls != 0 or int(test_metrics["astar_runtime_calls"]) != 0:
            raise IntegrityError("evaluation attempted to use A*")
        total_wall = time.perf_counter() - started_total
        logger.log("=" * 60)
        logger.log(f"[FINAL TEST] PBRS | Seed {seed}")
        logger.log(f"Success Rate : {float(test_metrics['success_rate']):.4f}")
        logger.log(f"SPL          : {float(test_metrics['spl']):.4f}")
        logger.log(f"Path Eff.    : {float(test_metrics['path_efficiency']):.4f}")
        logger.log(f"Avg. Steps   : {float(test_metrics['average_steps']):.4f}")
        logger.log(f"Timeout Rate : {float(test_metrics['timeout_rate']):.4f}")
        logger.log(f"Selected ckpt: {best_path}")
        logger.log(f"Training time: {_format_duration(training_wall)}")
        logger.log("=" * 60)
        manifest.update({"status": "completed", "executed_interaction_count": interaction, "ppo_update_count": agent.update_count, "selected_checkpoint": str(best_path), "selected_checkpoint_sha256": sha256_file(best_path), "selected_checkpoint_interaction": int(selected["interaction_count"]), "training_wall_clock_seconds": training_wall, "total_wall_clock_seconds": total_wall, "test_parameter_updates": 0, "test_metrics": test_metrics})
        write_json_atomic(run / "run_manifest.json", manifest)
        result = {"schema_version": "pbrs_baseline_result.v1", "method": METHOD, "seed": seed, "program_sha256": program_sha, "source_v7_program_sha256": EXPECTED_BASELINE_SHA256, "config_sha256": config.sha256, "protocol_sha256": protocol.sha256, "input_sha256": input_hashes, "training_interactions": interaction, "pbrs_alpha": environment.alpha, "gamma": environment.gamma, "pbrs_potential": "Phi(s)=-||position-goal||_2/grid_diagonal", "pbrs_terminal_handling": "Phi(s_next)=0 for success, collision, and timeout", "pbrs_clipping": False, "dynamic_reward": {"progress": 0.0, "revisit": 0.0, "oscillation": 0.0, "stagnation": 0.0}, "training_astar_calls": 0, "astar_training_query_attempts": 0, "astar_training_targets_generated": 0, "validation_astar_runtime_calls": 0, "test_astar_runtime_calls": 0, "test_parameter_updates": 0, "selected_checkpoint": str(best_path), "selected_checkpoint_sha256": sha256_file(best_path), "selected_checkpoint_interaction": int(selected["interaction_count"]), "training_wall_clock_seconds": training_wall, "total_wall_clock_seconds": total_wall, "test": test_metrics}
        result_path = run / f"formal_result_{METHOD}_seed{seed}.json"
        write_json_atomic(result_path, result)
        _write_seed_summary_csv(run / "seed_summary.csv", result)
        return result_path
    except Exception:
        manifest["status"] = "failed"
        manifest["executed_interaction_count"] = manifest.get("executed_interaction_count", 0)
        write_json_atomic(run / "run_manifest.json", manifest)
        raise
    finally:
        logger.close()


def preflight(config: FrozenConfig, protocol: FrozenProtocol, device: torch.device) -> Dict[str, Any]:
    validate_config(config, protocol)
    input_hashes = verify_inputs(config)
    train_tasks, validation_tasks, test_tasks = load_tasks(config)
    if not train_tasks or not validation_tasks or not test_tasks:
        raise IntegrityError("one or more frozen task splits are empty")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable")
    source = _source_v7_config()
    if config.number("reward.pbrs.alpha") != float(source["reward"]["dynamic"]["distance_weight"]):
        raise IntegrityError("PBRS alpha/source Progress coefficient mismatch")
    phi = potential((0, 0), (29, 29), 30)
    shaped, phi_current, phi_next = pbrs_shaping((0, 0), (0, 1), (29, 29), 30, config.number("reward.pbrs.alpha"), config.number("ppo.gamma"), False)
    terminal_shaped, _, terminal_phi_next = pbrs_shaping((0, 0), (0, 0), (29, 29), 30, config.number("reward.pbrs.alpha"), config.number("ppo.gamma"), True)
    if not all(math.isfinite(x) for x in (phi, shaped, phi_current, phi_next, terminal_shaped)):
        raise IntegrityError("PBRS preflight produced NaN or Inf")
    if terminal_phi_next != 0.0:
        raise IntegrityError("PBRS terminal Phi(s_next) assertion failed")
    dynamic = config.get("reward.dynamic_disabled")
    if any(float(value) != 0.0 for value in dynamic.values()):
        raise IntegrityError("Dynamic Reward is not completely disabled")
    maps = MapStore(config.integer("environment.map_size"))
    environment = PBRSNavigationEnvironment(config, maps)
    observation = environment.reset(train_tasks[0])
    mask = environment.action_mask()
    if observation.spatial.shape != (5, 30, 30) or observation.history.shape != (10,):
        raise IntegrityError("state representation mismatch during preflight")
    if mask.shape != (4,) or not mask.any():
        raise IntegrityError("action mask failed during preflight")
    model = ActorCritic(config).to(device)
    agent = PPOAgent(config, model, device)
    action, log_probability, value, logits = agent.act(observation, mask, deterministic=True)
    if not bool(mask[action]) or not all(math.isfinite(x) for x in (log_probability, value)) or not np.isfinite(logits).any():
        raise IntegrityError("masked policy preflight failed")
    next_observation, reward, done, info = environment.step(action)
    if not math.isfinite(reward) or not math.isfinite(float(info["reward"].pbrs_reward)):
        raise IntegrityError("PBRS environment reward is non-finite")
    smoke_buffer = RolloutBuffer()
    smoke_buffer.add(
        spatial=observation.spatial,
        history=observation.history,
        mask=mask,
        action=action,
        log_prob=log_probability,
        reward=reward,
        done=done,
        value=value,
        next_value=0.0 if done else agent.value(next_observation),
        planning_target=None,
    )
    smoke_losses = agent.update(smoke_buffer, None)
    if any(smoke_losses[key] != 0.0 for key in ("astar_auxiliary_loss", "distillation_loss", "planning_loss", "weighted_distillation_loss", "weighted_planning_loss", "effective_astar_samples")):
        raise IntegrityError("A* auxiliary loss is nonzero in PBRS preflight")
    result = {
        "status": "PASS",
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "program_sha256": sha256_file(SCRIPT_PATH),
        "source_v7_program_sha256": sha256_file(BASELINE_PATH),
        "config_sha256": config.sha256,
        "protocol_sha256": protocol.sha256,
        "input_sha256": input_hashes,
        "official_seeds": OFFICIAL_SEEDS,
        "task_counts": {"train": len(train_tasks), "validation": len(validation_tasks), "test": len(test_tasks)},
        "environment_reset": True,
        "action_mask": True,
        "pbrs_finite": True,
        "pbrs_unclipped": True,
        "terminal_phi_next_zero": True,
        "dynamic_reward_components": dynamic,
        "astar_query_counter": 0,
        "astar_auxiliary_loss": smoke_losses["astar_auxiliary_loss"],
        "ppo_smoke_update": True,
    }
    print("PREFLIGHT PASS", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict PPO+PBRS BARN baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FrozenConfig(args.config)
    protocol = FrozenProtocol(args.protocol)
    validate_config(config, protocol)
    if args.audit_only:
        result = {"status": "PASS", "program_sha256": sha256_file(SCRIPT_PATH), "source_v7_program_sha256": sha256_file(BASELINE_PATH), "config_sha256": config.sha256, "protocol_sha256": protocol.sha256, "input_sha256": verify_inputs(config), "official_seeds": OFFICIAL_SEEDS, "formal_execution_allowed": True}
        print(json.dumps(result, indent=2), flush=True)
        return
    device = torch.device(args.device)
    if args.preflight:
        preflight(config, protocol, device)
        return
    if args.seed is None or args.run_dir is None:
        raise ProtocolError("--seed and --run-dir are required for a formal run")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable")
    result_path = run_experiment(config, protocol, args.seed, device, args.run_dir.resolve())
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
