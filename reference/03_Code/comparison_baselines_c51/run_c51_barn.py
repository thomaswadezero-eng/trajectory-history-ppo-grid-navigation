#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from barn_protocol import (
    BarnEnvironment,
    IntegrityError,
    MapStore,
    ProtocolError,
    aggregate_metrics,
    checkpoint_key,
    episode_metrics,
    load_frozen_tasks,
    read_json,
    set_seed,
    sha256_file,
    write_json_atomic,
)
from c51_algorithm import C51Agent, epsilon_by_interaction, module_sha256
from c51_replay import C51ReplayBuffer


SCRIPT_PATH = Path(__file__).resolve()
SOURCE_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SOURCE_DIR.parent.parent
DEFAULT_CONFIG = SOURCE_DIR / "config" / "c51_barn_formal_v1.json"
METHOD = "C51"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TextLogger:
    def __init__(self, path: Path):
        self.handle = path.open("x", encoding="utf-8", buffering=1)

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        self.handle.write(line + "\n")

    def close(self) -> None:
        self.handle.close()


class CsvSink:
    def __init__(self, path: Path, fields: Sequence[str]):
        self.handle = path.open("x", encoding="utf-8-sig", newline="", buffering=1)
        self.writer = csv.DictWriter(self.handle, fieldnames=list(fields), extrasaction="ignore")
        self.writer.writeheader()

    def write(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow(dict(row))

    def close(self) -> None:
        self.handle.close()


EXPECTED_ENVIRONMENT = {
    "map_size": 30,
    "spatial_shape": [5, 30, 30],
    "history_size": 10,
    "action_count": 4,
    "move_deltas": [[-1, 0], [1, 0], [0, -1], [0, 1]],
    "max_steps": 180,
    "action_mask": "bounds_and_cspace_free_cell_all_action_paths",
    "timeout_is_terminal": True,
}
EXPECTED_STATE = {
    "visit_count_clip": 10,
    "recent_trace_decay": 0.85,
    "no_progress_counter_clip": 20,
    "minimum_progress_epsilon": 0.000001,
}
EXPECTED_REWARD = {
    "base": {"goal": 1.0, "collision": -0.2, "timeout": -0.2, "step": -0.002},
    "terminal_reward_adds_step_penalty": True,
    "reward_normalization": False,
    "return_normalization": False,
    "dynamic_reward": False,
    "astar": False,
    "intrinsic_reward": False,
}
EXPECTED_ENCODER = {
    "convolutions": [
        {"in_channels": 5, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
        {"in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1},
        {"in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1},
    ],
    "adaptive_pool": [4, 4],
    "spatial_projection": [1024, 256],
    "history_mlp": [10, 64, 64],
    "fusion": [320, 256],
    "activation": "ReLU",
    "fusion_normalization": "LayerNorm",
    "weight_initialization": "pytorch_default",
}
EXPECTED_NETWORK = {
    "shared_encoder": EXPECTED_ENCODER,
    "head": [256, 128, 204],
    "head_output_shape": [4, 51],
    "dueling": False,
    "planning_head": False,
    "recurrent": False,
    "noisy_network": False,
}
EXPECTED_C51 = {
    "support": {"n_atoms": 51, "v_min": -0.25, "v_max": 1.0, "delta_z": 0.025},
    "target_action_selection": "target_network_masked_expected_q_argmax",
    "target_distribution_evaluation": "same_target_network_selected_action_distribution",
    "double_dqn_decoupling": False,
    "gamma": 0.99,
    "optimizer": "Adam",
    "learning_rate": 0.0001,
    "adam_eps": 0.00000001,
    "replay_type": "uniform_disk_backed_explicit_transition_storage",
    "replay_capacity": 100000,
    "learning_starts": 50000,
    "batch_size": 32,
    "train_frequency": 4,
    "gradient_steps_per_update": 1,
    "target_update_type": "hard",
    "target_update_interval": 10000,
    "exploration_schedule": "linear",
    "exploration_fraction": 0.1,
    "exploration_initial_epsilon": 1.0,
    "exploration_final_epsilon": 0.05,
    "loss": "categorical_cross_entropy",
    "max_gradient_norm": 10.0,
    "prioritized_replay": False,
    "n_step_returns": False,
    "terminal_bootstrap": False,
}
EXPECTED_PROFILE = {
    "allowed_seeds": [42, 1004, 6010],
    "interaction_budget": 900000,
    "validation_interval": 100000,
    "validation_episodes": 100,
    "test_episodes": 1000,
    "diagnostic_interval": 1000,
}
EXPECTED_CHECKPOINT = {
    "selection_split": "validation",
    "ranking": ["success_rate_desc", "spl_desc", "timeout_rate_asc", "average_steps_asc", "interaction_count_asc"],
    "test_requires_frozen_best_validation_checkpoint": True,
}
EXPECTED_EVALUATION = {
    "deterministic_masked_argmax": True,
    "epsilon": 0.0,
    "parameter_updates": 0,
    "optimizer_steps": 0,
    "replay_writes": 0,
    "target_updates": 0,
    "gradient_computation": False,
    "dynamic_reward": False,
    "astar_runtime_calls": 0,
    "record_all_test_trajectories": True,
}
EXPECTED_ISOLATION = {
    "base_reward_only": True,
    "dynamic_reward": False,
    "astar": False,
    "planning_head": False,
    "curriculum": False,
    "imitation": False,
    "her": False,
    "demonstrations": False,
    "privileged_information": False,
    "test_time_adaptation": False,
    "automatic_early_stop": False,
    "automatic_retry": False,
    "automatic_multi_seed_queue": False,
}
EXPECTED_STREAMS = {
    "training_task_seed_offset": 104729,
    "exploration_seed_offset": 155921,
    "replay_seed_offset": 196613,
}
EXPECTED_INPUTS = {
    "training_tasks": {
        "path": "03_Code/training_tasks_v2.json",
        "sha256": "D29146F90A9B73749B1FADAA0E563618E547A817422EB7A5D9D2A6F18DAFDBF5",
        "count": 1000, "map_ids": [0, 19],
    },
    "validation_tasks": {
        "path": "03_Code/validation_tasks_v2.json",
        "sha256": "AAC69C346AEEFB3FDD27E8576493A60D30CFB0989CA23634750D7118D417194B",
        "count": 100, "map_ids": [20, 29],
    },
    "test_tasks": {
        "path": "03_Code/test_tasks_v9_unseen_final.json",
        "sha256": "CC3C33B00772F8D1B6314FFADC94F8B7D5B86230C1BF445C5C6BCF15570E8397",
        "count": 1000, "map_ids": [130, 229],
    },
}
EXPECTED_AUTHORITY = {
    "v7_program": {
        "path": "03_Code/final_two_factor_experiment_v7/final_two_factor_experiment_v7.py",
        "sha256": "96093A6B5F024F5D9505ACCEBEAE7592700228BA17587BF0590092A241B7544D",
    },
    "v7_config": {
        "path": "03_Code/final_two_factor_experiment_v7/experiment_config.json",
        "sha256": "CF229DE128FF16E29B73308A7E9B2756DE51294671573F067306274EDA8D4166",
    },
    "v7_protocol": {
        "path": "03_Code/final_two_factor_experiment_v7/experiment_protocol.json",
        "sha256": "2C8BAFE1A71695627B4F723583153067C0E1C7D08F905538E5BF8793A96B4C9B",
    },
    "v7_map_split": {
        "path": "03_Code/final_two_factor_experiment_v7/map_split.json",
        "sha256": "637792B5F70F086AD534E3CAE70C03F2640EE402C8F8D09B16979E1910C4EBB1",
    },
    "v7_task_split": {
        "path": "03_Code/final_two_factor_experiment_v7/task_split.json",
        "sha256": "268A76CF6D4BD2AC6A2BA6CFB6B281527326F06D0B0BA148BAD453D90414F693",
    },
    "ddqn_config": {
        "path": "03_Code/comparison_baselines_v1/configs/ddqn_v1.json",
        "sha256": "3D0556A6F70A38731162CFB97C89149926F6268484F909AB3A249412FD3408B8",
    },
    "d3qn_config": {
        "path": "03_Code/comparison_baselines_v1/configs/d3qn_v1.json",
        "sha256": "CBC3C5E47805C79BB539E1BBD10D048C381454291CD7A78E21CCA93B71C6BAC2",
    },
}


def validate_config_data(config: Mapping[str, Any]) -> None:
    expected_keys = {
        "schema_version", "status", "formal_execution_owner", "algorithm", "algorithm_identity",
        "reward_mode", "program", "authority", "references", "inputs", "random_streams",
        "environment", "state", "reward", "network", "c51", "profiles", "checkpoint",
        "evaluation", "isolation", "output_root",
    }
    if set(config) != expected_keys:
        raise ProtocolError("formal configuration top-level fields changed")
    scalar_expectations = {
        "schema_version": "c51_barn_external_baseline.v1",
        "status": "AUTHORIZED_FOR_MANUAL_EXECUTION_ONLY",
        "formal_execution_owner": "user_in_vscode",
        "algorithm": METHOD,
        "algorithm_identity": "Canonical Categorical DQN with target-network-only greedy selection and evaluation",
        "reward_mode": "Base Reward only",
        "output_root": r"redacted_local_root\comparison_baselines_c51_outputs",
    }
    for key, expected in scalar_expectations.items():
        if config.get(key) != expected:
            raise ProtocolError(f"frozen configuration mismatch: {key}")
    if config.get("environment") != EXPECTED_ENVIRONMENT or config.get("state") != EXPECTED_STATE:
        raise ProtocolError("environment or state protocol mismatch")
    if config.get("reward") != EXPECTED_REWARD:
        raise ProtocolError("Base Reward isolation mismatch")
    if config.get("network") != EXPECTED_NETWORK:
        raise ProtocolError("network identity mismatch")
    if config.get("c51") != EXPECTED_C51:
        raise ProtocolError("C51 algorithm configuration mismatch")
    if config.get("profiles") != {"formal": EXPECTED_PROFILE}:
        raise ProtocolError("formal profile mismatch")
    if config.get("checkpoint") != EXPECTED_CHECKPOINT:
        raise ProtocolError("checkpoint selection protocol mismatch")
    if config.get("evaluation") != EXPECTED_EVALUATION:
        raise ProtocolError("evaluation isolation mismatch")
    if config.get("isolation") != EXPECTED_ISOLATION:
        raise ProtocolError("innovation isolation mismatch")
    if config.get("random_streams") != EXPECTED_STREAMS:
        raise ProtocolError("random-stream offsets mismatch")
    if config.get("inputs") != EXPECTED_INPUTS:
        raise ProtocolError("frozen task identity mismatch")
    if config.get("authority") != EXPECTED_AUTHORITY:
        raise ProtocolError("authority identity mismatch")
    if config.get("program", {}).get("path") != "03_Code/comparison_baselines_c51/run_c51_barn.py":
        raise ProtocolError("program path mismatch")
    references = config.get("references")
    if references != {
        "paper": "https://proceedings.mlr.press/v70/bellemare17a.html",
        "cleanrl_cross_check": "https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py",
        "cleanrl_hyperparameters_adopted": False,
    }:
        raise ProtocolError("reference identity changed")


def load_config(path: Path) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    config = read_json(path)
    validate_config_data(config)
    if sha256_file(SCRIPT_PATH) != str(config["program"]["sha256"]).upper():
        raise IntegrityError("C51 runner SHA mismatch")
    for name, record in config["authority"].items():
        authority_path = (PROJECT_ROOT / str(record["path"])).resolve()
        if not authority_path.is_file() or sha256_file(authority_path) != str(record["sha256"]).upper():
            raise IntegrityError(f"authority SHA mismatch: {name}")
    return config, sha256_file(path)


def resolve_device(name: str) -> torch.device:
    requested = str(name).lower()
    if requested not in ("cpu", "cuda"):
        raise ProtocolError("device must be cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable")
    return torch.device(requested)


def checkpoint_identity(mode: str, seed: int, config_sha: str) -> dict[str, Any]:
    return {
        "algorithm": METHOD,
        "mode": mode,
        "seed": int(seed),
        "runner_sha256": sha256_file(SCRIPT_PATH),
        "algorithm_sha256": sha256_file(SOURCE_DIR / "c51_algorithm.py"),
        "protocol_boundary_sha256": sha256_file(SOURCE_DIR / "barn_protocol.py"),
        "replay_sha256": sha256_file(SOURCE_DIR / "c51_replay.py"),
        "config_sha256": config_sha,
    }


def _create_run_directory(root: Path, seed: int) -> Path:
    run = root / METHOD / f"seed_{seed}" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run.mkdir(parents=True, exist_ok=False)
    for name in ("models", "validation", "protocol_snapshot"):
        (run / name).mkdir()
    return run


def save_checkpoint(
    path: Path,
    agent: C51Agent,
    task_rng: np.random.RandomState,
    mode: str,
    seed: int,
    interaction: int,
    config_sha: str,
    validation: Mapping[str, Any],
) -> str:
    if path.exists():
        raise FileExistsError(path)
    payload = {
        "schema_version": "c51_barn_checkpoint.v1",
        **checkpoint_identity(mode, seed, config_sha),
        "interaction_count": int(interaction),
        "optimizer_step_count": agent.optimizer_step_count,
        "target_update_count": agent.target_update_count,
        "agent_state": agent.checkpoint_state(),
        "task_rng_state": task_rng.get_state(),
        "python_random_state": random.getstate(),
        "numpy_global_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        "validation_metrics": dict(validation),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return sha256_file(path)


def load_checkpoint(
    path: Path,
    agent: C51Agent,
    task_rng: np.random.RandomState,
    device: torch.device,
    *,
    mode: str,
    seed: int,
    config_sha: str,
    load_optimizer: bool,
    restore_rng: bool,
) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("schema_version") != "c51_barn_checkpoint.v1":
        raise IntegrityError("checkpoint schema mismatch")
    expected = checkpoint_identity(mode, seed, config_sha)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise IntegrityError("checkpoint source/config identity mismatch")
    agent.load_checkpoint_state(payload["agent_state"], load_optimizer=load_optimizer)
    if restore_rng:
        task_rng.set_state(payload["task_rng_state"])
        random.setstate(payload["python_random_state"])
        np.random.set_state(payload["numpy_global_rng_state"])
        torch.set_rng_state(payload["torch_cpu_rng_state"])
        if device.type == "cuda" and payload["torch_cuda_rng_state_all"] is not None:
            torch.cuda.set_rng_state_all(payload["torch_cuda_rng_state_all"])
    return payload


def evaluate(
    agent: C51Agent,
    tasks: Sequence[Any],
    config: Mapping[str, Any],
    maps: MapStore,
    *,
    split: str,
    seed: int,
    interaction: int,
    output_directory: Path,
    record_trajectories: bool,
) -> dict[str, Any]:
    if split not in ("validation", "test"):
        raise ProtocolError("evaluation split must be validation or test")
    expected_count = int(config["profiles"]["formal"][f"{split}_episodes"])
    if len(tasks) != expected_count:
        raise IntegrityError(f"{split} task count mismatch")
    output_directory.mkdir(parents=True, exist_ok=False)
    environment = BarnEnvironment(config, maps)
    rows = []
    online_before = module_sha256(agent.online)
    target_before = module_sha256(agent.target)
    optimizer_before = agent.optimizer_step_count
    target_updates_before = agent.target_update_count
    illegal_before = (
        agent.illegal_executed_action_count,
        agent.illegal_greedy_action_count,
        agent.illegal_target_argmax_count,
    )
    trajectory_handle = (output_directory / "trajectories.jsonl").open("x", encoding="utf-8") if record_trajectories else None
    agent.online.eval()
    try:
        with torch.no_grad():
            for episode, task in enumerate(tasks, 1):
                observation = environment.reset(task)
                trajectory = [list(environment.builder.position)]
                total_reward = 0.0
                done = False
                final: dict[str, Any] = {}
                while not done:
                    mask = environment.action_mask()
                    action = agent.deterministic_action(observation, mask)
                    observation, reward, done, final = environment.step(action)
                    total_reward += reward
                    trajectory.append(list(environment.builder.position))
                row = episode_metrics(
                    episode, task, bool(final["success"]), bool(final["collision"]),
                    bool(final["timeout"]), environment.builder.step_count,
                    int(final["path_length"]), total_reward,
                )
                rows.append(row)
                if trajectory_handle is not None:
                    import json
                    trajectory_handle.write(json.dumps({
                        "episode": episode, "task_id": task.task_id, "map_id": task.map_id,
                        "trajectory": trajectory, "outcome": row.to_dict(),
                    }, ensure_ascii=False) + "\n")
    finally:
        if trajectory_handle is not None:
            trajectory_handle.close()
        agent.online.train()
    metrics = aggregate_metrics(rows)
    online_after = module_sha256(agent.online)
    target_after = module_sha256(agent.target)
    illegal_after = (
        agent.illegal_executed_action_count,
        agent.illegal_greedy_action_count,
        agent.illegal_target_argmax_count,
    )
    if any((
        online_before != online_after,
        target_before != target_after,
        optimizer_before != agent.optimizer_step_count,
        target_updates_before != agent.target_update_count,
        illegal_before != illegal_after,
    )):
        raise IntegrityError("evaluation changed parameters/counters or selected an illegal action")
    fields = list(rows[0].to_dict())
    with (output_directory / "evaluation_episodes.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([row.to_dict() for row in rows])
    metrics.update({
        "split": split,
        "seed": int(seed),
        "checkpoint_interaction": int(interaction),
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "replay_writes": 0,
        "target_updates": 0,
        "gradient_computation": False,
        "epsilon": 0.0,
        "astar_runtime_calls": 0,
        "dynamic_reward_enabled": False,
        "online_network_sha256_before": online_before,
        "online_network_sha256_after": online_after,
        "target_network_sha256_before": target_before,
        "target_network_sha256_after": target_after,
        "illegal_executed_action_count": 0,
        "illegal_greedy_action_count": 0,
        "illegal_target_argmax_count": 0,
    })
    write_json_atomic(output_directory / "evaluation_summary.json", metrics, must_not_exist=True)
    return metrics


DIAGNOSTIC_FIELDS = [
    "interaction_count", "optimizer_step_count", "replay_size", "epsilon", "loss",
    "q_mean", "q_std", "q_min", "q_max", "pmf_sum_error_max", "pmf_min", "pmf_max",
    "distribution_entropy_mean", "lower_support_mass_mean", "upper_support_mass_mean",
    "projected_mass_sum_error_max", "projected_mass_min", "projected_mass_max",
    "target_support_clamp_lower_fraction", "target_support_clamp_upper_fraction",
    "gradient_norm_before_clip", "online_network_hash_before", "online_network_hash_after",
    "online_hash_changed", "target_update_count", "target_network_hash",
    "target_hash_changed_when_expected", "illegal_executed_action_count",
    "illegal_greedy_action_count", "illegal_target_argmax_count", "finite_status",
]


def _mean_update_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = [
        "loss", "q_mean", "q_std", "q_min", "q_max", "pmf_sum_error_max", "pmf_min", "pmf_max",
        "distribution_entropy_mean", "lower_support_mass_mean", "upper_support_mass_mean",
        "projected_mass_sum_error_max", "projected_mass_min", "projected_mass_max",
        "target_support_clamp_lower_fraction", "target_support_clamp_upper_fraction", "gradient_norm_before_clip",
    ]
    if not rows:
        return {name: "" for name in names}
    return {name: float(np.mean([float(row[name]) for row in rows])) for name in names}


def _snapshot_inputs(run: Path, config_path: Path) -> None:
    for source in (
        config_path.resolve(), SOURCE_DIR / "README.md", SOURCE_DIR / "USER_MANUAL_COMMANDS.md",
        SOURCE_DIR / "hash_manifest.json",
    ):
        if source.is_file():
            shutil.copy2(source, run / "protocol_snapshot" / source.name)


def train(config_path: Path, mode: str, seed: int, device: torch.device) -> Path:
    config, config_sha = load_config(config_path)
    if mode != "formal":
        raise ProtocolError("C51 has one formal profile only; no pilot profile exists")
    profile = config["profiles"]["formal"]
    if int(seed) not in profile["allowed_seeds"]:
        raise ProtocolError("seed is not in the frozen formal seed set")
    # Deliberate firewall: the training entry point loads train and validation only.
    tasks = load_frozen_tasks(config, PROJECT_ROOT, ("train", "validation"))
    if set(tasks) != {"train", "validation"}:
        raise IntegrityError("training split firewall failed")
    root = Path(str(config["output_root"])).resolve()
    run = _create_run_directory(root, int(seed))
    _snapshot_inputs(run, config_path)
    logger = TextLogger(run / "training.log")
    episode_sink = CsvSink(
        run / "training_episodes.csv",
        ["episode", "interaction_count", "episode_reward", "success", "collision", "timeout", "budget_truncated", "steps", "epsilon"],
    )
    diagnostic_sink = CsvSink(run / "diagnostics.csv", DIAGNOSTIC_FIELDS)
    started = time.time()
    interaction = 0
    replay: Optional[C51ReplayBuffer] = None
    identity = checkpoint_identity(mode, int(seed), config_sha)
    manifest: dict[str, Any] = {
        "schema_version": "c51_barn_run.v1", "status": "running", **identity,
        "run_directory": str(run), "output_root": str(root),
        "training_interaction_budget": int(profile["interaction_budget"]),
        "executed_training_interactions": 0, "test_status": "not_run",
        "test_split_loaded": False,
        "loaded_split_identifiers": {
            "train": {"task_count": len(tasks["train"]), "map_ids": sorted({task.map_id for task in tasks["train"]})},
            "validation": {"task_count": len(tasks["validation"]), "map_ids": sorted({task.map_id for task in tasks["validation"]})},
        },
        "important_input_sha256": {
            "training_tasks": config["inputs"]["training_tasks"]["sha256"],
            "validation_tasks": config["inputs"]["validation_tasks"]["sha256"],
        },
        "started_at": utc_now(),
    }
    write_json_atomic(run / "run_manifest.json", manifest, must_not_exist=True)
    try:
        set_seed(int(seed))
        maps = MapStore(30)
        agent = C51Agent(config, device, int(seed))
        task_rng = np.random.RandomState((int(seed) + int(config["random_streams"]["training_task_seed_offset"])) % (2**32))
        replay = C51ReplayBuffer(run / "replay", int(config["c51"]["replay_capacity"]))
        write_json_atomic(run / "replay_buffer_manifest.json", replay.manifest(), must_not_exist=True)
        environment = BarnEnvironment(config, maps)
        budget = int(profile["interaction_budget"])
        validation_interval = int(profile["validation_interval"])
        diagnostic_interval = int(profile["diagnostic_interval"])
        next_validation = validation_interval
        next_diagnostic = diagnostic_interval
        validation_count = 0
        best_record: Optional[dict[str, Any]] = None
        episode = 1
        task = tasks["train"][int(task_rng.randint(0, len(tasks["train"])))]
        observation = environment.reset(task)
        episode_reward = 0.0
        update_rows: list[dict[str, Any]] = []
        previous_online_hash = module_sha256(agent.online)
        latest_target_hash = module_sha256(agent.target)
        last_target_hash_changed_when_expected = True
        logger.log(
            f"start algorithm=C51 canonical=True atoms=51 support=[-0.25,1.0] mode=formal seed={seed} "
            f"budget={budget} device={device} test_split_loaded=False"
        )
        while interaction < budget:
            current_mask = environment.action_mask()
            epsilon = epsilon_by_interaction(
                interaction, budget, float(config["c51"]["exploration_initial_epsilon"]),
                float(config["c51"]["exploration_final_epsilon"]), float(config["c51"]["exploration_fraction"]),
            )
            action = agent.act(observation, current_mask, epsilon)
            current_observation = observation
            observation, reward, done, info = environment.step(action)
            next_mask = np.zeros(4, dtype=np.bool_) if done else environment.action_mask()
            replay.add(current_observation, current_mask, action, reward, done, observation, next_mask)
            interaction += 1
            episode_reward += reward
            if (
                interaction > int(config["c51"]["learning_starts"])
                and interaction % int(config["c51"]["train_frequency"]) == 0
            ):
                counter_before = interaction
                batch = replay.sample(int(config["c51"]["batch_size"]), agent.replay_rng)
                update_rows.append(agent.update(batch))
                if interaction != counter_before:
                    raise IntegrityError("optimizer update changed the environment interaction counter")
            if interaction % int(config["c51"]["target_update_interval"]) == 0:
                before_target = module_sha256(agent.target)
                online_hash = module_sha256(agent.online)
                expected_change = before_target != online_hash
                agent.hard_update_target()
                after_target = module_sha256(agent.target)
                if after_target != module_sha256(agent.online):
                    raise IntegrityError("hard target update did not exactly copy online parameters")
                last_target_hash_changed_when_expected = (not expected_change) or (after_target != before_target)
                latest_target_hash = after_target
            if done:
                episode_sink.write({
                    "episode": episode, "interaction_count": interaction, "episode_reward": episode_reward,
                    "success": int(info["success"]), "collision": int(info["collision"]),
                    "timeout": int(info["timeout"]), "budget_truncated": 0,
                    "steps": environment.builder.step_count, "epsilon": epsilon,
                })
                episode += 1
                episode_reward = 0.0
                if interaction < budget:
                    task = tasks["train"][int(task_rng.randint(0, len(tasks["train"])))]
                    observation = environment.reset(task)
            if interaction == next_diagnostic or interaction == budget:
                current_online_hash = module_sha256(agent.online)
                means = _mean_update_metrics(update_rows)
                finite_values = [float(value) for value in means.values() if value != ""]
                finite_status = all(np.isfinite(value) for value in finite_values)
                diagnostic_sink.write({
                    "interaction_count": interaction, "optimizer_step_count": agent.optimizer_step_count,
                    "replay_size": len(replay), "epsilon": epsilon, **means,
                    "online_network_hash_before": previous_online_hash,
                    "online_network_hash_after": current_online_hash,
                    "online_hash_changed": int(current_online_hash != previous_online_hash),
                    "target_update_count": agent.target_update_count,
                    "target_network_hash": latest_target_hash,
                    "target_hash_changed_when_expected": int(last_target_hash_changed_when_expected),
                    "illegal_executed_action_count": agent.illegal_executed_action_count,
                    "illegal_greedy_action_count": agent.illegal_greedy_action_count,
                    "illegal_target_argmax_count": agent.illegal_target_argmax_count,
                    "finite_status": "PASS" if finite_status else "FAIL",
                })
                logger.log(
                    f"diagnostic interactions={interaction} updates={agent.optimizer_step_count} replay={len(replay)} "
                    f"epsilon={epsilon:.6f} loss={means['loss']} pmf_error={means['pmf_sum_error_max']} "
                    f"projection_error={means['projected_mass_sum_error_max']} clamp_low={means['target_support_clamp_lower_fraction']} "
                    f"clamp_high={means['target_support_clamp_upper_fraction']} illegal=0 finite={'PASS' if finite_status else 'FAIL'}"
                )
                previous_online_hash = current_online_hash
                update_rows.clear()
                next_diagnostic += diagnostic_interval
            if interaction == next_validation:
                validation_count += 1
                metrics = evaluate(
                    agent, tasks["validation"], config, maps, split="validation", seed=int(seed),
                    interaction=interaction, output_directory=run / "validation" / f"interaction_{interaction:09d}",
                    record_trajectories=False,
                )
                checkpoint = run / "models" / f"checkpoint_interaction_{interaction:09d}.pt"
                checkpoint_sha = save_checkpoint(
                    checkpoint, agent, task_rng, mode, int(seed), interaction, config_sha, metrics
                )
                key = checkpoint_key(metrics, interaction)
                became_best = best_record is None or key < tuple(best_record["ranking_key"])
                record = {
                    "interaction_count": interaction, "validation_metrics": metrics,
                    "ranking_key": list(key), "became_best": bool(became_best),
                    "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                }
                write_json_atomic(run / "models" / "latest_checkpoint.json", record)
                if became_best:
                    best_record = record
                    write_json_atomic(run / "models" / "best_checkpoint.json", record)
                logger.log(
                    f"validation interaction={interaction} success={metrics['success_rate']:.6f} "
                    f"SPL={metrics['spl']:.6f} timeout={metrics['timeout_rate']:.6f} "
                    f"reward={metrics['average_episode_reward']:.6f} best={became_best}"
                )
                next_validation += validation_interval
        if not environment.done:
            episode_sink.write({
                "episode": episode, "interaction_count": interaction, "episode_reward": episode_reward,
                "success": 0, "collision": 0, "timeout": 0, "budget_truncated": 1,
                "steps": environment.builder.step_count, "epsilon": epsilon,
            })
        expected_validations = budget // validation_interval
        if interaction != budget or validation_count != expected_validations or best_record is None:
            raise IntegrityError("formal training budget, validation count, or checkpoint selection is incomplete")
        source_best = Path(str(best_record["checkpoint_path"])).resolve()
        if sha256_file(source_best) != str(best_record["checkpoint_sha256"]).upper():
            raise IntegrityError("selected validation checkpoint changed before freezing")
        frozen = run / "models" / "frozen_best_validation_checkpoint.pt"
        shutil.copy2(source_best, frozen)
        frozen_sha = sha256_file(frozen)
        replay.flush()
        completion = {
            "schema_version": "c51_barn_training_complete.v1", "status": "training_completed",
            **identity, "interaction_budget": budget, "executed_training_interactions": interaction,
            "validation_count": validation_count, "best_validation": best_record,
            "frozen_best_checkpoint": str(frozen), "frozen_best_checkpoint_sha256": frozen_sha,
            "optimizer_step_count": agent.optimizer_step_count, "target_update_count": agent.target_update_count,
            "illegal_executed_action_count": agent.illegal_executed_action_count,
            "illegal_greedy_action_count": agent.illegal_greedy_action_count,
            "illegal_target_argmax_count": agent.illegal_target_argmax_count,
            "training_astar_calls": 0, "validation_astar_runtime_calls": 0,
            "dynamic_reward_enabled": False, "test_split_loaded": False, "completed_at": utc_now(),
        }
        if any(completion[name] != 0 for name in (
            "illegal_executed_action_count", "illegal_greedy_action_count", "illegal_target_argmax_count"
        )):
            raise IntegrityError("formal training observed an illegal action")
        write_json_atomic(run / "training_complete_manifest.json", completion, must_not_exist=True)
        manifest.update({
            "status": "training_completed", "executed_training_interactions": interaction,
            "validation_count": validation_count, "best_checkpoint": str(frozen),
            "best_checkpoint_sha256": frozen_sha, "best_validation": best_record["validation_metrics"],
            "optimizer_step_count": agent.optimizer_step_count, "target_update_count": agent.target_update_count,
            "runtime_seconds": time.time() - started, "completed_at": utc_now(),
        })
        write_json_atomic(run / "run_manifest.json", manifest)
        logger.log(f"training complete interactions={interaction} test_status=not_run")
        return run
    except BaseException as exc:
        manifest.update({
            "status": "interrupted_by_user" if isinstance(exc, KeyboardInterrupt) else "failed",
            "executed_training_interactions": interaction,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "runtime_seconds": time.time() - started,
            "stopped_at": utc_now(),
        })
        write_json_atomic(run / "run_manifest.json", manifest)
        raise
    finally:
        if replay is not None:
            replay.close()
        episode_sink.close()
        diagnostic_sink.close()
        logger.close()


def validate_test_firewall(config: Mapping[str, Any], config_sha: str, run: Path) -> tuple[dict[str, Any], dict[str, Any], Path, int]:
    run = run.resolve()
    manifest_path = run / "run_manifest.json"
    completion_path = run / "training_complete_manifest.json"
    if not manifest_path.is_file() or not completion_path.is_file():
        raise IntegrityError("formal test requires run and training-complete manifests")
    manifest = read_json(manifest_path)
    completion = read_json(completion_path)
    if manifest.get("status") != "training_completed" or completion.get("status") != "training_completed":
        raise IntegrityError("formal test refused: training is not completed")
    if manifest.get("mode") != "formal" or completion.get("mode") != "formal":
        raise IntegrityError("formal test requires the sole frozen formal profile")
    if manifest.get("test_status") != "not_run" or manifest.get("test_split_loaded") is not False:
        raise IntegrityError("formal test is not in a pristine not-run state")
    if int(completion.get("executed_training_interactions", -1)) != 900000:
        raise IntegrityError("formal test requires exactly 900000 completed training interactions")
    seed = int(manifest["seed"])
    if seed not in config["profiles"]["formal"]["allowed_seeds"]:
        raise IntegrityError("formal run seed is not authorized")
    expected = checkpoint_identity("formal", seed, config_sha)
    if any(manifest.get(key) != value or completion.get(key) != value for key, value in expected.items()):
        raise IntegrityError("formal run identity does not match current frozen source/config")
    checkpoint = Path(str(completion["frozen_best_checkpoint"])).resolve()
    if checkpoint.parent != (run / "models").resolve():
        raise IntegrityError("frozen checkpoint is outside the run models directory")
    if not checkpoint.is_file() or sha256_file(checkpoint) != str(completion["frozen_best_checkpoint_sha256"]).upper():
        raise IntegrityError("frozen best-validation checkpoint SHA mismatch")
    if (run / "test").exists():
        raise FileExistsError(run / "test")
    return manifest, completion, checkpoint, seed


def formal_test(config_path: Path, run: Path, device: torch.device) -> Path:
    config, config_sha = load_config(config_path)
    run = run.resolve()
    # All completion/source/checkpoint checks occur before the test task file is loaded.
    manifest, completion, checkpoint, seed = validate_test_firewall(config, config_sha, run)
    tasks = load_frozen_tasks(config, PROJECT_ROOT, ("test",))
    maps = MapStore(30)
    set_seed(seed)
    agent = C51Agent(config, device, seed)
    task_rng = np.random.RandomState((seed + int(config["random_streams"]["training_task_seed_offset"])) % (2**32))
    payload = load_checkpoint(
        checkpoint, agent, task_rng, device, mode="formal", seed=seed,
        config_sha=config_sha, load_optimizer=False, restore_rng=False,
    )
    metrics = evaluate(
        agent, tasks["test"], config, maps, split="test", seed=seed,
        interaction=int(payload["interaction_count"]), output_directory=run / "test",
        record_trajectories=True,
    )
    if any((
        metrics["parameter_updates"] != 0, metrics["optimizer_steps"] != 0,
        metrics["replay_writes"] != 0, metrics["target_updates"] != 0,
        metrics["gradient_computation"] is not False, metrics["epsilon"] != 0.0,
        metrics["astar_runtime_calls"] != 0, metrics["dynamic_reward_enabled"] is not False,
        metrics["online_network_sha256_before"] != metrics["online_network_sha256_after"],
    )):
        raise IntegrityError("formal test isolation assertions failed")
    result = {
        "schema_version": "c51_barn_formal_result.v1", "algorithm": METHOD,
        "algorithm_identity": "Canonical Categorical DQN (C51), N=51", "seed": seed,
        "training_interactions": int(completion["executed_training_interactions"]),
        "selected_checkpoint": str(checkpoint), "selected_checkpoint_sha256": sha256_file(checkpoint),
        "selected_checkpoint_interaction": int(payload["interaction_count"]),
        "support": {"v_min": -0.25, "v_max": 1.0, "n_atoms": 51, "delta_z": 0.025},
        "reward_mode": "Base Reward only", "dynamic_reward_enabled": False,
        "astar_runtime_calls": 0, "planning_head_present": False,
        "curriculum_enabled": False, "imitation_enabled": False, "her_enabled": False,
        "test_parameter_updates": 0, "test_optimizer_steps": 0,
        "test_replay_writes": 0, "test_target_updates": 0,
        "test_gradient_computation": False, "test": metrics,
    }
    write_json_atomic(run / f"formal_result_C51_seed{seed}.json", result, must_not_exist=True)
    manifest.update({
        "status": "completed_with_formal_test", "test_split_loaded": True,
        "important_input_sha256": {
            **dict(manifest["important_input_sha256"]),
            "test_tasks": config["inputs"]["test_tasks"]["sha256"],
        },
        "loaded_split_identifiers": {
            **dict(manifest["loaded_split_identifiers"]),
            "test": {"task_count": len(tasks["test"]), "map_ids": sorted({task.map_id for task in tasks["test"]})},
        },
        "test_status": "completed", "test_metrics": metrics, "test_completed_at": utc_now(),
    })
    write_json_atomic(run / "run_manifest.json", manifest)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical C51 external baseline for frozen BARN V7")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="manual formal train+validation only")
    train_parser.add_argument("--mode", choices=("formal",), required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--device", default="cuda")
    test_parser = subparsers.add_parser("test", help="manual formal test after completed training")
    test_parser.add_argument("--run-dir", type=Path, required=True)
    test_parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.command == "train":
        run = train(args.config, args.mode, args.seed, device)
    else:
        run = formal_test(args.config, args.run_dir, device)
    print(str(run))


if __name__ == "__main__":
    main()
