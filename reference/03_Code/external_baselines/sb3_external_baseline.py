#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from sb3_baseline.audit import load_config, source_manifest, verify_environment, verify_frozen_files
from sb3_baseline.callbacks import V7ValidationAndBudgetCallback
from sb3_baseline.core import (
    BarnNavigationEnv,
    IntegrityError,
    cspace_aggregate_sha256,
    load_frozen_tasks,
    read_json,
    sha256_file,
    write_json_new,
    write_json_replace,
)
from sb3_baseline.evaluation import evaluate_tasks
from sb3_baseline.model import maskable_ppo_kwargs


SCRIPT_PATH = Path(__file__).resolve()
EXTERNAL_ROOT = SCRIPT_PATH.parent
PROJECT_ROOT = EXTERNAL_ROOT.parent.parent
DEFAULT_CONFIG = EXTERNAL_ROOT / "config" / "sb3_external_baseline_config_v1.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_all_tasks(config: dict[str, Any]):
    records = config["inputs"]
    train = load_frozen_tasks(
        PROJECT_ROOT / records["training_tasks"]["path"],
        records["training_tasks"]["sha256"],
        "train",
        records["training_tasks"]["count"],
        set(range(0, 20)),
    )
    validation = load_frozen_tasks(
        PROJECT_ROOT / records["validation_tasks"]["path"],
        records["validation_tasks"]["sha256"],
        "validation",
        records["validation_tasks"]["count"],
        set(range(20, 30)),
    )
    test = load_frozen_tasks(
        PROJECT_ROOT / records["test_tasks"]["path"],
        records["test_tasks"]["sha256"],
        "test",
        records["test_tasks"]["count"],
        set(range(130, 230)),
    )
    all_ids = [task.task_id for task in (*train, *validation, *test)]
    if len(all_ids) != len(set(all_ids)):
        raise IntegrityError("task IDs overlap across splits")
    return train, validation, test


def verify_cspace_snapshot(config: dict[str, Any], tasks) -> dict[str, Any]:
    aggregate, records = cspace_aggregate_sha256(tasks)
    expected = str(config["inputs"]["current_cspace_aggregate_sha256"]).upper()
    if aggregate != expected:
        raise IntegrityError(f"cspace aggregate mismatch: actual={aggregate}, expected={expected}")
    return {"algorithm": "SHA-256", "aggregate_sha256": aggregate, "file_count": len(records), "files": records}


def create_run_dir(output_root: Path, seed: int) -> Path:
    if not output_root.resolve().is_relative_to(EXTERNAL_ROOT.resolve()):
        raise IntegrityError(f"output root must remain inside external_baselines: {output_root}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run = output_root / "MaskablePPO" / f"seed_{seed}" / f"run_{stamp}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "models").mkdir()
    return run


def code_files() -> list[Path]:
    return [SCRIPT_PATH, DEFAULT_CONFIG, *sorted((EXTERNAL_ROOT / "sb3_baseline").glob("*.py"))]


def initialize_run(seed: int, device: str, config: dict[str, Any], output_root: Path):
    environment = verify_environment(device)
    frozen_hashes = verify_frozen_files(PROJECT_ROOT, config)
    train_tasks, validation_tasks, test_tasks = load_all_tasks(config)
    cspace_snapshot = verify_cspace_snapshot(config, (*train_tasks, *validation_tasks, *test_tasks))
    run = create_run_dir(output_root, seed)
    shutil.copy2(DEFAULT_CONFIG, run / DEFAULT_CONFIG.name)
    write_json_new(run / "cspace_file_manifest.json", cspace_snapshot)
    manifest = {
        "schema_version": "sb3_external_baseline_run.v1",
        "status": "initialized",
        "algorithm": "SB3-Contrib MaskablePPO",
        "seed": seed,
        "device": device,
        "environment": environment,
        "source_sha256": source_manifest(code_files()),
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "frozen_input_sha256": frozen_hashes,
        "cspace_aggregate_sha256": cspace_snapshot["aggregate_sha256"],
        "training_interaction_budget": 900000,
        "executed_interaction_count": 0,
        "dynamic_reward": False,
        "training_astar_calls": 0,
        "validation_astar_runtime_calls": 0,
        "test_astar_runtime_calls": 0,
        "validation_parameter_updates": 0,
        "test_parameter_updates": 0,
        "run_directory": str(run),
    }
    write_json_new(run / "run_manifest.json", manifest)
    return run, train_tasks, validation_tasks, test_tasks


def train(seed: int, device: str, config: dict[str, Any], output_root: Path):
    run, train_tasks, validation_tasks, test_tasks = initialize_run(seed, device, config, output_root)
    manifest_path = run / "run_manifest.json"
    manifest = read_json(manifest_path)
    manifest["status"] = "training"
    write_json_replace(manifest_path, manifest)
    set_seed(seed)
    raw_env = BarnNavigationEnv(train_tasks, "train", seed, training_sampling=True)
    monitor = Monitor(
        raw_env,
        filename=str(run / "training_monitor.csv"),
        info_keywords=("task_id", "map_id", "success", "collision", "timeout", "path_length", "outcome"),
    )
    kwargs = maskable_ppo_kwargs(seed, device)
    model = MaskablePPO(env=monitor, **kwargs)
    model.set_logger(configure(str(run / "sb3_logs"), ["stdout", "csv"]))
    callback = V7ValidationAndBudgetCallback(
        validation_tasks=validation_tasks,
        seed=seed,
        run_dir=run,
        manifest_path=manifest_path,
        interaction_budget=900000,
        validation_interval=100000,
    )
    try:
        model.learn(total_timesteps=900000, callback=callback, use_masking=True, reset_num_timesteps=True)
        if int(model.num_timesteps) != 900000:
            raise IntegrityError(f"unexpected final interaction count: {model.num_timesteps}")
        best_record = read_json(run / "models" / "best_checkpoint.json")
        manifest = read_json(manifest_path)
        manifest.update(
            {
                "status": "training_completed",
                "executed_interaction_count": int(model.num_timesteps),
                "ppo_update_count": int(model._n_updates),
                "selected_checkpoint": best_record["checkpoint_path"],
                "selected_checkpoint_sha256": best_record["checkpoint_sha256"],
                "selected_checkpoint_interaction": best_record["interaction_count"],
                "selected_validation_metrics": best_record["metrics"],
                "incomplete_final_rollout_discarded": True,
                "dynamic_reward": False,
                "training_astar_calls": 0,
            }
        )
        write_json_replace(manifest_path, manifest)
        return run, test_tasks
    except Exception as exc:
        manifest = read_json(manifest_path)
        manifest.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        write_json_replace(manifest_path, manifest)
        raise
    finally:
        monitor.close()


def formal_test(run: Path, test_tasks, device: str) -> dict[str, Any]:
    run = run.resolve()
    if not run.is_relative_to((EXTERNAL_ROOT / "outputs").resolve()):
        raise IntegrityError(f"formal test run must be inside external_baselines outputs: {run}")
    manifest_path = run / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") not in {"training_completed", "completed"}:
        raise IntegrityError("formal test requires a completed training run")
    if manifest.get("config_sha256") != sha256_file(DEFAULT_CONFIG):
        raise IntegrityError("current config no longer matches the training run")
    if manifest.get("source_sha256") != source_manifest(code_files()):
        raise IntegrityError("current source no longer matches the training run")
    test_dir = run / "test"
    if test_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing formal test: {test_dir}")
    checkpoint = Path(str(manifest["selected_checkpoint"]))
    if not checkpoint.resolve().is_relative_to(run):
        raise IntegrityError("selected checkpoint is outside its run directory")
    if sha256_file(checkpoint) != str(manifest["selected_checkpoint_sha256"]):
        raise IntegrityError("selected checkpoint SHA mismatch")
    model = MaskablePPO.load(checkpoint, env=None, device=device)
    result = evaluate_tasks(
        model=model,
        tasks=test_tasks,
        split="test",
        seed=int(manifest["seed"]),
        output_dir=test_dir,
        checkpoint_interaction=int(manifest["selected_checkpoint_interaction"]),
        record_trajectories=True,
    )
    manifest.update(
        {
            "status": "completed",
            "test": result,
            "test_astar_runtime_calls": 0,
            "test_parameter_updates": 0,
            "dynamic_reward": False,
        }
    )
    write_json_replace(manifest_path, manifest)
    write_json_new(
        run / f"formal_result_MaskablePPO_seed{manifest['seed']}.json",
        {
            "schema_version": "sb3_external_baseline_result.v1",
            "seed": manifest["seed"],
            "environment": manifest["environment"],
            "source_sha256": manifest["source_sha256"],
            "config_sha256": manifest["config_sha256"],
            "frozen_input_sha256": manifest["frozen_input_sha256"],
            "cspace_aggregate_sha256": manifest["cspace_aggregate_sha256"],
            "training_interactions": manifest["executed_interaction_count"],
            "selected_checkpoint": manifest["selected_checkpoint"],
            "selected_checkpoint_sha256": manifest["selected_checkpoint_sha256"],
            "selected_checkpoint_interaction": manifest["selected_checkpoint_interaction"],
            "test": result,
            "dynamic_reward": False,
            "training_astar_calls": 0,
            "validation_astar_runtime_calls": 0,
            "test_astar_runtime_calls": 0,
        },
    )
    return result


def audit_only(seed: int, device: str, config: dict[str, Any]) -> None:
    environment = verify_environment(device)
    hashes = verify_frozen_files(PROJECT_ROOT, config)
    train_tasks, validation_tasks, test_tasks = load_all_tasks(config)
    snapshot = verify_cspace_snapshot(config, (*train_tasks, *validation_tasks, *test_tasks))
    print(json.dumps({
        "audit": "PASS",
        "seed": seed,
        "environment": environment,
        "frozen_input_sha256": hashes,
        "task_counts": {"train": len(train_tasks), "validation": len(validation_tasks), "test": len(test_tasks)},
        "cspace_aggregate_sha256": snapshot["aggregate_sha256"],
        "training_started": False,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent SB3-Contrib MaskablePPO BARN baseline")
    parser.add_argument("--mode", choices=("train", "test", "train-and-test", "audit-only"), default="train-and-test")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-dir", type=Path, help="Required for --mode test")
    parser.add_argument("--total-interactions", type=int, default=900000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if config_path != DEFAULT_CONFIG.resolve():
        raise IntegrityError(f"formal entry only accepts the frozen config: {DEFAULT_CONFIG}")
    config = load_config(config_path)
    if args.seed not in config["official_seeds"]:
        raise IntegrityError(f"seed is not official: {args.seed}")
    if args.total_interactions != 900000:
        raise IntegrityError("formal entry requires exactly 900000 interactions")
    output_root = (args.output_root or (PROJECT_ROOT / config["output_root"])).resolve()
    if args.mode == "audit-only":
        audit_only(args.seed, args.device, config)
        return
    if args.mode == "test":
        if args.run_dir is None:
            raise IntegrityError("--run-dir is required for --mode test")
        verify_environment(args.device)
        verify_frozen_files(PROJECT_ROOT, config)
        run_manifest = read_json(args.run_dir.resolve() / "run_manifest.json")
        if int(run_manifest.get("seed", -1)) != args.seed:
            raise IntegrityError("--seed does not match the selected training run")
        train_tasks, validation_tasks, test_tasks = load_all_tasks(config)
        verify_cspace_snapshot(config, (*train_tasks, *validation_tasks, *test_tasks))
        formal_test(args.run_dir.resolve(), test_tasks, args.device)
        return
    run, test_tasks = train(args.seed, args.device, config, output_root)
    print(f"TRAINING_COMPLETED_RUN_DIR={run}", flush=True)
    if args.mode == "train-and-test":
        result = formal_test(run, test_tasks, args.device)
        print(json.dumps({"FORMAL_TEST_COMPLETED": True, "test": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
