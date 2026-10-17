#!/usr/bin/env python
"""Fail-closed, read-only audit of the 12 legacy component-ablation runs.

No environment, model forward pass, optimizer, CUDA context, training loop,
validation, or test evaluator is executed.  Selected checkpoint files are
deserialized on CPU only to verify their stored identity metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CODE_ROOT = PROJECT_ROOT / "03_Code" / "internal_ablation_v1"
LEGACY_OUTPUT_ROOT = Path(r"redacted_local_root\internal_ablation_v1")
METHODS = (
    "REWARD_WO_PROGRESS",
    "REWARD_WO_REVISIT",
    "REWARD_WO_OSCILLATION",
    "REWARD_WO_STAGNATION",
)
SEEDS = (42, 1004, 6010)
EXPECTED_INTERACTIONS = tuple(range(100000, 900001, 100000))
EXPECTED_PROGRAM_SHA = "A1FA5959884B782D356B28C359102DBC57D4FA8B32A3D714A0CEBC6C9BA4ADE4"
EXPECTED_CONFIG_SELF_SHA = "0BD21F4412873AE81CA8263ED8E8CDA7A697620A114EF4078521D45C70C5F4CB"
EXPECTED_PROTOCOL_SELF_SHA = "6E8F29D13320355477DB7E6A957DDBFFCE193F1FC1280102BDC9F13BD3404684"
EXPECTED_INPUTS = {
    "training_tasks": "D29146F90A9B73749B1FADAA0E563618E547A817422EB7A5D9D2A6F18DAFDBF5",
    "validation_tasks": "AAC69C346AEEFB3FDD27E8576493A60D30CFB0989CA23634750D7118D417194B",
    "test_tasks": "CC3C33B00772F8D1B6314FFADC94F8B7D5B86230C1BF445C5C6BCF15570E8397",
    "map_split": "637792B5F70F086AD534E3CAE70C03F2640EE402C8F8D09B16979E1910C4EBB1",
    "task_split": "268A76CF6D4BD2AC6A2BA6CFB6B281527326F06D0B0BA148BAD453D90414F693",
}
EXPECTED_SWITCHES = {
    "REWARD_WO_PROGRESS": {"progress": False, "revisit": True, "oscillation": True, "stagnation": True},
    "REWARD_WO_REVISIT": {"progress": True, "revisit": False, "oscillation": True, "stagnation": True},
    "REWARD_WO_OSCILLATION": {"progress": True, "revisit": True, "oscillation": False, "stagnation": True},
    "REWARD_WO_STAGNATION": {"progress": True, "revisit": True, "oscillation": True, "stagnation": False},
}


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"expected JSON object: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def close(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def load_task_ids(path: Path) -> List[str]:
    payload = read_json(path)
    tasks = payload.get("tasks")
    require(isinstance(tasks, list), f"missing task list: {path}")
    return [str(task["task_id"]) for task in tasks]


def recompute_summary(rows: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    count = len(rows)
    successes = sum(int(row["success"]) for row in rows)
    collisions = sum(int(row["collision"]) for row in rows)
    timeouts = sum(int(row["timeout"]) for row in rows)
    successful_efficiencies = [
        int(row["optimal_path_length"]) / float(max(int(row["optimal_path_length"]), int(row["path_length"])))
        for row in rows
        if int(row["success"]) == 1
    ]
    return {
        "episode_count": float(count),
        "success_count": float(successes),
        "collision_count": float(collisions),
        "timeout_count": float(timeouts),
        "success_rate": successes / float(count),
        "collision_rate": collisions / float(count),
        "timeout_rate": timeouts / float(count),
        "average_steps": statistics.mean(float(row["steps"]) for row in rows),
        "average_path_length": statistics.mean(float(row["path_length"]) for row in rows),
        "spl": statistics.mean(float(row["SPL"]) for row in rows),
        "path_efficiency": statistics.mean(successful_efficiencies) if successful_efficiencies else 0.0,
    }


def validate_summary(rows: Sequence[Mapping[str, str]], summary: Mapping[str, str], label: str) -> Dict[str, float]:
    computed = recompute_summary(rows)
    mapping = {
        "success_count": "success",
        "collision_count": "collision",
        "timeout_count": "timeout",
        "success_rate": "success_rate",
        "collision_rate": "collision_rate",
        "timeout_rate": "timeout_rate",
        "average_steps": "average_steps",
        "average_path_length": "average_path_length",
        "spl": "SPL",
        "path_efficiency": "path_efficiency",
    }
    for key, column in mapping.items():
        require(close(computed[key], summary[column]), f"{label}: summary mismatch for {key}")
    require(int(computed["success_count"] + computed["collision_count"] + computed["timeout_count"]) == len(rows), f"{label}: outcomes not exhaustive")
    return computed


def validate_checkpoint_metadata(path: Path, expected: Mapping[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    require(isinstance(payload, dict), f"checkpoint payload is not a dict: {path}")
    require(payload.get("schema_version") == "internal_ablation_checkpoint.v1", f"checkpoint schema mismatch: {path}")
    for key in ("program_sha256", "config_sha256", "protocol_sha256", "method", "configuration", "seed", "interaction_count"):
        require(payload.get(key) == expected[key], f"checkpoint {key} mismatch: {path}")
    require(isinstance(payload.get("model_state_dict"), dict) and payload["model_state_dict"], f"checkpoint model state missing: {path}")
    require(isinstance(payload.get("optimizer_state_dict"), dict), f"checkpoint optimizer state missing: {path}")
    return {
        "schema_version": payload["schema_version"],
        "interaction_count": int(payload["interaction_count"]),
        "ppo_update_count": int(payload.get("ppo_update_count", -1)),
        "model_tensor_count": len(payload["model_state_dict"]),
    }


def scan_reward_file(path: Path, method: str) -> Dict[str, Any]:
    switches = EXPECTED_SWITCHES[method]
    count = 0
    target_zero_violations = 0
    enabled_identity_violations = 0
    total_accounting_violations = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "distance_reward", "escape_reward", "revisit_penalty", "oscillation_penalty", "stagnation_penalty",
            "raw_distance_reward", "raw_escape_reward", "raw_revisit_penalty", "raw_oscillation_penalty", "raw_stagnation_penalty",
            "applied_progress_reward", "raw_progress_reward", "total_reward", "goal_reward", "collision_penalty", "step_penalty", "timeout_penalty", "other_reward_terms",
        }
        require(required.issubset(set(reader.fieldnames or [])), f"reward columns missing: {path}")
        for row in reader:
            count += 1
            applied = {
                "progress": float(row["applied_progress_reward"]),
                "revisit": float(row["revisit_penalty"]),
                "oscillation": float(row["oscillation_penalty"]),
                "stagnation": float(row["stagnation_penalty"]),
            }
            raw = {
                "progress": float(row["raw_progress_reward"]),
                "revisit": float(row["raw_revisit_penalty"]),
                "oscillation": float(row["raw_oscillation_penalty"]),
                "stagnation": float(row["raw_stagnation_penalty"]),
            }
            for component, enabled in switches.items():
                if not enabled and abs(applied[component]) > 1e-12:
                    target_zero_violations += 1
                if enabled and not close(applied[component], raw[component], 1e-12):
                    enabled_identity_violations += 1
            if method == "REWARD_WO_PROGRESS" and (abs(float(row["distance_reward"])) > 1e-12 or abs(float(row["escape_reward"])) > 1e-12):
                target_zero_violations += 1
            reconstructed = sum(float(row[key]) for key in (
                "goal_reward", "distance_reward", "collision_penalty", "step_penalty", "other_reward_terms",
            ))
            if not close(reconstructed, row["total_reward"], 1e-9):
                total_accounting_violations += 1
    require(count == 900000, f"reward row count must be 900000: {path} has {count}")
    require(target_zero_violations == 0, f"disabled reward component was nonzero: {path}")
    require(enabled_identity_violations == 0, f"enabled reward component differs from raw value: {path}")
    require(total_accounting_violations == 0, f"reward accounting mismatch: {path}")
    return {
        "rows": count,
        "target_zero_violations": target_zero_violations,
        "enabled_identity_violations": enabled_identity_violations,
        "total_accounting_violations": total_accounting_violations,
    }


def validate_run(method: str, seed: int, reward_scan: bool = True) -> Dict[str, Any]:
    seed_dir = LEGACY_OUTPUT_ROOT / method / f"seed_{seed}"
    runs = sorted(path for path in seed_dir.glob("run_*") if path.is_dir())
    require(len(runs) == 1, f"expected exactly one legacy run for {method}/seed_{seed}, found {len(runs)}")
    run = runs[0]
    manifest_path = run / "run_manifest.json"
    result_path = run / f"formal_result_{method}_seed{seed}.json"
    evaluation_path = run / "evaluation_metrics.csv"
    best_path = run / "models" / "best_checkpoint.json"
    for path in (manifest_path, result_path, evaluation_path, best_path):
        require(path.is_file(), f"required artifact missing: {path}")
    manifest = read_json(manifest_path)
    result = read_json(result_path)
    best = read_json(best_path)
    require(manifest.get("status") == "completed", f"run not completed: {run}")
    require(manifest.get("method") == method and manifest.get("configuration") == method, f"manifest method mismatch: {run}")
    require(int(manifest.get("seed", -1)) == seed, f"manifest seed mismatch: {run}")
    require(result.get("method") == method and result.get("configuration") == method, f"result method mismatch: {run}")
    require(int(result.get("seed", -1)) == seed, f"result seed mismatch: {run}")
    for payload, label in ((manifest, "manifest"), (result, "result")):
        require(payload.get("program_sha256") == EXPECTED_PROGRAM_SHA, f"{label} program identity mismatch: {run}")
        require(payload.get("config_sha256") == EXPECTED_CONFIG_SELF_SHA, f"{label} config identity mismatch: {run}")
        require(payload.get("protocol_sha256") == EXPECTED_PROTOCOL_SELF_SHA, f"{label} protocol identity mismatch: {run}")
        require(payload.get("input_sha256") == EXPECTED_INPUTS, f"{label} input identity mismatch: {run}")
        require(payload.get("reward_component_switches") == EXPECTED_SWITCHES[method], f"{label} reward switch mismatch: {run}")
        require(payload.get("distillation_switch") is False and payload.get("planning_loss_switch") is False, f"{label} A* loss switch mismatch: {run}")
        for key in ("training_astar_calls", "astar_training_query_attempts", "astar_training_targets_generated", "validation_astar_runtime_calls", "test_astar_runtime_calls"):
            require(int(payload.get(key, -1)) == 0, f"{label} nonzero {key}: {run}")
    require(manifest.get("astar_training_status") is False, f"A* training status enabled: {run}")
    require(int(manifest.get("training_interaction_budget", -1)) == 900000, f"budget mismatch: {run}")
    require(int(manifest.get("executed_interaction_count", -1)) == 900000, f"executed interactions mismatch: {run}")
    require(int(result.get("training_interactions", -1)) == 900000, f"result interactions mismatch: {run}")
    require(result.get("test") == manifest.get("test_metrics"), f"manifest/result test mismatch: {run}")
    test = result.get("test")
    require(isinstance(test, dict), f"missing test metrics: {run}")
    require(int(test.get("episode_count", -1)) == 1000, f"test count mismatch: {run}")
    require(int(test.get("astar_runtime_calls", -1)) == 0 and int(test.get("parameter_updates", -1)) == 0, f"test isolation mismatch: {run}")

    with evaluation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        evaluation = list(csv.DictReader(handle))
    episodes = [row for row in evaluation if row["record_type"] == "episode"]
    summaries = [row for row in evaluation if row["record_type"] == "summary"]
    validation_summaries = [row for row in summaries if row["split"] == "validation"]
    test_summaries = [row for row in summaries if row["split"] == "test"]
    require(len(evaluation) == 1910 and len(episodes) == 1900 and len(summaries) == 10, f"evaluation row counts mismatch: {run}")
    require(len(validation_summaries) == 9 and len(test_summaries) == 1, f"evaluation summary counts mismatch: {run}")
    require(tuple(int(row["checkpoint_interaction"]) for row in validation_summaries) == EXPECTED_INTERACTIONS, f"validation interactions mismatch: {run}")

    validation_ids = load_task_ids(PROJECT_ROOT / "03_Code" / "validation_tasks_v2.json")
    test_ids = load_task_ids(PROJECT_ROOT / "03_Code" / "test_tasks_v9_unseen_final.json")
    by_validation_interaction: Dict[int, List[Mapping[str, str]]] = {}
    for row in episodes:
        if row["split"] == "validation":
            by_validation_interaction.setdefault(int(row["checkpoint_interaction"]), []).append(row)
    for summary in validation_summaries:
        interaction = int(summary["checkpoint_interaction"])
        rows = by_validation_interaction.get(interaction, [])
        require(len(rows) == 100, f"validation episode count mismatch at {interaction}: {run}")
        require([row["task_id"] for row in rows] == validation_ids, f"validation task identity/order mismatch at {interaction}: {run}")
        validate_summary(rows, summary, f"{method}/seed{seed}/validation{interaction}")
    test_rows = [row for row in episodes if row["split"] == "test"]
    require(len(test_rows) == 1000, f"test episode count mismatch: {run}")
    require([row["task_id"] for row in test_rows] == test_ids, f"test task identity/order mismatch: {run}")
    recomputed_test = validate_summary(test_rows, test_summaries[0], f"{method}/seed{seed}/test")
    for key in ("episode_count", "success_count", "collision_count", "timeout_count", "success_rate", "collision_rate", "timeout_rate", "average_steps", "average_path_length", "spl", "path_efficiency"):
        require(close(recomputed_test[key], test[key]), f"formal test metric mismatch for {key}: {run}")

    checkpoints = sorted((run / "models").glob("checkpoint_interaction*.pth"))
    require(len(checkpoints) == 9, f"checkpoint count mismatch: {run}")
    for interaction in EXPECTED_INTERACTIONS:
        require((run / "models" / f"checkpoint_interaction{interaction}.pth").is_file(), f"checkpoint {interaction} missing: {run}")
    selected = Path(str(manifest.get("selected_checkpoint", "")))
    require(selected.is_file() and selected.parent.parent == run, f"selected checkpoint path invalid: {run}")
    selected_sha = sha256_file(selected)
    require(selected_sha == str(manifest.get("selected_checkpoint_sha256", "")).upper(), f"manifest selected checkpoint SHA mismatch: {run}")
    require(selected_sha == str(result.get("selected_checkpoint_sha256", "")).upper(), f"result selected checkpoint SHA mismatch: {run}")
    require(selected_sha == str(best.get("checkpoint_sha256", "")).upper(), f"best selected checkpoint SHA mismatch: {run}")
    selected_interaction = int(manifest["selected_checkpoint_interaction"])
    require(selected_interaction == int(best["interaction_count"]), f"best selected interaction mismatch: {run}")
    require(selected.name == f"checkpoint_interaction{selected_interaction}.pth", f"selected checkpoint filename mismatch: {run}")
    ranking = min(validation_summaries, key=lambda row: (-float(row["success_rate"]), -float(row["SPL"]), float(row["timeout_rate"]), float(row["average_steps"]), int(row["checkpoint_interaction"])))
    require(int(ranking["checkpoint_interaction"]) == selected_interaction, f"validation-selected checkpoint ranking mismatch: {run}")
    checkpoint_meta = validate_checkpoint_metadata(selected, {
        "program_sha256": EXPECTED_PROGRAM_SHA,
        "config_sha256": EXPECTED_CONFIG_SELF_SHA,
        "protocol_sha256": EXPECTED_PROTOCOL_SELF_SHA,
        "method": method,
        "configuration": method,
        "seed": seed,
        "interaction_count": selected_interaction,
    })
    trajectories = list((run / "trajectories").glob("episode_*.json"))
    require(len(trajectories) == 1000, f"test trajectory count mismatch: {run}")
    required_files = ("training_metrics.csv", "reward_components.csv", "astar_metrics.csv", "astar_loss_weight_schedule.csv", "astar_guidance_schedule.csv", "run_log.txt")
    for name in required_files:
        require((run / name).is_file(), f"required run artifact missing: {run / name}")
    reward_evidence = scan_reward_file(run / "reward_components.csv", method) if reward_scan else {"status": "SKIPPED"}
    return {
        "method": method,
        "seed": seed,
        "status": "REUSABLE",
        "run_directory": str(run),
        "program_sha256": EXPECTED_PROGRAM_SHA,
        "config_sha256": EXPECTED_CONFIG_SELF_SHA,
        "protocol_sha256": EXPECTED_PROTOCOL_SELF_SHA,
        "input_sha256": EXPECTED_INPUTS,
        "training_interactions": 900000,
        "validation_summaries": 9,
        "validation_episodes": 900,
        "test_episodes": 1000,
        "selected_checkpoint": str(selected),
        "selected_checkpoint_sha256": selected_sha,
        "selected_checkpoint_interaction": selected_interaction,
        "training_astar_calls": 0,
        "astar_training_query_attempts": 0,
        "validation_astar_runtime_calls": 0,
        "test_astar_runtime_calls": 0,
        "reward_scan": reward_evidence,
        "checkpoint_metadata": checkpoint_meta,
        "test_metrics": test,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
    }


def write_outputs(output_dir: Path, records: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "component_legacy_reuse_audit.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "audit_mode": "read_only_no_training_no_optimizer_no_cuda",
        "legacy_code_root": str(LEGACY_CODE_ROOT),
        "legacy_output_root": str(LEGACY_OUTPUT_ROOT),
        "expected_program_sha256": EXPECTED_PROGRAM_SHA,
        "expected_config_sha256": EXPECTED_CONFIG_SELF_SHA,
        "expected_protocol_sha256": EXPECTED_PROTOCOL_SELF_SHA,
        "records": list(records),
        "reusable_count": sum(record["status"] == "REUSABLE" for record in records),
        "not_reusable_count": sum(record["status"] != "REUSABLE" for record in records),
    }
    (output_dir / "legacy_run_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = [
        "method", "seed", "status", "reason", "run_directory", "program_sha256", "config_sha256", "protocol_sha256",
        "training_interactions", "validation_summaries", "validation_episodes", "test_episodes", "selected_checkpoint_interaction",
        "selected_checkpoint_sha256", "training_astar_calls", "astar_training_query_attempts", "validation_astar_runtime_calls", "test_astar_runtime_calls",
        "success_count", "episode_count", "success_rate", "spl", "path_efficiency", "average_steps", "average_path_length", "collision_rate", "timeout_rate",
    ]
    with (output_dir / "legacy_run_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, "") for key in fields}
            metrics = record.get("test_metrics", {})
            for key in ("success_count", "episode_count", "success_rate", "spl", "path_efficiency", "average_steps", "average_path_length", "collision_rate", "timeout_rate"):
                row[key] = metrics.get(key, "") if isinstance(metrics, dict) else ""
            writer.writerow(row)


def audit_all(reward_scan: bool = True) -> List[Dict[str, Any]]:
    require(sha256_file(LEGACY_CODE_ROOT / "internal_ablation_v1.py") == EXPECTED_PROGRAM_SHA, "current legacy program hash mismatch")
    records: List[Dict[str, Any]] = []
    for method in METHODS:
        for seed in SEEDS:
            try:
                record = validate_run(method, seed, reward_scan=reward_scan)
            except Exception as exc:
                record = {
                    "method": method,
                    "seed": seed,
                    "status": "NOT_REUSABLE",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "run_directory": str(LEGACY_OUTPUT_ROOT / method / f"seed_{seed}"),
                }
            records.append(record)
            print(f"[LEGACY AUDIT] {method} seed={seed}: {record['status']}", flush=True)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only audit of the 12 legacy component-ablation runs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-full-reward-scan", action="store_true", help="diagnostic only; final reuse audit must not use this flag")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = audit_all(reward_scan=not args.skip_full_reward_scan)
    write_outputs(args.output_dir, records)
    failures = [record for record in records if record["status"] != "REUSABLE"]
    print(json.dumps({"reusable": len(records) - len(failures), "not_reusable": len(failures), "output_dir": str(args.output_dir.resolve())}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
