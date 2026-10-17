#!/usr/bin/env python
"""Strictly serial, resumable one-command runner for component ablations.

This file is an execution entrypoint and must only be run manually by the user.
It never launches work without both ``--confirm-formal`` and a matching manual
runtime-preflight PASS marker.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Capture the user's launch environment before any legacy checkpoint audit can
# force CPU-only loading by mutating the parent process environment.
ORIGINAL_CHILD_ENV = os.environ.copy()

import audit_existing_runs as legacy
import component_ablation_8seed_v1 as impl


SCRIPT_PATH = Path(__file__).resolve()
OUTPUT_ROOT = Path(r"redacted_local_root\revision_component_ablation_8seed_v1")
PREFLIGHT_MARKER = OUTPUT_ROOT / "preflight" / "RUNTIME_PREFLIGHT_DONE.json"
METHODS = tuple(impl.CONFIGURATIONS)
SEEDS = tuple(impl.SEEDS)
METRICS = ("success_rate", "spl", "path_efficiency", "average_steps", "average_path_length", "collision_rate", "timeout_rate")


class BatchLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


def compact_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise impl.IntegrityError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def build_child_environment(original_environment: Mapping[str, str]) -> Dict[str, str]:
    """Return the formal child's environment from the pre-audit snapshot."""
    return {**original_environment, "PYTHONUNBUFFERED": "1"}


def expected_identity(config: impl.RevisionConfig, protocol: impl.RevisionProtocol, core: Any) -> Dict[str, Any]:
    return {
        "program_sha256": impl.sha256_file(impl.SCRIPT_PATH),
        "config_sha256": config.sha256,
        "protocol_sha256": protocol.sha256,
        "input_sha256": core.verify_inputs(config),
    }


def validate_preflight(identity: Mapping[str, Any], device: str) -> Dict[str, Any]:
    if not PREFLIGHT_MARKER.is_file():
        raise impl.IntegrityError(f"manual runtime preflight marker missing: {PREFLIGHT_MARKER}")
    marker = read_json(PREFLIGHT_MARKER)
    if marker.get("status") != "PASS" or marker.get("device") != device:
        raise impl.IntegrityError("runtime preflight status/device mismatch")
    for key in ("program_sha256", "config_sha256", "protocol_sha256", "input_sha256"):
        if marker.get(key) != identity[key]:
            raise impl.IntegrityError(f"runtime preflight identity mismatch for {key}")
    if int(marker.get("environment_transitions", -1)) != 1 or int(marker.get("ppo_optimizer_updates", -1)) != 1:
        raise impl.IntegrityError("runtime preflight did not record exactly one transition and one isolated update")
    if int(marker.get("formal_training_interactions", -1)) != 0:
        raise impl.IntegrityError("runtime preflight contains formal training activity")
    return marker


def validate_revision_result(path: Path, method: str, seed: int, identity: Mapping[str, Any]) -> Dict[str, Any]:
    result = read_json(path)
    if result.get("schema_version") != "internal_ablation_result.v1":
        raise impl.IntegrityError(f"revision result schema mismatch: {path}")
    if result.get("method") != method or result.get("configuration") != method or int(result.get("seed", -1)) != seed:
        raise impl.IntegrityError(f"revision method/seed mismatch: {path}")
    for key in ("program_sha256", "config_sha256", "protocol_sha256", "input_sha256"):
        if result.get(key) != identity[key]:
            raise impl.IntegrityError(f"revision result identity mismatch for {key}: {path}")
    if result.get("reward_component_switches") != impl.CONFIGURATIONS[method]["reward_components"]:
        raise impl.IntegrityError(f"revision reward switch mismatch: {path}")
    if result.get("distillation_switch") is not False or result.get("planning_loss_switch") is not False:
        raise impl.IntegrityError(f"revision A* loss switch mismatch: {path}")
    if int(result.get("training_interactions", -1)) != 900000:
        raise impl.IntegrityError(f"revision incomplete training budget: {path}")
    for key in ("training_astar_calls", "astar_training_query_attempts", "astar_training_targets_generated", "validation_astar_runtime_calls", "test_astar_runtime_calls"):
        if int(result.get(key, -1)) != 0:
            raise impl.IntegrityError(f"revision nonzero {key}: {path}")
    test = result.get("test")
    if not isinstance(test, dict) or int(test.get("episode_count", -1)) != 1000:
        raise impl.IntegrityError(f"revision final test missing/incomplete: {path}")
    if int(test.get("astar_runtime_calls", -1)) != 0 or int(test.get("parameter_updates", -1)) != 0:
        raise impl.IntegrityError(f"revision final-test isolation mismatch: {path}")
    for metric in METRICS:
        if metric not in test:
            raise impl.IntegrityError(f"revision missing metric {metric}: {path}")
    run = path.parent
    manifest = read_json(run / "run_manifest.json")
    if manifest.get("status") != "completed" or int(manifest.get("executed_interaction_count", -1)) != 900000:
        raise impl.IntegrityError(f"revision manifest incomplete: {path}")
    for key in ("program_sha256", "config_sha256", "protocol_sha256", "input_sha256"):
        if manifest.get(key) != identity[key]:
            raise impl.IntegrityError(f"revision manifest identity mismatch for {key}: {path}")
    if manifest.get("astar_training_status") is not False:
        raise impl.IntegrityError(f"revision A* training status enabled: {path}")
    evaluation_path = run / "evaluation_metrics.csv"
    with evaluation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validation_summaries = [row for row in rows if row["record_type"] == "summary" and row["split"] == "validation"]
    validation_episodes = [row for row in rows if row["record_type"] == "episode" and row["split"] == "validation"]
    test_summaries = [row for row in rows if row["record_type"] == "summary" and row["split"] == "test"]
    test_episodes = [row for row in rows if row["record_type"] == "episode" and row["split"] == "test"]
    if (len(validation_summaries), len(validation_episodes), len(test_summaries), len(test_episodes)) != (9, 900, 1, 1000):
        raise impl.IntegrityError(f"revision evaluation record counts mismatch: {path}")
    if tuple(int(row["checkpoint_interaction"]) for row in validation_summaries) != tuple(range(100000, 900001, 100000)):
        raise impl.IntegrityError(f"revision validation interactions mismatch: {path}")
    checkpoints = list((run / "models").glob("checkpoint_interaction*.pth"))
    trajectories = list((run / "trajectories").glob("episode_*.json"))
    if len(checkpoints) != 9 or len(trajectories) != 1000:
        raise impl.IntegrityError(f"revision checkpoint/trajectory counts mismatch: {path}")
    selected = Path(str(result.get("selected_checkpoint", "")))
    if not selected.is_file() or impl.sha256_file(selected) != str(result.get("selected_checkpoint_sha256", "")).upper():
        raise impl.IntegrityError(f"revision selected checkpoint identity mismatch: {path}")
    best = read_json(run / "models" / "best_checkpoint.json")
    if str(best.get("checkpoint_sha256", "")).upper() != str(result.get("selected_checkpoint_sha256", "")).upper():
        raise impl.IntegrityError(f"revision best-checkpoint mismatch: {path}")
    return result


def validate_done(path: Path, method: str, seed: int, identity: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        done = read_json(path)
        if done.get("status") != "COMPLETED" or done.get("method") != method or int(done.get("seed", -1)) != seed:
            return None
        for key in ("program_sha256", "config_sha256", "protocol_sha256", "input_sha256"):
            if done.get(key) != identity[key]:
                return None
        result_path = Path(str(done.get("result_path", "")))
        if not result_path.is_file() or impl.sha256_file(result_path) != str(done.get("result_sha256", "")).upper():
            return None
        result = validate_revision_result(result_path, method, seed, identity)
        if done.get("final_metrics") != result["test"]:
            return None
        return result
    except Exception:
        return None


def resume_decision(legacy_reusable: bool, done_exists: bool, done_valid: bool) -> str:
    """Pure decision table used by the runner and static resume simulation."""
    if legacy_reusable:
        return "SKIP_REUSABLE_LEGACY"
    if done_valid:
        return "SKIP_VALID_DONE"
    if done_exists:
        return "REFUSE_INVALID_DONE"
    return "CREATE_NEW_ATTEMPT"


def make_done(seed_dir: Path, method: str, seed: int, result_path: Path, result: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": "component_ablation_batch_done.v1",
        "status": "COMPLETED",
        "method": method,
        "seed": seed,
        **identity,
        "result_path": str(result_path),
        "result_sha256": impl.sha256_file(result_path),
        "selected_checkpoint": result["selected_checkpoint"],
        "selected_checkpoint_sha256": result["selected_checkpoint_sha256"],
        "final_metrics": result["test"],
        "completion_timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_exclusive(seed_dir / "DONE.json", payload)


def record_failure(seed_dir: Path, method: str, seed: int, return_code: int, attempt: Optional[Path], error: str, identity: Mapping[str, Any]) -> Path:
    payload = {
        "schema_version": "component_ablation_batch_failure.v1",
        "status": "FAILED",
        "method": method,
        "seed": seed,
        **identity,
        "return_code": return_code,
        "attempt_directory": "" if attempt is None else str(attempt),
        "error": error,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    primary = seed_dir / "FAILED.json"
    target = primary if not primary.exists() else seed_dir / f"FAILED_{compact_timestamp()}.json"
    write_json_exclusive(target, payload)
    return target


def recent_training_progress(run: Path) -> Optional[Tuple[int, float, float]]:
    path = run / "training_metrics.csv"
    if not path.is_file():
        return None
    recent: deque[Dict[str, str]] = deque(maxlen=100)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                recent.append(row)
    except (OSError, csv.Error, ValueError):
        return None
    if not recent:
        return None
    interaction = int(recent[-1]["interaction_count"])
    reward = statistics.mean(float(row["reward"]) for row in recent)
    success = statistics.mean(float(row["success"]) for row in recent)
    return interaction, reward, success


def report_new_validations(run: Path, seen: set, logger: BatchLogger) -> None:
    path = run / "evaluation_metrics.csv"
    if not path.is_file():
        return
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [row for row in csv.DictReader(handle) if row["record_type"] == "summary" and row["split"] == "validation"]
    except (OSError, csv.Error, ValueError):
        return
    best_path = run / "models" / "best_checkpoint.json"
    best_interaction = None
    if best_path.is_file():
        try:
            best_interaction = int(read_json(best_path)["interaction_count"])
        except Exception:
            best_interaction = None
    for row in rows:
        interaction = int(row["checkpoint_interaction"])
        if interaction in seen:
            continue
        seen.add(interaction)
        update = "UPDATED" if best_interaction == interaction else "NOT UPDATED"
        logger.log(
            f"[COMPONENT VALIDATION] interaction={interaction} success={float(row['success_rate']):.4f} "
            f"SPL={float(row['SPL']):.4f} timeout={float(row['timeout_rate']):.4f} checkpoint={update} A*_queries=0"
        )


def run_child(method: str, seed: int, device: str, method_index: int, seed_index: int, completed: int, logger: BatchLogger) -> Tuple[int, Optional[Path]]:
    seed_dir = OUTPUT_ROOT / method / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    before = {path.resolve() for path in seed_dir.glob("run_*") if path.is_dir()}
    log_path = seed_dir / f"attempt_{compact_timestamp()}.log"
    command = [sys.executable, "-u", str(impl.SCRIPT_PATH), "--method", method, "--seed", str(seed), "--device", device, "--confirm-formal"]
    started = time.time()
    with log_path.open("x", encoding="utf-8", buffering=1) as child_log:
        child_log.write(f"COMMAND: {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            stdout=child_log,
            stderr=subprocess.STDOUT,
            text=True,
            env=build_child_environment(ORIGINAL_CHILD_ENV),
        )
        attempt: Optional[Path] = None
        offset = 0
        seen_validations: set = set()
        try:
            while process.poll() is None:
                time.sleep(30)
                candidates = [path.resolve() for path in seed_dir.glob("run_*") if path.is_dir() and path.resolve() not in before]
                if len(candidates) == 1:
                    attempt = candidates[0]
                if log_path.is_file():
                    with log_path.open("r", encoding="utf-8", errors="replace") as reader:
                        reader.seek(offset)
                        for line in reader:
                            logger.log(line.rstrip("\r\n"))
                        offset = reader.tell()
                if attempt is not None:
                    report_new_validations(attempt, seen_validations, logger)
                    progress = recent_training_progress(attempt)
                    if progress is not None:
                        interaction, recent_reward, recent_success = progress
                        elapsed = time.time() - started
                        eta = (elapsed / interaction * (900000 - interaction)) if interaction > 0 else float("nan")
                        logger.log(
                            f"[COMPONENT BATCH] method={method} ({method_index}/4) seed={seed} ({seed_index}/8) "
                            f"interaction={interaction}/900000 ({100.0 * interaction / 900000:.2f}%) "
                            f"elapsed={elapsed/3600:.2f}h ETA={eta/3600:.2f}h recent100_reward={recent_reward:.6f} "
                            f"recent100_success={recent_success:.4f} A*_queries=0 batch_completed={completed}/32"
                        )
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        return_code = int(process.wait())
    candidates = [path.resolve() for path in seed_dir.glob("run_*") if path.is_dir() and path.resolve() not in before]
    attempt = candidates[0] if len(candidates) == 1 else None
    return return_code, attempt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict serial resumable runner for four component ablations × eight paired seeds")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_formal:
        raise impl.ProtocolError("refusing to start formal training without --confirm-formal")
    core, config, protocol = impl.prepare()
    identity = expected_identity(config, protocol, core)
    validate_preflight(identity, args.device)
    legacy_records = legacy.audit_all(reward_scan=False)
    legacy_by_key = {(record["method"], int(record["seed"])): record for record in legacy_records if record["status"] == "REUSABLE"}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    logger = BatchLogger(OUTPUT_ROOT / "batch.log")
    try:
        completed = len(legacy_by_key)
        logger.log(f"[COMPONENT BATCH] validated reusable legacy runs: {completed}/12")
        for method_index, method in enumerate(METHODS, 1):
            for seed_index, seed in enumerate(SEEDS, 1):
                seed_dir = OUTPUT_ROOT / method / f"seed_{seed}"
                done_path = seed_dir / "DONE.json"
                existing_result = validate_done(done_path, method, seed, identity)
                decision = resume_decision((method, seed) in legacy_by_key, done_path.exists(), existing_result is not None)
                if decision == "SKIP_REUSABLE_LEGACY":
                    logger.log(f"[COMPONENT BATCH] {method} seed={seed} SKIPPED: validated reusable legacy result")
                    continue
                seed_dir.mkdir(parents=True, exist_ok=True)
                if decision == "SKIP_VALID_DONE":
                    completed += 1
                    logger.log(f"[COMPONENT BATCH] {method} seed={seed} SKIPPED: valid DONE.json")
                    continue
                if decision == "REFUSE_INVALID_DONE":
                    failure = record_failure(seed_dir, method, seed, -1, None, "invalid DONE.json; refusing to overwrite", identity)
                    logger.log(f"[COMPONENT BATCH] {method} seed={seed} FAILED: invalid DONE ({failure})")
                    continue
                logger.log(f"[COMPONENT BATCH] START method={method} ({method_index}/4) seed={seed} ({seed_index}/8) completed={completed}/32")
                attempt: Optional[Path] = None
                try:
                    return_code, attempt = run_child(method, seed, args.device, method_index, seed_index, completed, logger)
                    if return_code != 0 or attempt is None:
                        raise impl.IntegrityError(f"child return code={return_code}; unique attempt={attempt}")
                    matches = list(attempt.glob(f"formal_result_{method}_seed{seed}.json"))
                    if len(matches) != 1:
                        raise impl.IntegrityError(f"expected one formal result in {attempt}, found {len(matches)}")
                    result = validate_revision_result(matches[0], method, seed, identity)
                    make_done(seed_dir, method, seed, matches[0], result, identity)
                    completed += 1
                    logger.log(f"[COMPONENT BATCH] DONE method={method} seed={seed} completed={completed}/32")
                except KeyboardInterrupt:
                    failure = record_failure(seed_dir, method, seed, -2, attempt, "KeyboardInterrupt", identity)
                    logger.log(f"[COMPONENT BATCH] INTERRUPTED method={method} seed={seed}; recorded {failure}")
                    raise
                except Exception as exc:
                    failure = record_failure(seed_dir, method, seed, -1, attempt, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", identity)
                    logger.log(f"[COMPONENT BATCH] FAILED method={method} seed={seed}; retained attempt; recorded {failure}")
        from aggregate_and_analyze import aggregate_and_analyze
        aggregate_and_analyze(OUTPUT_ROOT, identity, legacy_records, logger.log)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
