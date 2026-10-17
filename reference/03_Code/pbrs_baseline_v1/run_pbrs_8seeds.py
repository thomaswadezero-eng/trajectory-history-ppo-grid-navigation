#!/usr/bin/env python
"""One-command, strictly serial, resumable runner for the eight formal PBRS seeds."""

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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pbrs_baseline_v1 as pbrs


SCRIPT_PATH = Path(__file__).resolve()
SEEDS = [42, 1004, 6010, 68882, 19816, 74494, 81399, 31524]
METRICS = ["success_rate", "spl", "path_efficiency", "average_steps", "timeout_rate"]


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def compact_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class BatchLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8", buffering=1)

    def log(self, message: str = "") -> None:
        print(message, flush=True)
        self.handle.write(message + "\n")

    def child_line(self, line: str) -> None:
        text = line.rstrip("\r\n")
        print(text, flush=True)
        self.handle.write(text + "\n")

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise pbrs.IntegrityError(f"expected JSON object: {path}")
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


def expected_identity(config: pbrs.FrozenConfig, protocol: pbrs.FrozenProtocol) -> Dict[str, Any]:
    return {
        "program_sha256": pbrs.sha256_file(pbrs.SCRIPT_PATH),
        "config_sha256": config.sha256,
        "protocol_sha256": protocol.sha256,
        "input_sha256": pbrs.verify_inputs(config),
    }


def validate_result(path: Path, seed: int, identity: Mapping[str, Any]) -> Dict[str, Any]:
    result = read_json(path)
    if result.get("schema_version") != "pbrs_baseline_result.v1":
        raise pbrs.IntegrityError(f"invalid PBRS result schema: {path}")
    if result.get("method") != pbrs.METHOD or int(result.get("seed", -1)) != seed:
        raise pbrs.IntegrityError(f"PBRS result method/seed mismatch: {path}")
    for key in ("program_sha256", "config_sha256", "protocol_sha256", "input_sha256"):
        if result.get(key) != identity[key]:
            raise pbrs.IntegrityError(f"PBRS result identity mismatch for {key}: {path}")
    if int(result.get("training_interactions", -1)) != 900000:
        raise pbrs.IntegrityError(f"PBRS result has incomplete training budget: {path}")
    if any(int(result.get(key, -1)) != 0 for key in ("training_astar_calls", "astar_training_query_attempts", "astar_training_targets_generated", "validation_astar_runtime_calls", "test_astar_runtime_calls", "test_parameter_updates")):
        raise pbrs.IntegrityError(f"PBRS result violates A*/test isolation: {path}")
    if result.get("dynamic_reward") != {"progress": 0.0, "revisit": 0.0, "oscillation": 0.0, "stagnation": 0.0}:
        raise pbrs.IntegrityError(f"PBRS result Dynamic Reward isolation mismatch: {path}")
    if result.get("pbrs_clipping") is not False:
        raise pbrs.IntegrityError(f"PBRS result clipping declaration mismatch: {path}")
    test = result.get("test")
    if not isinstance(test, dict):
        raise pbrs.IntegrityError(f"PBRS result lacks final test metrics: {path}")
    for metric in METRICS:
        if metric not in test:
            raise pbrs.IntegrityError(f"PBRS result missing {metric}: {path}")
    checkpoint = Path(str(result.get("selected_checkpoint", "")))
    if not checkpoint.is_file() or pbrs.sha256_file(checkpoint) != str(result.get("selected_checkpoint_sha256", "")).upper():
        raise pbrs.IntegrityError(f"PBRS selected checkpoint identity mismatch: {path}")
    run_dir = path.parent
    run_manifest = read_json(run_dir / "run_manifest.json")
    if run_manifest.get("status") != "completed" or int(run_manifest.get("executed_interaction_count", -1)) != 900000:
        raise pbrs.IntegrityError(f"PBRS run manifest is not completed at 900000 interactions: {path}")
    for key in ("program_sha256", "config_sha256", "protocol_sha256", "input_sha256"):
        if run_manifest.get(key) != identity[key]:
            raise pbrs.IntegrityError(f"PBRS run manifest identity mismatch for {key}: {path}")
    evaluation_path = run_dir / "evaluation_metrics.csv"
    with evaluation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validation_summaries = [row for row in rows if row["record_type"] == "summary" and row["split"] == "validation"]
    test_summaries = [row for row in rows if row["record_type"] == "summary" and row["split"] == "test"]
    test_episodes = [row for row in rows if row["record_type"] == "episode" and row["split"] == "test"]
    if (len(validation_summaries), len(test_summaries), len(test_episodes)) != (9, 1, 1000):
        raise pbrs.IntegrityError(f"PBRS evaluation record counts mismatch: {path}")
    checkpoints = list((run_dir / "models").glob("checkpoint_interaction*.pth"))
    trajectories = list((run_dir / "trajectories").glob("episode_*.json"))
    if len(checkpoints) != 9 or len(trajectories) != 1000:
        raise pbrs.IntegrityError(f"PBRS checkpoint/trajectory counts mismatch: {path}")
    for required in ("training_metrics.csv", "reward_components.csv", "astar_metrics.csv", "astar_loss_weight_schedule.csv", "astar_guidance_schedule.csv", "seed_summary.csv", "run_log.txt"):
        if not (run_dir / required).is_file():
            raise pbrs.IntegrityError(f"PBRS required artifact missing: {run_dir / required}")
    return result


def validate_done(path: Path, seed: int, identity: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        done = read_json(path)
        if done.get("status") != "COMPLETED" or int(done.get("seed", -1)) != seed:
            return None
        for key in ("program_sha256", "config_sha256", "protocol_sha256", "input_sha256"):
            if done.get(key) != identity[key]:
                return None
        result_path = Path(str(done.get("result_path", "")))
        if not result_path.is_file() or pbrs.sha256_file(result_path) != str(done.get("result_sha256", "")).upper():
            return None
        result = validate_result(result_path, seed, identity)
        if done.get("final_metrics") != result["test"]:
            return None
        return result
    except Exception:
        return None


def run_subprocess(command: Sequence[str], log_path: Path, logger: BatchLogger) -> int:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8", buffering=1) as seed_log:
        seed_log.write(f"\n[{timestamp()}] COMMAND: {' '.join(command)}\n")
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )
        if process.stdout is None:
            raise RuntimeError("child stdout pipe was not created")
        for line in process.stdout:
            seed_log.write(line)
            seed_log.flush()
            logger.child_line(line)
        return int(process.wait())


def run_preflight(device: str, logger: BatchLogger) -> None:
    logger.log("[PBRS BATCH] Running mandatory preflight...")
    command = [sys.executable, "-u", str(pbrs.SCRIPT_PATH), "--preflight", "--device", device]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if process.stdout is None:
        raise RuntimeError("preflight stdout pipe was not created")
    for line in process.stdout:
        logger.child_line(line)
    return_code = int(process.wait())
    if return_code != 0:
        raise pbrs.IntegrityError(f"preflight failed with return code {return_code}")
    logger.log("[PBRS BATCH] Preflight PASS. Formal serial execution is unlocked.")


def make_done(seed_dir: Path, seed: int, result_path: Path, result: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": "pbrs_batch_done.v1",
        "status": "COMPLETED",
        "seed": seed,
        **identity,
        "result_path": str(result_path),
        "result_sha256": pbrs.sha256_file(result_path),
        "selected_checkpoint": result["selected_checkpoint"],
        "selected_checkpoint_sha256": result["selected_checkpoint_sha256"],
        "final_metrics": result["test"],
        "training_wall_clock_seconds": result["training_wall_clock_seconds"],
        "completion_timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    write_json_exclusive(seed_dir / "DONE.json", payload)


def record_failure(seed_dir: Path, seed: int, return_code: int, log_path: Path, attempt_dir: Path, error: str, identity: Mapping[str, Any]) -> Path:
    payload = {
        "schema_version": "pbrs_batch_failure.v1",
        "status": "FAILED",
        "seed": seed,
        **identity,
        "return_code": return_code,
        "error": error,
        "error_log": str(log_path),
        "attempt_directory": str(attempt_dir),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    primary = seed_dir / "FAILED.json"
    target = primary if not primary.exists() else seed_dir / f"FAILED_{compact_timestamp()}.json"
    write_json_exclusive(target, payload)
    return target


def collect_records(root: Path, identity: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    records: List[Dict[str, Any]] = []
    statuses: Dict[int, str] = {}
    for seed in SEEDS:
        result = validate_done(root / f"seed_{seed}" / "DONE.json", seed, identity)
        if result is None:
            statuses[seed] = "FAILED_OR_INCOMPLETE"
        else:
            statuses[seed] = "DONE"
            records.append(result)
    return records, statuses


def write_csv_atomic(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def aggregate(root: Path, identity: Mapping[str, Any], logger: BatchLogger) -> Tuple[List[Dict[str, Any]], Dict[int, str], Path]:
    records, statuses = collect_records(root, identity)
    by_seed = {int(record["seed"]): record for record in records}
    aggregate_dir = root / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    per_seed_rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        result = by_seed.get(seed)
        row: Dict[str, Any] = {"seed": seed, "status": statuses[seed]}
        if result is None:
            row.update({"success_rate": "", "spl": "", "path_efficiency": "", "average_steps": "", "timeout_rate": "", "selected_checkpoint": "", "training_wall_clock_seconds": ""})
        else:
            row.update({metric: result["test"][metric] for metric in METRICS})
            row["selected_checkpoint"] = result["selected_checkpoint"]
            row["training_wall_clock_seconds"] = result["training_wall_clock_seconds"]
        per_seed_rows.append(row)
    per_seed_path = aggregate_dir / "pbrs_per_seed_metrics.csv"
    write_csv_atomic(per_seed_path, list(per_seed_rows[0]), per_seed_rows)
    mean_std_rows: List[Dict[str, Any]] = []
    actual_n = len(records)
    for metric in METRICS:
        values = [float(record["test"][metric]) for record in records]
        mean_std_rows.append({"metric": metric, "actual_n": actual_n, "planned_n": len(SEEDS), "mean": statistics.mean(values) if values else "", "sample_std": statistics.stdev(values) if len(values) >= 2 else ""})
    mean_std_path = aggregate_dir / "pbrs_mean_std.csv"
    write_csv_atomic(mean_std_path, ["metric", "actual_n", "planned_n", "mean", "sample_std"], mean_std_rows)
    status_payload = {"schema_version": "pbrs_batch_status.v1", "updated_at": datetime.now().isoformat(timespec="seconds"), "official_seeds": SEEDS, "actual_n": actual_n, "planned_n": len(SEEDS), "statuses": {str(seed): statuses[seed] for seed in SEEDS}, "all_eight_completed": actual_n == len(SEEDS)}
    write_json_atomic(aggregate_dir / "batch_status.json", status_payload)
    manifest_payload = {"schema_version": "pbrs_batch_manifest.v1", **identity, "batch_runner_sha256": pbrs.sha256_file(SCRIPT_PATH), "official_seeds": SEEDS, "actual_n": actual_n, "result_files": {str(seed): str(Path(by_seed[seed]["selected_checkpoint"]).parent.parent / f"formal_result_{pbrs.METHOD}_seed{seed}.json") for seed in by_seed}, "aggregate_files": [str(per_seed_path), str(mean_std_path), str(aggregate_dir / "batch_status.json")]}
    write_json_atomic(aggregate_dir / "run_manifest.json", manifest_payload)
    logger.log(f"[PBRS BATCH] Aggregate generated with actual n={actual_n}; planned n={len(SEEDS)}")
    return records, statuses, aggregate_dir


def print_final_summary(records: Sequence[Mapping[str, Any]], statuses: Mapping[int, str], aggregate_dir: Path, logger: BatchLogger) -> None:
    by_seed = {int(record["seed"]): record for record in records}
    logger.log("=" * 76)
    logger.log("PBRS 8-SEED FINAL SUMMARY")
    logger.log("=" * 76)
    logger.log(f"{'Seed':<9}{'Success':<13}{'SPL':<11}{'PathEff':<13}{'AvgSteps':<14}{'Status'}")
    for seed in SEEDS:
        result = by_seed.get(seed)
        if result is None:
            logger.log(f"{seed:<9}{'-':<13}{'-':<11}{'-':<13}{'-':<14}{statuses[seed]}")
        else:
            test = result["test"]
            logger.log(f"{seed:<9}{float(test['success_rate']):<13.4f}{float(test['spl']):<11.4f}{float(test['path_efficiency']):<13.4f}{float(test['average_steps']):<14.3f}DONE")
    actual_n = len(records)
    logger.log("")
    logger.log(f"Successful runs: {actual_n} / {len(SEEDS)}")
    if records:
        logger.log(f"Mean ± sample SD (actual n={actual_n}):")
        labels = {"success_rate": "Success", "spl": "SPL", "path_efficiency": "Path Eff.", "average_steps": "Avg Steps", "timeout_rate": "Timeout"}
        for metric in METRICS:
            values = [float(record["test"][metric]) for record in records]
            sd = statistics.stdev(values) if len(values) >= 2 else None
            suffix = f"{sd:.6f}" if sd is not None else "N/A"
            logger.log(f"{labels[metric]:<11}: {statistics.mean(values):.6f} ± {suffix}")
    logger.log("")
    logger.log(f"Aggregate files: {aggregate_dir}")
    if actual_n != len(SEEDS):
        logger.log(f"WARNING: aggregate actual n={actual_n}; no n=8 result has been claimed.")
    logger.log("=" * 76)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict serial runner for the eight formal PPO+PBRS seeds")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--confirm-formal", action="store_true", help="required safety acknowledgement before any 900k formal seed can start")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_formal:
        raise pbrs.ProtocolError("refusing to start formal training without --confirm-formal")
    config = pbrs.FrozenConfig(pbrs.DEFAULT_CONFIG)
    protocol = pbrs.FrozenProtocol(pbrs.DEFAULT_PROTOCOL)
    pbrs.validate_config(config, protocol)
    if SEEDS != pbrs.OFFICIAL_SEEDS or list(config.get("official_seeds")) != SEEDS:
        raise pbrs.ProtocolError("runner/config 8-seed list mismatch")
    root = Path(str(config.get("output_root"))).resolve()
    if root != Path(r"redacted_local_root\revision_pbrs_v1").resolve():
        raise pbrs.ProtocolError("unexpected PBRS output root")
    pbrs.ensure_archive(config, protocol, root)
    logger = BatchLogger(root / "batch.log")
    try:
        identity = expected_identity(config, protocol)
        completed_before = sum(validate_done(root / f"seed_{seed}" / "DONE.json", seed, identity) is not None for seed in SEEDS)
        logger.log("=" * 60)
        logger.log("PBRS 8-SEED BATCH")
        logger.log(f"Output root : {root}")
        logger.log(f"Device      : {args.device}")
        logger.log(f"Seeds       : {SEEDS}")
        logger.log("Budget      : 900000 interactions / seed")
        logger.log(f"Completed   : {completed_before} / {len(SEEDS)}")
        logger.log("=" * 60)
        run_preflight(args.device, logger)
        completed_count = completed_before
        for index, seed in enumerate(SEEDS, 1):
            seed_dir = root / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            done_path = seed_dir / "DONE.json"
            existing_result = validate_done(done_path, seed, identity)
            if existing_result is not None:
                logger.log(f"[PBRS BATCH] Seed {seed} SKIPPED: valid DONE.json and matching identities")
                continue
            if done_path.exists():
                failure_path = record_failure(seed_dir, seed, -1, seed_dir / "seed.log", seed_dir, "invalid DONE.json; refusing to overwrite it", identity)
                logger.log(f"[PBRS BATCH] Seed {seed} FAILED: invalid DONE.json ({failure_path})")
                logger.log("Continuing to next seed...")
                continue
            attempt_dir = seed_dir / f"attempt_{compact_timestamp()}"
            seed_log = seed_dir / "seed.log"
            logger.log("-")
            logger.log(f"[PBRS BATCH] Starting seed {seed}")
            logger.log(f"Progress: seed {index} / {len(SEEDS)}")
            logger.log(f"Start time: {timestamp()}")
            logger.log("-" * 31)
            command = [sys.executable, "-u", str(pbrs.SCRIPT_PATH), "--seed", str(seed), "--device", args.device, "--run-dir", str(attempt_dir)]
            started = time.perf_counter()
            return_code = -1
            try:
                return_code = run_subprocess(command, seed_log, logger)
                if return_code != 0:
                    raise RuntimeError(f"child process returned {return_code}")
                matches = list(attempt_dir.glob(f"formal_result_{pbrs.METHOD}_seed{seed}.json"))
                if len(matches) != 1:
                    raise pbrs.IntegrityError(f"expected exactly one formal result, found {matches}")
                result = validate_result(matches[0], seed, identity)
                make_done(seed_dir, seed, matches[0], result, identity)
                completed_count += 1
                test = result["test"]
                next_seed = SEEDS[index] if index < len(SEEDS) else "none"
                logger.log("-")
                logger.log(f"[PBRS BATCH] Seed {seed} COMPLETE")
                logger.log(f"Success      : {float(test['success_rate']):.4f}")
                logger.log(f"SPL          : {float(test['spl']):.4f}")
                logger.log(f"Path Eff.    : {float(test['path_efficiency']):.4f}")
                logger.log(f"Avg. Steps   : {float(test['average_steps']):.4f}")
                logger.log(f"Wall time    : {pbrs._format_duration(time.perf_counter() - started)}")
                logger.log(f"Batch status : {completed_count} / {len(SEEDS)} seeds complete")
                logger.log(f"Next seed    : {next_seed}")
                logger.log("-" * 19)
            except Exception as exc:
                failure_path = record_failure(seed_dir, seed, return_code, seed_log, attempt_dir, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", identity)
                logger.log(f"[PBRS BATCH] Seed {seed} FAILED")
                logger.log(f"Return code: {return_code}")
                logger.log(f"Error log: {seed_log}")
                logger.log(f"Failure record: {failure_path}")
                logger.log("Continuing to next seed...")
        records, statuses, aggregate_dir = aggregate(root, identity, logger)
        print_final_summary(records, statuses, aggregate_dir, logger)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
