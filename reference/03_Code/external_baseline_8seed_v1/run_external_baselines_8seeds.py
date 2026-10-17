#!/usr/bin/env python
"""Strictly serial, batch-resumable external-baseline 8-seed runner.

This file is an execution entrypoint.  It never launches a formal child unless
the user supplies --confirm-formal and a matching manual preflight PASS marker.
"""

from __future__ import annotations

import os

# This snapshot must precede every import that could transitively alter the
# parent environment during a legacy audit.
ORIGINAL_CHILD_ENV = os.environ.copy()

import argparse
import csv
import json
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import audit_existing_runs
import common


SCRIPT_PATH = Path(__file__).resolve()
PREFLIGHT_MARKER = common.OUTPUT_ROOT / "preflight" / "RUNTIME_PREFLIGHT_PASS.json"
MASKABLE_PYTHON = common.PROJECT_ROOT / "03_Code" / "external_baselines" / ".venv_py310" / "python.exe"
VALUE_C51_PYTHON = Path(r"redacted_local_root\python.exe")
FORMAL_BUDGET = 900000
TOTAL_RECORDS = len(common.METHODS) * len(common.SEEDS)
TOTAL_NEW_RUNS = len(common.METHODS) * len(common.NEW_SEEDS)


def child_environment(snapshot: Mapping[str, str]) -> Dict[str, str]:
    return {**snapshot, "PYTHONUNBUFFERED": "1"}


def current_identity() -> Dict[str, Any]:
    files = (
        "common.py", "audit_existing_runs.py", "run_value_wrapper.py", "run_c51_wrapper.py",
        "run_maskable_ppo_wrapper.py", "run_external_baselines_8seeds.py", "runtime_preflight.py",
        "runtime_probe.py",
        "aggregate_and_analyze.py", "external_baseline_hypothesis_plan_v1.json",
    )
    return {name: common.sha256_file(common.REVISION_ROOT / name) for name in files}


def validate_preflight(device: str) -> Dict[str, Any]:
    if not PREFLIGHT_MARKER.is_file():
        raise common.AuditError(f"runtime preflight marker missing: {PREFLIGHT_MARKER}")
    marker = common.read_json(PREFLIGHT_MARKER)
    if marker.get("status") != "PASS" or marker.get("device") != device:
        raise common.AuditError("runtime preflight status/device mismatch")
    if marker.get("revision_identity") != current_identity():
        raise common.AuditError("runtime preflight revision identity mismatch")
    if marker.get("formal_training_interactions") != 0 or marker.get("formal_result_root_written") is not False:
        raise common.AuditError("preflight marker does not prove formal isolation")
    if sorted(marker.get("methods", [])) != sorted(common.METHODS):
        raise common.AuditError("preflight method coverage mismatch")
    bootstrap = marker.get("formal_wrapper_bootstrap", {})
    if (
        bootstrap.get("status") != "PASS"
        or bootstrap.get("common_path")
        != str((common.REVISION_ROOT / "common.py").resolve())
        or bootstrap.get("authority_path")
        != str(common.AUTHORITY["MaskablePPO"]["program"].resolve())
        or bootstrap.get("authority_sha256")
        != common.AUTHORITY["MaskablePPO"]["program_sha256"]
        or bootstrap.get("formal_training_interactions") != 0
        or bootstrap.get("formal_result_created") is not False
    ):
        raise common.AuditError("preflight lacks valid MaskablePPO formal-wrapper bootstrap coverage")
    return marker


def resume_decision(legacy_status: str, done_states: Sequence[Tuple[Path, bool]]) -> str:
    if legacy_status == "REUSABLE_LEGACY":
        return "SKIP_REUSABLE_LEGACY"
    if any(not valid for _path, valid in done_states):
        return "REFUSE_INVALID_DONE"
    if len(done_states) == 1 and done_states[0][1]:
        return "SKIP_VALID_DONE"
    if len(done_states) > 1:
        return "REFUSE_MULTIPLE_DONE"
    return "CREATE_NEW_ATTEMPT"


def validate_result(attempt: Path, method: str, seed: int) -> Dict[str, Any]:
    result_path = attempt / common.result_name(method, seed)
    manifest_path = attempt / "run_manifest.json"
    provenance_path = attempt / "extension_provenance.json"
    if not result_path.is_file() or not manifest_path.is_file() or not provenance_path.is_file():
        raise common.AuditError("formal result, manifest, or extension provenance missing")
    result = common.read_json(result_path)
    manifest = common.read_json(manifest_path)
    provenance = common.read_json(provenance_path)
    actual_method = str(result.get("method") or result.get("algorithm") or method)
    if method not in actual_method or int(result.get("seed", -1)) != seed:
        raise common.AuditError("revision result method/seed mismatch")
    if str(manifest.get("status")) not in {"completed", "completed_with_formal_test"}:
        raise common.AuditError("revision manifest is incomplete")
    executed = int(manifest.get("executed_interaction_count", manifest.get("executed_training_interactions", -1)))
    if executed != 900000 or int(result.get("training_interactions", -1)) != 900000:
        raise common.AuditError("revision training budget mismatch")
    if provenance.get("algorithm_or_hyperparameter_changes") != []:
        raise common.AuditError("revision provenance reports an algorithm change")
    test = result.get("test")
    if not isinstance(test, dict) or int(test.get("episode_count", -1)) != 1000:
        raise common.AuditError("revision test result is incomplete")
    if int(test.get("parameter_updates", result.get("test_parameter_updates", 0))) != 0:
        raise common.AuditError("revision test parameter updates are nonzero")
    if int(test.get("astar_runtime_calls", result.get("test_astar_runtime_calls", 0))) != 0:
        raise common.AuditError("revision test A* runtime calls are nonzero")
    checkpoint = Path(str(result.get("selected_checkpoint", "")))
    checkpoint_sha = str(result.get("selected_checkpoint_sha256", "")).upper()
    if not checkpoint.is_file() or common.sha256_file(checkpoint) != checkpoint_sha:
        raise common.AuditError("revision selected checkpoint mismatch")
    rows = common.load_episode_rows(common.test_csv_path(attempt))
    recomputed = common.recompute_metrics(rows)
    problems = common.compare_metrics(recomputed, test)
    if problems or not common.verify_test_order(rows):
        raise common.AuditError("revision episode-level audit failed: " + "; ".join(problems))
    return {"result": result, "manifest": manifest, "test": test, "result_path": result_path, "checkpoint": checkpoint}


def validate_done(path: Path, method: str, seed: int) -> bool:
    try:
        done = common.read_json(path)
        if done.get("status") != "COMPLETED" or done.get("method") != method or int(done.get("seed", -1)) != seed:
            return False
        attempt = path.parent
        validated = validate_result(attempt, method, seed)
        if str(done.get("formal_result_sha256", "")).upper() != common.sha256_file(validated["result_path"]):
            return False
        if done.get("result_file_hashes") != result_file_hashes(attempt, validated):
            return False
        return done.get("final_metrics") == validated["test"]
    except Exception:
        return False


def result_file_hashes(attempt: Path, validated: Mapping[str, Any]) -> Dict[str, str]:
    paths = {
        "formal_result": validated["result_path"],
        "run_manifest": attempt / "run_manifest.json",
        "extension_provenance": attempt / "extension_provenance.json",
        "test_episode_csv": common.test_csv_path(attempt),
        "selected_checkpoint": validated["checkpoint"],
    }
    return {name: common.sha256_file(Path(path)) for name, path in paths.items()}


def write_done(attempt: Path, method: str, seed: int, validated: Mapping[str, Any]) -> Path:
    target = attempt / "DONE.json"
    if target.exists():
        raise FileExistsError(target)
    payload = {
        "schema_version": "external_baseline_8seed_done.v1", "status": "COMPLETED",
        "method": method, "seed": seed,
        "legacy_program_sha256": common.AUTHORITY[method]["program_sha256"],
        "legacy_config_sha256": common.AUTHORITY[method]["config_sha256"],
        "legacy_protocol_sha256": common.AUTHORITY[method].get("protocol_sha256"),
        "authority_identity": {
            key: value for key, value in common.AUTHORITY[method].items()
            if key.endswith("_sha256")
        },
        "formal_result": str(validated["result_path"]),
        "formal_result_sha256": common.sha256_file(validated["result_path"]),
        "executed_budget": 900000,
        "selected_checkpoint": str(validated["checkpoint"]),
        "selected_checkpoint_sha256": common.sha256_file(validated["checkpoint"]),
        "input_hashes": common.TASK_HASHES,
        "final_metrics": validated["test"],
        "result_file_hashes": result_file_hashes(attempt, validated),
        "completed_at": common.now_iso(),
    }
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return target


def write_failed(attempt: Path, method: str, seed: int, command: Sequence[str], return_code: int, error: str) -> Path:
    target = attempt / "FAILED.json"
    if target.exists():
        target = attempt / f"FAILED_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    payload = {
        "schema_version": "external_baseline_8seed_failed.v1", "status": "FAILED",
        "method": method, "seed": seed, "timestamp": common.now_iso(),
        "command": list(command), "return_code": return_code, "exception_or_stderr_tail": error[-8000:],
        "attempt_path": str(attempt),
    }
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return target


def command_for(method: str, seed: int, device: str, attempt: Path) -> Sequence[str]:
    if method in ("DDQN", "D3QN"):
        if not VALUE_C51_PYTHON.is_file():
            raise common.AuditError(f"frozen value-baseline interpreter missing: {VALUE_C51_PYTHON}")
        return [str(VALUE_C51_PYTHON), "-u", str(common.REVISION_ROOT / "run_value_wrapper.py"), "--method", method, "--seed", str(seed), "--device", device, "--attempt-dir", str(attempt), "--confirm-formal"]
    if method == "C51":
        if not VALUE_C51_PYTHON.is_file():
            raise common.AuditError(f"frozen C51 interpreter missing: {VALUE_C51_PYTHON}")
        return [str(VALUE_C51_PYTHON), "-u", str(common.REVISION_ROOT / "run_c51_wrapper.py"), "--seed", str(seed), "--device", device, "--attempt-dir", str(attempt), "--confirm-formal"]
    if not MASKABLE_PYTHON.is_file():
        raise common.AuditError(f"frozen MaskablePPO interpreter missing: {MASKABLE_PYTHON}")
    return [str(MASKABLE_PYTHON), "-u", str(common.REVISION_ROOT / "run_maskable_ppo_wrapper.py"), "--seed", str(seed), "--device", device, "--attempt-dir", str(attempt), "--confirm-formal"]


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ProgressDisplay:
    """Read-only observer for child text/files; it never calls legacy code."""

    def __init__(
        self, attempt: Path, method: str, seed: int, method_index: int,
        seed_index: int, new_run_index: int, completed: int,
        completed_new_durations: Sequence[float],
    ) -> None:
        self.attempt = attempt
        self.method = method
        self.seed = seed
        self.method_index = method_index
        self.seed_index = seed_index
        self.new_run_index = new_run_index
        self.completed = completed
        self.completed_new_durations = list(completed_new_durations)
        self.started = time.monotonic()
        self.last_interaction = -1
        self.last_test_count = -1
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._poll_files, daemon=True)

    def start(self) -> None:
        overall_eta = "estimating"
        if self.completed_new_durations:
            remaining = TOTAL_NEW_RUNS - len(self.completed_new_durations)
            overall_eta = _duration(
                (sum(self.completed_new_durations) / len(self.completed_new_durations)) * remaining
            )
        print(
            f"[EXTERNAL BATCH] method={self.method} ({self.method_index}/4) "
            f"seed={self.seed} ({self.seed_index}/8) new_run={self.new_run_index}/{TOTAL_NEW_RUNS} "
            f"batch_completed={self.completed}/{TOTAL_RECORDS} interaction=0/{FORMAL_BUDGET} "
            f"progress=0.00% elapsed=00:00:00 ETA=estimating batch_ETA={overall_eta}",
            flush=True,
        )
        self.thread.start()

    def stop(self) -> float:
        self.stop_event.set()
        self.thread.join()
        return time.monotonic() - self.started

    def _training(self, interaction: int, source: str) -> None:
        interaction = max(0, min(FORMAL_BUDGET, int(interaction)))
        if interaction <= self.last_interaction:
            return
        self.last_interaction = interaction
        elapsed = time.monotonic() - self.started
        eta = "estimating" if interaction <= 0 else _duration(elapsed * (FORMAL_BUDGET - interaction) / interaction)
        print(
            f"[TRAIN] method={self.method} seed={self.seed} stage=training "
            f"interaction={interaction}/{FORMAL_BUDGET} progress={100.0 * interaction / FORMAL_BUDGET:.2f}% "
            f"elapsed={_duration(elapsed)} ETA={eta} source={source}",
            flush=True,
        )

    def observe_line(self, line: str) -> None:
        validation = re.search(
            r"validation\s+interaction=(\d+).*?success=([0-9.eE+-]+).*?SPL=([0-9.eE+-]+)",
            line,
            re.IGNORECASE,
        )
        if validation:
            interaction = int(validation.group(1))
            self._training(interaction, "legacy_validation_log")
            best = self._best_checkpoint()
            print(
                f"[VALIDATION] method={self.method} seed={self.seed} interaction={interaction} "
                f"success={validation.group(2)} SPL={validation.group(3)} best_checkpoint={best}",
                flush=True,
            )
            return
        diagnostic = re.search(r"diagnostic\s+interactions?=(\d+)", line, re.IGNORECASE)
        if diagnostic:
            self._training(int(diagnostic.group(1)), "legacy_diagnostic_log")
            return
        sb3 = re.search(r"total_timesteps\s*\|\s*(\d+)", line, re.IGNORECASE)
        if sb3:
            interaction = int(sb3.group(1))
            if interaction - self.last_interaction >= 10000 or interaction >= FORMAL_BUDGET:
                self._training(interaction, "frozen_sb3_stdout")

    def _best_checkpoint(self) -> str:
        path = self.attempt / "models" / "best_checkpoint.json"
        try:
            payload = common.read_json(path)
            return str(payload.get("checkpoint_path") or payload.get("path") or "pending")
        except Exception:
            return "pending"

    def _validation_interaction(self) -> int:
        root = self.attempt / "validation"
        values = []
        if root.is_dir():
            for path in root.glob("interaction_*"):
                digits = "".join(char for char in path.name if char.isdigit())
                if digits:
                    values.append(int(digits))
        return max(values, default=-1)

    def _test_episode_count(self) -> int:
        candidates = (
            self.attempt / "test" / "evaluation_metrics.csv",
            self.attempt / "test" / "evaluation_episodes.csv",
        )
        for path in candidates:
            if path.is_file():
                try:
                    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                        return max(0, sum(1 for _line in handle) - 1)
                except OSError:
                    return 0
        return -1

    def _poll_files(self) -> None:
        while not self.stop_event.wait(2.0):
            interaction = self._validation_interaction()
            if interaction > self.last_interaction:
                self._training(interaction, "validation_checkpoint")
            test_count = self._test_episode_count()
            if test_count >= 0 and (
                self.last_test_count < 0 or test_count >= self.last_test_count + 100 or test_count >= 1000
            ):
                self.last_test_count = test_count
                print(
                    f"[TEST] method={self.method} seed={self.seed} episodes={min(test_count, 1000)}/1000 "
                    f"elapsed={_duration(time.monotonic() - self.started)}",
                    flush=True,
                )


def run_child(
    command: Sequence[str], attempt: Path, progress: ProgressDisplay
) -> Tuple[int, str, float]:
    log_path = attempt / "attempt.log"
    tail: list[str] = []
    progress.start()
    try:
        with log_path.open("x", encoding="utf-8", buffering=1) as log:
            log.write("COMMAND: " + json.dumps(list(command)) + "\n")
            log.write("PYTHON: " + str(command[0]) + "\n")
            log.write("WORKING_DIRECTORY: " + str(common.PROJECT_ROOT) + "\n")
            log.write("CUDA_VISIBLE_DEVICES: " + repr(ORIGINAL_CHILD_ENV.get("CUDA_VISIBLE_DEVICES")) + "\n")
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                cwd=str(common.PROJECT_ROOT), env=child_environment(ORIGINAL_CHILD_ENV),
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                progress.observe_line(line)
                tail.append(line)
                if len(tail) > 200:
                    tail.pop(0)
            return_code = int(process.wait())
    finally:
        elapsed = progress.stop()
    return return_code, "".join(tail), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strictly serial external-baseline 8-seed extension")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_formal:
        raise common.AuditError("refusing formal execution without --confirm-formal")
    validate_preflight(args.device)
    audit = audit_existing_runs.run_audit()
    legacy = {(record["method"], int(record["seed"])): record["status"] for record in audit["legacy_runs"]}
    if audit["full_reference_summary"]["valid"] != 8:
        raise common.AuditError("FULL read-only reference provenance is not valid for all eight seeds")
    completed = sum(status == "REUSABLE_LEGACY" for status in legacy.values())
    completed_new_durations: list[float] = []
    for method_index, method in enumerate(common.METHODS, 1):
        for seed_index, seed in enumerate(common.SEEDS, 1):
            seed_dir = common.RUNS_ROOT / method / f"seed_{seed}"
            done_paths = sorted(seed_dir.glob("run_*/DONE.json")) if seed_dir.is_dir() else []
            done_states = [(path, validate_done(path, method, seed)) for path in done_paths]
            decision = resume_decision(legacy.get((method, seed), "MISSING"), done_states)
            print(f"[EXTERNAL BATCH] method={method} ({method_index}/4) seed={seed} ({seed_index}/8) completed={completed}/32 decision={decision}", flush=True)
            if decision == "SKIP_REUSABLE_LEGACY":
                continue
            if decision == "SKIP_VALID_DONE":
                completed += 1
                continue
            if decision.startswith("REFUSE"):
                raise common.AuditError(f"{method} seed {seed}: {decision}; no file was overwritten")
            attempt = seed_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            attempt.mkdir(parents=True, exist_ok=False)
            command = command_for(method, seed, args.device, attempt)
            new_run_index = (method_index - 1) * len(common.NEW_SEEDS) + common.NEW_SEEDS.index(seed) + 1
            progress = ProgressDisplay(
                attempt, method, seed, method_index, seed_index,
                new_run_index, completed, completed_new_durations,
            )
            try:
                return_code, tail, elapsed = run_child(command, attempt, progress)
                if return_code != 0:
                    write_failed(attempt, method, seed, command, return_code, tail)
                    print(f"[EXTERNAL BATCH] FAILED method={method} seed={seed} return_code={return_code}", flush=True)
                    continue
                validated = validate_result(attempt, method, seed)
                write_done(attempt, method, seed, validated)
                completed += 1
                completed_new_durations.append(elapsed)
                test = validated["test"]
                print(
                    f"[DONE] method={method} seed={seed} Success={test.get('success_rate')} "
                    f"SPL={test.get('spl')} PathEfficiency={test.get('path_efficiency')} "
                    f"AvgSteps={test.get('average_steps')} "
                    f"new_completed={completed - len(common.LEGACY_SEEDS) * len(common.METHODS)}/{TOTAL_NEW_RUNS} "
                    f"total_completed={completed}/{TOTAL_RECORDS} elapsed={_duration(elapsed)}",
                    flush=True,
                )
            except KeyboardInterrupt:
                write_failed(attempt, method, seed, command, 130, "KeyboardInterrupt")
                raise
            except Exception as exc:
                write_failed(attempt, method, seed, command, -1, traceback.format_exc())
                print(f"[EXTERNAL BATCH] FAILED method={method} seed={seed}: {exc}", flush=True)
    if completed != TOTAL_RECORDS:
        print(
            f"[AGGREGATE] skipped records={completed}/{TOTAL_RECORDS} paired_seeds=INCOMPLETE "
            "statistics=INCOMPLETE_PAIRED_DATA; rerun the same batch command to resume",
            flush=True,
        )
        return
    from aggregate_and_analyze import AGGREGATE_ROOT, aggregate
    aggregate()
    statistics_payload = common.read_json(AGGREGATE_ROOT / "external_baseline_statistics.json")
    print(
        f"[AGGREGATE] records={TOTAL_RECORDS}/{TOTAL_RECORDS} paired_seeds=8/8 "
        f"statistics={statistics_payload.get('status')}",
        flush=True,
    )
    summary_path = AGGREGATE_ROOT / "external_baseline_summary.csv"
    with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for method in (*common.METHODS, "FULL"):
        values = {
            row["metric"]: row for row in rows
            if row.get("method") == method
            and row.get("metric") in {"success_rate", "spl", "path_efficiency", "average_steps"}
        }
        rendered = " ".join(
            f"{metric}={values[metric]['mean']}+-{values[metric]['sample_sd']}"
            for metric in ("success_rate", "spl", "path_efficiency", "average_steps")
            if metric in values
        )
        print(f"[SUMMARY] method={method} n=8 {rendered}", flush=True)


if __name__ == "__main__":
    main()
