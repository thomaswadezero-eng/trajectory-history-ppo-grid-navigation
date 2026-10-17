#!/usr/bin/env python
"""Single explicit entry point for the five missing MovingAI64 seed runs.

This file is a strict serial dispatcher.  It never implements an environment,
model, reward, PPO update, validation routine, or test routine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from extension_utils import (
    CONFIG_PATH,
    LEGACY_WRAPPER_PATH,
    METHODS,
    NEW_RESULT_ROOT,
    NEW_SEEDS,
    PROTOCOL_PATH,
    IntegrityError,
    ProtocolError,
    resolve_new_done,
    validate_extension_identity,
    validate_legacy_runs,
    verify_task_protocol,
    write_json_replace,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    value = max(0, int(seconds))
    return "%02d:%02d:%02d" % (value // 3600, (value % 3600) // 60, value % 60)


def assert_pristine_new_root() -> None:
    if not NEW_RESULT_ROOT.exists():
        return
    entries = list(NEW_RESULT_ROOT.iterdir())
    if entries:
        raise IntegrityError(
            "new output root is not pristine; automatic resume, retry, and skip are disabled. "
            "Preserve the directory and obtain explicit user authorization for a recovery plan: %s"
            % NEW_RESULT_ROOT
        )


def run_one_child(method: str, seed: int, device: str, log_path: Path) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-u",
        str(LEGACY_WRAPPER_PATH),
        "--config",
        str(CONFIG_PATH),
        "--protocol",
        str(PROTOCOL_PATH),
        "--method",
        method,
        "--seed",
        str(seed),
        "--device",
        device,
        "--confirm-formal-child",
    ]
    started_at = utc_now()
    started = time.monotonic()
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        log.write("command=" + json.dumps(command, ensure_ascii=False) + "\n")
        log.write("started_at_utc=" + started_at + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("child stdout pipe was not created")
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        ended_at = utc_now()
        elapsed = time.monotonic() - started
        log.write("ended_at_utc=" + ended_at + "\n")
        log.write("exit_code=%d\n" % return_code)
        log.write("elapsed_seconds=%.6f\n" % elapsed)
    return {
        "method": method,
        "seed": seed,
        "command": command,
        "started_at_utc": started_at,
        "ended_at_utc": ended_at,
        "elapsed_seconds": elapsed,
        "exit_code": return_code,
        "log_path": str(log_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strict serial launcher for the five missing seeds in the MovingAI64 eight-seed extension"
    )
    parser.add_argument("--run", action="store_true", help="explicitly authorize the formal serial batch")
    parser.add_argument("--device", choices=("cuda",), required=True, help="formal execution is CUDA-only")
    args = parser.parse_args()
    if not args.run:
        raise ProtocolError("--run is required; importing or invoking without it cannot start formal work")

    identity = validate_extension_identity()
    validate_legacy_runs(identity)
    task_audit = verify_task_protocol(identity["config"])
    assert_pristine_new_root()

    NEW_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    control_dir = NEW_RESULT_ROOT / "batch_control"
    control_dir.mkdir()
    batch_id = "batch_%s" % datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_dir = control_dir / batch_id
    batch_dir.mkdir()
    state_path = batch_dir / "batch_state.json"
    matrix = [(method, seed) for method in METHODS for seed in NEW_SEEDS]
    state: Dict[str, Any] = {
        "schema_version": "movingai_8seed_extension_batch.v1",
        "status": "RUNNING",
        "formal_experiment_status_before_launch": "NOT_STARTED",
        "started_at_utc": utc_now(),
        "device": args.device,
        "serial_execution": True,
        "maximum_concurrent_children": 1,
        "automatic_retry": False,
        "skip_failed_runs": False,
        "resume": False,
        "methods": list(METHODS),
        "new_seeds": list(NEW_SEEDS),
        "planned_new_run_count": len(matrix),
        "legacy_completed_run_count": 9,
        "task_audit": task_audit,
        "runs": [],
    }
    write_json_replace(state_path, state)
    batch_started = time.monotonic()

    try:
        for position, (method, seed) in enumerate(matrix, 1):
            print(
                "[MOVINGAI 8-SEED] starting %d/%d method=%s seed=%d; strict serial; no retry"
                % (position, len(matrix), method, seed),
                flush=True,
            )
            log_path = batch_dir / ("%02d_%s_seed%d.log" % (position, method, seed))
            run_record = run_one_child(method, seed, args.device, log_path)
            state["runs"].append(run_record)
            write_json_replace(state_path, state)
            if run_record["exit_code"] != 0:
                state["status"] = "FAILED_STOPPED_IMMEDIATELY"
                state["ended_at_utc"] = utc_now()
                state["failed_method"] = method
                state["failed_seed"] = seed
                write_json_replace(state_path, state)
                raise SystemExit(run_record["exit_code"] or 2)
            resolve_new_done(identity, method, seed)
            print(
                "[MOVINGAI 8-SEED] completed %d/%d method=%s seed=%d elapsed=%s"
                % (position, len(matrix), method, seed, format_duration(run_record["elapsed_seconds"])),
                flush=True,
            )

        aggregate_command = [
            sys.executable,
            "-u",
            str(Path(__file__).resolve().parent / "aggregate_movingai_8seed_results.py"),
            "--aggregate",
        ]
        aggregate_started = utc_now()
        aggregate_process = subprocess.run(aggregate_command, check=False)
        state["aggregate"] = {
            "command": aggregate_command,
            "started_at_utc": aggregate_started,
            "ended_at_utc": utc_now(),
            "exit_code": aggregate_process.returncode,
        }
        if aggregate_process.returncode != 0:
            state["status"] = "AGGREGATION_FAILED"
            state["ended_at_utc"] = utc_now()
            write_json_replace(state_path, state)
            raise SystemExit(aggregate_process.returncode or 3)
        state["status"] = "COMPLETE_24_OF_24_AND_AGGREGATED"
        state["ended_at_utc"] = utc_now()
        state["elapsed_seconds"] = time.monotonic() - batch_started
        write_json_replace(state_path, state)
        print(
            "[MOVINGAI 8-SEED COMPLETE] new_runs=15 total_matrix=24 elapsed=%s"
            % format_duration(state["elapsed_seconds"]),
            flush=True,
        )
    except BaseException:
        if state.get("status") == "RUNNING":
            state["status"] = "INTERRUPTED_OR_FAILED_STOPPED_IMMEDIATELY"
            state["ended_at_utc"] = utc_now()
            write_json_replace(state_path, state)
        raise


if __name__ == "__main__":
    main()

