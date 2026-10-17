#!/usr/bin/env python
"""Strictly serial MovingAI64 method-by-seed batch launcher with resume."""

from __future__ import print_function

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from protocol_utils import ALLOWED_METHODS, RESULT_ROOT, REVISION_DIR, IntegrityError, ProtocolError, map_hash_aggregate, read_json, sha256_file, validate_confirmed


def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def validate_done(path, identity, method, seed):
    done = read_json(path)
    expected = {
        "status": "DONE",
        "method": method,
        "seed": seed,
        "config_sha256": identity["config_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "training_interaction_budget": 900000,
        "executed_training_interactions": 900000,
        "validation_astar_runtime_calls": 0,
        "test_astar_runtime_calls": 0,
    }
    for key, value in expected.items():
        if done.get(key) != value:
            raise IntegrityError("invalid DONE field %s: %s" % (key, path))
    expected_inputs = {name: record["sha256"] for name, record in identity["inputs"].items()}
    if done.get("input_sha256") != expected_inputs:
        raise IntegrityError("invalid DONE input identity: %s" % path)
    if done.get("map_hash_aggregate") != map_hash_aggregate(identity["config"]):
        raise IntegrityError("invalid DONE processed-map identity: %s" % path)
    for path_key, hash_key in (
        ("selected_checkpoint_path", "selected_checkpoint_sha256"),
        ("formal_test_result_path", "formal_test_result_sha256"),
        ("episode_csv_path", "episode_csv_sha256"),
    ):
        artifact = Path(str(done[path_key])).resolve()
        if not artifact.is_file() or sha256_file(artifact) != str(done[hash_key]).upper():
            raise IntegrityError("DONE artifact mismatch: %s" % artifact)
    return done


def completed_run(identity, method, seed):
    seed_root = RESULT_ROOT / method / ("seed_%d" % seed)
    done_files = sorted(seed_root.glob("run_*/DONE.json")) if seed_root.is_dir() else []
    if len(done_files) > 1:
        raise IntegrityError("multiple DONE ambiguity: %s/seed_%d" % (method, seed))
    if not done_files:
        return None
    validate_done(done_files[0], identity, method, seed)
    return done_files[0]


def latest_interaction(run_dir):
    if run_dir is None:
        return 0
    path = run_dir / "reward_components.csv"
    if not path.is_file() or path.stat().st_size < 10:
        return 0
    with path.open("rb") as handle:
        size = handle.seek(0, os.SEEK_END)
        handle.seek(max(0, size - 16384), os.SEEK_SET)
        text = handle.read().decode("utf-8-sig", errors="ignore")
    lines = [line for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        fields = line.split(",")
        if len(fields) > 2 and fields[2].strip().isdigit():
            return int(fields[2].strip())
    return 0


def latest_evaluation_event(run_dir):
    if run_dir is None:
        return None
    path = run_dir / "evaluation_metrics.csv"
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeDecodeError):
        return None
    complete = [row for row in rows if row.get("record_type") in ("episode", "summary")]
    if not complete:
        return None
    row = complete[-1]
    if row.get("split") == "validation" and row.get("record_type") == "summary":
        return ("validation", row.get("checkpoint_interaction"), row.get("success_rate"), row.get("SPL"), row.get("path_efficiency"))
    if row.get("split") == "test" and row.get("record_type") == "episode":
        return ("test", row.get("episode"))
    return None


def newest_run(method, seed, before):
    root = RESULT_ROOT / method / ("seed_%d" % seed)
    if not root.is_dir():
        return None
    candidates = [path for path in root.glob("run_*") if path.is_dir() and path not in before]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def verify_preflight(identity):
    marker_path = RESULT_ROOT / "preflight" / "PREFLIGHT_PASS.json"
    if not marker_path.is_file():
        raise ProtocolError("RUNTIME_VERIFICATION_PENDING_USER: missing %s" % marker_path)
    marker = read_json(marker_path)
    if marker.get("status") != "PASS" or marker.get("config_sha256") != identity["config_sha256"] or marker.get("protocol_sha256") != identity["protocol_sha256"]:
        raise IntegrityError("preflight marker identity mismatch")
    for relative, expected in marker.get("code_sha256", {}).items():
        path = REVISION_DIR / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise IntegrityError("preflight code identity is stale: %s" % path)
    if marker.get("input_sha256") != {name: value["sha256"] for name, value in identity["inputs"].items()}:
        raise IntegrityError("preflight task identity is stale")
    if marker.get("map_hash_aggregate") != map_hash_aggregate(identity["config"]):
        raise IntegrityError("preflight map identity is stale")


def run_child(config_path, protocol_path, method, seed, device, method_position, seed_position, position, total, completed_before, remaining_after):
    seed_root = RESULT_ROOT / method / ("seed_%d" % seed)
    before = set(seed_root.glob("run_*")) if seed_root.is_dir() else set()
    command = [
        sys.executable, "-u", str(REVISION_DIR / "run_movingai64_wrapper.py"),
        "--config", str(config_path), "--protocol", str(protocol_path),
        "--method", method, "--seed", str(seed), "--device", device, "--confirm-formal-child",
    ]
    print("[MOVINGAI64 BATCH] method=%s method_position=%d/3 seed=%d seed_position=%d run=%d/%d completed=%d batch_ETA=unknown" % (method, method_position, seed, seed_position, position, total, completed_before), flush=True)
    started = time.time()
    process = subprocess.Popen(command)
    run_dir = None
    last_interaction = -1
    last_evaluation = None
    while process.poll() is None:
        if run_dir is None:
            run_dir = newest_run(method, seed, before)
        interaction = latest_interaction(run_dir)
        if interaction != last_interaction:
            elapsed = time.time() - started
            eta = elapsed * (900000 - interaction) / interaction if interaction > 0 else None
            projected_run = elapsed * 900000.0 / interaction if interaction > 0 else None
            batch_eta = eta + projected_run * remaining_after if eta is not None else None
            print("[TRAIN] method=%s seed=%d interaction=%d/900000 progress=%.2f%% elapsed=%s ETA=%s batch_ETA=%s" % (method, seed, interaction, 100.0 * interaction / 900000.0, format_duration(elapsed), format_duration(eta), format_duration(batch_eta)), flush=True)
            last_interaction = interaction
        evaluation = latest_evaluation_event(run_dir)
        if evaluation is not None and evaluation != last_evaluation:
            if evaluation[0] == "validation":
                print("[VALIDATION] interaction=%s success=%s SPL=%s PathEfficiency=%s best_checkpoint=read_from_frozen_run_after_selection" % evaluation[1:], flush=True)
            else:
                print("[TEST] episodes=%s/in_progress" % evaluation[1], flush=True)
            last_evaluation = evaluation
        time.sleep(5.0)
    return_code = process.wait()
    if return_code != 0:
        print("[FAILED] method=%s seed=%d return_code=%d; attempt preserved" % (method, seed, return_code), flush=True)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Serial MovingAI64 PPO/PPO+Reward/FULL formal batch")
    parser.add_argument("--config", type=Path, default=REVISION_DIR / "movingai64_config_confirmed.json")
    parser.add_argument("--protocol", type=Path, default=REVISION_DIR / "movingai64_protocol_confirmed.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--confirm-formal", action="store_true")
    args = parser.parse_args()
    if not args.confirm_formal:
        raise ProtocolError("--confirm-formal is required")
    identity = validate_confirmed(args.config, args.protocol)
    verify_preflight(identity)
    seeds = list(identity["config"]["official_seeds"])
    matrix = [(method, seed) for method in ALLOWED_METHODS for seed in seeds]
    completed = 0
    pending = []
    for method, seed in matrix:
        done = completed_run(identity, method, seed)
        if done is not None:
            completed += 1
            print("[SKIP] valid DONE method=%s seed=%d" % (method, seed), flush=True)
        else:
            pending.append((method, seed))
    failures = []
    batch_started = time.time()
    initial_completed = completed
    for index, (method, seed) in enumerate(pending, 1):
        position = initial_completed + index
        ok = run_child(args.config, args.protocol, method, seed, args.device, ALLOWED_METHODS.index(method) + 1, seeds.index(seed) + 1, position, len(matrix), completed, len(pending) - index)
        if not ok:
            failures.append((method, seed))
            continue
        done_path = completed_run(identity, method, seed)
        if done_path is None:
            raise IntegrityError("child exited successfully without a valid DONE")
        done = read_json(done_path)
        completed += 1
        metrics = done["final_metrics"]
        print("[DONE] method=%s seed=%d Success=%.6f SPL=%.6f PathEfficiency=%.6f AvgSteps=%.6f completed=%d/%d" % (method, seed, float(metrics["success_rate"]), float(metrics["spl"]), float(metrics["path_efficiency"]), float(metrics["average_steps"]), completed, len(matrix)), flush=True)
    if failures:
        print("[BATCH INCOMPLETE] failures=%s; rerun the same command for batch-level resume" % failures, flush=True)
        raise SystemExit(2)
    aggregate_command = [sys.executable, "-u", str(REVISION_DIR / "aggregate_and_analyze.py"), "--config", str(args.config), "--protocol", str(args.protocol), "--confirm-aggregate"]
    subprocess.run(aggregate_command, check=True)
    print("[MOVINGAI64 BATCH COMPLETE] runs=%d elapsed=%s" % (len(matrix), format_duration(time.time() - batch_started)), flush=True)


if __name__ == "__main__":
    main()
