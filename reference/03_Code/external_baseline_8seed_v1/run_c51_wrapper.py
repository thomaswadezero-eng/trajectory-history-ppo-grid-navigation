#!/usr/bin/env python
"""Thin C51 adapter; only seed authorization and output routing are overlaid."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import common


def load_legacy_module():
    root = common.PROJECT_ROOT / "03_Code" / "comparison_baselines_c51"
    sys.path.insert(0, str(root))
    path = root / "run_c51_barn.py"
    spec = importlib.util.spec_from_file_location("frozen_c51_barn", path)
    if spec is None or spec.loader is None:
        raise common.AuditError("cannot load frozen C51 authority")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def formal(seed: int, device_name: str, attempt_dir: Path) -> Path:
    if seed not in common.NEW_SEEDS:
        raise common.AuditError("C51 wrapper accepts only the five authorized extension seeds")
    problems = __import__("audit_existing_runs")._verify_current_authority("C51")
    if problems:
        raise common.AuditError("legacy C51 authority drift: " + "; ".join(problems))
    attempt = common.require_attempt_path(attempt_dir, "C51", seed)
    legacy = load_legacy_module()
    original_load_config = legacy.load_config

    def load_overlay(path: Path):
        frozen, digest = original_load_config(path)
        overlay = copy.deepcopy(frozen)
        overlay["profiles"]["formal"]["allowed_seeds"] = list(common.SEEDS)
        overlay["output_root"] = str(common.RUNS_ROOT)
        return overlay, digest

    def create_exact(_root: Path, actual_seed: int) -> Path:
        if actual_seed != seed:
            raise common.AuditError("C51 legacy runner requested an unexpected seed")
        (attempt / "models").mkdir(exist_ok=False)
        (attempt / "validation").mkdir(exist_ok=False)
        (attempt / "protocol_snapshot").mkdir(exist_ok=False)
        return attempt

    legacy.load_config = load_overlay
    legacy._create_run_directory = create_exact
    device = legacy.resolve_device(device_name)
    run = legacy.train(legacy.DEFAULT_CONFIG, "formal", seed, device)
    legacy.formal_test(legacy.DEFAULT_CONFIG, run, device)
    provenance = {
        "schema_version": "external_baseline_8seed_extension_provenance.v1",
        "method": "C51", "seed": seed,
        "delegation": "frozen run_c51_barn.train plus frozen run_c51_barn.formal_test",
        "legacy_program": str(common.AUTHORITY["C51"]["program"]),
        "legacy_program_sha256": common.AUTHORITY["C51"]["program_sha256"],
        "legacy_algorithm_sha256": common.AUTHORITY["C51"]["algorithm_sha256"],
        "legacy_replay_sha256": common.AUTHORITY["C51"]["replay_sha256"],
        "legacy_protocol_sha256": common.AUTHORITY["C51"]["protocol_sha256"],
        "legacy_config_sha256": common.AUTHORITY["C51"]["config_sha256"],
        "allowed_overlay": {"profiles.formal.allowed_seeds": list(common.SEEDS), "output_root": str(common.RUNS_ROOT)},
        "algorithm_or_hyperparameter_changes": [],
    }
    common.write_json_atomic(run / "extension_provenance.json", provenance)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen C51 delegation wrapper")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_formal:
        raise common.AuditError("formal execution requires --confirm-formal")
    run = formal(args.seed, args.device, args.attempt_dir)
    print(json.dumps({"status": "COMPLETED", "run": str(run)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
