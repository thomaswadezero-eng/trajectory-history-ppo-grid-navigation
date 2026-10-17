#!/usr/bin/env python
"""Thin MaskablePPO adapter delegating the frozen SB3-Contrib implementation."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

# The frozen Python 3.10 distribution uses python310._pth isolation and does
# not add the executed script directory to sys.path.  Bootstrap the revision
# root explicitly before importing the revision-local ``common.py``.  Removing
# a preloaded generic module of the same name prevents ambient module capture.
REVISION_ROOT = Path(__file__).resolve().parent
sys.modules.pop("common", None)
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != REVISION_ROOT]
sys.path.insert(0, str(REVISION_ROOT))
import common

EXPECTED_COMMON = REVISION_ROOT / "common.py"
ACTUAL_COMMON = Path(common.__file__).resolve()
if ACTUAL_COMMON != EXPECTED_COMMON.resolve():
    raise RuntimeError(
        f"revision common identity mismatch: expected={EXPECTED_COMMON.resolve()} actual={ACTUAL_COMMON}"
    )


def bootstrap_check(result_path: Path) -> dict:
    """Verify only import/path/authority identities; never import the legacy runtime."""
    resolved = result_path.resolve()
    preflight_root = (common.OUTPUT_ROOT / "preflight").resolve()
    try:
        resolved.relative_to(preflight_root)
    except ValueError:
        raise common.AuditError(f"bootstrap result must stay inside preflight root: {resolved}")
    authority_path = Path(common.AUTHORITY["MaskablePPO"]["program"]).resolve()
    expected_authority = (
        common.PROJECT_ROOT / "03_Code" / "external_baselines" / "sb3_external_baseline.py"
    ).resolve()
    authority_sha = common.sha256_file(authority_path)
    expected_sha = str(common.AUTHORITY["MaskablePPO"]["program_sha256"]).upper()
    if authority_path != expected_authority or authority_sha != expected_sha:
        raise common.AuditError("frozen MaskablePPO authority identity mismatch")
    payload = {
        "schema_version": "external_maskable_wrapper_bootstrap.v1",
        "status": "PASS",
        "revision_root": str(REVISION_ROOT),
        "common_path": str(ACTUAL_COMMON),
        "common_sha256": common.sha256_file(ACTUAL_COMMON),
        "authority_path": str(authority_path),
        "authority_sha256": authority_sha,
        "legacy_module_imported": False,
        "torch_imported": False,
        "formal_training_interactions": 0,
        "formal_result_created": False,
    }
    common.write_json_atomic(resolved, payload)
    return payload


def load_legacy_module():
    root = common.PROJECT_ROOT / "03_Code" / "external_baselines"
    sys.path.insert(0, str(root))
    path = root / "sb3_external_baseline.py"
    spec = importlib.util.spec_from_file_location("frozen_maskable_ppo", path)
    if spec is None or spec.loader is None:
        raise common.AuditError("cannot load frozen MaskablePPO authority")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def formal(seed: int, device_name: str, attempt_dir: Path) -> Path:
    if seed not in common.NEW_SEEDS:
        raise common.AuditError("MaskablePPO wrapper accepts only the five authorized extension seeds")
    problems = __import__("audit_existing_runs")._verify_current_authority("MaskablePPO")
    if problems:
        raise common.AuditError("legacy MaskablePPO authority drift: " + "; ".join(problems))
    attempt = common.require_attempt_path(attempt_dir, "MaskablePPO", seed)
    legacy = load_legacy_module()
    frozen_config = legacy.load_config(legacy.DEFAULT_CONFIG)
    config = copy.deepcopy(frozen_config)
    config["official_seeds"] = list(common.SEEDS)
    config["output_root"] = str(common.RUNS_ROOT)
    original_code_files = tuple(legacy.code_files())

    def create_exact(_root: Path, actual_seed: int) -> Path:
        if actual_seed != seed:
            raise common.AuditError("MaskablePPO legacy runner requested an unexpected seed")
        (attempt / "models").mkdir(exist_ok=False)
        return attempt

    legacy.create_run_dir = create_exact
    legacy.code_files = lambda: list(original_code_files)
    run, test_tasks = legacy.train(seed, device_name, config, common.RUNS_ROOT)
    # The original formal-test firewall allows only EXTERNAL_ROOT/outputs.  Point
    # that path-only guard at the isolated revision root while retaining the
    # captured original source list used by the source-identity check.
    legacy.EXTERNAL_ROOT = common.OUTPUT_ROOT
    legacy.formal_test(run, test_tasks, device_name)
    provenance = {
        "schema_version": "external_baseline_8seed_extension_provenance.v1",
        "method": "MaskablePPO", "seed": seed,
        "delegation": "frozen sb3_external_baseline.train plus frozen sb3_external_baseline.formal_test",
        "legacy_program": str(common.AUTHORITY["MaskablePPO"]["program"]),
        "legacy_program_sha256": common.AUTHORITY["MaskablePPO"]["program_sha256"],
        "legacy_config_sha256": common.AUTHORITY["MaskablePPO"]["config_sha256"],
        "allowed_overlay": {"official_seeds": list(common.SEEDS), "output_root": str(common.RUNS_ROOT), "formal_test_path_guard_only": str(common.RUNS_ROOT)},
        "algorithm_or_hyperparameter_changes": [],
    }
    common.write_json_atomic(run / "extension_provenance.json", provenance)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen MaskablePPO delegation wrapper")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--attempt-dir", type=Path)
    parser.add_argument("--confirm-formal", action="store_true")
    parser.add_argument("--bootstrap-check", action="store_true")
    parser.add_argument("--bootstrap-result", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_check:
        if args.bootstrap_result is None or args.confirm_formal:
            raise common.AuditError("bootstrap check requires --bootstrap-result and forbids --confirm-formal")
        payload = bootstrap_check(args.bootstrap_result)
        print(json.dumps(payload, indent=2), flush=True)
        return
    if not args.confirm_formal:
        raise common.AuditError("formal execution requires --confirm-formal")
    if args.seed is None or args.device is None or args.attempt_dir is None:
        raise common.AuditError("formal execution requires seed, device, and attempt directory")
    run = formal(args.seed, args.device, args.attempt_dir)
    print(json.dumps({"status": "COMPLETED", "run": str(run)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
