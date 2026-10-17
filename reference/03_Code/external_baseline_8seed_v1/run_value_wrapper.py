#!/usr/bin/env python
"""Thin DDQN/D3QN adapter delegating the frozen legacy runner unchanged."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

import common as revision_common


def load_legacy():
    root = revision_common.PROJECT_ROOT / "03_Code" / "comparison_baselines_v1"
    revision_dir = str(Path(__file__).resolve().parent)
    sys.modules.pop("common", None)
    sys.path = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != revision_dir]
    sys.path.insert(0, str(root))
    protocol = importlib.import_module("common.protocol")
    runner = importlib.import_module("runner")
    return protocol, runner


def formal(method: str, seed: int, device_name: str, attempt_dir: Path) -> Path:
    if method not in ("DDQN", "D3QN") or seed not in revision_common.NEW_SEEDS:
        raise revision_common.AuditError("value wrapper accepts only the five authorized extension seeds")
    authority_problems = __import__("audit_existing_runs")._verify_current_authority(method)
    if authority_problems:
        raise revision_common.AuditError("legacy authority drift: " + "; ".join(authority_problems))
    attempt = revision_common.require_attempt_path(attempt_dir, method, seed)
    protocol, runner = load_legacy()
    bundle = protocol.load_bundle(method)
    overlay = copy.deepcopy(bundle.protocol)
    overlay["official_seeds"] = list(revision_common.SEEDS)
    overlay["output_root"] = str(revision_common.RUNS_ROOT)
    overlay["protected_output_roots"] = list(overlay["protected_output_roots"])
    extended = protocol.ProtocolBundle(
        protocol=overlay,
        config=bundle.config,
        protocol_sha256=bundle.protocol_sha256,
        config_sha256=bundle.config_sha256,
        framework_sha256=bundle.framework_sha256,
    )

    def create_exact(_root: Path, actual_method: str, actual_seed: int) -> Path:
        if actual_method != method or actual_seed != seed:
            raise revision_common.AuditError("legacy runner requested an unexpected method/seed")
        (attempt / "models").mkdir(exist_ok=False)
        return attempt

    runner._create_run_directory = create_exact
    run = runner.run_formal_experiment(extended, method, seed, runner.torch.device(device_name))
    provenance = {
        "schema_version": "external_baseline_8seed_extension_provenance.v1",
        "method": method,
        "seed": seed,
        "delegation": "frozen comparison_baselines_v1 runner.run_formal_experiment",
        "legacy_program": str(revision_common.AUTHORITY[method]["program"]),
        "legacy_program_sha256": revision_common.AUTHORITY[method]["program_sha256"],
        "legacy_framework_sha256": revision_common.AUTHORITY[method]["framework_sha256"],
        "legacy_config_sha256": revision_common.AUTHORITY[method]["config_sha256"],
        "legacy_protocol_sha256": revision_common.AUTHORITY[method]["protocol_sha256"],
        "allowed_overlay": {"official_seeds": list(revision_common.SEEDS), "output_root": str(revision_common.RUNS_ROOT)},
        "algorithm_or_hyperparameter_changes": [],
    }
    revision_common.write_json_atomic(run / "extension_provenance.json", provenance)
    return run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen DDQN/D3QN delegation wrapper")
    parser.add_argument("--method", choices=("DDQN", "D3QN"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_formal:
        raise revision_common.AuditError("formal execution requires --confirm-formal")
    run = formal(args.method, args.seed, args.device, args.attempt_dir)
    print(json.dumps({"status": "COMPLETED", "run": str(run)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
