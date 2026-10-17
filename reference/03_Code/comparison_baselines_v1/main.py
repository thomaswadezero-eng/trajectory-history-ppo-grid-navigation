from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common.errors import ProtocolError
from common.protocol import METHODS, OFFICIAL_SEEDS, load_and_verify_tasks, load_bundle, verify_v7_authority
from runner import run_formal_experiment


FORMAL_CONFIRMATION = "I_UNDERSTAND_THIS_STARTS_A_FORMAL_RUN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen DDQN/D3QN/A2C comparison framework aligned to Final Two-Factor Experiment V7"
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--audit-only", action="store_true", help="static hashes/config/split audit; no map load or environment interaction")
    parser.add_argument("--confirm-formal-run", default="", help="required safety phrase for formal execution")
    return parser.parse_args()


def static_audit() -> None:
    rows = {}
    for method in METHODS:
        bundle = load_bundle(method)
        v7_hashes = verify_v7_authority(bundle.protocol)
        _, split_audit = load_and_verify_tasks(bundle.protocol)
        rows[method] = {
            "config_sha256": bundle.config_sha256,
            "protocol_sha256": bundle.protocol_sha256,
            "framework_sha256": bundle.framework_sha256,
            "v7_authority_sha256": v7_hashes,
            "split_audit": split_audit,
            "formal_run_started": False,
            "environment_interactions": 0,
        }
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.audit_only:
        if args.confirm_formal_run:
            raise ProtocolError("--audit-only cannot be combined with formal-run confirmation")
        static_audit()
        return
    if args.method is None or args.seed is None or args.device is None:
        raise ProtocolError("--method, --seed, and --device are required for a formal run")
    if args.seed not in OFFICIAL_SEEDS:
        raise ProtocolError(f"seed must be one of {OFFICIAL_SEEDS}")
    if args.confirm_formal_run != FORMAL_CONFIRMATION:
        raise ProtocolError(
            "formal execution blocked: pass --confirm-formal-run " + FORMAL_CONFIRMATION
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable")
    bundle = load_bundle(args.method)
    run = run_formal_experiment(bundle, args.method, args.seed, torch.device(args.device))
    print(run)


if __name__ == "__main__":
    main()

