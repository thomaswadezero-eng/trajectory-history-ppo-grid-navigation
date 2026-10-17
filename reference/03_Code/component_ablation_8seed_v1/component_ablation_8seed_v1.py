#!/usr/bin/env python
"""Eight-seed extension for the four V7 Dynamic Reward leave-one-out runs.

The training implementation is deliberately delegated to the frozen
``internal_ablation_v1.py`` program.  This module verifies that dependency by
SHA-256, supplies only the authorized seed/output/identity overlay, and makes
the two A*-loss configurations unreachable.  Importing this module never
starts training.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Mapping


SCRIPT_PATH = Path(__file__).resolve()
SOURCE_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SOURCE_DIR.parent.parent
INTERNAL_ROOT = PROJECT_ROOT / "03_Code" / "internal_ablation_v1"
V7_ROOT = PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7"
INTERNAL_PROGRAM = INTERNAL_ROOT / "internal_ablation_v1.py"
V7_PROGRAM = V7_ROOT / "final_two_factor_experiment_v7.py"
DEFAULT_CONFIG = SOURCE_DIR / "experiment_config.json"
DEFAULT_PROTOCOL = SOURCE_DIR / "experiment_protocol.json"

EXPECTED_INTERNAL_PROGRAM_SHA256 = "A1FA5959884B782D356B28C359102DBC57D4FA8B32A3D714A0CEBC6C9BA4ADE4"
EXPECTED_INTERNAL_CONFIG_FILE_SHA256 = "D68ABAB4D57C2319D43116C2F5E63A31B223E3E4B49D09355E116F400936A898"
EXPECTED_INTERNAL_CONFIG_SELF_SHA256 = "0BD21F4412873AE81CA8263ED8E8CDA7A697620A114EF4078521D45C70C5F4CB"
EXPECTED_INTERNAL_PROTOCOL_FILE_SHA256 = "2843E7E5D42B73B256F017F6471EE8C44EF5BDD5C390F25CBA0854DFF00C9446"
EXPECTED_INTERNAL_PROTOCOL_SELF_SHA256 = "6E8F29D13320355477DB7E6A957DDBFFCE193F1FC1280102BDC9F13BD3404684"
EXPECTED_V7_PROGRAM_SHA256 = "96093A6B5F024F5D9505ACCEBEAE7592700228BA17587BF0590092A241B7544D"

SEEDS = [42, 1004, 6010, 68882, 19816, 74494, 81399, 31524]
REWARD_COMPONENTS = ("progress", "revisit", "oscillation", "stagnation")
CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    "REWARD_WO_PROGRESS": {
        "reward_components": {"progress": False, "revisit": True, "oscillation": True, "stagnation": True},
        "astar_supervision": False,
        "distillation_loss": False,
        "planning_loss": False,
    },
    "REWARD_WO_REVISIT": {
        "reward_components": {"progress": True, "revisit": False, "oscillation": True, "stagnation": True},
        "astar_supervision": False,
        "distillation_loss": False,
        "planning_loss": False,
    },
    "REWARD_WO_OSCILLATION": {
        "reward_components": {"progress": True, "revisit": True, "oscillation": False, "stagnation": True},
        "astar_supervision": False,
        "distillation_loss": False,
        "planning_loss": False,
    },
    "REWARD_WO_STAGNATION": {
        "reward_components": {"progress": True, "revisit": True, "oscillation": True, "stagnation": False},
        "astar_supervision": False,
        "distillation_loss": False,
        "planning_loss": False,
    },
}


class ProtocolError(RuntimeError):
    pass


class IntegrityError(RuntimeError):
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
        raise ProtocolError(f"expected JSON object: {path}")
    return payload


def load_internal_module() -> Any:
    if sha256_file(INTERNAL_PROGRAM) != EXPECTED_INTERNAL_PROGRAM_SHA256:
        raise IntegrityError("frozen internal-ablation parent program SHA mismatch")
    spec = importlib.util.spec_from_file_location("component_ablation_frozen_parent", INTERNAL_PROGRAM)
    if spec is None or spec.loader is None:
        raise IntegrityError(f"cannot import frozen parent: {INTERNAL_PROGRAM}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _replace_identity_fields(parent: Dict[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    data = copy.deepcopy(parent)
    data["schema_version"] = str(overlay["schema_version"])
    data["status"] = "FROZEN_AND_AUTHORIZED"
    data["formal_execution_allowed"] = True
    data["authorized_program"] = str(overlay["authorized_program"])
    data["authorized_program_sha256"] = str(overlay["authorized_program_sha256"])
    data["official_seeds"] = list(SEEDS)
    data["output_root"] = str(overlay["output_root"])
    data["configurations"] = copy.deepcopy(CONFIGURATIONS)
    data["revision_identity"] = copy.deepcopy(dict(overlay["revision_identity"]))
    return data


class RevisionConfig:
    def __init__(self, path: Path, core: Any):
        self.path = path.resolve()
        self.sha256 = core.verified_self_hash(self.path)
        self.overlay = read_json(self.path)
        parent_path = INTERNAL_ROOT / "experiment_config.json"
        if sha256_file(parent_path) != EXPECTED_INTERNAL_CONFIG_FILE_SHA256:
            raise IntegrityError("frozen internal config physical SHA mismatch")
        parent = core.FrozenConfig(parent_path)
        if parent.sha256 != EXPECTED_INTERNAL_CONFIG_SELF_SHA256:
            raise IntegrityError("frozen internal config self-hash mismatch")
        self.data = _replace_identity_fields(parent.data, self.overlay)

    def get(self, dotted: str) -> Any:
        value: Any = self.data
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ProtocolError(f"missing configuration field: {dotted}")
            value = value[part]
        return value

    def integer(self, dotted: str) -> int:
        value = self.get(dotted)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProtocolError(f"expected integer field: {dotted}")
        return int(value)

    def number(self, dotted: str) -> float:
        value = self.get(dotted)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ProtocolError(f"expected numeric field: {dotted}")
        return float(value)


class RevisionProtocol:
    def __init__(self, path: Path, core: Any):
        self.path = path.resolve()
        self.sha256 = core.verified_protocol_hash(self.path)
        self.overlay = read_json(self.path)
        parent_path = INTERNAL_ROOT / "experiment_protocol.json"
        if sha256_file(parent_path) != EXPECTED_INTERNAL_PROTOCOL_FILE_SHA256:
            raise IntegrityError("frozen internal protocol physical SHA mismatch")
        parent = core.FrozenProtocol(parent_path)
        if parent.sha256 != EXPECTED_INTERNAL_PROTOCOL_SELF_SHA256:
            raise IntegrityError("frozen internal protocol self-hash mismatch")
        self.data = copy.deepcopy(parent.data)
        self.data["schema_version"] = str(self.overlay["schema_version"])
        self.data["status"] = "FROZEN_AND_AUTHORIZED"
        self.data["formal_execution_allowed"] = True
        self.data["authorized_program"] = str(self.overlay["authorized_program"])
        self.data["authorized_program_sha256"] = str(self.overlay["authorized_program_sha256"])
        self.data["official_seeds"] = list(SEEDS)
        self.data["configurations"] = copy.deepcopy(CONFIGURATIONS)
        self.data["revision_identity"] = copy.deepcopy(dict(self.overlay["revision_identity"]))

    def get(self, dotted: str) -> Any:
        value: Any = self.data
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ProtocolError(f"missing protocol field: {dotted}")
            value = value[part]
        return value


def validate_config(config: RevisionConfig, protocol: RevisionProtocol) -> None:
    expected_program = "03_Code/component_ablation_8seed_v1/component_ablation_8seed_v1.py"
    current_program_sha = sha256_file(SCRIPT_PATH)
    if config.overlay.get("schema_version") != "component_ablation_8seed.v1":
        raise ProtocolError("unsupported component-ablation config schema")
    if protocol.overlay.get("schema_version") != "component_ablation_8seed.v1.protocol":
        raise ProtocolError("unsupported component-ablation protocol schema")
    for payload, label in ((config.overlay, "config"), (protocol.overlay, "protocol")):
        if payload.get("status") != "FROZEN_AND_AUTHORIZED" or payload.get("formal_execution_allowed") is not True:
            raise ProtocolError(f"{label} is not frozen and authorized")
        if payload.get("authorized_program") != expected_program:
            raise ProtocolError(f"{label} authorized program path mismatch")
        if str(payload.get("authorized_program_sha256", "")).upper() != current_program_sha:
            raise ProtocolError(f"{label} authorized program SHA mismatch")
        if list(payload.get("official_seeds", [])) != SEEDS:
            raise ProtocolError(f"{label} official seed list mismatch")
        if payload.get("configurations") != CONFIGURATIONS:
            raise ProtocolError(f"{label} configuration matrix mismatch")
    if config.overlay.get("output_root") != r"redacted_local_root\revision_component_ablation_8seed_v1":
        raise ProtocolError("independent revision output root mismatch")
    if config.integer("training.interaction_budget") != 900000:
        raise ProtocolError("training interaction budget drift")
    if config.integer("training.validation_interval_interactions") != 100000:
        raise ProtocolError("validation interval drift")
    if any(record["astar_supervision"] for record in CONFIGURATIONS.values()):
        raise ProtocolError("A* must be disabled for every component ablation")
    if any(record["distillation_loss"] or record["planning_loss"] for record in CONFIGURATIONS.values()):
        raise ProtocolError("A* auxiliary losses must be disabled")
    for name, record in CONFIGURATIONS.items():
        disabled = [key for key, enabled in record["reward_components"].items() if not enabled]
        if len(disabled) != 1 or name != f"REWARD_WO_{disabled[0].upper()}":
            raise ProtocolError(f"not exactly one correctly named disabled component: {name}")
    if sha256_file(V7_PROGRAM) != EXPECTED_V7_PROGRAM_SHA256:
        raise IntegrityError("frozen V7 authority SHA mismatch")


def configure_frozen_core(core: Any) -> None:
    core.SCRIPT_PATH = SCRIPT_PATH
    core.PROJECT_ROOT = PROJECT_ROOT
    core.SOURCE_DIR = INTERNAL_ROOT
    core.CONFIGURATIONS = copy.deepcopy(CONFIGURATIONS)

    def ensure_archive(config: RevisionConfig, protocol: RevisionProtocol, root: Path) -> None:
        archive = root / "protocol"
        archive.mkdir(parents=True, exist_ok=True)
        sources = {
            "component_ablation_8seed_v1.py": SCRIPT_PATH,
            "experiment_config.json": config.path,
            "experiment_protocol.json": protocol.path,
            "parent_internal_ablation_v1.py": INTERNAL_PROGRAM,
            "parent_internal_experiment_config.json": INTERNAL_ROOT / "experiment_config.json",
            "parent_internal_experiment_protocol.json": INTERNAL_ROOT / "experiment_protocol.json",
            "frozen_v7_authority.py": V7_PROGRAM,
            "map_split.json": INTERNAL_ROOT / "map_split.json",
            "task_split.json": INTERNAL_ROOT / "task_split.json",
            "hypothesis_plan.json": SOURCE_DIR / "component_hypothesis_plan_v1.json",
            "hash_manifest.json": SOURCE_DIR / "hash_manifest.json",
        }
        for name, source in sources.items():
            destination = archive / name
            if destination.exists():
                if sha256_file(destination) != sha256_file(source):
                    raise IntegrityError(f"protocol archive conflict: {destination}")
            else:
                shutil.copy2(source, destination)
        (root / "analysis_ready").mkdir(parents=True, exist_ok=True)

    core.ensure_archive = ensure_archive


def prepare(config_path: Path = DEFAULT_CONFIG, protocol_path: Path = DEFAULT_PROTOCOL) -> tuple[Any, RevisionConfig, RevisionProtocol]:
    core = load_internal_module()
    config = RevisionConfig(config_path, core)
    protocol = RevisionProtocol(protocol_path, core)
    validate_config(config, protocol)
    configure_frozen_core(core)
    return core, config, protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V7-equivalent 8-seed Dynamic Reward component leave-one-out extension")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--method", choices=tuple(CONFIGURATIONS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    core, config, protocol = prepare(args.config, args.protocol)
    input_hashes = core.verify_inputs(config)
    core.load_tasks(config)
    if args.audit_only:
        print(json.dumps({
            "status": "STATIC_IDENTITY_PASS",
            "program_sha256": sha256_file(SCRIPT_PATH),
            "config_sha256": config.sha256,
            "protocol_sha256": protocol.sha256,
            "parent_internal_program_sha256": sha256_file(INTERNAL_PROGRAM),
            "v7_program_sha256": sha256_file(V7_PROGRAM),
            "input_sha256": input_hashes,
            "official_seeds": SEEDS,
            "methods": list(CONFIGURATIONS),
            "runtime_preflight": "RUNTIME_PREFLIGHT_PENDING",
        }, indent=2))
        return
    if not args.confirm_formal:
        raise ProtocolError("refusing to start formal training without --confirm-formal")
    if args.method is None or args.seed is None or args.device is None:
        raise ProtocolError("--method, --seed, and --device are required")
    if args.seed not in SEEDS:
        raise ProtocolError("unauthorized seed")
    if args.device == "cuda" and not core.torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable")
    run = core.run_experiment(config, protocol, args.method, args.seed, core.torch.device(args.device))
    print(run, flush=True)


if __name__ == "__main__":
    main()
