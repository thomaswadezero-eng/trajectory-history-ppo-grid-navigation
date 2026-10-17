#!/usr/bin/env python
"""Formal BARN 30x30 PPO+Progress runtime-compatible entry point.

The frozen V7 program remains responsible for the network, PPO update,
training loop, checkpoint selection, evaluation, and output schema. The
already-audited internal-ablation environment supplies only scalar reward
component switches. This adapter preserves the frozen V7 constructor contract
``NavigationEnvironment(config, maps, reward_enabled)`` and translates the
boolean internally. Importing this module never starts training.
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
from typing import Any, Dict, Mapping, Optional


SCRIPT_PATH = Path(__file__).resolve()
SOURCE_DIR = SCRIPT_PATH.parent
PROJECT_ROOT = SOURCE_DIR.parent.parent
V7_ROOT = PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7"
INTERNAL_ROOT = PROJECT_ROOT / "03_Code" / "internal_ablation_v1"
V7_PROGRAM = V7_ROOT / "final_two_factor_experiment_v7.py"
V7_CONFIG = V7_ROOT / "experiment_config.json"
V7_PROTOCOL = V7_ROOT / "experiment_protocol.json"
INTERNAL_PROGRAM = INTERNAL_ROOT / "internal_ablation_v1.py"
DEFAULT_CONFIG = SOURCE_DIR / "experiment_config.json"
DEFAULT_PROTOCOL = SOURCE_DIR / "experiment_protocol.json"
HASH_MANIFEST = SOURCE_DIR / "hash_manifest.json"

EXPECTED_V7_PROGRAM_SHA256 = "96093A6B5F024F5D9505ACCEBEAE7592700228BA17587BF0590092A241B7544D"
EXPECTED_V7_CONFIG_FILE_SHA256 = "CF229DE128FF16E29B73308A7E9B2756DE51294671573F067306274EDA8D4166"
EXPECTED_V7_CONFIG_SELF_SHA256 = "489431F4C5609FC1F7148D56D152D22C1252F6B6AF79CE8AE09716E1F8D84B8D"
EXPECTED_V7_PROTOCOL_FILE_SHA256 = "2C8BAFE1A71695627B4F723583153067C0E1C7D08F905538E5BF8793A96B4C9B"
EXPECTED_V7_PROTOCOL_SELF_SHA256 = "7887F254E31A1C6A0CDB95994EF3C18CC3A8A43775D8D6BD16475FDFB042ADB7"
EXPECTED_INTERNAL_PROGRAM_SHA256 = "A1FA5959884B782D356B28C359102DBC57D4FA8B32A3D714A0CEBC6C9BA4ADE4"

METHOD_ID = "PPO_PROGRESS"
DISPLAY_NAME = "PPO+Progress"
SEEDS = [42, 1004, 6010, 68882, 19816, 74494, 81399, 31524]
OUTPUT_ROOT = Path(r"redacted_local_root\revision_progress_only_8seed_v2")
REWARD_COMPONENTS = {
    "progress": True,
    "revisit": False,
    "oscillation": False,
    "stagnation": False,
}
EVALUATION_REWARD_COMPONENTS = {
    "progress": False,
    "revisit": False,
    "oscillation": False,
    "stagnation": False,
}
CORE_CONFIGURATIONS = {METHOD_ID: (False, True)}
DECLARED_CONFIGURATION = {
    METHOD_ID: {
        "display_name": DISPLAY_NAME,
        "reward_components": REWARD_COMPONENTS,
        "astar_supervision": False,
        "distillation_loss": False,
        "planning_loss": False,
    }
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


def write_json_exclusive(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_module(name: str, path: Path, expected_sha256: str) -> Any:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise IntegrityError(f"frozen dependency SHA mismatch: {path}: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IntegrityError(f"cannot import frozen dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_parents() -> tuple[Any, Any]:
    if sha256_file(V7_CONFIG) != EXPECTED_V7_CONFIG_FILE_SHA256:
        raise IntegrityError("frozen V7 config physical SHA mismatch")
    if sha256_file(V7_PROTOCOL) != EXPECTED_V7_PROTOCOL_FILE_SHA256:
        raise IntegrityError("frozen V7 protocol physical SHA mismatch")
    v7 = load_module("ppo_progress_v2_frozen_v7", V7_PROGRAM, EXPECTED_V7_PROGRAM_SHA256)
    internal = load_module(
        "ppo_progress_v2_reward_switch_parent",
        INTERNAL_PROGRAM,
        EXPECTED_INTERNAL_PROGRAM_SHA256,
    )
    return v7, internal


class RevisionConfig:
    def __init__(self, path: Path, v7: Any):
        self.path = path.resolve()
        self.sha256 = v7.verified_self_hash(self.path)
        self.overlay = read_json(self.path)
        parent = v7.FrozenConfig(V7_CONFIG)
        if parent.sha256 != EXPECTED_V7_CONFIG_SELF_SHA256:
            raise IntegrityError("frozen V7 config self-hash mismatch")
        self.parent_data = copy.deepcopy(parent.data)
        self.data = copy.deepcopy(parent.data)
        for key in (
            "schema_version",
            "status",
            "formal_execution_allowed",
            "authorized_program",
            "authorized_program_sha256",
            "output_root",
            "official_seeds",
            "configurations",
            "revision_identity",
        ):
            self.data[key] = copy.deepcopy(self.overlay[key])
        self.data["fairness_constraints"]["only_factor_switches"] = [
            "revisit",
            "oscillation",
            "stagnation",
        ]

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
    def __init__(self, path: Path, v7: Any):
        self.path = path.resolve()
        self.sha256 = v7.verified_protocol_hash(self.path)
        self.overlay = read_json(self.path)
        parent = v7.FrozenProtocol(V7_PROTOCOL)
        if parent.sha256 != EXPECTED_V7_PROTOCOL_SELF_SHA256:
            raise IntegrityError("frozen V7 protocol self-hash mismatch")
        self.parent_data = copy.deepcopy(parent.data)
        self.data = copy.deepcopy(parent.data)
        for key in (
            "schema_version",
            "status",
            "formal_execution_allowed",
            "authorized_program",
            "authorized_program_sha256",
            "configuration",
            "configurations",
            "official_seeds",
            "revision_identity",
            "isolation_assertions",
        ):
            self.data[key] = copy.deepcopy(self.overlay[key])

    def get(self, dotted: str) -> Any:
        value: Any = self.data
        for part in dotted.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ProtocolError(f"missing protocol field: {dotted}")
            value = value[part]
        return value


ORIGINAL_V7_ENVIRONMENT: Optional[type] = None


def validate_overlay(config: RevisionConfig, protocol: RevisionProtocol) -> None:
    expected_program = "03_Code/ppo_progress_8seed_v2/ppo_progress_8seed_v2.py"
    current_sha = sha256_file(SCRIPT_PATH)
    for payload, label in ((config.overlay, "config"), (protocol.overlay, "protocol")):
        if payload.get("status") != "FROZEN_AND_AUTHORIZED":
            raise ProtocolError(f"{label} is not frozen")
        if payload.get("formal_execution_allowed") is not True:
            raise ProtocolError(f"{label} does not authorize future formal execution")
        if payload.get("authorized_program") != expected_program:
            raise ProtocolError(f"{label} authorized program path mismatch")
        if str(payload.get("authorized_program_sha256", "")).upper() != current_sha:
            raise ProtocolError(f"{label} authorized program SHA mismatch")
        if list(payload.get("official_seeds", [])) != SEEDS:
            raise ProtocolError(f"{label} seed list mismatch")
        if payload.get("configurations") != DECLARED_CONFIGURATION:
            raise ProtocolError(f"{label} method definition mismatch")
    if config.overlay.get("schema_version") != "ppo_progress_8seed.v2":
        raise ProtocolError("unsupported PPO+Progress config schema")
    if protocol.overlay.get("schema_version") != "ppo_progress_8seed.v2.protocol":
        raise ProtocolError("unsupported PPO+Progress protocol schema")
    if config.overlay.get("output_root") != str(OUTPUT_ROOT):
        raise ProtocolError("formal output root mismatch")
    if protocol.overlay.get("configuration", {}).get("self_field_zeroed_sha256") != config.sha256:
        raise ProtocolError("protocol-to-config identity mismatch")
    if config.integer("training.interaction_budget") != 900000:
        raise ProtocolError("training budget drift")
    if config.integer("training.validation_interval_interactions") != 100000:
        raise ProtocolError("validation interval drift")
    if config.get("training.checkpoint_selection_split") != "validation":
        raise ProtocolError("checkpoint selection split drift")
    if config.integer("evaluation.parameter_updates") != 0:
        raise ProtocolError("evaluation parameter-update contract drift")
    if config.get("reward.dynamic.distance_metric") != "euclidean":
        raise ProtocolError("Progress distance metric drift")
    if config.number("reward.dynamic.distance_weight") != 1.0:
        raise ProtocolError("Progress coefficient drift")
    if config.number("reward.dynamic.distance_progress_clip") != 0.05:
        raise ProtocolError("Progress clipping drift")
    if any(REWARD_COMPONENTS[name] for name in ("revisit", "oscillation", "stagnation")):
        raise ProtocolError("history-conditioned correction unexpectedly enabled")
    if any(EVALUATION_REWARD_COMPONENTS.values()):
        raise ProtocolError("evaluation dynamic reward component unexpectedly enabled")


def configure_v7(v7: Any, internal: Any) -> None:
    global ORIGINAL_V7_ENVIRONMENT
    ORIGINAL_V7_ENVIRONMENT = v7.NavigationEnvironment

    class ProgressOnlyNavigationEnvironment(internal.NavigationEnvironment):
        """V7-compatible facade over the component-switch environment."""

        def __init__(self, config: RevisionConfig, maps: Any, reward_enabled: bool):
            self.reward_enabled = bool(reward_enabled)
            switches = REWARD_COMPONENTS if self.reward_enabled else EVALUATION_REWARD_COMPONENTS
            super().__init__(config, maps, switches)

    v7.SCRIPT_PATH = SCRIPT_PATH
    v7.PROJECT_ROOT = PROJECT_ROOT
    v7.SOURCE_DIR = V7_ROOT
    v7.CONFIGURATIONS = copy.deepcopy(CORE_CONFIGURATIONS)
    v7.NavigationEnvironment = ProgressOnlyNavigationEnvironment

    def ensure_archive(config: RevisionConfig, protocol: RevisionProtocol, root: Path) -> None:
        archive = root / "protocol"
        archive.mkdir(parents=True, exist_ok=True)
        sources = {
            "ppo_progress_8seed_v2.py": SCRIPT_PATH,
            "experiment_config.json": config.path,
            "experiment_protocol.json": protocol.path,
            "hash_manifest.json": HASH_MANIFEST,
            "frozen_v7_authority.py": V7_PROGRAM,
            "frozen_v7_experiment_config.json": V7_CONFIG,
            "frozen_v7_experiment_protocol.json": V7_PROTOCOL,
            "reward_switch_parent_internal_ablation_v1.py": INTERNAL_PROGRAM,
            "map_split.json": V7_ROOT / "map_split.json",
            "task_split.json": V7_ROOT / "task_split.json",
        }
        for name, source in sources.items():
            destination = archive / name
            if destination.exists():
                if sha256_file(destination) != sha256_file(source):
                    raise IntegrityError(f"protocol archive conflict: {destination}")
            else:
                shutil.copy2(source, destination)
        (root / "analysis_ready").mkdir(parents=True, exist_ok=True)

    v7.ensure_archive = ensure_archive


def prepare(
    config_path: Path = DEFAULT_CONFIG,
    protocol_path: Path = DEFAULT_PROTOCOL,
) -> tuple[Any, Any, RevisionConfig, RevisionProtocol]:
    v7, internal = load_parents()
    config = RevisionConfig(config_path, v7)
    protocol = RevisionProtocol(protocol_path, v7)
    validate_overlay(config, protocol)
    configure_v7(v7, internal)
    return v7, internal, config, protocol


def seed_directory(config: RevisionConfig, seed: int) -> Path:
    return Path(str(config.get("output_root"))).resolve() / METHOD_ID / f"seed_{seed}"


def validate_completed_seed(config: RevisionConfig, seed: int) -> Dict[str, Any]:
    directory = seed_directory(config, seed)
    done_path = directory / "DONE.json"
    if not done_path.is_file():
        raise IntegrityError(f"missing DONE marker: {done_path}")
    done = read_json(done_path)
    if done.get("status") != "COMPLETED" or done.get("method") != METHOD_ID or int(done.get("seed", -1)) != seed:
        raise IntegrityError("DONE marker identity/status mismatch")
    result_path = Path(str(done.get("formal_result"))).resolve()
    manifest_path = Path(str(done.get("run_manifest"))).resolve()
    result = read_json(result_path)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed" or int(manifest.get("executed_interaction_count", -1)) != 900000:
        raise IntegrityError("run manifest is not a completed 900k run")
    if result.get("method") != METHOD_ID or int(result.get("seed", -1)) != seed:
        raise IntegrityError("formal result identity mismatch")
    test = result.get("test", {})
    if int(test.get("episode_count", -1)) != 1000:
        raise IntegrityError("formal test episode count mismatch")
    if sum(int(test.get(key, -1)) for key in ("success_count", "collision_count", "timeout_count")) != 1000:
        raise IntegrityError("formal test outcomes are not exhaustive")
    for key in (
        "training_astar_calls",
        "astar_training_query_attempts",
        "astar_training_targets_generated",
        "validation_astar_runtime_calls",
        "test_astar_runtime_calls",
    ):
        if int(result.get(key, -1)) != 0:
            raise IntegrityError(f"nonzero A* counter: {key}")
    if int(test.get("parameter_updates", -1)) != 0:
        raise IntegrityError("test parameter updates are nonzero")
    return done


def write_done_marker(config: RevisionConfig, seed: int, run: Path) -> Path:
    result_path = run / f"formal_result_{METHOD_ID}_seed{seed}.json"
    manifest_path = run / "run_manifest.json"
    result = read_json(result_path)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed" or result.get("method") != METHOD_ID:
        raise IntegrityError("refusing DONE marker for incomplete/mismatched run")
    marker = seed_directory(config, seed) / "DONE.json"
    write_json_exclusive(
        marker,
        {
            "status": "COMPLETED",
            "method": METHOD_ID,
            "display_name": DISPLAY_NAME,
            "seed": seed,
            "formal_result": str(result_path.resolve()),
            "formal_result_sha256": sha256_file(result_path),
            "run_manifest": str(manifest_path.resolve()),
            "run_manifest_sha256": sha256_file(manifest_path),
        },
    )
    return marker


def print_formal_preflight(config: RevisionConfig, input_hashes: Mapping[str, str], seed: int) -> None:
    print("[PPO+PROGRESS V2 FORMAL PREFLIGHT]", flush=True)
    print(f"method = {DISPLAY_NAME}", flush=True)
    print(f"seed = {seed}", flush=True)
    print("Progress = ON", flush=True)
    print("Revisit = OFF", flush=True)
    print("Oscillation = OFF", flush=True)
    print("Stagnation = OFF", flush=True)
    print("A* = OFF", flush=True)
    print(f"training budget = {config.integer('training.interaction_budget')}", flush=True)
    print(f"validation interval = {config.integer('training.validation_interval_interactions')}", flush=True)
    print("test episodes = 1000", flush=True)
    print(f"training task SHA = {input_hashes['training_tasks']}", flush=True)
    print(f"validation task SHA = {input_hashes['validation_tasks']}", flush=True)
    print(f"test task SHA = {input_hashes['test_tasks']}", flush=True)
    print(f"output directory = {seed_directory(config, seed)}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal BARN 30x30 PPO+Progress v2 compatibility adapter")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--method", choices=(METHOD_ID,))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verify-completed", action="store_true")
    parser.add_argument("--confirm-formal", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v7, _, config, protocol = prepare(args.config, args.protocol)
    input_hashes = v7.verify_inputs(config)
    train_tasks, validation_tasks, test_tasks = v7.load_tasks(config)
    if len(test_tasks) != 1000:
        raise ProtocolError("formal test task count is not 1000")
    if args.audit_only:
        print(json.dumps({
            "status": "STATIC_IDENTITY_PASS",
            "training_started": False,
            "cuda_started": False,
            "program_sha256": sha256_file(SCRIPT_PATH),
            "config_sha256": config.sha256,
            "protocol_sha256": protocol.sha256,
            "v7_program_sha256": sha256_file(V7_PROGRAM),
            "reward_switch_parent_sha256": sha256_file(INTERNAL_PROGRAM),
            "input_sha256": input_hashes,
            "task_counts": {"training": len(train_tasks), "validation": len(validation_tasks), "test": len(test_tasks)},
            "official_seeds": SEEDS,
            "method": METHOD_ID,
            "display_name": DISPLAY_NAME,
            "training_reward_components": REWARD_COMPONENTS,
            "evaluation_reward_components": EVALUATION_REWARD_COMPONENTS,
            "astar_enabled": False,
            "output_root": str(OUTPUT_ROOT),
            "output_root_exists": OUTPUT_ROOT.exists(),
        }, indent=2), flush=True)
        return
    if args.verify_completed:
        if args.seed is None or args.seed not in SEEDS:
            raise ProtocolError("--verify-completed requires one authorized --seed")
        print(json.dumps(validate_completed_seed(config, args.seed), indent=2), flush=True)
        return
    if not args.confirm_formal:
        raise ProtocolError("refusing to start formal training without --confirm-formal")
    if args.method != METHOD_ID or args.seed not in SEEDS or args.device is None:
        raise ProtocolError("--method PPO_PROGRESS, an authorized --seed, and --device are required")
    directory = seed_directory(config, args.seed)
    if directory.exists():
        raise ProtocolError(f"refusing to overwrite or retry existing seed directory: {directory}")
    if args.device == "cuda" and not v7.torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable")
    print_formal_preflight(config, input_hashes, args.seed)
    run = v7.run_experiment(config, protocol, METHOD_ID, args.seed, v7.torch.device(args.device))
    marker = write_done_marker(config, args.seed, run)
    validate_completed_seed(config, args.seed)
    print(f"completed_run = {run}", flush=True)
    print(f"done_marker = {marker}", flush=True)


if __name__ == "__main__":
    main()
