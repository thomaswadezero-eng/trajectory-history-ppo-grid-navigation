#!/usr/bin/env python
"""Read-only identity and result validators for the MovingAI 8-seed extension."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
CONFIG_PATH = HERE / "movingai_8seed_config.json"
PROTOCOL_PATH = HERE / "movingai_8seed_protocol.json"
LEGACY_REGISTRY_PATH = HERE / "legacy_canonical_runs.json"
LEGACY_CONFIG_PATH = PROJECT_ROOT / "03_Code" / "movingai_64x64_robustness_v1" / "movingai64_config_confirmed.json"
LEGACY_PROTOCOL_PATH = PROJECT_ROOT / "03_Code" / "movingai_64x64_robustness_v1" / "movingai64_protocol_confirmed.json"
LEGACY_WRAPPER_PATH = PROJECT_ROOT / "03_Code" / "movingai_64x64_robustness_v1" / "run_movingai64_wrapper.py"
LEGACY_PROTOCOL_UTILS_PATH = PROJECT_ROOT / "03_Code" / "movingai_64x64_robustness_v1" / "protocol_utils.py"
AUTHORITY_PATH = PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7" / "final_two_factor_experiment_v7.py"
LEGACY_RESULT_ROOT = Path(r"redacted_local_root\revision_movingai64_robustness_v1")
NEW_RESULT_ROOT = Path(r"redacted_local_root\revision_movingai_8seed_extension")
METHODS = ("PPO", "PPO_Reward", "FULL")
ALL_SEEDS = (42, 1004, 6010, 68882, 19816, 74494, 81399, 31524)
LEGACY_SEEDS = (42, 1004, 6010)
NEW_SEEDS = (68882, 19816, 74494, 81399, 31524)
METRICS = (
    "success_rate",
    "spl",
    "path_efficiency",
    "average_steps",
    "average_path_length",
    "collision_rate",
    "timeout_rate",
    "episode_reward",
)
CONFIG_HASH_PATTERN = re.compile(rb'("config_sha256"\s*:\s*")([0-9A-Fa-f]{64})(")')
PROTOCOL_HASH_PATTERN = re.compile(rb'("protocol_sha256"\s*:\s*")([0-9A-Fa-f]{64})(")')


class ProtocolError(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ProtocolError("expected JSON object: %s" % path)
    return payload


def write_json_new(path: Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_json_replace(path: Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def verified_self_hash(path: Path, field: str) -> str:
    raw = Path(path).read_bytes()
    pattern = CONFIG_HASH_PATTERN if field == "config_sha256" else PROTOCOL_HASH_PATTERN
    matches = list(pattern.finditer(raw))
    if len(matches) != 1:
        raise ProtocolError("expected exactly one %s field: %s" % (field, path))
    match = matches[0]
    stored = match.group(2).decode("ascii").upper()
    zeroed = raw[: match.start(2)] + b"0" * 64 + raw[match.end(2) :]
    calculated = hashlib.sha256(zeroed).hexdigest().upper()
    if stored != calculated:
        raise IntegrityError("%s mismatch: stored=%s calculated=%s" % (field, stored, calculated))
    return stored


def expected_input_hashes(config: Mapping[str, Any]) -> Dict[str, str]:
    return {name: str(record["sha256"]).upper() for name, record in config["inputs"].items()}


def verify_inputs(config: Mapping[str, Any]) -> Dict[str, str]:
    actual: Dict[str, str] = {}
    for name, record in config["inputs"].items():
        path = Path(str(record["path"])).resolve()
        if not path.is_file():
            raise IntegrityError("missing frozen input: %s" % path)
        digest = sha256_file(path)
        if digest != str(record["sha256"]).upper():
            raise IntegrityError("input SHA mismatch: %s" % path)
        actual[name] = digest
    return actual


def verify_task_protocol(config: Mapping[str, Any]) -> Dict[str, Any]:
    split_names = ("training_tasks", "validation_tasks", "test_tasks")
    expected_counts = {"training_tasks": 800, "validation_tasks": 200, "test_tasks": 750}
    expected_subsets = {
        "training_tasks": {"DAO": 380, "Room": 190, "Maze": 230},
        "validation_tasks": {"DAO": 80, "Room": 40, "Maze": 80},
        "test_tasks": {"DAO": 380, "Room": 140, "Maze": 230},
    }
    all_task_ids: List[str] = []
    source_sha_by_split: Dict[str, set[str]] = {}
    crop_signatures_by_split: Dict[str, set[Tuple[Any, ...]]] = {}
    summary: Dict[str, Any] = {}
    for name in split_names:
        payload = read_json(Path(str(config["inputs"][name]["path"])))
        tasks = payload.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != expected_counts[name]:
            raise IntegrityError("frozen task count mismatch: %s" % name)
        ids = [str(task["task_id"]) for task in tasks]
        if len(ids) != len(set(ids)):
            raise IntegrityError("duplicate task IDs within %s" % name)
        all_task_ids.extend(ids)
        subset_counts: Dict[str, int] = {}
        source_shas: set[str] = set()
        crop_signatures: set[Tuple[Any, ...]] = set()
        for task in tasks:
            subset = str(task["subset"])
            subset_counts[subset] = subset_counts.get(subset, 0) + 1
            if int(task["crop_height"]) != 64 or int(task["crop_width"]) != 64:
                raise IntegrityError("non-64x64 frozen crop: %s" % task["task_id"])
            if int(task["max_steps"]) != 400:
                raise IntegrityError("frozen horizon mismatch: %s" % task["task_id"])
            source_shas.add(str(task["source_map_sha256"]).upper())
            crop_signatures.add(
                (
                    str(task["source_map_sha256"]).upper(),
                    int(task["crop_top"]),
                    int(task["crop_left"]),
                    int(task["crop_bottom"]),
                    int(task["crop_right"]),
                    tuple(task["start"]),
                    tuple(task["goal"]),
                )
            )
        if subset_counts != expected_subsets[name]:
            raise IntegrityError("subset composition mismatch: %s %s" % (name, subset_counts))
        source_sha_by_split[name] = source_shas
        crop_signatures_by_split[name] = crop_signatures
        summary[name] = {"task_count": len(tasks), "subset_counts": subset_counts}
    if len(all_task_ids) != len(set(all_task_ids)):
        raise IntegrityError("task IDs overlap across splits")
    for left, right in ((split_names[0], split_names[1]), (split_names[0], split_names[2]), (split_names[1], split_names[2])):
        if source_sha_by_split[left] & source_sha_by_split[right]:
            raise IntegrityError("source maps overlap: %s/%s" % (left, right))
        if crop_signatures_by_split[left] & crop_signatures_by_split[right]:
            raise IntegrityError("crop signatures overlap: %s/%s" % (left, right))
    summary["split_isolation"] = "PASS"
    return summary


def validate_extension_identity(config_path: Path = CONFIG_PATH, protocol_path: Path = PROTOCOL_PATH) -> Dict[str, Any]:
    config = read_json(config_path)
    protocol = read_json(protocol_path)
    config_sha = verified_self_hash(config_path, "config_sha256")
    protocol_sha = verified_self_hash(protocol_path, "protocol_sha256")
    if config.get("status") != "CONFIRMED_BY_USER" or config.get("formal_execution_allowed") is not True:
        raise ProtocolError("extension config is not confirmed")
    if protocol.get("status") != "CONFIRMED_BY_USER" or protocol.get("formal_execution_allowed") is not True:
        raise ProtocolError("extension protocol is not confirmed")
    if tuple(config.get("official_seeds", ())) != ALL_SEEDS:
        raise ProtocolError("eight-seed order mismatch")
    if tuple(protocol.get("seed_design", {}).get("confirmed_seeds", ())) != ALL_SEEDS:
        raise ProtocolError("protocol eight-seed order mismatch")
    if tuple(protocol.get("seed_design", {}).get("legacy_seeds", ())) != LEGACY_SEEDS:
        raise ProtocolError("legacy seed identity mismatch")
    if tuple(protocol.get("seed_design", {}).get("new_seeds", ())) != NEW_SEEDS:
        raise ProtocolError("new seed identity mismatch")
    if list(config.get("configurations", {}).keys()) != list(METHODS):
        raise ProtocolError("method order or identity mismatch")
    if Path(str(config.get("output_root"))).resolve() != NEW_RESULT_ROOT.resolve():
        raise ProtocolError("new output root mismatch")
    if int(config["training"]["interaction_budget"]) != 900000:
        raise ProtocolError("interaction budget drift")
    if int(config["environment"]["max_steps"]) != 400 or int(config["environment"]["map_size"]) != 64:
        raise ProtocolError("MovingAI64 environment drift")
    if protocol.get("statistics", {}).get("inferential_status") != "NOT_AUTHORIZED":
        raise ProtocolError("inferential statistics must remain unauthorized")

    legacy = read_json(LEGACY_CONFIG_PATH)
    allowed_changes = {
        "schema_version",
        "config_sha256_rule",
        "config_sha256",
        "adapter_program",
        "extension_launcher",
        "extension_aggregator",
        "output_root",
        "official_seeds",
        "seed_status",
        "extension_scope",
        "program_sha256",
    }
    for key in set(legacy) | set(config):
        if key in allowed_changes:
            continue
        if config.get(key) != legacy.get(key):
            raise IntegrityError("config drift outside seed/output metadata: %s" % key)

    expected_programs = config.get("program_sha256", {})
    actual_programs = {
        "authority": sha256_file(AUTHORITY_PATH),
        "legacy_wrapper": sha256_file(LEGACY_WRAPPER_PATH),
        "legacy_protocol_utils": sha256_file(LEGACY_PROTOCOL_UTILS_PATH),
        "launcher": sha256_file(HERE / "run_movingai_8seed_extension.py"),
        "aggregator": sha256_file(HERE / "aggregate_movingai_8seed_results.py"),
        "extension_utils": sha256_file(HERE / "extension_utils.py"),
    }
    if expected_programs != actual_programs:
        raise IntegrityError("extension program identity mismatch")
    if str(protocol.get("configuration", {}).get("self_field_zeroed_sha256", "")).upper() != config_sha:
        raise IntegrityError("protocol does not bind extension config")
    inputs = verify_inputs(config)
    return {
        "config": config,
        "protocol": protocol,
        "config_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "input_sha256": inputs,
        "program_sha256": actual_programs,
    }


def validate_done_artifacts(
    done_path: Path,
    method: str,
    seed: int,
    config_sha: str,
    protocol_sha: str,
    input_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    done = read_json(done_path)
    expected = {
        "status": "DONE",
        "method": method,
        "seed": seed,
        "authority_program_sha256": sha256_file(AUTHORITY_PATH),
        "wrapper_program_sha256": sha256_file(LEGACY_WRAPPER_PATH),
        "config_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "training_interaction_budget": 900000,
        "executed_training_interactions": 900000,
        "validation_count": 9,
        "test_episode_count": 750,
        "validation_astar_runtime_calls": 0,
        "test_astar_runtime_calls": 0,
    }
    for key, expected_value in expected.items():
        if done.get(key) != expected_value:
            raise IntegrityError("DONE field mismatch %s: %s" % (key, done_path))
    if done.get("input_sha256") != dict(input_hashes):
        raise IntegrityError("DONE input identity mismatch: %s" % done_path)
    if done.get("map_hash_aggregate") != {
        "count": 1750,
        "sha256": "A0AEBE9DEC2ACD9F8B87B8E9E9C5C3F29F0EE0AB1221F41C79D6E4DDC1C6A71A",
    }:
        raise IntegrityError("DONE processed-map identity mismatch: %s" % done_path)
    for path_key, hash_key in (
        ("selected_checkpoint_path", "selected_checkpoint_sha256"),
        ("formal_test_result_path", "formal_test_result_sha256"),
        ("authority_evaluation_csv_path", "authority_evaluation_csv_sha256"),
        ("episode_csv_path", "episode_csv_sha256"),
    ):
        artifact = Path(str(done[path_key])).resolve()
        if not artifact.is_file() or sha256_file(artifact) != str(done[hash_key]).upper():
            raise IntegrityError("DONE artifact mismatch: %s" % artifact)
    return done


def validate_legacy_runs(identity: Mapping[str, Any]) -> List[Dict[str, Any]]:
    registry = read_json(LEGACY_REGISTRY_PATH)
    if tuple(registry.get("methods", ())) != METHODS or tuple(registry.get("seeds", ())) != LEGACY_SEEDS:
        raise IntegrityError("legacy registry matrix mismatch")
    if sha256_file(Path(registry["formal_audit_json"]["path"])) != registry["formal_audit_json"]["sha256"]:
        raise IntegrityError("legacy formal-audit identity mismatch")
    audit = read_json(Path(registry["formal_audit_json"]["path"]))
    if audit.get("overall_pass") is not True or int(audit.get("canonical_run_count", 0)) != 9:
        raise IntegrityError("legacy formal audit is not PASS 9/9")
    legacy_config_sha = verified_self_hash(LEGACY_CONFIG_PATH, "config_sha256")
    legacy_protocol_sha = verified_self_hash(LEGACY_PROTOCOL_PATH, "protocol_sha256")
    results: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()
    for record in registry.get("runs", []):
        method, seed = str(record["method"]), int(record["seed"])
        key = (method, seed)
        if key in seen:
            raise IntegrityError("duplicate legacy registry entry: %s" % (key,))
        seen.add(key)
        done_path = Path(str(record["done_path"])).resolve()
        if sha256_file(done_path) != str(record["done_sha256"]).upper():
            raise IntegrityError("legacy DONE SHA mismatch: %s" % done_path)
        done = validate_done_artifacts(
            done_path,
            method,
            seed,
            legacy_config_sha,
            legacy_protocol_sha,
            identity["input_sha256"],
        )
        results.append({"source": "legacy_3seed", "method": method, "seed": seed, "done_path": done_path, "done": done})
    expected = {(method, seed) for method in METHODS for seed in LEGACY_SEEDS}
    if seen != expected:
        raise IntegrityError("legacy canonical matrix is not exactly 9/9")
    return results


def resolve_new_done(identity: Mapping[str, Any], method: str, seed: int) -> Dict[str, Any]:
    seed_root = NEW_RESULT_ROOT / method / ("seed_%d" % seed)
    run_dirs = sorted(path for path in seed_root.glob("run_*") if path.is_dir()) if seed_root.is_dir() else []
    if len(run_dirs) != 1:
        raise IntegrityError("expected exactly one new attempt for %s/%d, found %d" % (method, seed, len(run_dirs)))
    done_path = run_dirs[0] / "DONE.json"
    if not done_path.is_file():
        raise IntegrityError("new attempt is incomplete; recovery requires user authorization: %s" % run_dirs[0])
    done = validate_done_artifacts(
        done_path,
        method,
        seed,
        identity["config_sha256"],
        identity["protocol_sha256"],
        identity["input_sha256"],
    )
    return {"source": "new_5seed", "method": method, "seed": seed, "done_path": done_path, "done": done}


def load_all_24_records(identity: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records = validate_legacy_runs(identity)
    for method in METHODS:
        for seed in NEW_SEEDS:
            records.append(resolve_new_done(identity, method, seed))
    keys = {(record["method"], record["seed"]) for record in records}
    expected = {(method, seed) for method in METHODS for seed in ALL_SEEDS}
    if len(records) != 24 or keys != expected:
        raise IntegrityError("formal aggregate requires exactly 24 unique method-by-seed combinations")
    return records


def read_episode_rows(record: Mapping[str, Any], test_tasks: Mapping[str, Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    done = record["done"]
    path = Path(str(done["episode_csv_path"]))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 750:
        raise IntegrityError("episode CSV row count mismatch: %s" % path)
    task_ids = [str(row["task_id"]) for row in rows]
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(test_tasks):
        raise IntegrityError("test task coverage mismatch: %s" % path)
    normalized: List[Dict[str, Any]] = []
    successes: List[int] = []
    collisions: List[int] = []
    timeouts: List[int] = []
    steps: List[int] = []
    path_lengths: List[int] = []
    spls: List[float] = []
    efficiencies: List[float] = []
    rewards: List[float] = []
    for row in rows:
        task = test_tasks[str(row["task_id"])]
        if row["method"] != record["method"] or int(row["seed"]) != record["seed"]:
            raise IntegrityError("episode method/seed mismatch: %s" % path)
        if row["source_map_sha256"].upper() != str(task["source_map_sha256"]).upper():
            raise IntegrityError("episode source-map mismatch: %s" % row["task_id"])
        if row["processed_map_sha256"].upper() != str(task["processed_map_sha256"]).upper():
            raise IntegrityError("episode processed-map mismatch: %s" % row["task_id"])
        success, collision, timeout = int(row["success"]), int(row["collision"]), int(row["timeout"])
        if success + collision + timeout != 1:
            raise IntegrityError("episode outcome is not exhaustive: %s" % row["task_id"])
        step_count = int(row["steps"])
        path_length = int(row["path_length"])
        optimal = int(row["shortest_path_length"])
        expected_spl = success * optimal / float(max(optimal, path_length))
        expected_efficiency = optimal / float(max(optimal, path_length)) if success else 0.0
        if not math.isclose(float(row["SPL"]), expected_spl, rel_tol=0.0, abs_tol=1e-12):
            raise IntegrityError("episode SPL mismatch: %s" % row["task_id"])
        if not math.isclose(float(row["path_efficiency"]), expected_efficiency, rel_tol=0.0, abs_tol=1e-12):
            raise IntegrityError("episode path-efficiency mismatch: %s" % row["task_id"])
        successes.append(success)
        collisions.append(collision)
        timeouts.append(timeout)
        steps.append(step_count)
        path_lengths.append(path_length)
        spls.append(expected_spl)
        if success:
            efficiencies.append(expected_efficiency)
        rewards.append(float(row["episode_reward"]))
        normalized.append(
            {
                "method": record["method"],
                "seed": record["seed"],
                "source": record["source"],
                "task_id": str(row["task_id"]),
                "subset": str(task["subset"]),
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "steps": step_count,
                "path_length": path_length,
                "shortest_path_length": optimal,
                "spl": expected_spl,
                "path_efficiency": expected_efficiency,
                "episode_reward": float(row["episode_reward"]),
            }
        )
    metrics = {
        "success_rate": sum(successes) / 750.0,
        "spl": sum(spls) / 750.0,
        "path_efficiency": sum(efficiencies) / float(len(efficiencies)) if efficiencies else 0.0,
        "average_steps": sum(steps) / 750.0,
        "average_path_length": sum(path_lengths) / 750.0,
        "collision_rate": sum(collisions) / 750.0,
        "timeout_rate": sum(timeouts) / 750.0,
        "episode_reward": sum(rewards) / 750.0,
    }
    for metric, value in metrics.items():
        formal = float(done["final_metrics"][metric])
        if not math.isclose(value, formal, rel_tol=0.0, abs_tol=1e-12):
            raise IntegrityError("recomputed metric mismatch %s: %s" % (metric, path))
    return normalized, metrics

