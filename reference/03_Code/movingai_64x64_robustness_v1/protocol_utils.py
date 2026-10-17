#!/usr/bin/env python
"""Shared identity helpers. This module never imports Torch or the V7 runtime."""

from __future__ import print_function

import hashlib
import json
import re
from pathlib import Path


REVISION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = REVISION_DIR.parent.parent
AUTHORITY_PATH = PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7" / "final_two_factor_experiment_v7.py"
AUTHORITY_SHA256 = "96093A6B5F024F5D9505ACCEBEAE7592700228BA17587BF0590092A241B7544D"
EXTENSION_PATH = PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7_8seed" / "final_two_factor_experiment_v7_8seed.py"
EXTENSION_SHA256 = "19290D3A23DCC8E10FD4264EF9D8A3FEC4A691CFFD8A69C6FDEE8DE65C8EF8DD"
RESULT_ROOT = Path(r"redacted_local_root\revision_movingai64_robustness_v1")
ALLOWED_METHODS = ("PPO", "PPO_Reward", "FULL")
CONFIG_HASH_PATTERN = re.compile(rb'("config_sha256"\s*:\s*")([0-9A-Fa-f]{64})(")')
PROTOCOL_HASH_PATTERN = re.compile(rb'("protocol_sha256"\s*:\s*")([0-9A-Fa-f]{64})(")')


class ProtocolError(RuntimeError):
    pass


class IntegrityError(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError("expected JSON object: %s" % path)
    return value


def write_json_new(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def verified_self_hash(path, field):
    raw = Path(path).read_bytes()
    pattern = CONFIG_HASH_PATTERN if field == "config_sha256" else PROTOCOL_HASH_PATTERN
    matches = list(pattern.finditer(raw))
    if len(matches) != 1:
        raise ProtocolError("expected exactly one %s field" % field)
    match = matches[0]
    stored = match.group(2).decode("ascii").upper()
    zeroed = raw[:match.start(2)] + b"0" * 64 + raw[match.end(2):]
    calculated = hashlib.sha256(zeroed).hexdigest().upper()
    if stored != calculated:
        raise ProtocolError("%s mismatch: stored=%s calculated=%s" % (field, stored, calculated))
    return stored


def resolve_declared_path(value):
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def verify_authorities():
    actual = {
        "legacy_v7": sha256_file(AUTHORITY_PATH),
        "eight_seed_extension": sha256_file(EXTENSION_PATH),
    }
    if actual["legacy_v7"] != AUTHORITY_SHA256:
        raise IntegrityError("AUTHORITY_IDENTITY_MISMATCH: Legacy V7")
    if actual["eight_seed_extension"] != EXTENSION_SHA256:
        raise IntegrityError("AUTHORITY_IDENTITY_MISMATCH: 8-seed extension")
    return actual


def task_identity_from_config(config):
    result = {}
    for name, record in config["inputs"].items():
        expected = str(record["sha256"]).upper()
        if expected == "TASK_ARTIFACTS_PENDING_USER_GENERATION":
            raise ProtocolError("TASK_ARTIFACTS_PENDING_USER_GENERATION: %s" % name)
        path = resolve_declared_path(record["path"])
        if not path.is_file():
            raise IntegrityError("missing frozen input: %s" % path)
        actual = sha256_file(path)
        if actual != expected:
            raise IntegrityError("input SHA mismatch: %s" % path)
        result[name] = {"path": str(path), "sha256": actual}
    return result


def map_hash_aggregate(config):
    digest = hashlib.sha256()
    count = 0
    for input_name in ("training_tasks", "validation_tasks", "test_tasks"):
        path = resolve_declared_path(config["inputs"][input_name]["path"])
        tasks = read_json(path).get("tasks", [])
        for task in sorted(tasks, key=lambda item: str(item["task_id"])):
            map_path = Path(str(task["cspace_file"])).resolve()
            actual = sha256_file(map_path)
            if actual != str(task["processed_map_sha256"]).upper():
                raise IntegrityError("processed map SHA mismatch: %s" % map_path)
            digest.update((str(task["task_id"]) + "\t" + actual + "\n").encode("utf-8"))
            count += 1
    return {"count": count, "sha256": digest.hexdigest().upper()}


def validate_confirmed(config_path, protocol_path):
    config_path = Path(config_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    config = read_json(config_path)
    protocol = read_json(protocol_path)
    config_sha = verified_self_hash(config_path, "config_sha256")
    protocol_sha = verified_self_hash(protocol_path, "protocol_sha256")
    verify_authorities()
    if config.get("status") != "CONFIRMED_BY_USER" or config.get("formal_execution_allowed") is not True:
        raise ProtocolError("configuration is not user-confirmed for formal execution")
    if protocol.get("status") != "CONFIRMED_BY_USER" or protocol.get("formal_execution_allowed") is not True:
        raise ProtocolError("protocol is not user-confirmed for formal execution")
    seeds = protocol.get("seed_design", {}).get("confirmed_seeds")
    horizon = protocol.get("horizon", {}).get("confirmed_value")
    if not isinstance(seeds, list) or not seeds or seeds != config.get("official_seeds"):
        raise ProtocolError("confirmed seed identity mismatch")
    if protocol.get("seed_design", {}).get("status") != "CONFIRMED_BY_USER":
        raise ProtocolError("seed protocol remains draft")
    if horizon is None or int(horizon) != int(config["environment"]["max_steps"]):
        raise ProtocolError("confirmed horizon identity mismatch")
    if protocol.get("horizon", {}).get("status") != "CONFIRMED_BY_USER":
        raise ProtocolError("horizon remains pending")
    if list(config.get("configurations", {}).keys()) != list(ALLOWED_METHODS):
        raise ProtocolError("method set must be exactly PPO, PPO_Reward, FULL")
    if int(config["environment"]["map_size"]) != 64:
        raise ProtocolError("map_size must be 64")
    if config["state"]["spatial_shape"] != [5, 64, 64] or config["network"]["spatial_input_shape"] != [5, 64, 64]:
        raise ProtocolError("64x64 shape identity mismatch")
    if int(config["training"]["interaction_budget"]) != 900000:
        raise ProtocolError("interaction budget drift")
    identities = task_identity_from_config(config)
    return {
        "config": config,
        "protocol": protocol,
        "config_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "inputs": identities,
    }


def normalized_config_drift(candidate, legacy):
    """Return fields whose values drift outside the explicitly allowed shape/data scope."""
    allowed_top = {
        "schema_version", "status", "formal_execution_allowed", "config_sha256_rule", "config_sha256",
        "authority_program", "authority_program_sha256", "adapter_program", "output_root", "official_seeds",
        "seed_status", "configurations", "fairness_constraints", "inputs", "environment", "state", "network",
    }
    drift = []
    for key in legacy:
        if key in allowed_top:
            continue
        if candidate.get(key) != legacy.get(key):
            drift.append(key)
    for key in ("ppo", "reward", "astar", "training", "evaluation", "random_streams"):
        if candidate.get(key) != legacy.get(key):
            drift.append(key)
    legacy_network = dict(legacy["network"])
    candidate_network = dict(candidate["network"])
    legacy_network["spatial_input_shape"] = [5, 64, 64]
    if candidate_network != legacy_network:
        drift.append("network_except_spatial_input_shape")
    return sorted(set(drift))
