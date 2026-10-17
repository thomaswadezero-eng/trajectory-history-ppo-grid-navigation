#!/usr/bin/env python
"""Thin formal child delegating training/evaluation to the frozen Legacy V7 core."""

from __future__ import print_function

import argparse
import csv
import importlib.util
import json
import sys
from statistics import mean
from pathlib import Path

from protocol_utils import (
    ALLOWED_METHODS,
    AUTHORITY_PATH,
    AUTHORITY_SHA256,
    REVISION_DIR,
    IntegrityError,
    ProtocolError,
    map_hash_aggregate,
    normalized_config_drift,
    read_json,
    sha256_file,
    validate_confirmed,
    write_json_new,
)


def load_authority_module():
    if sha256_file(AUTHORITY_PATH) != AUTHORITY_SHA256:
        raise IntegrityError("AUTHORITY_IDENTITY_MISMATCH")
    name = "movingai64_frozen_v7_authority"
    spec = importlib.util.spec_from_file_location(name, str(AUTHORITY_PATH))
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V7 authority")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_zero_drift(candidate):
    legacy_path = AUTHORITY_PATH.parent / "experiment_config.json"
    legacy = read_json(legacy_path)
    for key in ("ppo", "reward", "astar", "training", "evaluation", "random_streams"):
        if candidate.get(key) != legacy.get(key):
            raise ProtocolError("zero-drift violation in %s" % key)
    legacy_network = dict(legacy["network"])
    legacy_network["spatial_input_shape"] = [5, 64, 64]
    if candidate.get("network") != legacy_network:
        raise ProtocolError("trainable network drift detected")
    legacy_state = dict(legacy["state"])
    legacy_state["spatial_shape"] = [5, 64, 64]
    if candidate.get("state") != legacy_state:
        raise ProtocolError("state drift outside spatial shape")
    legacy_environment = dict(legacy["environment"])
    legacy_environment["map_size"] = 64
    legacy_environment["max_steps"] = candidate["environment"]["max_steps"]
    candidate_environment = dict(candidate["environment"])
    candidate_environment.pop("horizon_status", None)
    candidate_environment.pop("historical_reference_max_steps", None)
    candidate_environment.pop("candidate_horizons", None)
    if candidate_environment != legacy_environment:
        raise ProtocolError("environment drift outside map size/confirmed horizon")


def read_evaluation_rows(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_enriched_test_evidence(run, rows, config, method, seed, result):
    task_path = Path(str(config["inputs"]["test_tasks"]["path"])).resolve()
    tasks = {str(task["task_id"]): task for task in read_json(task_path)["tasks"]}
    episode_rows = [row for row in rows if row.get("record_type") == "episode" and row.get("split") == "test"]
    output = run / "formal_test_episodes_enriched.csv"
    fields = [
        "method", "seed", "episode", "map", "map_id", "task_id", "success", "steps",
        "path_length", "shortest_path_length", "SPL", "path_efficiency", "collision",
        "timeout", "episode_reward", "source_map_sha256", "processed_map_sha256",
    ]
    rewards = []
    with output.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in episode_rows:
            task = tasks[str(row["task_id"])]
            success = int(row["success"])
            collision = int(row["collision"])
            timeout = int(row["timeout"])
            steps = int(row["steps"])
            optimal = int(row["optimal_path_length"])
            path_length = int(row["path_length"])
            efficiency = optimal / float(max(optimal, path_length)) if success else 0.0
            reward = steps * float(config["reward"]["base"]["step"])
            reward += success * float(config["reward"]["base"]["goal"])
            reward += collision * float(config["reward"]["base"]["collision"])
            reward += timeout * float(config["reward"]["base"]["timeout"])
            rewards.append(reward)
            writer.writerow({
                "method": method, "seed": seed, "episode": row["episode"],
                "map": task["source_map_name"], "map_id": row["map_id"], "task_id": row["task_id"],
                "success": success, "steps": steps, "path_length": path_length,
                "shortest_path_length": optimal, "SPL": row["SPL"], "path_efficiency": efficiency,
                "collision": collision, "timeout": timeout, "episode_reward": reward,
                "source_map_sha256": task["source_map_sha256"],
                "processed_map_sha256": task["processed_map_sha256"],
            })
    metrics = dict(result["test"])
    metrics["episode_reward"] = mean(rewards) if rewards else 0.0
    enhanced = {
        "schema_version": "movingai64_formal_result.v1",
        "method": method,
        "seed": seed,
        "authority_result_path": str((run / ("formal_result_%s_seed%d.json" % (method, seed))).resolve()),
        "authority_result_sha256": sha256_file(run / ("formal_result_%s_seed%d.json" % (method, seed))),
        "test_episode_csv_path": str(output.resolve()),
        "test_episode_csv_sha256": sha256_file(output),
        "final_metrics": metrics,
        "reward_reconstruction": "steps*base.step + success*base.goal + collision*base.collision + timeout*base.timeout; V7 evaluator disables Dynamic Reward for all methods",
        "validation_astar_runtime_calls": result["validation_astar_runtime_calls"],
        "test_astar_runtime_calls": result["test_astar_runtime_calls"],
    }
    enhanced_path = run / "movingai64_formal_result.json"
    write_json_new(enhanced_path, enhanced)
    return output, enhanced_path, enhanced


def create_done(run, config_path, protocol_path, identity, method, seed):
    manifest_path = run / "run_manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed" or int(manifest.get("executed_interaction_count", -1)) != 900000:
        raise IntegrityError("completed V7 manifest is invalid")
    authority_result_path = run / ("formal_result_%s_seed%d.json" % (method, seed))
    result = read_json(authority_result_path)
    evaluation_path = run / "evaluation_metrics.csv"
    rows = read_evaluation_rows(evaluation_path)
    validation_count = sum(1 for row in rows if row.get("record_type") == "summary" and row.get("split") == "validation")
    test_summaries = [row for row in rows if row.get("record_type") == "summary" and row.get("split") == "test"]
    test_episodes = sum(1 for row in rows if row.get("record_type") == "episode" and row.get("split") == "test")
    if validation_count != 9 or len(test_summaries) != 1:
        raise IntegrityError("validation/test summary count mismatch")
    if int(result.get("validation_astar_runtime_calls", -1)) != 0 or int(result.get("test_astar_runtime_calls", -1)) != 0:
        raise IntegrityError("A* was used during validation/test")
    selected_checkpoint = Path(str(result["selected_checkpoint"])).resolve()
    if sha256_file(selected_checkpoint) != str(result["selected_checkpoint_sha256"]).upper():
        raise IntegrityError("selected checkpoint SHA mismatch")
    episode_path, result_path, enhanced = write_enriched_test_evidence(run, rows, identity["config"], method, seed, result)
    done = {
        "schema_version": "movingai64_done.v1",
        "status": "DONE",
        "method": method,
        "seed": seed,
        "authority_program_sha256": AUTHORITY_SHA256,
        "wrapper_program_sha256": sha256_file(Path(__file__).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": identity["config_sha256"],
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": identity["protocol_sha256"],
        "input_sha256": {name: value["sha256"] for name, value in identity["inputs"].items()},
        "map_hash_aggregate": map_hash_aggregate(identity["config"]),
        "training_interaction_budget": 900000,
        "executed_training_interactions": int(result["training_interactions"]),
        "validation_count": validation_count,
        "test_episode_count": test_episodes,
        "selected_checkpoint_path": str(selected_checkpoint),
        "selected_checkpoint_sha256": sha256_file(selected_checkpoint),
        "formal_test_result_path": str(result_path.resolve()),
        "formal_test_result_sha256": sha256_file(result_path),
        "authority_evaluation_csv_path": str(evaluation_path.resolve()),
        "authority_evaluation_csv_sha256": sha256_file(evaluation_path),
        "episode_csv_path": str(episode_path.resolve()),
        "episode_csv_sha256": sha256_file(episode_path),
        "training_astar_query_count": int(result.get("training_astar_calls", 0)),
        "validation_astar_runtime_calls": int(result.get("validation_astar_runtime_calls", 0)),
        "test_astar_runtime_calls": int(result.get("test_astar_runtime_calls", 0)),
        "final_metrics": enhanced["final_metrics"],
    }
    write_json_new(run / "DONE.json", done)
    return done


def main():
    parser = argparse.ArgumentParser(description="One MovingAI64 formal child using frozen Legacy V7")
    parser.add_argument("--config", type=Path, default=REVISION_DIR / "movingai64_config_confirmed.json")
    parser.add_argument("--protocol", type=Path, default=REVISION_DIR / "movingai64_protocol_confirmed.json")
    parser.add_argument("--method", choices=ALLOWED_METHODS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--confirm-formal-child", action="store_true")
    args = parser.parse_args()
    if not args.confirm_formal_child:
        raise ProtocolError("--confirm-formal-child is required")
    identity = validate_confirmed(args.config, args.protocol)
    validate_zero_drift(identity["config"])
    if args.seed not in identity["config"]["official_seeds"]:
        raise ProtocolError("seed is not confirmed")
    authority = load_authority_module()
    authority.SOURCE_DIR = Path(identity["inputs"]["map_split"]["path"]).resolve().parent
    authority.DEFAULT_CONFIG = Path(args.config).resolve()
    authority.DEFAULT_PROTOCOL = Path(args.protocol).resolve()
    config = authority.FrozenConfig(args.config)
    protocol = authority.FrozenProtocol(args.protocol)
    if args.device == "cuda" and not authority.torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable")
    run = authority.run_experiment(config, protocol, args.method, args.seed, authority.torch.device(args.device))
    done = create_done(run, args.config, args.protocol, identity, args.method, args.seed)
    print(json.dumps(done, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
