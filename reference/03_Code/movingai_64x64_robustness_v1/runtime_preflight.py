#!/usr/bin/env python
"""User-run runtime preflight. Codex must never execute this file."""

from __future__ import print_function

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from protocol_utils import RESULT_ROOT, REVISION_DIR, IntegrityError, ProtocolError, map_hash_aggregate, sha256_file, validate_confirmed, write_json_new
from run_movingai64_wrapper import load_authority_module, validate_zero_drift


def formal_snapshot():
    result = []
    for method in ("PPO", "PPO_Reward", "FULL"):
        root = RESULT_ROOT / method
        if root.exists():
            result.extend(str(path.resolve()) for path in root.rglob("*") if path.is_file())
    return sorted(result)


def code_identity():
    names = (
        "protocol_utils.py", "movingai64_task_rules.py", "audit_movingai64_task_candidates.py",
        "generate_movingai64_tasks.py", "confirm_protocol.py",
        "runtime_preflight.py", "run_movingai64_wrapper.py", "run_movingai64_robustness.py",
        "aggregate_and_analyze.py", "feasibility_audit.py", "static_audit.py",
    )
    return {name: sha256_file(REVISION_DIR / name) for name in names}


def main():
    parser = argparse.ArgumentParser(description="MovingAI64 runtime preflight; creates no formal run")
    parser.add_argument("--config", type=Path, default=REVISION_DIR / "movingai64_config_confirmed.json")
    parser.add_argument("--protocol", type=Path, default=REVISION_DIR / "movingai64_protocol_confirmed.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--confirm-preflight", action="store_true")
    args = parser.parse_args()
    if not args.confirm_preflight:
        raise ProtocolError("--confirm-preflight is required")
    if sys.version_info[:3] != (3, 8, 10):
        raise ProtocolError("runtime requires Python 3.8.10; found %s" % (sys.version.split()[0],))
    identity = validate_confirmed(args.config, args.protocol)
    validate_zero_drift(identity["config"])
    before = formal_snapshot()
    preflight_root = RESULT_ROOT / "preflight"
    preflight_root.mkdir(parents=True, exist_ok=True)
    marker = preflight_root / "PREFLIGHT_PASS.json"
    if marker.exists():
        raise FileExistsError("preflight marker already exists and will not be overwritten: %s" % marker)
    session = preflight_root / ("preflight_%s" % datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    session.mkdir(parents=False, exist_ok=False)
    (session / "models").mkdir()
    (session / "trajectories").mkdir()
    authority = load_authority_module()
    authority.SOURCE_DIR = Path(identity["inputs"]["map_split"]["path"]).resolve().parent
    config = authority.FrozenConfig(args.config)
    protocol = authority.FrozenProtocol(args.protocol)
    if args.device == "cuda" and not authority.torch.cuda.is_available():
        raise ProtocolError("CUDA requested but unavailable")
    device = authority.torch.device(args.device)
    train_tasks, validation_tasks, _ = authority.load_tasks(config)
    maps = authority.MapStore(config.integer("environment.map_size"))
    model = authority.ActorCritic(config)
    agent = authority.PPOAgent(config, model, device)
    environment = authority.NavigationEnvironment(config, maps, reward_enabled=True)
    observation = environment.reset(train_tasks[0])
    if list(observation.spatial.shape) != [5, 64, 64] or list(observation.history.shape) != [10]:
        raise IntegrityError("observation shape mismatch")
    mask = environment.action_mask()
    if mask.shape != (4,) or not bool(mask.any()):
        raise IntegrityError("action mask mismatch")
    supervisor = authority.AStarPlanningSupervisor(config, np.random.RandomState(123))
    target, audit = supervisor.collect_target(environment.grid, str(train_tasks[0]["cspace_file"]), environment.builder.position, environment.builder.goal, mask, environment.builder.max_steps, 0)
    if target is None or not audit.get("target_available"):
        raise IntegrityError("FULL training-only A* target path did not execute")
    action, log_probability, value, _ = agent.act(observation, mask, deterministic=False)
    old = observation
    observation, reward, done, _ = environment.step(action)
    next_value = 0.0 if done else agent.value(observation)
    buffer = authority.RolloutBuffer()
    buffer.add(spatial=old.spatial, history=old.history, mask=mask, action=action, log_prob=log_probability, reward=reward, done=done, value=value, next_value=next_value, planning_target=target)
    losses = agent.update(buffer, supervisor)
    if not all(np.isfinite(float(value)) for value in losses.values()):
        raise IntegrityError("nonfinite minimal update")
    logger = authority.RunLogger(session)
    try:
        evaluator = authority.Evaluator(config, model, device, maps, logger, "FULL", int(identity["config"]["official_seeds"][0]))
        validation = evaluator.evaluate(validation_tasks[:1], "validation", 0)
    finally:
        logger.close()
    if evaluator.runtime_astar_calls != 0 or int(validation["astar_runtime_calls"]) != 0:
        raise IntegrityError("validation attempted runtime A*")
    after = formal_snapshot()
    if before != after:
        raise IntegrityError("formal result area changed during preflight")
    result = {
        "schema_version": "movingai64_preflight.v1",
        "status": "PASS",
        "python": sys.version,
        "torch_version": authority.torch.__version__,
        "cuda_build": authority.torch.version.cuda,
        "cuda_available": bool(authority.torch.cuda.is_available()),
        "requested_device": args.device,
        "current_gpu": authority.torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "authority_program_sha256": sha256_file(Path(authority.SCRIPT_PATH)),
        "config_sha256": identity["config_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "input_sha256": {name: value["sha256"] for name, value in identity["inputs"].items()},
        "map_hash_aggregate": map_hash_aggregate(identity["config"]),
        "code_sha256": code_identity(),
        "method_set": ["PPO", "PPO_Reward", "FULL"],
        "observation_shape": [5, 64, 64],
        "history_shape": [10],
        "action_mask_legal_count": int(mask.sum()),
        "minimal_optimizer_update": "PASS",
        "full_training_only_astar_target": "PASS",
        "validation_episode_count": 1,
        "validation_astar_runtime_calls": 0,
        "formal_training_interactions": 0,
        "formal_result_area_unchanged": True,
        "session_directory": str(session.resolve()),
    }
    write_json_new(session / "preflight_result.json", result)
    write_json_new(marker, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
