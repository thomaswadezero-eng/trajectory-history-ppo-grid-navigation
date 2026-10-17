#!/usr/bin/env python
"""User-invoked runtime preflight; never called automatically by Codex or runner."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

import component_ablation_8seed_v1 as impl


SCRIPT_PATH = Path(__file__).resolve()
MARKER = Path(r"redacted_local_root\revision_component_ablation_8seed_v1\preflight\RUNTIME_PREFLIGHT_DONE.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual one-transition/one-update component-ablation runtime preflight")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--confirm-runtime-preflight", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_runtime_preflight:
        raise impl.ProtocolError("runtime preflight requires --confirm-runtime-preflight")
    core, config, protocol = impl.prepare()
    if args.device == "cuda" and not core.torch.cuda.is_available():
        raise impl.ProtocolError("CUDA requested but unavailable")
    input_hashes = core.verify_inputs(config)
    train_tasks, _, _ = core.load_tasks(config)
    reward_checks = {}
    expected_applied = {
        "REWARD_WO_PROGRESS": (0.0, -0.02, -0.03, -0.04),
        "REWARD_WO_REVISIT": (0.05, 0.0, -0.03, -0.04),
        "REWARD_WO_OSCILLATION": (0.05, -0.02, 0.0, -0.04),
        "REWARD_WO_STAGNATION": (0.05, -0.02, -0.03, 0.0),
    }
    for method, expected in expected_applied.items():
        record = core.RewardRecord(
            raw_distance_reward=0.04,
            raw_escape_reward=0.01,
            raw_revisit_penalty=-0.02,
            raw_oscillation_penalty=-0.03,
            raw_stagnation_penalty=-0.04,
        )
        core.apply_dynamic_reward_switches(record, impl.CONFIGURATIONS[method]["reward_components"])
        actual = (record.applied_progress_reward, record.revisit_penalty, record.oscillation_penalty, record.stagnation_penalty)
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-15):
            raise impl.IntegrityError(f"reward switch runtime preflight failed for {method}: {actual}")
        reward_checks[method] = {"actual": actual, "expected": expected, "pass": True}

    core.set_seed(42)
    device = core.torch.device(args.device)
    maps = core.MapStore(config.integer("environment.map_size"))
    method = "REWARD_WO_PROGRESS"
    environment = core.NavigationEnvironment(config, maps, impl.CONFIGURATIONS[method]["reward_components"])
    observation = environment.reset(train_tasks[0])
    model = core.ActorCritic(config)
    agent = core.PPOAgent(config, model, device)
    mask = environment.action_mask()
    action, log_probability, value, _ = agent.act(observation, mask, deterministic=False)
    old_observation = observation
    observation, reward, done, _ = environment.step(action)
    next_value = 0.0 if done else agent.value(observation)
    buffer = core.RolloutBuffer()
    buffer.add(
        spatial=old_observation.spatial,
        history=old_observation.history,
        mask=mask,
        action=action,
        log_prob=log_probability,
        reward=reward,
        done=done,
        value=value,
        next_value=next_value,
        planning_target=None,
    )
    before_updates = agent.update_count
    losses = agent.update(buffer, None)
    if agent.update_count != before_updates + 1:
        raise impl.IntegrityError("exactly one isolated PPO update was not executed")
    if not all(np.isfinite(float(value)) for value in losses.values()):
        raise impl.IntegrityError("non-finite loss in runtime preflight")

    with tempfile.TemporaryDirectory(prefix="component_ablation_runtime_preflight_") as temporary:
        run = Path(temporary)
        (run / "trajectories").mkdir()
        logger = core.RunLogger(run)
        try:
            evaluator = core.Evaluator(config, model, device, maps, logger, method, 42)
            metrics = evaluator.evaluate(train_tasks[:1], "runtime_preflight", 1)
            if evaluator.runtime_astar_calls != 0 or int(metrics["astar_runtime_calls"]) != 0:
                raise impl.IntegrityError("evaluation A* runtime call detected")
        finally:
            logger.close()

    payload = {
        "schema_version": "component_ablation_runtime_preflight.v1",
        "status": "PASS",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "preflight_program_sha256": impl.sha256_file(SCRIPT_PATH),
        "program_sha256": impl.sha256_file(impl.SCRIPT_PATH),
        "config_sha256": config.sha256,
        "protocol_sha256": protocol.sha256,
        "input_sha256": input_hashes,
        "environment_transitions": 1,
        "ppo_optimizer_updates": 1,
        "formal_training_interactions": 0,
        "formal_validation_episodes": 0,
        "formal_test_episodes": 0,
        "reward_component_integrity": reward_checks,
        "astar_training_queries": 0,
        "evaluation_astar_runtime_calls": 0,
        "curriculum": "absent",
        "escape_independent_reward": "absent",
    }
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    temporary_marker = MARKER.with_suffix(".json.tmp")
    if temporary_marker.exists():
        raise FileExistsError(temporary_marker)
    temporary_marker.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_marker.replace(MARKER)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
