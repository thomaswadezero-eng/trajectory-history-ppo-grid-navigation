from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch

from algorithms import A2CAgent, D3QNAgent, DDQNAgent
from algorithms.ddqn import epsilon_by_interaction
from checkpoint.selection import CheckpointSelector, load_selected_checkpoint, save_checkpoint
from common.errors import IntegrityError, ProtocolError
from common.io_utils import sha256_file, write_json
from common.protocol import (
    FRAMEWORK_MANIFEST_PATH,
    FRAMEWORK_ROOT,
    PROTOCOL_PATH,
    ProtocolBundle,
    cspace_manifest,
    load_and_verify_tasks,
    method_config_path,
    runtime_identity,
    set_seed,
    verify_v7_authority,
)
from environment.navigation import MapStore, NavigationEnvironment, Observation
from evaluation.evaluator import Evaluator
from replay_buffer.uniform import UniformReplayBuffer
from run_logging import RunLogger


def _validate_output_root(root: Path, protocol: Mapping[str, Any]) -> None:
    expected = Path(str(protocol["output_root"])).resolve()
    if root.resolve() != expected:
        raise ProtocolError(f"output root is not the frozen comparison root: {root}")
    protected = [Path(value).resolve() for value in protocol["protected_output_roots"]]
    root_text = str(root.resolve()).lower()
    for item in protected:
        item_text = str(item).lower()
        if root_text == item_text or root_text.startswith(item_text + "\\") or item_text.startswith(root_text + "\\"):
            raise ProtocolError(f"comparison output overlaps protected output: {item}")


def _create_run_directory(root: Path, method: str, seed: int) -> Path:
    run = root / method / f"seed_{seed}" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run.mkdir(parents=True, exist_ok=False)
    (run / "models").mkdir()
    (run / "validation").mkdir()
    return run


def _snapshot_inputs(run: Path, method: str) -> None:
    snapshot = run / "protocol_snapshot"
    snapshot.mkdir()
    for source in (PROTOCOL_PATH, method_config_path(method), FRAMEWORK_MANIFEST_PATH):
        shutil.copy2(source, snapshot / source.name)


def _checkpoint_identity(bundle: ProtocolBundle, method: str, seed: int) -> Dict[str, Any]:
    return {
        "method": method,
        "seed": int(seed),
        "protocol_sha256": bundle.protocol_sha256,
        "config_sha256": bundle.config_sha256,
        "framework_sha256": bundle.framework_sha256,
    }


def _make_agent(bundle: ProtocolBundle, method: str, device: torch.device):
    encoder = bundle.protocol["network"]["shared_encoder"]
    if method == "DDQN":
        return DDQNAgent(encoder, bundle.config, device)
    if method == "D3QN":
        return D3QNAgent(encoder, bundle.config, device)
    if method == "A2C":
        return A2CAgent(encoder, bundle.config, device)
    raise ProtocolError(f"unsupported method: {method}")


def _validate_and_checkpoint(
    agent,
    evaluator: Evaluator,
    validation_tasks: Sequence[Mapping[str, Any]],
    interaction: int,
    run: Path,
    selector: CheckpointSelector,
    identity: Mapping[str, Any],
    logger: RunLogger,
) -> Dict[str, Any]:
    validation_directory = run / "validation" / f"interaction_{interaction:09d}"
    metrics = evaluator.evaluate(
        agent,
        validation_tasks,
        "validation",
        interaction,
        validation_directory,
        record_trajectories=False,
    )
    checkpoint = run / "models" / f"checkpoint_interaction_{interaction:09d}.pth"
    checkpoint_sha = save_checkpoint(checkpoint, agent, interaction, identity, metrics)
    selector.consider(checkpoint, checkpoint_sha, interaction, metrics)
    logger.log(
        f"validation interaction={interaction} success={metrics['success_rate']:.6f} "
        f"SPL={metrics['spl']:.6f} timeout={metrics['timeout_rate']:.6f}"
    )
    return metrics


def _run_value_based_training(
    method: str,
    seed: int,
    bundle: ProtocolBundle,
    tasks: Mapping[str, Sequence[Mapping[str, Any]]],
    run: Path,
    logger: RunLogger,
    device: torch.device,
    maps: MapStore,
) -> Tuple[Any, CheckpointSelector, int, Dict[str, Any]]:
    agent = _make_agent(bundle, method, device)
    algorithm = bundle.config["algorithm"]
    training = bundle.protocol["training"]
    budget = int(training["interaction_budget"])
    validation_interval = int(training["validation_interval_interactions"])
    task_rng = np.random.RandomState((seed + int(bundle.protocol["random_streams"]["training_task_seed_offset"])) % (2**32))
    exploration_rng = np.random.RandomState((seed + int(bundle.protocol["random_streams"]["exploration_seed_offset"])) % (2**32))
    replay_rng = np.random.RandomState((seed + int(bundle.protocol["random_streams"]["replay_seed_offset"])) % (2**32))
    environment = NavigationEnvironment(bundle.protocol, maps)
    replay = UniformReplayBuffer(
        run / "replay_buffer",
        capacity=int(algorithm["replay_capacity"]),
        spatial_shape=tuple(bundle.protocol["state"]["spatial_shape"]),
        history_size=int(bundle.protocol["state"]["history_size"]),
    )
    write_json(
        run / "replay_buffer_manifest.json",
        {
            "type": "uniform_disk_backed_memory_efficient_float32",
            "configured_transition_capacity": replay.capacity,
            "storage_slots": replay.slots,
            "allocated_bytes": replay.allocated_bytes,
            "prioritized": False,
            "n_step": False,
        },
    )
    evaluator = Evaluator(bundle.protocol, maps, method, seed)
    selector = CheckpointSelector(run / "models")
    identity = _checkpoint_identity(bundle, method, seed)
    interaction = 0
    episode = 0
    latest_update: Dict[str, Any] = {}
    try:
        while interaction < budget:
            episode += 1
            task = tasks["train"][int(task_rng.randint(0, len(tasks["train"])))]
            observation = environment.reset(task)
            done = False
            episode_reward = 0.0
            final: Dict[str, Any] = {}
            while not done and interaction < budget:
                mask = environment.action_mask()
                epsilon = epsilon_by_interaction(
                    interaction,
                    budget,
                    float(algorithm["exploration_initial_epsilon"]),
                    float(algorithm["exploration_final_epsilon"]),
                    float(algorithm["exploration_fraction"]),
                )
                action = agent.select_action(observation, mask, epsilon, exploration_rng)
                if not bool(mask[action]):
                    raise IntegrityError("training selected an illegal action")
                next_observation, reward, done, final = environment.step(action)
                next_mask = np.zeros(4, dtype=np.bool_) if done else environment.action_mask()
                replay.add(observation, action, reward, done, next_observation, next_mask)
                observation = next_observation
                interaction += 1
                episode_reward += reward
                if interaction >= int(algorithm["learning_starts"]) and interaction % int(algorithm["train_frequency"]) == 0:
                    batch = replay.sample(int(algorithm["batch_size"]), replay_rng)
                    latest_update = agent.update(batch)
                    logger.updates.write(
                        {
                            "interaction_count": interaction,
                            "update_count": agent.update_count,
                            "method": method,
                            **latest_update,
                            "epsilon": epsilon,
                            "replay_size": len(replay),
                            "target_update_count": agent.target_update_count,
                            "rollout_interactions": "",
                        }
                    )
                if interaction % int(algorithm["target_update_interval"]) == 0:
                    agent.hard_update_target()
                if interaction % validation_interval == 0:
                    _validate_and_checkpoint(
                        agent,
                        evaluator,
                        tasks["validation"],
                        interaction,
                        run,
                        selector,
                        identity,
                        logger,
                    )
            logger.episodes.write(
                {
                    "episode": episode,
                    "interaction_count": interaction,
                    "reward": episode_reward,
                    "success": int(bool(final.get("success", False))) if done else 0,
                    "collision": int(bool(final.get("collision", False))) if done else 0,
                    "timeout": int(bool(final.get("timeout", False))) if done else 0,
                    "truncated": int(not done),
                    "steps": environment.builder.step_count,
                    "method": method,
                    "seed": seed,
                }
            )
        replay.flush()
    finally:
        replay.close()
    audit = {
        "optimizer_updates": agent.update_count,
        "target_network_updates": agent.target_update_count,
        "replay_total_added": replay.total_added,
        "replay_final_size": len(replay),
        "episodes_started": episode,
        "latest_update": latest_update,
    }
    return agent, selector, interaction, audit


def _run_a2c_training(
    method: str,
    seed: int,
    bundle: ProtocolBundle,
    tasks: Mapping[str, Sequence[Mapping[str, Any]]],
    run: Path,
    logger: RunLogger,
    device: torch.device,
    maps: MapStore,
) -> Tuple[Any, CheckpointSelector, int, Dict[str, Any]]:
    agent = _make_agent(bundle, method, device)
    algorithm = bundle.config["algorithm"]
    training = bundle.protocol["training"]
    budget = int(training["interaction_budget"])
    interval = int(training["validation_interval_interactions"])
    n_envs = int(algorithm["synchronous_environments"])
    n_steps = int(algorithm["rollout_steps_per_environment"])
    rollout_interactions = n_envs * n_steps
    if budget % rollout_interactions != 0 or interval % rollout_interactions != 0:
        raise ProtocolError("A2C rollout does not divide the budget and validation interval exactly")
    task_rng = np.random.RandomState((seed + int(bundle.protocol["random_streams"]["training_task_seed_offset"])) % (2**32))
    environments = [NavigationEnvironment(bundle.protocol, maps) for _ in range(n_envs)]
    observations: List[Observation] = []
    episode_rewards = np.zeros(n_envs, dtype=np.float64)
    episode_numbers = np.zeros(n_envs, dtype=np.int64)
    global_episode = 0
    for environment in environments:
        task = tasks["train"][int(task_rng.randint(0, len(tasks["train"])))]
        observations.append(environment.reset(task))
    evaluator = Evaluator(bundle.protocol, maps, method, seed)
    selector = CheckpointSelector(run / "models")
    identity = _checkpoint_identity(bundle, method, seed)
    interaction = 0
    latest_update: Dict[str, Any] = {}
    while interaction < budget:
        spatial = np.zeros((n_steps, n_envs, 5, 30, 30), dtype=np.float32)
        history = np.zeros((n_steps, n_envs, 10), dtype=np.float32)
        masks = np.zeros((n_steps, n_envs, 4), dtype=np.bool_)
        actions = np.zeros((n_steps, n_envs), dtype=np.int64)
        rewards = np.zeros((n_steps, n_envs), dtype=np.float32)
        dones = np.zeros((n_steps, n_envs), dtype=np.float32)
        values = np.zeros((n_steps, n_envs), dtype=np.float32)
        for step in range(n_steps):
            current_masks = np.stack([environment.action_mask() for environment in environments])
            sampled_actions, _, sampled_values = agent.act_batch(observations, current_masks)
            spatial[step] = np.stack([observation.spatial for observation in observations])
            history[step] = np.stack([observation.history for observation in observations])
            masks[step] = current_masks
            actions[step] = sampled_actions
            values[step] = sampled_values
            for worker, environment in enumerate(environments):
                action = int(sampled_actions[worker])
                if not bool(current_masks[worker, action]):
                    raise IntegrityError("A2C sampling selected an illegal action")
                next_observation, reward, done, info = environment.step(action)
                rewards[step, worker] = reward
                dones[step, worker] = float(done)
                episode_rewards[worker] += reward
                observations[worker] = next_observation
                if done:
                    global_episode += 1
                    episode_numbers[worker] += 1
                    logger.episodes.write(
                        {
                            "episode": global_episode,
                            "interaction_count": interaction + (step + 1) * n_envs,
                            "reward": episode_rewards[worker],
                            "success": int(bool(info["success"])),
                            "collision": int(bool(info["collision"])),
                            "timeout": int(bool(info["timeout"])),
                            "truncated": 0,
                            "steps": environment.builder.step_count,
                            "method": method,
                            "seed": seed,
                        }
                    )
                    episode_rewards[worker] = 0.0
                    task = tasks["train"][int(task_rng.randint(0, len(tasks["train"])))]
                    observations[worker] = environment.reset(task)
        final_values = agent.values(observations)
        interaction += rollout_interactions
        rollout = {
            "spatial": spatial,
            "history": history,
            "masks": masks,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "final_values": final_values,
        }
        latest_update = agent.update(rollout, interaction, budget)
        logger.updates.write(
            {
                "interaction_count": interaction,
                "update_count": agent.update_count,
                "method": method,
                **latest_update,
                "epsilon": "",
                "replay_size": "",
                "target_update_count": "",
                "rollout_interactions": rollout_interactions,
            }
        )
        if interaction % interval == 0:
            _validate_and_checkpoint(
                agent,
                evaluator,
                tasks["validation"],
                interaction,
                run,
                selector,
                identity,
                logger,
            )
    write_json(
        run / "a2c_partial_episode_state_at_budget.json",
        {
            "worker_episode_counts": episode_numbers.tolist(),
            "unfinished_episode_rewards": episode_rewards.tolist(),
            "unfinished_episode_steps": [environment.builder.step_count for environment in environments],
            "note": "Budget is defined by transitions; partial episodes are not converted into terminal outcomes.",
        },
    )
    audit = {
        "optimizer_updates": agent.update_count,
        "rollout_interactions": rollout_interactions,
        "completed_episodes": global_episode,
        "latest_update": latest_update,
    }
    return agent, selector, interaction, audit


def run_formal_experiment(bundle: ProtocolBundle, method: str, seed: int, device: torch.device) -> Path:
    if seed not in bundle.protocol["official_seeds"]:
        raise ProtocolError("seed is not in the frozen official seed set")
    verify_v7_authority(bundle.protocol)
    tasks, split_audit = load_and_verify_tasks(bundle.protocol)
    root = Path(str(bundle.protocol["output_root"])).resolve()
    _validate_output_root(root, bundle.protocol)
    run = _create_run_directory(root, method, seed)
    _snapshot_inputs(run, method)
    logger = RunLogger(run)
    started = time.time()
    identity = _checkpoint_identity(bundle, method, seed)
    manifest: Dict[str, Any] = {
        "schema_version": "comparison_baselines.run.v1",
        "status": "running",
        **identity,
        "run_directory": str(run),
        "output_root": str(root),
        "runtime": runtime_identity(device),
        "interaction_budget": int(bundle.protocol["training"]["interaction_budget"]),
        "executed_interaction_count": 0,
        "dynamic_reward_enabled": False,
        "astar_training_calls": 0,
        "validation_astar_runtime_calls": 0,
        "test_astar_runtime_calls": 0,
        "curriculum_enabled": False,
        "planning_head_present": False,
        "test_parameter_updates": 0,
        "failure_reason": None,
    }
    write_json(run / "run_manifest.json", manifest)
    try:
        set_seed(seed)
        cspace = cspace_manifest(tasks)
        write_json(run / "cspace_file_manifest.json", cspace)
        maps = MapStore(int(bundle.protocol["environment"]["map_size"]))
        logger.log(f"start method={method} seed={seed} budget={manifest['interaction_budget']} device={device}")
        if method in ("DDQN", "D3QN"):
            agent, selector, interaction, training_audit = _run_value_based_training(
                method, seed, bundle, tasks, run, logger, device, maps
            )
        else:
            agent, selector, interaction, training_audit = _run_a2c_training(
                method, seed, bundle, tasks, run, logger, device, maps
            )
        expected_validations = int(bundle.protocol["training"]["interaction_budget"]) // int(
            bundle.protocol["training"]["validation_interval_interactions"]
        )
        if interaction != int(bundle.protocol["training"]["interaction_budget"]):
            raise IntegrityError("formal run did not execute the exact transition budget")
        if len(selector.history) != expected_validations or selector.best_path is None:
            raise IntegrityError("validation/checkpoint count is incomplete")
        selected = load_selected_checkpoint(selector.best_path, agent, device, identity)
        evaluator = Evaluator(bundle.protocol, maps, method, seed)
        test_metrics = evaluator.evaluate(
            agent,
            tasks["test"],
            "test",
            int(selected["interaction_count"]),
            run / "test",
            record_trajectories=True,
        )
        if test_metrics["parameter_updates"] != 0 or test_metrics["astar_runtime_calls"] != 0:
            raise IntegrityError("frozen test isolation failed")
        manifest.update(
            {
                "status": "completed",
                "executed_interaction_count": interaction,
                "selected_checkpoint": str(selector.best_path),
                "selected_checkpoint_sha256": sha256_file(selector.best_path),
                "selected_checkpoint_interaction": int(selected["interaction_count"]),
                "validation_count": len(selector.history),
                "training_audit": training_audit,
                "split_audit": split_audit,
                "cspace_aggregate_sha256": cspace["aggregate_sha256"],
                "test_metrics": test_metrics,
                "runtime_seconds": time.time() - started,
            }
        )
        write_json(run / "run_manifest.json", manifest)
        result = {
            "schema_version": "comparison_baselines.formal_result.v1",
            **identity,
            "training_interactions": interaction,
            "selected_checkpoint": str(selector.best_path),
            "selected_checkpoint_sha256": sha256_file(selector.best_path),
            "selected_checkpoint_interaction": int(selected["interaction_count"]),
            "dynamic_reward_enabled": False,
            "astar_training_calls": 0,
            "validation_astar_runtime_calls": 0,
            "test_astar_runtime_calls": 0,
            "curriculum_enabled": False,
            "test_parameter_updates": 0,
            "test": test_metrics,
        }
        write_json(run / f"formal_result_{method}_seed{seed}.json", result)
        logger.log(f"completed method={method} seed={seed} success={test_metrics['success_rate']:.6f} SPL={test_metrics['spl']:.6f}")
        return run
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["failure_reason"] = f"{type(exc).__name__}: {exc}"
        manifest["runtime_seconds"] = time.time() - started
        write_json(run / "run_manifest.json", manifest)
        raise
    finally:
        logger.close()
