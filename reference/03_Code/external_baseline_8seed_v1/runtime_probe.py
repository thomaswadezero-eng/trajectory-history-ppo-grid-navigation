#!/usr/bin/env python
"""Method-local runtime probe used only by ``runtime_preflight.py``.

This is deliberately a separate process: framework imports, CUDA state and any
temporary environment-variable changes cannot leak between methods or into the
later formal-training children.  It is never imported by the formal runner.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


REVISION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REVISION_ROOT.parents[1]
PREFLIGHT_ROOT = Path(r"redacted_local_root\revision_external_baselines_8seed_v1\preflight")


def _inside_preflight(path: Path) -> Path:
    resolved = path.resolve()
    # pathlib.Path.is_relative_to was added in Python 3.9.  DDQN, D3QN and
    # C51 are deliberately pinned to the legacy Python 3.8.10 interpreter, so
    # use the equivalent Python-3.8-compatible containment check here.
    try:
        resolved.relative_to(PREFLIGHT_ROOT.resolve())
    except ValueError:
        raise RuntimeError(f"probe output must stay inside preflight root: {resolved}")
    return resolved


def _runtime_banner(method: str, requested_device: str, work: Path, result_path: Path) -> None:
    cuda_key = "CUDA_VISIBLE_DEVICES"
    fields = {
        "method": method,
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "working_directory": os.getcwd(),
        "requested_device": requested_device,
        "cuda_visible_devices_present": cuda_key in os.environ,
        "cuda_visible_devices": os.environ.get(cuda_key),
        "pythonpath_present": "PYTHONPATH" in os.environ,
        "pythonpath": os.environ.get("PYTHONPATH"),
        "sys_path": list(sys.path),
        "work_dir": str(work),
        "result_json": str(result_path),
    }
    print("[RUNTIME PROBE] " + json.dumps(fields, ensure_ascii=False), flush=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _value_probe(method: str, device_name: str, work: Path) -> dict[str, Any]:
    framework = PROJECT_ROOT / "03_Code" / "comparison_baselines_v1"
    sys.path.insert(0, str(framework))
    sys.modules.pop("common", None)
    print("[IMPORT PATH] " + json.dumps({
        "framework_path": str(framework), "sys_path_head": sys.path[:3]
    }, ensure_ascii=False), flush=True)

    import numpy as np
    import torch
    from common.protocol import (
        FRAMEWORK_MANIFEST_PATH,
        PROTOCOL_PATH,
        load_and_verify_tasks,
        load_bundle,
        method_config_path,
        verify_v7_authority,
    )
    from environment.navigation import MapStore, NavigationEnvironment
    from evaluation.evaluator import Evaluator
    from runner import _make_agent

    cuda_available = bool(torch.cuda.is_available())
    print("[TORCH] " + json.dumps({
        "torch_version": str(torch.__version__),
        "torch_cuda_build": str(torch.version.cuda),
        "cuda_available": cuda_available,
        "requested_device": device_name,
        "framework_path": str(framework),
    }, ensure_ascii=False), flush=True)
    if device_name == "cuda" and not cuda_available:
        raise RuntimeError(
            f"requested device=cuda but torch.cuda.is_available() is False in {sys.executable}"
        )
    device = torch.device(device_name)
    bundle = load_bundle(method)
    print("[FROZEN INPUTS] " + json.dumps({
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": bundle.protocol_sha256,
        "config_path": str(method_config_path(method)),
        "config_sha256": bundle.config_sha256,
        "framework_manifest_path": str(FRAMEWORK_MANIFEST_PATH),
        "framework_sha256": bundle.framework_sha256,
    }, ensure_ascii=False), flush=True)
    authority = verify_v7_authority(bundle.protocol)
    tasks, split_audit = load_and_verify_tasks(bundle.protocol)
    maps = MapStore(int(bundle.protocol["environment"]["map_size"]))
    environment = NavigationEnvironment(bundle.protocol, maps)
    observation = environment.reset(tasks["train"][0])
    mask = environment.action_mask()
    print("[OBSERVATION] " + json.dumps({
        "spatial_shape": list(observation.spatial.shape),
        "history_shape": list(observation.history.shape),
        "action_mask": mask.tolist(),
        "legal_action_count": int(mask.sum()),
    }), flush=True)
    agent = _make_agent(bundle, method, device)
    action = int(agent.deterministic_action(observation, mask))
    print(f"[FORWARD] action={action}", flush=True)
    next_observation, reward, done, _ = environment.step(action)
    next_mask = environment.action_mask()
    count = 2
    batch = {
        "spatial": np.repeat(observation.spatial[None], count, axis=0),
        "history": np.repeat(observation.history[None], count, axis=0),
        "actions": np.full(count, action, dtype=np.int64),
        "rewards": np.full(count, reward, dtype=np.float32),
        "dones": np.full(count, float(done), dtype=np.float32),
        "next_spatial": np.repeat(next_observation.spatial[None], count, axis=0),
        "next_history": np.repeat(next_observation.history[None], count, axis=0),
        "next_masks": np.repeat(next_mask[None], count, axis=0),
    }
    update = agent.update(batch)
    print("[REPLAY UPDATE] " + json.dumps(_json_safe(update), ensure_ascii=False), flush=True)
    evaluator = Evaluator(bundle.protocol, maps, method, 42)
    evaluation = evaluator.evaluate(
        agent, tasks["validation"][:1], "validation", 0,
        work / "one_episode_evaluation", record_trajectories=False,
    )
    print("[VALIDATION PROBE] " + json.dumps({
        "episode_count": int(evaluation["episode_count"]),
        "success_rate": evaluation.get("success_rate"),
        "spl": evaluation.get("spl"),
    }, ensure_ascii=False), flush=True)
    return {
        "status": "PASS",
        "method": method,
        "device": str(device),
        "observation_shapes": {
            "spatial": list(observation.spatial.shape),
            "history": list(observation.history.shape),
        },
        "action_dimension": 4,
        "legal_action_count": int(mask.sum()),
        "forward_action": action,
        "optimizer_update": _json_safe(update),
        "evaluator_episode_count": int(evaluation["episode_count"]),
        "config_sha256": bundle.config_sha256,
        "protocol_sha256": bundle.protocol_sha256,
        "framework_sha256": bundle.framework_sha256,
        "v7_authority_sha256": authority,
        "split_audit": split_audit,
        "preflight_environment_interactions": 1,
        "formal_training_interactions": 0,
    }


def _c51_probe(device_name: str, work: Path) -> dict[str, Any]:
    source = PROJECT_ROOT / "03_Code" / "comparison_baselines_c51"
    sys.path.insert(0, str(source))

    import copy
    import numpy as np
    from barn_protocol import BarnEnvironment, MapStore, load_frozen_tasks
    from c51_algorithm import C51Agent
    import run_c51_barn as legacy

    device = legacy.resolve_device(device_name)
    config, config_sha = legacy.load_config(legacy.DEFAULT_CONFIG)
    tasks = load_frozen_tasks(config, legacy.PROJECT_ROOT, ("train", "validation"))
    maps = MapStore(30)
    environment = BarnEnvironment(config, maps)
    observation = environment.reset(tasks["train"][0])
    mask = environment.action_mask()
    agent = C51Agent(config, device, 42)
    action = int(agent.deterministic_action(observation, mask))
    next_observation, reward, done, _ = environment.step(action)
    next_mask = mask if done else environment.action_mask()
    count = 2
    batch = {
        "spatial": np.repeat(observation.spatial[None], count, axis=0),
        "history": np.repeat(observation.history[None], count, axis=0),
        "actions": np.full(count, action, dtype=np.int64),
        "rewards": np.full(count, reward, dtype=np.float32),
        "dones": np.full(count, float(done), dtype=np.float32),
        "next_spatial": np.repeat(next_observation.spatial[None], count, axis=0),
        "next_history": np.repeat(next_observation.history[None], count, axis=0),
        "current_masks": np.repeat(mask[None], count, axis=0),
        "next_masks": np.repeat(next_mask[None], count, axis=0),
    }
    update = agent.update(batch)
    probe_config = copy.deepcopy(config)
    probe_config["profiles"]["formal"]["validation_episodes"] = 1
    evaluation = legacy.evaluate(
        agent, tasks["validation"][:1], probe_config, maps,
        split="validation", seed=42, interaction=0,
        output_directory=work / "one_episode_evaluation",
        record_trajectories=False,
    )
    return {
        "status": "PASS",
        "method": "C51",
        "device": str(device),
        "observation_shapes": {
            "spatial": list(observation.spatial.shape),
            "history": list(observation.history.shape),
        },
        "action_dimension": 4,
        "legal_action_count": int(mask.sum()),
        "forward_action": action,
        "optimizer_update": _json_safe(update),
        "evaluator_episode_count": int(evaluation["episode_count"]),
        "config_sha256": config_sha,
        "preflight_environment_interactions": 1,
        "formal_training_interactions": 0,
    }


def _maskable_probe(device_name: str, work: Path) -> dict[str, Any]:
    source = PROJECT_ROOT / "03_Code" / "external_baselines"
    sys.path.insert(0, str(source))

    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.monitor import Monitor
    import sb3_external_baseline as legacy
    from sb3_baseline.audit import load_config, verify_environment, verify_frozen_files
    from sb3_baseline.core import BarnNavigationEnv
    from sb3_baseline.evaluation import evaluate_tasks
    from sb3_baseline.model import maskable_ppo_kwargs

    environment_identity = verify_environment(device_name)
    config = load_config(legacy.DEFAULT_CONFIG)
    frozen_hashes = verify_frozen_files(legacy.PROJECT_ROOT, config)
    train_tasks, validation_tasks, test_tasks = legacy.load_all_tasks(config)
    snapshot = legacy.verify_cspace_snapshot(config, (*train_tasks, *validation_tasks, *test_tasks))
    raw_environment = BarnNavigationEnv(train_tasks, "train", 42, training_sampling=True)
    observation, _ = raw_environment.reset()
    mask = raw_environment.action_masks()
    kwargs = maskable_ppo_kwargs(42, device_name)
    # A preflight-only rollout size exercises the unchanged SB3 train path and
    # optimizer without starting or approximating a formal 900k run.
    kwargs.update({"n_steps": 8, "batch_size": 8, "n_epochs": 1, "verbose": 0})
    monitored = Monitor(raw_environment)
    model = MaskablePPO(env=monitored, **kwargs)
    action, _ = model.predict(observation, action_masks=mask, deterministic=True)
    model.learn(total_timesteps=8, use_masking=True, reset_num_timesteps=True)
    evaluation = evaluate_tasks(
        model=model, tasks=validation_tasks[:1], split="validation", seed=42,
        output_dir=work / "one_episode_evaluation", checkpoint_interaction=0,
        record_trajectories=False,
    )
    monitored.close()
    return {
        "status": "PASS",
        "method": "MaskablePPO",
        "device": device_name,
        "environment": environment_identity,
        "observation_shapes": {
            "spatial": list(observation["spatial"].shape),
            "history": list(observation["history"].shape),
        },
        "action_dimension": int(raw_environment.action_space.n),
        "legal_action_count": int(mask.sum()),
        "forward_action": int(action),
        "optimizer_update_count": int(model._n_updates),
        "evaluator_episode_count": int(evaluation["episode_count"]),
        "config_sha256": legacy.sha256_file(legacy.DEFAULT_CONFIG),
        "frozen_input_sha256": frozen_hashes,
        "cspace_aggregate_sha256": snapshot["aggregate_sha256"],
        "preflight_environment_interactions": 8,
        "formal_training_interactions": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One isolated external-baseline runtime probe")
    parser.add_argument("--method", choices=("DDQN", "D3QN", "C51", "MaskablePPO"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--confirm-runtime-probe", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_runtime_probe:
        raise RuntimeError("runtime probe requires explicit confirmation")
    work = _inside_preflight(args.work_dir)
    result_path = _inside_preflight(args.result_json)
    _runtime_banner(args.method, args.device, work, result_path)
    work.mkdir(parents=True, exist_ok=False)
    if result_path.exists():
        raise FileExistsError(result_path)
    if args.method in ("DDQN", "D3QN"):
        result = _value_probe(args.method, args.device, work)
    elif args.method == "C51":
        result = _c51_probe(args.device, work)
    else:
        result = _maskable_probe(args.device, work)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(_json_safe(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(result_path)


if __name__ == "__main__":
    main()
