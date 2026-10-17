#!/usr/bin/env python
"""User-only atomic generator for deterministic Moving AI 64x64 tasks."""

from __future__ import print_function

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

from movingai64_task_rules import CANDIDATE_HORIZONS, CROP_RULE_ID, SUBSETS, SPLITS, binary_crop, collect_candidates, parse_map, select_tasks, shortest_path, split_source_maps
from protocol_utils import REVISION_DIR, IntegrityError, ProtocolError, read_json, sha256_file, write_json_new


def canonical_json_sha256(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest().upper()


def task_target_identity(protocol):
    return {
        "source_map_split_rule": protocol["data"]["source_map_split_rule"],
        "draft_source_map_quotas": protocol["data"]["draft_source_map_quotas"],
        "draft_task_targets": protocol["data"]["draft_task_targets"],
        "maximum_tasks_per_source_map": protocol["data"]["maximum_tasks_per_source_map"],
    }


def validate_candidate_audit(report_path, protocol, horizon):
    path = Path(report_path).resolve()
    report = read_json(path)
    evidence = protocol.get("candidate_audit_evidence", {})
    if report.get("schema_version") != "movingai64_candidate_audit.v2" or report.get("status") != "PASS" or report.get("candidate_enumeration", {}).get("horizon_filter_applied") is not False:
        raise ProtocolError("candidate audit must PASS after horizon-free reachable enumeration")
    if sha256_file(path) != evidence.get("sha256") or str(path).lower() != str(Path(evidence.get("path", "")).resolve()).lower():
        raise ProtocolError("candidate audit report does not match the reviewed draft identity")
    if int(evidence.get("selected_horizon", -1)) != int(horizon):
        raise ProtocolError("candidate audit evidence is not bound to the reviewed horizon")
    if list(report.get("candidate_horizons", [])) != list(CANDIDATE_HORIZONS) or int(horizon) not in CANDIDATE_HORIZONS:
        raise ProtocolError("candidate horizon set mismatch")
    identity = report.get("identity", {})
    if identity.get("raw_root", "").lower() != str(Path(protocol["data"]["raw_root"]).resolve()).lower():
        raise ProtocolError("candidate audit raw-root identity mismatch")
    if int(identity.get("construction_seed", -1)) != int(protocol["data"]["construction_seed"]):
        raise ProtocolError("candidate audit construction seed mismatch")
    if identity.get("crop_rule_id") != CROP_RULE_ID or identity.get("source_map_split_rule") != protocol["data"]["source_map_split_rule"]:
        raise ProtocolError("candidate audit crop/split rule mismatch")
    if identity.get("task_target_identity_sha256") != canonical_json_sha256(task_target_identity(protocol)):
        raise ProtocolError("candidate audit task target identity mismatch")
    horizon_record = report.get("quota_feasibility_by_horizon", {}).get(str(horizon), {})
    if horizon_record.get("quota_feasibility_status") != "QUOTA_FEASIBLE_AT_HORIZON":
        raise ProtocolError("QUOTA_NOT_FEASIBLE_AT_HORIZON")
    return path, report, horizon_record


def safe_cleanup_staging(staging, final_root):
    staging = Path(staging).resolve()
    final_root = Path(final_root).resolve()
    expected_prefix = final_root.name + ".__staging_"
    if staging.parent != final_root.parent or not staging.name.startswith(expected_prefix) or staging == final_root:
        raise IntegrityError("refusing unsafe staging cleanup: %s" % staging)
    if staging.exists():
        shutil.rmtree(str(staging))


def materialize(selected, staging_root, final_root, horizon, construction_seed):
    manifests = staging_root / "manifests"
    outputs = {}
    next_map_id = 640000
    all_ids = set()
    source_sets = {}
    crop_signatures = set()
    for split in SPLITS:
        split_maps = staging_root / split / "maps"
        split_maps.mkdir(parents=True, exist_ok=False)
        tasks = []
        source_sets[split] = set()
        for index, candidate in enumerate(selected[split], 1):
            parsed = parse_map(Path(candidate["source_map_path"]))
            crop = binary_crop(parsed, int(candidate["crop_top"]), int(candidate["crop_left"]), 64)
            start, goal = tuple(candidate["start"]), tuple(candidate["goal"])
            verified_length = shortest_path(crop, start, goal)
            if verified_length != int(candidate["shortest_path_length"]) or verified_length > int(horizon):
                raise IntegrityError("BFS identity changed during materialization")
            signature = (candidate["source_map_sha256"], candidate["crop_top"], candidate["crop_left"], start, goal)
            if signature in crop_signatures:
                raise IntegrityError("duplicate crop signature")
            crop_signatures.add(signature)
            task_id = "movingai64_%s_%04d_%s" % (split, index, candidate["ranking_sha256"][:12].lower())
            if task_id in all_ids:
                raise IntegrityError("duplicate task ID")
            all_ids.add(task_id)
            staged_map_path = split_maps / (task_id + ".npy")
            final_map_path = final_root / split / "maps" / staged_map_path.name
            np.save(str(staged_map_path), crop, allow_pickle=False)
            next_map_id += 1
            source_sets[split].add(candidate["source_map_sha256"])
            task = dict(candidate)
            task.update({
                "task_id": task_id, "split": split, "map_id": next_map_id,
                "cspace_file": str(final_map_path.resolve()), "processed_map_sha256": sha256_file(staged_map_path),
                "crop_bottom": int(candidate["crop_top"]) + 64, "crop_right": int(candidate["crop_left"]) + 64,
                "crop_height": 64, "crop_width": 64,
                "shortest_path_length_4connected": int(verified_length), "max_steps": int(horizon),
                "construction_seed": int(construction_seed),
            })
            tasks.append(task)
        filename = "training_tasks.json" if split == "train" else split + "_tasks.json"
        staged_task_path = manifests / filename
        final_task_path = final_root / "manifests" / filename
        write_json_new(staged_task_path, {"schema_version": "movingai64_tasks.v1", "status": "GENERATED_NEEDS_USER_CONFIRMATION", "split": split, "tasks": tasks})
        outputs[split] = {"path": str(final_task_path.resolve()), "staged_path": str(staged_task_path.resolve()), "sha256": sha256_file(staged_task_path), "task_count": len(tasks), "map_ids": [task["map_id"] for task in tasks]}
    if source_sets["train"] & source_sets["validation"] or source_sets["train"] & source_sets["test"] or source_sets["validation"] & source_sets["test"]:
        raise IntegrityError("source-map leakage detected")
    return outputs, source_sets


def validate_staging(staging_root, selected, outputs, assignment, protocol, horizon):
    targets = protocol["data"]["draft_task_targets"]
    expected_totals = {split: sum(int(targets[split][subset]) for subset in SUBSETS) for split in SPLITS}
    for split in SPLITS:
        if len(selected[split]) != expected_totals[split] or outputs[split]["task_count"] != expected_totals[split]:
            raise IntegrityError("task target incomplete for %s" % split)
        if len(outputs[split]["map_ids"]) != len(set(outputs[split]["map_ids"])):
            raise IntegrityError("duplicate map IDs for %s" % split)
        staged_tasks = Path(outputs[split]["staged_path"])
        if not staged_tasks.is_file() or sha256_file(staged_tasks) != outputs[split]["sha256"]:
            raise IntegrityError("task manifest hash mismatch for %s" % split)
        for task in read_json(staged_tasks)["tasks"]:
            staged_map = staging_root / split / "maps" / (task["task_id"] + ".npy")
            if not staged_map.is_file() or sha256_file(staged_map) != task["processed_map_sha256"]:
                raise IntegrityError("processed map hash mismatch: %s" % task["task_id"])
            if int(task["max_steps"]) != int(horizon) or int(task["shortest_path_length_4connected"]) > int(horizon):
                raise IntegrityError("invalid task horizon: %s" % task["task_id"])
    assigned = {split: set(name for subset in SUBSETS for name in assignment[split][subset]) for split in SPLITS}
    if assigned["train"] & assigned["validation"] or assigned["train"] & assigned["test"] or assigned["validation"] & assigned["test"]:
        raise IntegrityError("source map name leakage detected")


def main():
    parser = argparse.ArgumentParser(description="Generate deterministic Moving AI 64x64 tasks without loading a model")
    parser.add_argument("--protocol", type=Path, default=REVISION_DIR / "movingai64_protocol_draft.json")
    parser.add_argument("--candidate-audit-report", type=Path, default=REVISION_DIR / "candidate_audit_report.json")
    parser.add_argument("--confirmed-horizon", type=int, required=True)
    parser.add_argument("--confirm-generation", action="store_true")
    args = parser.parse_args()
    if not args.confirm_generation:
        raise ProtocolError("--confirm-generation is required; task generation is user-only")
    protocol = read_json(args.protocol)
    if protocol["status"] != "DRAFT_NEEDS_USER_CONFIRMATION" or protocol["formal_execution_allowed"] is not False:
        raise ProtocolError("generator requires the reviewed fail-closed draft protocol")
    recommended_horizon = protocol.get("horizon", {}).get("recommended_value")
    if recommended_horizon is None or args.confirmed_horizon != int(recommended_horizon):
        raise ProtocolError("--confirmed-horizon must equal the reviewed draft recommended horizon")
    if args.confirmed_horizon not in CANDIDATE_HORIZONS:
        raise ProtocolError("reviewed horizon must be one of the audited candidate horizons")
    audit_path, audit_report, horizon_record = validate_candidate_audit(args.candidate_audit_report, protocol, args.confirmed_horizon)
    raw_root = Path(protocol["data"]["raw_root"]).resolve()
    final_root = Path(protocol["data"]["processed_root"]).resolve()
    if final_root.exists():
        raise FileExistsError("refusing to overwrite existing processed root: %s" % final_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    staging_root = final_root.parent / (final_root.name + ".__staging_" + stamp)
    if staging_root.exists():
        raise FileExistsError("refusing to reuse staging root: %s" % staging_root)
    finalized = False
    try:
        staging_root.mkdir(parents=False, exist_ok=False)
        (staging_root / "manifests").mkdir()
        by_subset_map, inventory = collect_candidates(raw_root, int(protocol["data"]["construction_seed"]), args.confirmed_horizon)
        if canonical_json_sha256(inventory) != audit_report["identity"]["source_inventory_sha256"]:
            raise IntegrityError("candidate audit source inventory identity mismatch")
        assignment = split_source_maps(by_subset_map, protocol)
        selected = select_tasks(by_subset_map, assignment, protocol)
        outputs, source_sets = materialize(selected, staging_root, final_root, args.confirmed_horizon, int(protocol["data"]["construction_seed"]))
        map_split_path = staging_root / "manifests" / "map_split.json"
        task_split_path = staging_root / "manifests" / "task_split.json"
        write_json_new(map_split_path, {
            "schema_version": "movingai64_map_split.v1", "status": "GENERATED_NEEDS_USER_CONFIRMATION",
            "train_map_ids": outputs["train"]["map_ids"], "validation_map_ids": outputs["validation"]["map_ids"], "test_map_ids": outputs["test"]["map_ids"],
            "source_map_assignment": assignment, "source_map_sha_disjoint": True,
        })
        write_json_new(task_split_path, {
            "schema_version": "movingai64_task_split.v1", "status": "GENERATED_NEEDS_USER_CONFIRMATION",
            "splits": {split: {"path": outputs[split]["path"], "sha256": outputs[split]["sha256"], "task_count": outputs[split]["task_count"], "map_count": len(outputs[split]["map_ids"])} for split in SPLITS},
            "task_constraints": {"task_ids_disjoint": True, "source_maps_disjoint": True, "map_size": 64, "max_steps": args.confirmed_horizon},
        })
        shutil.copy2(str(REVISION_DIR / "hash_manifest.json"), str(staging_root / "manifests" / "hash_manifest.json"))
        identity = {
            "schema_version": "movingai64_generated_identity.v1", "status": "GENERATED_NEEDS_USER_CONFIRMATION",
            "horizon": args.confirmed_horizon, "construction_seed": int(protocol["data"]["construction_seed"]), "raw_inventory": inventory,
            "task_files": {split: {key: value for key, value in outputs[split].items() if key not in ("map_ids", "staged_path")} for split in SPLITS},
            "map_split": {"path": str((final_root / "manifests" / "map_split.json").resolve()), "sha256": sha256_file(map_split_path)},
            "task_split": {"path": str((final_root / "manifests" / "task_split.json").resolve()), "sha256": sha256_file(task_split_path)},
            "source_map_sha_counts": {split: len(source_sets[split]) for split in SPLITS},
            "source_map_sha_pairwise_disjoint": True, "model_loaded": False, "checkpoint_loaded": False,
            "candidate_audit_report": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
            "candidate_audit_horizon_status": horizon_record["quota_feasibility_status"],
            "candidate_audit_margin_status": horizon_record["quota_margin_status"],
        }
        write_json_new(staging_root / "manifests" / "generated_task_identity.json", identity)
        validate_staging(staging_root, selected, outputs, assignment, protocol, args.confirmed_horizon)
        required = ["training_tasks.json", "validation_tasks.json", "test_tasks.json", "map_split.json", "task_split.json", "generated_task_identity.json", "hash_manifest.json"]
        if not all((staging_root / "manifests" / name).is_file() for name in required):
            raise IntegrityError("required staging manifests are incomplete")
        if final_root.exists():
            raise FileExistsError("final processed root appeared during generation; refusing overwrite: %s" % final_root)
        staging_root.rename(final_root)
        finalized = True
        print(json.dumps(identity, ensure_ascii=False, indent=2))
    finally:
        if not finalized and staging_root.exists():
            safe_cleanup_staging(staging_root, final_root)


if __name__ == "__main__":
    main()
