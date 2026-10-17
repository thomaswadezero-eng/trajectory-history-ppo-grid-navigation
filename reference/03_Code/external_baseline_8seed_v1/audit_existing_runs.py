#!/usr/bin/env python
"""Static provenance audit for the 12 legacy external-baseline runs and FULL.

No legacy module is imported.  This script reads files, hashes bytes, parses
JSON/CSV/DOCX OOXML, and recomputes summaries from recorded episodes only.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import common


EXPECTED_VALIDATIONS = tuple(range(100000, 900001, 100000))
LEGACY_STATUS_VALUES = {"REUSABLE_LEGACY", "NOT_REUSABLE", "AMBIGUOUS"}


def _file_hash_check(path: Path, expected: str, problems: List[str], label: str) -> None:
    if not path.is_file():
        problems.append(f"missing {label}: {path}")
    elif common.sha256_file(path) != expected.upper():
        problems.append(f"{label} SHA mismatch: {path}")


def _verify_current_authority(method: str) -> List[str]:
    authority = common.AUTHORITY[method]
    problems: List[str] = []
    _file_hash_check(Path(authority["program"]), str(authority["program_sha256"]), problems, "program")
    _file_hash_check(Path(authority["config"]), str(authority["config_sha256"]), problems, "config")
    if method in ("DDQN", "D3QN"):
        _file_hash_check(Path(authority["runtime"]), str(authority["runtime_sha256"]), problems, "runner")
        _file_hash_check(Path(authority["protocol"]), str(authority["protocol_sha256"]), problems, "protocol")
        manifest = common.read_json(Path(authority["framework_manifest"]))
        if str(manifest.get("aggregate_sha256", "")).upper() != authority["framework_sha256"]:
            problems.append("framework aggregate identity mismatch")
        for record in manifest.get("files", []):
            _file_hash_check(common.PROJECT_ROOT / str(record["path"]), str(record["sha256"]), problems, "framework member")
    elif method == "C51":
        for role in ("algorithm", "replay", "protocol"):
            _file_hash_check(Path(authority[role]), str(authority[f"{role}_sha256"]), problems, role)
    return problems


def _legacy_identity_problems(method: str, manifest: Mapping[str, Any], result: Mapping[str, Any]) -> List[str]:
    a = common.AUTHORITY[method]
    problems: List[str] = []
    if method in ("DDQN", "D3QN"):
        expected = {
            "framework_sha256": a["framework_sha256"],
            "protocol_sha256": a["protocol_sha256"],
            "config_sha256": a["config_sha256"],
        }
        for key, value in expected.items():
            if str(manifest.get(key, "")).upper() != value or str(result.get(key, "")).upper() != value:
                problems.append(f"{key} legacy provenance mismatch")
    elif method == "C51":
        expected = {
            "runner_sha256": a["program_sha256"],
            "algorithm_sha256": a["algorithm_sha256"],
            "protocol_boundary_sha256": a["protocol_sha256"],
            "replay_sha256": a["replay_sha256"],
            "config_sha256": a["config_sha256"],
        }
        for key, value in expected.items():
            if str(manifest.get(key, "")).upper() != value:
                problems.append(f"{key} legacy provenance mismatch")
    else:
        sources = manifest.get("source_sha256")
        if not isinstance(sources, dict):
            problems.append("MaskablePPO source manifest missing")
        else:
            program_matches = [value for path, value in sources.items() if Path(path).name == "sb3_external_baseline.py"]
            if program_matches != [a["program_sha256"]]:
                problems.append("MaskablePPO program provenance mismatch")
            for raw_path, expected_hash in sources.items():
                _file_hash_check(Path(raw_path), str(expected_hash), problems, "MaskablePPO source member")
        if str(manifest.get("config_sha256", "")).upper() != a["config_sha256"]:
            problems.append("MaskablePPO config provenance mismatch")
        if result.get("source_sha256") != manifest.get("source_sha256"):
            problems.append("MaskablePPO result/source provenance mismatch")
    return problems


def _input_problems(method: str, run: Path, manifest: Mapping[str, Any], result: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    for key, path in common.TASK_PATHS.items():
        _file_hash_check(path, common.TASK_HASHES[key], problems, key)
    if method == "MaskablePPO":
        frozen = manifest.get("frozen_input_sha256", {})
        for key, expected in common.TASK_HASHES.items():
            if str(frozen.get(key, "")).upper() != expected:
                problems.append(f"frozen input mismatch: {key}")
        if result.get("frozen_input_sha256") != frozen:
            problems.append("result frozen-input identity mismatch")
    elif method == "C51":
        frozen = manifest.get("important_input_sha256", {})
        for key, expected in common.TASK_HASHES.items():
            if str(frozen.get(key, "")).upper() != expected:
                problems.append(f"frozen input mismatch: {key}")
    else:
        protocol = common.read_json(Path(common.AUTHORITY[method]["protocol"]))
        for split, key in (("train", "training_tasks"), ("validation", "validation_tasks"), ("test", "test_tasks")):
            if str(protocol["tasks"][split]["sha256"]).upper() != common.TASK_HASHES[key]:
                problems.append(f"protocol input mismatch: {key}")
    cspace_path = run / "cspace_file_manifest.json"
    if cspace_path.is_file():
        cspace = common.read_json(cspace_path)
        recorded = str(manifest.get("cspace_aggregate_sha256", result.get("cspace_aggregate_sha256", ""))).upper()
        actual = str(cspace.get("aggregate_sha256", "")).upper()
        if recorded and actual != recorded:
            problems.append("cspace aggregate mismatch")
    return problems


def _selected_checkpoint(method: str, manifest: Mapping[str, Any], result: Mapping[str, Any]) -> tuple[Path, str, int]:
    path = Path(str(result.get("selected_checkpoint") or manifest.get("selected_checkpoint") or manifest.get("best_checkpoint")))
    digest = str(result.get("selected_checkpoint_sha256") or manifest.get("selected_checkpoint_sha256") or manifest.get("best_checkpoint_sha256") or "").upper()
    interaction = int(result.get("selected_checkpoint_interaction") or manifest.get("selected_checkpoint_interaction") or result.get("test", {}).get("checkpoint_interaction") or -1)
    return path, digest, interaction


def audit_one(method: str, seed: int) -> Dict[str, Any]:
    discovery, run, problems = common.discover_run(common.LEGACY_ROOTS[method], seed)
    record: Dict[str, Any] = {
        "method": method,
        "seed": seed,
        "status": discovery if discovery != "CANDIDATE" else "AMBIGUOUS",
        "authority_program_path": str(common.AUTHORITY[method]["program"]),
        "authority_program_sha256": common.AUTHORITY[method]["program_sha256"],
        "old_result_root": str(common.LEGACY_ROOTS[method]),
        "run_path": "" if run is None else str(run),
        "reasons": list(problems),
    }
    if run is None:
        return record
    manifest_path = run / "run_manifest.json"
    result_path = run / common.result_name(method, seed)
    if not manifest_path.is_file() or not result_path.is_file():
        record["status"] = "NOT_REUSABLE"
        record["reasons"].append("manifest or formal result missing")
        return record
    manifest = common.read_json(manifest_path)
    result = common.read_json(result_path)
    problems = _verify_current_authority(method)
    problems.extend(_legacy_identity_problems(method, manifest, result))
    problems.extend(_input_problems(method, run, manifest, result))

    actual_method = str(manifest.get("method") or manifest.get("algorithm") or method)
    if method not in actual_method:
        problems.append("method identity mismatch")
    if int(manifest.get("seed", -1)) != seed or int(result.get("seed", -1)) != seed:
        problems.append("seed mismatch")
    allowed_status = {"completed", "completed_with_formal_test"}
    if str(manifest.get("status")) not in allowed_status:
        problems.append("run manifest is not completed")
    executed = int(manifest.get("executed_interaction_count", manifest.get("executed_training_interactions", -1)))
    training = int(result.get("training_interactions", -1))
    if executed != 900000 or training != 900000:
        problems.append("training budget is not exactly 900000")
    interactions = common.validation_interactions(run)
    if interactions != EXPECTED_VALIDATIONS:
        problems.append(f"validation checkpoints mismatch: {interactions}")
    if int(manifest.get("validation_count", len(interactions))) != 9:
        problems.append("validation count mismatch")

    checkpoint, checkpoint_sha, checkpoint_interaction = _selected_checkpoint(method, manifest, result)
    if not checkpoint.is_file():
        problems.append("selected checkpoint missing")
    elif common.sha256_file(checkpoint) != checkpoint_sha:
        problems.append("selected checkpoint SHA mismatch")
    test = result.get("test")
    if not isinstance(test, dict):
        problems.append("formal test summary missing")
        test = {}
    if int(test.get("episode_count", -1)) != 1000:
        problems.append("formal test episode count mismatch")
    if int(test.get("parameter_updates", result.get("test_parameter_updates", 0))) != 0:
        problems.append("test parameter updates are nonzero")
    astar = int(test.get("astar_runtime_calls", result.get("test_astar_runtime_calls", 0)))
    if astar != 0:
        problems.append("test A* runtime calls are nonzero")
    try:
        csv_path = common.test_csv_path(run)
        rows = common.load_episode_rows(csv_path)
        recomputed = common.recompute_metrics(rows)
        problems.extend(common.compare_metrics(recomputed, test))
        if not common.verify_test_order(rows):
            problems.append("test task identity/order mismatch")
    except Exception as exc:
        csv_path = Path()
        recomputed = {}
        problems.append(f"episode-level audit failed: {exc}")

    # Formal result and manifest must agree on the final summary and selected model.
    manifest_test = manifest.get("test_metrics", manifest.get("test", {}))
    if manifest_test and common.compare_metrics({k: recomputed[k] for k in recomputed if k != "path_efficiency_success_count"}, manifest_test):
        problems.append("manifest/formal-result final metrics disagree")
    manifest_cp = str(manifest.get("selected_checkpoint_sha256") or manifest.get("best_checkpoint_sha256") or "").upper()
    if manifest_cp and manifest_cp != checkpoint_sha:
        problems.append("manifest/formal-result checkpoint identity disagrees")

    record.update({
        "status": "REUSABLE_LEGACY" if not problems else "NOT_REUSABLE",
        "reasons": problems,
        "manifest_path": str(manifest_path),
        "manifest_sha256": common.sha256_file(manifest_path),
        "formal_result_path": str(result_path),
        "formal_result_sha256": common.sha256_file(result_path),
        "config_sha256": str(manifest.get("config_sha256", "")),
        "protocol_sha256": str(manifest.get("protocol_sha256", manifest.get("protocol_boundary_sha256", ""))),
        "framework_or_source_identity": str(manifest.get("framework_sha256", manifest.get("runner_sha256", "source_manifest"))),
        "training_interactions": training,
        "validation_interactions": list(interactions),
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": checkpoint_sha,
        "selected_checkpoint_interaction": checkpoint_interaction,
        "test_csv_path": str(csv_path),
        "test_episode_count": int(test.get("episode_count", -1)),
        "success_count": int(test.get("success_count", -1)),
        "collision_count": int(test.get("collision_count", -1)),
        "timeout_count": int(test.get("timeout_count", -1)),
        "success_rate": test.get("success_rate"),
        "spl": test.get("spl"),
        "path_efficiency": test.get("path_efficiency"),
        "average_steps": test.get("average_steps"),
        "average_path_length": test.get("average_path_length"),
        "collision_rate": test.get("collision_rate"),
        "timeout_rate": test.get("timeout_rate"),
        "episode_reward": test.get("average_episode_reward"),
        "episode_reaggregation_match": not any("recomputation mismatch" in item for item in problems),
        "test_task_order_match": "test task identity/order mismatch" not in problems,
    })
    return record


def audit_legacy() -> List[Dict[str, Any]]:
    return [audit_one(method, seed) for method in common.METHODS for seed in common.LEGACY_SEEDS]


def _full_identity(seed: int) -> Dict[str, str]:
    if seed in common.LEGACY_SEEDS:
        return {
            "program_sha256": "96093A6B5F024F5D9505ACCEBEAE7592700228BA17587BF0590092A241B7544D",
            "config_sha256": "489431F4C5609FC1F7148D56D152D22C1252F6B6AF79CE8AE09716E1F8D84B8D",
            "protocol_sha256": "7887F254E31A1C6A0CDB95994EF3C18CC3A8A43775D8D6BD16475FDFB042ADB7",
        }
    return {
        "program_sha256": "19290D3A23DCC8E10FD4264EF9D8A3FEC4A691CFFD8A69C6FDEE8DE65C8EF8DD",
        "config_sha256": "A1B343EE2D5FA67D09D0D1D4119261DB045BD892A5A231F1E534399C2C86B5B0",
        "protocol_sha256": "CDEE0BDB835466E8610E5952DFE6683480D25E78A13DD9BBCD822F09897421BA",
    }


def audit_full_reference() -> List[Dict[str, Any]]:
    source_files = {
        "96093A6B5F024F5D9505ACCEBEAE7592700228BA17587BF0590092A241B7544D": common.PROJECT_ROOT / "03_Code/final_two_factor_experiment_v7/final_two_factor_experiment_v7.py",
        "19290D3A23DCC8E10FD4264EF9D8A3FEC4A691CFFD8A69C6FDEE8DE65C8EF8DD": common.PROJECT_ROOT / "03_Code/final_two_factor_experiment_v7_8seed/final_two_factor_experiment_v7_8seed.py",
    }
    records: List[Dict[str, Any]] = []
    for seed in common.SEEDS:
        discovery, run, reasons = common.discover_run(common.FULL_ROOT, seed)
        if run is None:
            records.append({"seed": seed, "status": discovery, "run_path": "", "reasons": reasons})
            continue
        manifest_path = run / "run_manifest.json"
        result_path = run / f"formal_result_FULL_seed{seed}.json"
        problems: List[str] = []
        if not manifest_path.is_file() or not result_path.is_file():
            problems.append("FULL manifest or result missing")
            manifest = {}
            result = {}
        else:
            manifest = common.read_json(manifest_path)
            result = common.read_json(result_path)
        expected = _full_identity(seed)
        for key, value in expected.items():
            if str(manifest.get(key, "")).upper() != value or str(result.get(key, "")).upper() != value:
                problems.append(f"FULL {key} mismatch")
        program = source_files[expected["program_sha256"]]
        _file_hash_check(program, expected["program_sha256"], problems, "FULL program")
        if manifest.get("status") != "completed" or int(manifest.get("executed_interaction_count", -1)) != 900000:
            problems.append("FULL run incomplete")
        if int(result.get("training_interactions", -1)) != 900000:
            problems.append("FULL result budget mismatch")
        for key, expected_hash in common.TASK_HASHES.items():
            if str(result.get("input_sha256", {}).get(key, "")).upper() != expected_hash:
                problems.append(f"FULL input mismatch: {key}")
        checkpoint = Path(str(result.get("selected_checkpoint", "")))
        checkpoint_sha = str(result.get("selected_checkpoint_sha256", "")).upper()
        if not checkpoint.is_file() or common.sha256_file(checkpoint) != checkpoint_sha:
            problems.append("FULL selected checkpoint mismatch")
        test = result.get("test", {})
        if int(test.get("episode_count", -1)) != 1000 or int(test.get("parameter_updates", -1)) != 0 or int(test.get("astar_runtime_calls", -1)) != 0:
            problems.append("FULL test protocol mismatch")
        try:
            rows = common.load_episode_rows(run / "evaluation_metrics.csv")
            recomputed = common.recompute_metrics(rows)
            problems.extend("FULL " + item for item in common.compare_metrics(recomputed, test))
            if not common.verify_test_order(rows):
                problems.append("FULL test task order mismatch")
        except Exception as exc:
            recomputed = {}
            problems.append(f"FULL episode audit failed: {exc}")
        records.append({
            "seed": seed,
            "status": "VALID_READ_ONLY_REFERENCE" if not problems else "NOT_VALID_REFERENCE",
            "run_path": str(run),
            "manifest_path": str(manifest_path),
            "formal_result_path": str(result_path),
            "formal_result_sha256": common.sha256_file(result_path) if result_path.is_file() else "",
            "program_sha256": expected["program_sha256"],
            "config_sha256": expected["config_sha256"],
            "protocol_sha256": expected["protocol_sha256"],
            "selected_checkpoint_sha256": checkpoint_sha,
            "test": test,
            "reasons": problems,
        })
    return records


def paper_consistency(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    problems: List[str] = []
    _file_hash_check(common.PAPER_AUTHORITY, common.PAPER_AUTHORITY_SHA256, problems, "paper authority")
    table = common.extract_external_table(common.PAPER_AUTHORITY)
    by_method: Dict[str, List[Mapping[str, Any]]] = {}
    for method in common.METHODS:
        by_method[method] = sorted((record for record in records if record["method"] == method), key=lambda item: int(item["seed"]))
    # The current v8 Table 4 intentionally contains four metrics.  Timeout was
    # present in older drafts and must not be used to select an obsolete paper.
    metrics = ("success_rate", "spl", "path_efficiency", "average_steps")
    checks: Dict[str, Any] = {}
    for method, method_records in by_method.items():
        if len(method_records) != 3:
            problems.append(f"paper comparison lacks three legacy records for {method}")
            continue
        means = []
        sample_sds = []
        for metric in metrics:
            values = [float(record[metric]) for record in method_records]
            factor = 100.0 if metric in ("success_rate", "timeout_rate") else 1.0
            means.append(statistics.fmean(values) * factor)
            sample_sds.append(statistics.stdev(values) * factor)
        paper_means = table[method]
        tolerances = (0.011, 0.00051, 0.00051, 0.051)
        match = all(abs(value - paper) <= tolerance for value, paper, tolerance in zip(means, paper_means, tolerances))
        if not match:
            problems.append(f"current paper external table mismatch for {method}")
        checks[method] = {"means": means, "sample_sds": sample_sds, "paper_means": paper_means, "match": match}
    return {"status": "MATCH" if not problems else "NEEDS_USER_REVIEW", "paper": str(common.PAPER_AUTHORITY), "table": table, "checks": checks, "problems": problems}


def run_audit() -> Dict[str, Any]:
    legacy = audit_legacy()
    full = audit_full_reference()
    paper = paper_consistency(legacy)
    reusable = sum(record["status"] == "REUSABLE_LEGACY" for record in legacy)
    full_valid = sum(record["status"] == "VALID_READ_ONLY_REFERENCE" for record in full)
    return {
        "schema_version": "external_baseline_8seed_legacy_audit.v1",
        "generated_at": common.now_iso(),
        "audit_mode": "STATIC_FILE_PROVENANCE_ONLY_NO_MODEL_NO_ENVIRONMENT_NO_CUDA",
        "legacy_runs": legacy,
        "legacy_summary": {
            "expected": 12,
            "reusable_legacy": reusable,
            "not_reusable": sum(record["status"] == "NOT_REUSABLE" for record in legacy),
            "ambiguous": sum(record["status"] == "AMBIGUOUS" for record in legacy),
            "new_formal_runs_required": len(common.METHODS) * len(common.SEEDS) - reusable,
            "zero_drift_extension_eligible": reusable == 12,
        },
        "full_reference_runs": full,
        "full_reference_summary": {"expected": 8, "valid": full_valid, "read_only": True},
        "paper_consistency": paper,
        "runtime_preflight": "RUNTIME_PREFLIGHT_PENDING",
        "formal_training": "NOT_STARTED",
    }


def csv_rows(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in payload["legacy_runs"]:
        rows.append({
            "method": record["method"], "seed": record["seed"], "status": record["status"],
            "authority_program_path": record["authority_program_path"], "authority_program_sha256": record["authority_program_sha256"],
            "old_result_root": record["old_result_root"], "run_path": record["run_path"],
            "training_interactions": record.get("training_interactions", ""), "validation_interactions": json.dumps(record.get("validation_interactions", [])),
            "selected_checkpoint": record.get("selected_checkpoint", ""), "selected_checkpoint_sha256": record.get("selected_checkpoint_sha256", ""),
            "test_episode_count": record.get("test_episode_count", ""), "success_count": record.get("success_count", ""),
            "collision_count": record.get("collision_count", ""), "timeout_count": record.get("timeout_count", ""),
            "success_rate": record.get("success_rate", ""), "spl": record.get("spl", ""), "path_efficiency": record.get("path_efficiency", ""),
            "average_steps": record.get("average_steps", ""), "average_path_length": record.get("average_path_length", ""),
            "collision_rate": record.get("collision_rate", ""), "timeout_rate": record.get("timeout_rate", ""),
            "episode_reward": record.get("episode_reward", ""), "episode_reaggregation_match": record.get("episode_reaggregation_match", ""),
            "test_task_order_match": record.get("test_task_order_match", ""), "reasons": " | ".join(record.get("reasons", [])),
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static provenance audit of legacy external baselines")
    parser.add_argument("--write", action="store_true", help="write legacy_run_audit.json/csv beside this script")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_audit()
    if args.write:
        common.write_json_atomic(common.REVISION_ROOT / "legacy_run_audit.json", payload)
        rows = csv_rows(payload)
        common.write_csv_atomic(common.REVISION_ROOT / "legacy_run_audit.csv", list(rows[0]), rows)
    print(json.dumps(payload["legacy_summary"], ensure_ascii=False, indent=2))
    print(json.dumps(payload["full_reference_summary"], ensure_ascii=False, indent=2))
    print(json.dumps({"paper_consistency": payload["paper_consistency"]["status"]}, ensure_ascii=False, indent=2))
    if payload["legacy_summary"]["reusable_legacy"] != 12 or payload["full_reference_summary"]["valid"] != 8 or payload["paper_consistency"]["status"] != "MATCH":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
