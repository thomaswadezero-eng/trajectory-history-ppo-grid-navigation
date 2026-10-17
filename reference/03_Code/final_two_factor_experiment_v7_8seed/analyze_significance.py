#!/usr/bin/env python
"""Strict seed-level significance analysis for the frozen V7 8-seed plan.

This program is read-only with respect to formal run directories. It refuses to
analyze incomplete, duplicate, non-completed, unexpected-seed, or identity-
mismatched run sets. The independent statistical unit is one trained policy
per random seed (n=8 paired seeds), never an individual test episode.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Pattern, Sequence, Set, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
LEGACY_DIR = PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7"
LEGACY_PROGRAM = LEGACY_DIR / "final_two_factor_experiment_v7.py"
LEGACY_CONFIG = LEGACY_DIR / "experiment_config.json"
LEGACY_PROTOCOL = LEGACY_DIR / "experiment_protocol.json"
EXTENSION_PROGRAM = SCRIPT_DIR / "final_two_factor_experiment_v7_8seed.py"
EXTENSION_CONFIG = SCRIPT_DIR / "experiment_config_v7_8seed.json"
EXTENSION_PROTOCOL = SCRIPT_DIR / "experiment_protocol_v7_8seed.json"
EXTENSION_HASH_MANIFEST = SCRIPT_DIR / "hash_manifest_v7_8seed.json"
DEFAULT_PLAN = SCRIPT_DIR / "significance_plan.json"
DEFAULT_OUTPUT_ROOT = Path(r"redacted_local_root\final_two_factor_experiment_v7")
DEFAULT_ANALYSIS_OUTPUT = DEFAULT_OUTPUT_ROOT / "analysis_ready" / "significance_8seed"

PLAN_SHA256 = "FDACCBA34368184DFDDC1A6044F280D1CB4064775B689E46A98B19872E57DB49"
DIFF_AUDIT_SHA256 = "6105769FDFF43AC6A8BB00D9047E5E5CB43208922B380617752C3B8E1D09964D"
LEGACY_PROGRAM_SHA256 = "96093A6B5F024F5D9505ACCEBEAE7592700228BA17587BF0590092A241B7544D"
LEGACY_CONFIG_SHA256 = "489431F4C5609FC1F7148D56D152D22C1252F6B6AF79CE8AE09716E1F8D84B8D"
LEGACY_PROTOCOL_SHA256 = "7887F254E31A1C6A0CDB95994EF3C18CC3A8A43775D8D6BD16475FDFB042ADB7"
EXTENSION_PROGRAM_SHA256 = "19290D3A23DCC8E10FD4264EF9D8A3FEC4A691CFFD8A69C6FDEE8DE65C8EF8DD"
EXTENSION_CONFIG_SHA256 = "A1B343EE2D5FA67D09D0D1D4119261DB045BD892A5A231F1E534399C2C86B5B0"
EXTENSION_PROTOCOL_SHA256 = "CDEE0BDB835466E8610E5952DFE6683480D25E78A13DD9BBCD822F09897421BA"
EXTENSION_CONFIG_FILE_SHA256 = "6F39A3DEAC6AF9A7CC7BE3237DEE2DC61AE5449B51AE1EDA49D0B456F2D8E731"
EXTENSION_PROTOCOL_FILE_SHA256 = "B51B1BB2DD72D89B964FD368F1ED36660A944EEEA1A1CC1EE33BD32555528B97"
EXTENSION_HASH_MANIFEST_SHA256 = "FB1F0316EE16F77A0D976A0E25C05E8F64BFB752F9CA0D7745A17F7FC3C54A8C"

LEGACY_SEEDS = (42, 1004, 6010)
NEW_SEEDS = (68882, 19816, 74494, 81399, 31524)
EXPECTED_SEEDS = LEGACY_SEEDS + NEW_SEEDS
ABANDONED_SEEDS = (2026, 12345, 27182, 31415, 54321)
METHODS = ("PPO", "PPO_Reward", "PPO_Astar", "FULL")
METHOD_DISPLAY = {
    "PPO": "PPO",
    "PPO_Reward": "PPO+Reward",
    "PPO_Astar": "PPO+A*",
    "FULL": "FULL",
}
METRICS = (
    "success_rate",
    "spl",
    "path_efficiency",
    "average_steps",
    "timeout_rate",
    "collision_rate",
)
METRIC_DIRECTIONS = {
    "success_rate": "higher_is_better",
    "spl": "higher_is_better",
    "path_efficiency": "higher_is_better",
    "average_steps": "lower_is_better",
    "timeout_rate": "lower_is_better",
    "collision_rate": "lower_is_better",
}
HYPOTHESES = (
    {
        "hypothesis": "H1",
        "comparison": "PPO vs PPO+Reward",
        "method_a": "PPO",
        "method_b": "PPO_Reward",
        "metrics": ("success_rate", "spl", "timeout_rate"),
    },
    {
        "hypothesis": "H2",
        "comparison": "PPO+A* vs FULL",
        "method_a": "PPO_Astar",
        "method_b": "FULL",
        "metrics": ("success_rate", "spl", "timeout_rate"),
    },
    {
        "hypothesis": "H3",
        "comparison": "PPO+Reward vs FULL",
        "method_a": "PPO_Reward",
        "method_b": "FULL",
        "metrics": ("spl", "path_efficiency", "average_steps"),
    },
)
INTERACTION_BUDGET = 900000
TEST_EPISODES = 1000
ALPHA = 0.05
BOOTSTRAP_REPLICATES = 100000
ANALYSIS_SEED = 20260828

CONFIG_HASH_PATTERN = re.compile(rb'("config_sha256"\s*:\s*")([0-9A-Fa-f]{64})(")')
PROTOCOL_HASH_PATTERN = re.compile(rb'("protocol_sha256"\s*:\s*")([0-9A-Fa-f]{64})(")')


class AnalysisIntegrityError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AnalysisIntegrityError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AnalysisIntegrityError(f"Expected JSON object: {path}")
    return payload


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AnalysisIntegrityError(
            f"{label} mismatch: expected {expected!r}, found {actual!r}"
        )


def verified_self_hash(path: Path, pattern: Pattern[bytes], field_name: str) -> str:
    raw = path.read_bytes()
    matches = list(pattern.finditer(raw))
    if len(matches) != 1:
        raise AnalysisIntegrityError(
            f"{path} must contain exactly one {field_name} field"
        )
    match = matches[0]
    stored = match.group(2).decode("ascii").upper()
    zeroed = raw[: match.start(2)] + b"0" * 64 + raw[match.end(2) :]
    calculated = hashlib.sha256(zeroed).hexdigest().upper()
    require_equal(stored, calculated, f"{path.name} self-field-zeroed SHA")
    return stored


def normalize_extension_program(text: str) -> str:
    replacements = (
        (
            "03_Code/final_two_factor_experiment_v7_8seed/final_two_factor_experiment_v7_8seed.py",
            "03_Code/final_two_factor_experiment_v7/final_two_factor_experiment_v7.py",
        ),
        (
            "03_Code/final_two_factor_experiment_v7_8seed/experiment_config_v7_8seed.json",
            "03_Code/final_two_factor_experiment_v7/experiment_config.json",
        ),
        ("experiment_config_v7_8seed.json", "experiment_config.json"),
        ("experiment_protocol_v7_8seed.json", "experiment_protocol.json"),
        (
            "[42, 1004, 6010, 68882, 19816, 74494, 81399, 31524]",
            "[42, 1004, 6010]",
        ),
        ("hash_manifest_v7_8seed.json", "hash_manifest.json"),
    )
    for new, legacy in replacements:
        text = text.replace(new, legacy)
    return text


def verify_extension_equivalence(plan: Mapping[str, Any]) -> None:
    require_equal(sha256_file(DEFAULT_PLAN), PLAN_SHA256, "Frozen plan SHA")
    require_equal(sha256_file(LEGACY_PROGRAM), LEGACY_PROGRAM_SHA256, "Legacy program SHA")
    require_equal(
        sha256_file(EXTENSION_PROGRAM), EXTENSION_PROGRAM_SHA256, "Extension program SHA"
    )
    require_equal(
        sha256_file(EXTENSION_CONFIG),
        EXTENSION_CONFIG_FILE_SHA256,
        "Extension config file SHA",
    )
    require_equal(
        sha256_file(EXTENSION_PROTOCOL),
        EXTENSION_PROTOCOL_FILE_SHA256,
        "Extension protocol file SHA",
    )
    require_equal(
        sha256_file(EXTENSION_HASH_MANIFEST),
        EXTENSION_HASH_MANIFEST_SHA256,
        "Extension hash-manifest SHA",
    )

    audit_path = Path(str(plan["diff_audit"]["path"]))
    require_equal(sha256_file(audit_path), DIFF_AUDIT_SHA256, "Diff-audit SHA")
    require_equal(plan["diff_audit"]["status"], "PASS", "Diff-audit status")

    normalized_program = normalize_extension_program(
        EXTENSION_PROGRAM.read_text(encoding="utf-8")
    )
    legacy_program = LEGACY_PROGRAM.read_text(encoding="utf-8")
    require_equal(
        normalized_program,
        legacy_program,
        "Program after allowed authorization/identity normalization",
    )
    normalized_sha = hashlib.sha256(normalized_program.encode("utf-8")).hexdigest().upper()
    require_equal(normalized_sha, LEGACY_PROGRAM_SHA256, "Normalized program SHA")

    legacy_config = read_json(LEGACY_CONFIG)
    extension_config = read_json(EXTENSION_CONFIG)
    normalized_config = copy.deepcopy(extension_config)
    for key in (
        "config_sha256",
        "authorized_program",
        "authorized_program_sha256",
        "official_seeds",
    ):
        normalized_config[key] = copy.deepcopy(legacy_config[key])
    require_equal(normalized_config, legacy_config, "Normalized configuration")

    legacy_protocol = read_json(LEGACY_PROTOCOL)
    extension_protocol = read_json(EXTENSION_PROTOCOL)
    normalized_protocol = copy.deepcopy(extension_protocol)
    for key in (
        "protocol_sha256",
        "authorized_program",
        "authorized_program_sha256",
        "official_seeds",
    ):
        normalized_protocol[key] = copy.deepcopy(legacy_protocol[key])
    normalized_protocol["configuration"] = copy.deepcopy(legacy_protocol["configuration"])
    require_equal(normalized_protocol, legacy_protocol, "Normalized protocol")

    require_equal(
        verified_self_hash(LEGACY_CONFIG, CONFIG_HASH_PATTERN, "config_sha256"),
        LEGACY_CONFIG_SHA256,
        "Legacy config self hash",
    )
    require_equal(
        verified_self_hash(LEGACY_PROTOCOL, PROTOCOL_HASH_PATTERN, "protocol_sha256"),
        LEGACY_PROTOCOL_SHA256,
        "Legacy protocol self hash",
    )
    require_equal(
        verified_self_hash(EXTENSION_CONFIG, CONFIG_HASH_PATTERN, "config_sha256"),
        EXTENSION_CONFIG_SHA256,
        "Extension config self hash",
    )
    require_equal(
        verified_self_hash(EXTENSION_PROTOCOL, PROTOCOL_HASH_PATTERN, "protocol_sha256"),
        EXTENSION_PROTOCOL_SHA256,
        "Extension protocol self hash",
    )

    manifest = read_json(EXTENSION_HASH_MANIFEST)
    for relative_path, expected_sha in manifest["files"].items():
        actual_path = PROJECT_ROOT / str(relative_path)
        require_equal(sha256_file(actual_path), str(expected_sha), f"Manifest file {relative_path}")
    require_equal(
        manifest["configuration_self_field_zeroed_sha256"],
        EXTENSION_CONFIG_SHA256,
        "Hash-manifest config self hash",
    )
    require_equal(
        manifest["protocol_self_field_zeroed_sha256"],
        EXTENSION_PROTOCOL_SHA256,
        "Hash-manifest protocol self hash",
    )


def verify_plan(plan_path: Path, output_root: Path) -> Dict[str, Any]:
    require_equal(plan_path.resolve(), DEFAULT_PLAN.resolve(), "Plan path")
    plan = read_json(plan_path)
    require_equal(sha256_file(plan_path), PLAN_SHA256, "Plan SHA")
    require_equal(plan.get("status"), "FROZEN_BEFORE_NEW_RESULTS", "Plan status")
    require_equal(Path(plan["output_root"]), output_root, "Plan output root")
    require_equal(tuple(plan["existing_seeds"]), LEGACY_SEEDS, "Plan legacy seeds")
    require_equal(tuple(plan["new_seeds"]), NEW_SEEDS, "Plan new seeds")
    require_equal(tuple(plan["final_seed_list"]), EXPECTED_SEEDS, "Plan final seeds")
    require_equal(
        tuple(plan["abandoned_seeds_do_not_use"]), ABANDONED_SEEDS, "Plan abandoned seeds"
    )
    require_equal(plan["training_interaction_budget"], INTERACTION_BUDGET, "Plan budget")
    require_equal(
        plan["statistical_analysis"]["required_pair_count"], 8, "Plan required pair count"
    )
    require_equal(plan["statistical_analysis"]["alternative"], "two-sided", "Plan alternative")
    require_equal(float(plan["statistical_analysis"]["alpha"]), ALPHA, "Plan alpha")
    require_equal(
        plan["statistical_analysis"]["test"],
        "two-sided paired Wilcoxon signed-rank test",
        "Plan statistical test",
    )
    require_equal(
        plan["statistical_analysis"]["multiple_comparison_correction"],
        "Holm correction applied separately within each three-metric hypothesis family",
        "Plan multiplicity correction",
    )
    require_equal(
        int(plan["statistical_analysis"]["bootstrap"]["replicates"]),
        BOOTSTRAP_REPLICATES,
        "Plan bootstrap replicates",
    )
    require_equal(
        int(plan["statistical_analysis"]["bootstrap"]["analysis_seed"]),
        ANALYSIS_SEED,
        "Plan analysis seed",
    )
    planned_methods = tuple(item["cli_token"] for item in plan["methods"])
    require_equal(planned_methods, METHODS, "Plan method tokens and order")
    planned_matrix = tuple(
        (int(item["run_index"]), str(item["method"]), int(item["seed"]))
        for item in plan["run_matrix"]
    )
    expected_matrix = tuple(
        (index, method, seed)
        for index, (seed, method) in enumerate(
            ((seed, method) for seed in NEW_SEEDS for method in METHODS), start=1
        )
    )
    require_equal(planned_matrix, expected_matrix, "Plan 20-run matrix")
    planned_families = tuple(
        (
            item["hypothesis"],
            item["comparison"],
            item["method_a"],
            item["method_b"],
            tuple(item["primary_metrics"]),
        )
        for item in plan["statistical_analysis"]["hypothesis_families"]
    )
    expected_families = tuple(
        (
            item["hypothesis"],
            item["comparison"],
            item["method_a"],
            item["method_b"],
            tuple(item["metrics"]),
        )
        for item in HYPOTHESES
    )
    require_equal(planned_families, expected_families, "Plan hypothesis families")
    verify_extension_equivalence(plan)
    return plan


def expected_identity(seed: int) -> Tuple[str, str, str]:
    if seed in LEGACY_SEEDS:
        return LEGACY_PROGRAM_SHA256, LEGACY_CONFIG_SHA256, LEGACY_PROTOCOL_SHA256
    if seed in NEW_SEEDS:
        return EXTENSION_PROGRAM_SHA256, EXTENSION_CONFIG_SHA256, EXTENSION_PROTOCOL_SHA256
    raise AnalysisIntegrityError(f"Unexpected seed: {seed}")


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise AnalysisIntegrityError(f"Missing {label}: {path}")


def validate_completed_run(
    output_root: Path, method: str, seed: int, run_dir: Path
) -> Dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    formal_path = run_dir / f"formal_result_{method}_seed{seed}.json"
    evaluation_path = run_dir / "evaluation_metrics.csv"
    best_path = run_dir / "models" / "best_checkpoint.json"
    for path, label in (
        (manifest_path, "manifest"),
        (formal_path, "formal result"),
        (evaluation_path, "evaluation metrics"),
        (best_path, "best-checkpoint record"),
    ):
        require_file(path, label)

    formal_matches = list(run_dir.glob("formal_result_*.json"))
    if formal_matches != [formal_path]:
        raise AnalysisIntegrityError(
            f"Formal-result uniqueness failure in {run_dir}: {formal_matches}"
        )

    manifest = read_json(manifest_path)
    formal = read_json(formal_path)
    best = read_json(best_path)
    program_sha, config_sha, protocol_sha = expected_identity(seed)

    require_equal(manifest.get("schema_version"), "final_two_factor_run.v7", "Manifest schema")
    require_equal(manifest.get("status"), "completed", "Manifest status")
    require_equal(manifest.get("method"), method, "Manifest method")
    require_equal(int(manifest.get("seed", -1)), seed, "Manifest seed")
    require_equal(manifest.get("program_sha256"), program_sha, "Manifest program SHA")
    require_equal(manifest.get("config_sha256"), config_sha, "Manifest config SHA")
    require_equal(manifest.get("protocol_sha256"), protocol_sha, "Manifest protocol SHA")
    require_equal(
        int(manifest.get("training_interaction_budget", -1)),
        INTERACTION_BUDGET,
        "Manifest budget",
    )
    require_equal(
        int(manifest.get("executed_interaction_count", -1)),
        INTERACTION_BUDGET,
        "Manifest interactions",
    )
    require_equal(int(manifest.get("validation_astar_runtime_calls", -1)), 0, "Validation A* calls")
    require_equal(int(manifest.get("test_astar_runtime_calls", -1)), 0, "Test A* calls")

    require_equal(formal.get("schema_version"), "final_two_factor_result.v7", "Formal schema")
    require_equal(formal.get("method"), method, "Formal method")
    require_equal(int(formal.get("seed", -1)), seed, "Formal seed")
    require_equal(formal.get("program_sha256"), program_sha, "Formal program SHA")
    require_equal(formal.get("config_sha256"), config_sha, "Formal config SHA")
    require_equal(formal.get("protocol_sha256"), protocol_sha, "Formal protocol SHA")
    require_equal(
        int(formal.get("training_interactions", -1)), INTERACTION_BUDGET, "Formal interactions"
    )
    require_equal(formal.get("test"), manifest.get("test_metrics"), "Manifest/formal test metrics")
    test = formal.get("test")
    if not isinstance(test, dict):
        raise AnalysisIntegrityError(f"Missing test metrics: {formal_path}")
    require_equal(int(test.get("episode_count", -1)), TEST_EPISODES, "Test episode count")
    require_equal(int(test.get("astar_runtime_calls", -1)), 0, "Test A* runtime calls")
    require_equal(int(test.get("parameter_updates", -1)), 0, "Test parameter updates")
    for metric in METRICS:
        if metric not in test:
            raise AnalysisIntegrityError(f"Missing metric {metric}: {formal_path}")
        value = float(test[metric])
        if not math.isfinite(value):
            raise AnalysisIntegrityError(f"Non-finite metric {metric}: {formal_path}")

    selected_checkpoint = Path(str(manifest.get("selected_checkpoint", "")))
    require_file(selected_checkpoint, "selected checkpoint")
    require_equal(
        sha256_file(selected_checkpoint),
        str(manifest.get("selected_checkpoint_sha256", "")).upper(),
        "Selected-checkpoint SHA",
    )
    best_checkpoint = Path(str(best.get("checkpoint_path", "")))
    require_file(best_checkpoint, "best checkpoint")
    require_equal(
        sha256_file(best_checkpoint),
        str(best.get("checkpoint_sha256", "")).upper(),
        "Best-checkpoint SHA",
    )

    expected_interactions = tuple(range(100000, 900001, 100000))
    checkpoint_files = sorted((run_dir / "models").glob("checkpoint_interaction*.pth"))
    require_equal(len(checkpoint_files), 9, "Checkpoint count")
    for interaction in expected_interactions:
        require_file(
            run_dir / "models" / f"checkpoint_interaction{interaction}.pth",
            f"checkpoint {interaction}",
        )

    trajectories = list((run_dir / "trajectories").glob("episode_*.json"))
    require_equal(len(trajectories), TEST_EPISODES, "Trajectory count")
    require_file(run_dir / "trajectories" / "episode_0001.json", "first trajectory")
    require_file(run_dir / "trajectories" / "episode_1000.json", "last trajectory")

    validation_interactions: List[int] = []
    test_summary_count = 0
    with evaluation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("record_type") == "summary" and row.get("split") == "validation":
                validation_interactions.append(int(row["checkpoint_interaction"]))
            elif row.get("record_type") == "summary" and row.get("split") == "test":
                test_summary_count += 1
    require_equal(tuple(sorted(validation_interactions)), expected_interactions, "Validation history")
    require_equal(test_summary_count, 1, "Test summary count")

    ready_pattern = f"{method}_seed{seed}_{run_dir.name}.json"
    ready_path = output_root / "analysis_ready" / ready_pattern
    require_file(ready_path, "analysis_ready result")
    ready = read_json(ready_path)
    require_equal(ready.get("method"), method, "analysis_ready method")
    require_equal(int(ready.get("seed", -1)), seed, "analysis_ready seed")
    require_equal(ready.get("program_sha256"), program_sha, "analysis_ready program SHA")

    return {
        "method": method,
        "seed": seed,
        "run_directory": str(run_dir),
        "formal_result": str(formal_path),
        "program_sha256": program_sha,
        "config_sha256": config_sha,
        "protocol_sha256": protocol_sha,
        "metrics": {metric: float(test[metric]) for metric in METRICS},
    }


def validate_run(output_root: Path, method: str, seed: int) -> Dict[str, Any]:
    seed_dir = output_root / method / f"seed_{seed}"
    if not seed_dir.is_dir():
        raise AnalysisIntegrityError(f"Missing run: {method}/seed_{seed}")
    run_dirs = sorted(path for path in seed_dir.glob("run_*") if path.is_dir())
    if not run_dirs:
        raise AnalysisIntegrityError(f"Missing run: {method}/seed_{seed}")

    completed_records: List[Dict[str, Any]] = []
    excluded_runs: List[Tuple[Path, str]] = []
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        formal_matches = sorted(run_dir.glob("formal_result_*.json"))
        ready_path = (
            output_root
            / "analysis_ready"
            / f"{method}_seed{seed}_{run_dir.name}.json"
        )
        if not manifest_path.is_file():
            if formal_matches or ready_path.is_file():
                raise AnalysisIntegrityError(
                    f"Run without manifest has formal completion artifacts: {run_dir}"
                )
            excluded_runs.append((run_dir, "missing_manifest"))
            continue

        manifest = read_json(manifest_path)
        status = str(manifest.get("status", "missing"))
        if status != "completed":
            if formal_matches or ready_path.is_file():
                raise AnalysisIntegrityError(
                    f"Non-completed run has formal completion artifacts: "
                    f"status={status}, run={run_dir}"
                )
            excluded_runs.append((run_dir, f"manifest_status_{status}"))
            continue

        completed_records.append(
            validate_completed_run(output_root, method, seed, run_dir)
        )

    if len(completed_records) != 1:
        completed_paths = [record["run_directory"] for record in completed_records]
        raise AnalysisIntegrityError(
            f"Expected exactly one fully validated completed run for "
            f"{method}/seed_{seed}; found {len(completed_records)}: "
            f"{completed_paths}"
        )

    for run_dir, reason in excluded_runs:
        print(
            f"EXCLUDED_NONCOMPLETED_RUN method={method} seed={seed} "
            f"reason={reason} path={run_dir}"
        )
    return completed_records[0]


def collect_complete_runs(output_root: Path) -> List[Dict[str, Any]]:
    if not output_root.is_dir():
        raise AnalysisIntegrityError(f"Missing output root: {output_root}")
    records: List[Dict[str, Any]] = []
    expected_seed_set = set(EXPECTED_SEEDS)
    for method in METHODS:
        method_dir = output_root / method
        if not method_dir.is_dir():
            raise AnalysisIntegrityError(f"Missing method directory: {method_dir}")
        found_seeds: Set[int] = set()
        for path in method_dir.glob("seed_*"):
            if not path.is_dir():
                continue
            match = re.fullmatch(r"seed_(\d+)", path.name)
            if not match:
                raise AnalysisIntegrityError(f"Unexpected seed-directory name: {path}")
            found_seeds.add(int(match.group(1)))
        require_equal(found_seeds, expected_seed_set, f"Strict seed set for {method}")
        for seed in EXPECTED_SEEDS:
            records.append(validate_run(output_root, method, seed))
    require_equal(len(records), 32, "Total formal core-run count")
    return records


def values_for(
    records: Sequence[Mapping[str, Any]], method: str, metric: str
) -> List[float]:
    by_seed = {
        int(record["seed"]): float(record["metrics"][metric])
        for record in records
        if record["method"] == method
    }
    require_equal(set(by_seed), set(EXPECTED_SEEDS), f"Paired seeds for {method}/{metric}")
    return [by_seed[seed] for seed in EXPECTED_SEEDS]


def rank_biserial(benefit_differences: Sequence[float]) -> float:
    from scipy.stats import rankdata

    nonzero = [float(value) for value in benefit_differences if float(value) != 0.0]
    if not nonzero:
        return 0.0
    ranks = rankdata([abs(value) for value in nonzero], method="average")
    positive = sum(float(rank) for rank, value in zip(ranks, nonzero) if value > 0.0)
    negative = sum(float(rank) for rank, value in zip(ranks, nonzero) if value < 0.0)
    denominator = positive + negative
    return 0.0 if denominator == 0.0 else (positive - negative) / denominator


def wilcoxon_two_sided(benefit_differences: Sequence[float]) -> Tuple[float, float]:
    from scipy.stats import wilcoxon

    values = [float(value) for value in benefit_differences]
    if all(value == 0.0 for value in values):
        return 0.0, 1.0
    try:
        result = wilcoxon(
            values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
    except TypeError:
        result = wilcoxon(
            values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            mode="auto",
        )
    return float(result.statistic), float(result.pvalue)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def bootstrap_mean_ci(
    differences: Sequence[float], rng: Any, replicates: int
) -> Tuple[float, float]:
    import numpy as np

    values = np.asarray(differences, dtype=np.float64)
    samples = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return float(lower), float(upper)


def format_array(values: Iterable[float]) -> str:
    return json.dumps([float(value) for value in values], separators=(",", ":"))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def analyze(records: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import numpy as np

    main_rows: List[Dict[str, Any]] = []
    for method in METHODS:
        row: Dict[str, Any] = {
            "method": method,
            "display_name": METHOD_DISPLAY[method],
            "n_seeds": 8,
            "seeds": ",".join(str(seed) for seed in EXPECTED_SEEDS),
        }
        for metric in METRICS:
            values = values_for(records, method, metric)
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_sample_sd"] = statistics.stdev(values)
        main_rows.append(row)

    rng = np.random.default_rng(ANALYSIS_SEED)
    significance_rows: List[Dict[str, Any]] = []
    for family in HYPOTHESES:
        family_rows: List[Dict[str, Any]] = []
        raw_p_values: List[float] = []
        for metric in family["metrics"]:
            method_a = str(family["method_a"])
            method_b = str(family["method_b"])
            values_a = values_for(records, method_a, metric)
            values_b = values_for(records, method_b, metric)
            raw_differences = [b - a for a, b in zip(values_a, values_b)]
            higher_is_better = METRIC_DIRECTIONS[metric] == "higher_is_better"
            benefit_differences = [
                difference if higher_is_better else -difference
                for difference in raw_differences
            ]
            statistic, raw_p = wilcoxon_two_sided(benefit_differences)
            ci_lower, ci_upper = bootstrap_mean_ci(
                benefit_differences, rng, BOOTSTRAP_REPLICATES
            )
            row = {
                "hypothesis": family["hypothesis"],
                "comparison": family["comparison"],
                "metric": metric,
                "better_direction": METRIC_DIRECTIONS[metric],
                "n_pairs": 8,
                "seeds": ",".join(str(seed) for seed in EXPECTED_SEEDS),
                "method_a": method_a,
                "method_a_display": METHOD_DISPLAY[method_a],
                "method_a_mean": statistics.mean(values_a),
                "method_a_sample_sd": statistics.stdev(values_a),
                "method_b": method_b,
                "method_b_display": METHOD_DISPLAY[method_b],
                "method_b_mean": statistics.mean(values_b),
                "method_b_sample_sd": statistics.stdev(values_b),
                "raw_delta_definition": "method_b_minus_method_a",
                "raw_paired_mean_difference_b_minus_a": statistics.mean(raw_differences),
                "raw_paired_median_difference_b_minus_a": statistics.median(raw_differences),
                "benefit_delta_definition": (
                    "method_b_minus_method_a"
                    if higher_is_better
                    else "method_a_minus_method_b"
                ),
                "benefit_oriented_mean_difference": statistics.mean(benefit_differences),
                "benefit_oriented_median_difference": statistics.median(benefit_differences),
                "wilcoxon_statistic": statistic,
                "wilcoxon_zero_method": "wilcox",
                "wilcoxon_method": "auto",
                "raw_p_value": raw_p,
                "holm_adjusted_p_value": None,
                "alpha": ALPHA,
                "holm_reject": None,
                "rank_biserial_effect_size": rank_biserial(benefit_differences),
                "rank_biserial_orientation": "positive_means_method_b_is_better",
                "bootstrap_ci_level": 0.95,
                "bootstrap_mean_benefit_delta_lower": ci_lower,
                "bootstrap_mean_benefit_delta_upper": ci_upper,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "analysis_seed": ANALYSIS_SEED,
                "method_a_values_by_seed": format_array(values_a),
                "method_b_values_by_seed": format_array(values_b),
                "raw_deltas_by_seed": format_array(raw_differences),
                "benefit_deltas_by_seed": format_array(benefit_differences),
            }
            family_rows.append(row)
            raw_p_values.append(raw_p)
        adjusted = holm_adjust(raw_p_values)
        for row, adjusted_p in zip(family_rows, adjusted):
            row["holm_adjusted_p_value"] = adjusted_p
            row["holm_reject"] = bool(adjusted_p < ALPHA)
            significance_rows.append(row)
    return main_rows, significance_rows


def markdown_number(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{float(value):.8g}"
    return str(value)


def write_report(
    path: Path,
    plan: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    main_rows: Sequence[Mapping[str, Any]],
    significance_rows: Sequence[Mapping[str, Any]],
) -> None:
    generated_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    lines = [
        "# V7 Core Two-Factor Significance Report — 8 Paired Seeds",
        "",
        f"Generated: {generated_at}",
        f"Pre-registered plan creation time: {plan['plan_created_at']}",
        "",
        "## Integrity result",
        "",
        "- Complete unique formal core runs: 32/32 (4 methods × 8 seeds).",
        "- Statistical unit: independently trained policy/random seed; n = 8 paired seeds.",
        "- Test episodes were not treated as independent statistical samples.",
        "- Legacy seeds use the authorized legacy V7 identity; new seeds use the authorized V7 8-seed extension identity.",
        "- The locked extension diff audit passed before analysis.",
        "",
        "## Main 8-seed descriptive results",
        "",
        "| Method | Metric | Mean | Sample SD | Direction |",
        "|---|---|---:|---:|---|",
    ]
    for row in main_rows:
        for metric in METRICS:
            lines.append(
                "| {method} | {metric} | {mean} | {sd} | {direction} |".format(
                    method=row["display_name"],
                    metric=metric,
                    mean=markdown_number(row[f"{metric}_mean"]),
                    sd=markdown_number(row[f"{metric}_sample_sd"]),
                    direction=METRIC_DIRECTIONS[metric],
                )
            )
    lines.extend(
        [
            "",
            "## Pre-defined paired tests",
            "",
            "Two-sided paired Wilcoxon signed-rank tests were applied to n=8 seed pairs. Holm correction was applied separately within H1, H2, and H3. Rank-biserial effect sizes and benefit-oriented deltas are positive when method B is better.",
            "",
            "| Hypothesis | Comparison | Metric | Direction | Raw delta B−A | Benefit delta | W | Raw p | Holm p | Holm reject at 0.05 | Rank-biserial | 95% bootstrap CI of mean benefit delta |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in significance_rows:
        lines.append(
            "| {hypothesis} | {comparison} | {metric} | {direction} | {raw_delta} | {benefit_delta} | {w} | {raw_p} | {holm_p} | {reject} | {effect} | [{lower}, {upper}] |".format(
                hypothesis=row["hypothesis"],
                comparison=row["comparison"],
                metric=row["metric"],
                direction=row["better_direction"],
                raw_delta=markdown_number(row["raw_paired_mean_difference_b_minus_a"]),
                benefit_delta=markdown_number(row["benefit_oriented_mean_difference"]),
                w=markdown_number(row["wilcoxon_statistic"]),
                raw_p=markdown_number(row["raw_p_value"]),
                holm_p=markdown_number(row["holm_adjusted_p_value"]),
                reject=markdown_number(row["holm_reject"]),
                effect=markdown_number(row["rank_biserial_effect_size"]),
                lower=markdown_number(row["bootstrap_mean_benefit_delta_lower"]),
                upper=markdown_number(row["bootstrap_mean_benefit_delta_upper"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Raw and Holm-adjusted p-values are both reported; no result was selected for reporting based on significance.",
            "- A p-value below 0.05 is not a 95% probability that a method is better.",
            "- Lower-is-better metrics were sign-adjusted only for benefit-oriented deltas and effect-size orientation; raw B−A deltas are also retained.",
            "- H3 tests incremental path-quality/path-efficiency benefit of training-only A* cost-to-go supervision on top of Dynamic Reward; it is not pre-registered as a claim that A* primarily improves task completion.",
            "- This report is an objective statistical output and does not automatically generate a manuscript claim of significant improvement.",
            "",
            "## Formal run inventory",
            "",
        ]
    )
    for record in records:
        lines.append(
            f"- `{record['method']}` seed `{record['seed']}`: `{record['run_directory']}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the complete frozen V7 4-method × 8-seed formal result set"
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ANALYSIS_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    plan = verify_plan(args.plan.resolve(), output_root)
    records = collect_complete_runs(output_root)

    # No output is created until the complete 32-run identity and structure audit passes.
    main_rows, significance_rows = analyze(records)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    main_fields = ["method", "display_name", "n_seeds", "seeds"]
    for metric in METRICS:
        main_fields.extend([f"{metric}_mean", f"{metric}_sample_sd"])
    significance_fields = list(significance_rows[0].keys())

    main_path = output_dir / "main_results_8seed.csv"
    significance_path = output_dir / "significance_results.csv"
    report_path = output_dir / "significance_report.md"
    write_csv(main_path, main_rows, main_fields)
    write_csv(significance_path, significance_rows, significance_fields)
    write_report(report_path, plan, records, main_rows, significance_rows)

    print(main_path)
    print(significance_path)
    print(report_path)


if __name__ == "__main__":
    main()
