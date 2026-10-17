#!/usr/bin/env python
"""Shared read-only helpers for PPO+Progress post-run processing."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V7_ROOT = PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7"
V7_8SEED_ROOT = PROJECT_ROOT / "03_Code" / "final_two_factor_experiment_v7_8seed"
PROGRESS_SOURCE_ROOT = PROJECT_ROOT / "03_Code" / "ppo_progress_8seed_v2"
COMPONENT_STAT_ROOT = PROJECT_ROOT / "07_Output" / "component_statistics_4families_final_20260924_v1"

PROGRESS_FORMAL_ROOT = Path(r"redacted_local_root\revision_progress_only_8seed_v2\PPO_PROGRESS")
V7_FORMAL_ROOT = Path(r"redacted_local_root\final_two_factor_experiment_v7")

SEEDS = (42, 1004, 6010, 68882, 19816, 74494, 81399, 31524)
METHOD_PROGRESS = "PPO_PROGRESS"
EXPECTED_INPUT_SHA = {
    "training_tasks": "D29146F90A9B73749B1FADAA0E563618E547A817422EB7A5D9D2A6F18DAFDBF5",
    "validation_tasks": "AAC69C346AEEFB3FDD27E8576493A60D30CFB0989CA23634750D7118D417194B",
    "test_tasks": "CC3C33B00772F8D1B6314FFADC94F8B7D5B86230C1BF445C5C6BCF15570E8397",
    "map_split": "637792B5F70F086AD534E3CAE70C03F2640EE402C8F8D09B16979E1910C4EBB1",
    "task_split": "268A76CF6D4BD2AC6A2BA6CFB6B281527326F06D0B0BA148BAD453D90414F693",
}
EXPECTED_SHA = {
    "v7_program": "96093A6B5F024F5D9505ACCEBEAE7592700228BA17587BF0590092A241B7544D",
    "v7_config_file": "CF229DE128FF16E29B73308A7E9B2756DE51294671573F067306274EDA8D4166",
    "v7_protocol_file": "2C8BAFE1A71695627B4F723583153067C0E1C7D08F905538E5BF8793A96B4C9B",
    "v7_config_identity": "489431F4C5609FC1F7148D56D152D22C1252F6B6AF79CE8AE09716E1F8D84B8D",
    "v7_protocol_identity": "7887F254E31A1C6A0CDB95994EF3C18CC3A8A43775D8D6BD16475FDFB042ADB7",
    "v7_8seed_program": "19290D3A23DCC8E10FD4264EF9D8A3FEC4A691CFFD8A69C6FDEE8DE65C8EF8DD",
    "progress_program": "E7F0E8C38BD403D1DF5E836307484B1A9B31B6D13EE6AD33532F727787C1C129",
    "progress_config_file": "F35D7C275367ADED734BE4CFAC042D682537A4825EFD592BFF1800348380D62F",
    "progress_protocol_file": "554E8BD6AB6040820244B3F451F7134C87A19BFE4C4882EE39D080E1DEB6F5DC",
    "progress_config_identity": "F69BC09C6AFF103A1D8A665B5E8C948F02DA15E36AD37807B21DBDF7034DAD5A",
    "progress_protocol_identity": "F43DC728A82894344357F2EEBAC039E22B3F29A4EBC5E64085DC674590912C66",
    "component_four_family_program": "8E3C5C1EE1D62C701248BA485E87D435A0DF672D03D8FEFCF3C89B68424C6DAB",
    "main_significance_program": "5AA340D550625BB9D54068FB318FC827AA6B280FAB2EEBE7055C9C9BC146BA29",
    "forward_ablation_plan": "74B80E7CD53061FD6134EDF5FDAC1D647053D253E99C75EAF85EB80CEF94AAA1",
}


class PostrunError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PostrunError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Dict[str, Any]:
    require(path.is_file(), "missing JSON file: {}".format(path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "expected JSON object: {}".format(path))
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def read_csv(path: Path) -> List[Dict[str, str]]:
    require(path.is_file(), "missing CSV file: {}".format(path))
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_number(value: Any, label: str) -> float:
    require(not isinstance(value, bool), "boolean is not numeric: {}".format(label))
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise PostrunError("non-numeric value for {}: {!r}".format(label, value))
    require(math.isfinite(number), "non-finite value for {}".format(label))
    return number


def assert_finite_tree(value: Any, label: str = "root") -> None:
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite value at {}".format(label))
    elif isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, "{}.{}".format(label, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, "{}[{}]".format(label, index))


def close(left: Any, right: Any, label: str, tolerance: float = 1e-12) -> None:
    a = finite_number(left, label + ".left")
    b = finite_number(right, label + ".right")
    require(math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance),
            "metric mismatch for {}: {} vs {}".format(label, a, b))


def static_authority_checks() -> List[Dict[str, Any]]:
    paths = {
        "v7_program": V7_ROOT / "final_two_factor_experiment_v7.py",
        "v7_config_file": V7_ROOT / "experiment_config.json",
        "v7_protocol_file": V7_ROOT / "experiment_protocol.json",
        "v7_8seed_program": V7_8SEED_ROOT / "final_two_factor_experiment_v7_8seed.py",
        "main_significance_program": V7_8SEED_ROOT / "analyze_significance.py",
        "progress_program": PROGRESS_SOURCE_ROOT / "ppo_progress_8seed_v2.py",
        "progress_config_file": PROGRESS_SOURCE_ROOT / "experiment_config.json",
        "progress_protocol_file": PROGRESS_SOURCE_ROOT / "experiment_protocol.json",
        "component_four_family_program": COMPONENT_STAT_ROOT / "recalculate_component_statistics_4families.py",
        "forward_ablation_plan": Path(__file__).resolve().parent / "forward_ablation_plan_v1.json",
    }
    rows = []
    for identity, path in paths.items():
        actual = sha256_file(path)
        expected = EXPECTED_SHA[identity]
        require(actual == expected, "authority SHA mismatch: {}: {}".format(path, actual))
        rows.append({"check": identity, "status": "PASS", "evidence": str(path), "value": actual})

    config = read_json(PROGRESS_SOURCE_ROOT / "experiment_config.json")
    protocol = read_json(PROGRESS_SOURCE_ROOT / "experiment_protocol.json")
    expected_components = {"progress": True, "revisit": False, "oscillation": False, "stagnation": False}
    for payload, name in ((config, "config"), (protocol, "protocol")):
        require(payload.get("official_seeds") == list(SEEDS), "{} seed identity mismatch".format(name))
        definition = payload.get("configurations", {}).get(METHOD_PROGRESS, {})
        require(definition.get("reward_components") == expected_components,
                "{} reward switches mismatch".format(name))
        require(definition.get("astar_supervision") is False, "{} A* must be disabled".format(name))
        require(definition.get("planning_loss") is False, "{} planning loss must be disabled".format(name))
        require(definition.get("distillation_loss") is False, "{} distillation loss must be disabled".format(name))
    require(config.get("config_sha256") == EXPECTED_SHA["progress_config_identity"], "progress config identity mismatch")
    require(protocol.get("protocol_sha256") == EXPECTED_SHA["progress_protocol_identity"], "progress protocol identity mismatch")
    revision = protocol.get("revision_identity", {})
    require(int(revision.get("interaction_budget", -1)) == 900000, "interaction budget drift")
    require(int(revision.get("validation_interval", -1)) == 100000, "validation interval drift")
    require(int(revision.get("test_episodes", -1)) == 1000, "test episode drift")
    rows.append({"check": "ppo_progress_method_semantics", "status": "PASS",
                 "evidence": "immutable adapter/config/protocol", "value": json.dumps(expected_components, sort_keys=True)})
    plan = read_json(Path(__file__).resolve().parent / "forward_ablation_plan_v1.json")
    require(plan.get("status") == "FROZEN_BEFORE_PPO_PROGRESS_FORMAL_RESULTS", "forward plan is not frozen")
    require(plan.get("seeds") == list(SEEDS) and int(plan.get("n_pairs", -1)) == 8, "forward plan seed identity mismatch")
    families = plan.get("families", [])
    require([item.get("family_id") for item in families] == ["F1_PPO_TO_PPO_PROGRESS", "F2_PPO_PROGRESS_TO_PPO_REWARD"], "forward family identity mismatch")
    require(all(item.get("metrics") == ["success_rate", "spl", "timeout_rate"] for item in families), "forward metric family mismatch")
    require(plan.get("multiple_comparison_correction", {}).get("global_six_test_family_forbidden") is True,
            "forward plan does not forbid a global six-test family")
    input_paths = {
        "training_tasks": PROJECT_ROOT / "03_Code" / "training_tasks_v2.json",
        "validation_tasks": PROJECT_ROOT / "03_Code" / "validation_tasks_v2.json",
        "test_tasks": PROJECT_ROOT / "03_Code" / "test_tasks_v9_unseen_final.json",
        "map_split": V7_ROOT / "map_split.json",
        "task_split": V7_ROOT / "task_split.json",
    }
    for name, path in input_paths.items():
        actual = sha256_file(path)
        require(actual == EXPECTED_INPUT_SHA[name], "frozen input SHA mismatch: {}".format(path))
        rows.append({"check": "input_{}".format(name), "status": "PASS", "evidence": str(path), "value": actual})
    return rows


def metric_aggregate(values: Sequence[float]) -> Tuple[float, float]:
    require(len(values) == 8, "aggregate requires exactly eight seed values")
    return statistics.mean(values), statistics.stdev(values)


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


def wilcoxon_two_sided(benefit_differences: Sequence[float]) -> Tuple[float, float, str]:
    from scipy.stats import wilcoxon
    values = [float(value) for value in benefit_differences]
    require(len(values) == 8, "Wilcoxon test requires exactly eight paired seeds")
    if all(value == 0.0 for value in values):
        return 0.0, 1.0, "all_zero_shortcut"
    requested_method = "auto" if any(value == 0.0 for value in values) else "exact"
    try:
        result = wilcoxon(values, zero_method="wilcox", correction=False,
                          alternative="two-sided", method=requested_method)
    except TypeError:
        result = wilcoxon(values, zero_method="wilcox", correction=False,
                          alternative="two-sided", mode=requested_method)
    label = "exact" if requested_method == "exact" else "auto_zero_fallback"
    return float(result.statistic), float(result.pvalue), label


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    require(count == 3, "each forward-ablation Holm family must contain exactly three tests")
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def bootstrap_mean_ci(differences: Sequence[float], rng: Any,
                      replicates: int = 100000) -> Tuple[float, float]:
    import numpy as np
    values = np.asarray([float(value) for value in differences], dtype=np.float64)
    require(values.shape == (8,), "bootstrap requires exactly eight paired differences")
    indices = rng.integers(0, 8, size=(replicates, 8))
    means = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(lower), float(upper)


def ensure_new_output_dir(path: Path) -> None:
    if path.exists():
        require(path.is_dir(), "output path is not a directory: {}".format(path))
        require(not any(path.iterdir()), "refusing non-empty output directory: {}".format(path))
    else:
        path.mkdir(parents=True)
