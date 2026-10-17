"""Pure-file utilities and frozen identities for the external-baseline extension.

This module deliberately imports only the Python standard library.  Importing it
cannot create an environment, load a checkpoint, initialize CUDA, or execute a
model.  Runtime entrypoints import the legacy implementations only after their
own explicit confirmation gates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from xml.etree import ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(r"redacted_local_root\revision_external_baselines_8seed_v1")
RUNS_ROOT = OUTPUT_ROOT / "outputs"

METHODS: Tuple[str, ...] = ("DDQN", "D3QN", "C51", "MaskablePPO")
SEEDS: Tuple[int, ...] = (42, 1004, 6010, 68882, 19816, 74494, 81399, 31524)
LEGACY_SEEDS: Tuple[int, ...] = (42, 1004, 6010)
NEW_SEEDS: Tuple[int, ...] = (68882, 19816, 74494, 81399, 31524)
METRICS: Tuple[str, ...] = (
    "success_rate",
    "spl",
    "path_efficiency",
    "average_steps",
    "average_path_length",
    "collision_rate",
    "timeout_rate",
    "episode_reward",
)

TASK_HASHES = {
    "training_tasks": "D29146F90A9B73749B1FADAA0E563618E547A817422EB7A5D9D2A6F18DAFDBF5",
    "validation_tasks": "AAC69C346AEEFB3FDD27E8576493A60D30CFB0989CA23634750D7118D417194B",
    "test_tasks": "CC3C33B00772F8D1B6314FFADC94F8B7D5B86230C1BF445C5C6BCF15570E8397",
}
TASK_PATHS = {
    "training_tasks": PROJECT_ROOT / "03_Code" / "training_tasks_v2.json",
    "validation_tasks": PROJECT_ROOT / "03_Code" / "validation_tasks_v2.json",
    "test_tasks": PROJECT_ROOT / "03_Code" / "test_tasks_v9_unseen_final.json",
}

LEGACY_ROOTS = {
    "DDQN": Path(r"redacted_local_root\comparison_baselines_v1_outputs\DDQN"),
    "D3QN": Path(r"redacted_local_root\comparison_baselines_v1_outputs\D3QN"),
    "C51": Path(r"redacted_local_root\comparison_baselines_c51_outputs\C51"),
    "MaskablePPO": PROJECT_ROOT / "03_Code" / "external_baselines" / "outputs" / "MaskablePPO",
}
FULL_ROOT = Path(r"redacted_local_root\final_two_factor_experiment_v7\FULL")

AUTHORITY = {
    "DDQN": {
        "program": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "main.py",
        "program_sha256": "4D19655BEA498C10CE7BB21E21EAC7577AA789C297FB7B260F6557FD766F0FBF",
        "runtime": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "runner.py",
        "runtime_sha256": "7BCA589666B63A2784CAD1B6CE7B955EF6A10FB0ED70AFAB6C21022B84CE8A0F",
        "framework_manifest": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "framework_hash_manifest.json",
        "framework_sha256": "E0523F2220A73ADBC59261F469ABA11DA2111FDE255EFFBF0D2088AA88950329",
        "protocol": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "comparison_protocol_v1.json",
        "protocol_sha256": "CAD180D2E69477153DE8A28F8E1FA632DC6BDF92D1BCD7757F859DF38C474604",
        "config": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "configs" / "ddqn_v1.json",
        "config_sha256": "3D0556A6F70A38731162CFB97C89149926F6268484F909AB3A249412FD3408B8",
    },
    "D3QN": {
        "program": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "main.py",
        "program_sha256": "4D19655BEA498C10CE7BB21E21EAC7577AA789C297FB7B260F6557FD766F0FBF",
        "runtime": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "runner.py",
        "runtime_sha256": "7BCA589666B63A2784CAD1B6CE7B955EF6A10FB0ED70AFAB6C21022B84CE8A0F",
        "framework_manifest": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "framework_hash_manifest.json",
        "framework_sha256": "E0523F2220A73ADBC59261F469ABA11DA2111FDE255EFFBF0D2088AA88950329",
        "protocol": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "comparison_protocol_v1.json",
        "protocol_sha256": "CAD180D2E69477153DE8A28F8E1FA632DC6BDF92D1BCD7757F859DF38C474604",
        "config": PROJECT_ROOT / "03_Code" / "comparison_baselines_v1" / "configs" / "d3qn_v1.json",
        "config_sha256": "CBC3C5E47805C79BB539E1BBD10D048C381454291CD7A78E21CCA93B71C6BAC2",
    },
    "C51": {
        "program": PROJECT_ROOT / "03_Code" / "comparison_baselines_c51" / "run_c51_barn.py",
        "program_sha256": "62D8FB2B214D12E96F73C587835DEC673E4EDAB8B511D86EC262F12490540938",
        "algorithm": PROJECT_ROOT / "03_Code" / "comparison_baselines_c51" / "c51_algorithm.py",
        "algorithm_sha256": "8E8C5E602DFA30C3E6E3D851C7E9D1D04AC44A9515992573CF5E2B507CBB1BA5",
        "replay": PROJECT_ROOT / "03_Code" / "comparison_baselines_c51" / "c51_replay.py",
        "replay_sha256": "51149CD01E07FEA119E0C10E35D52768CC4A6E55FD96CCC3C8765D86EDF5F5A9",
        "protocol": PROJECT_ROOT / "03_Code" / "comparison_baselines_c51" / "barn_protocol.py",
        "protocol_sha256": "1B7B88235B97A079A97911C5515DD655C63968C3F3DA96C8985A2F44847DEC7F",
        "config": PROJECT_ROOT / "03_Code" / "comparison_baselines_c51" / "config" / "c51_barn_formal_v1.json",
        "config_sha256": "77FE5C3E17E137D00A06B1C9BC1D3510F8C7BF48F4E93D12BC03FD1ECB5C81C6",
    },
    "MaskablePPO": {
        "program": PROJECT_ROOT / "03_Code" / "external_baselines" / "sb3_external_baseline.py",
        "program_sha256": "6499A0F4E3108C0790EFD2E46F3536B36A783C8447D22CC61EF5FDC652A2BDB8",
        "config": PROJECT_ROOT / "03_Code" / "external_baselines" / "config" / "sb3_external_baseline_config_v1.json",
        "config_sha256": "ACC652D56B4D0400F740219E97BFA48C021BE495B57EBAF741E445B1D9EEC041",
    },
}

PAPER_AUTHORITY = PROJECT_ROOT / "07_Output" / "论文0827_english_v8_final_language_minimal_patch_text_revision_codex_v4.docx"
PAPER_AUTHORITY_SHA256 = "A17E5CEC6DCE50002D3310FC02094534AF74892AB35E9E7B43068D520CF8B63B"


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AuditError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def result_name(method: str, seed: int) -> str:
    return f"formal_result_{method}_seed{seed}.json"


def discover_run(root: Path, seed: int) -> Tuple[str, Path | None, List[str]]:
    seed_dir = root / f"seed_{seed}"
    if not seed_dir.is_dir():
        return "NOT_REUSABLE", None, [f"missing seed directory: {seed_dir}"]
    runs = sorted(path for path in seed_dir.glob("run_*") if path.is_dir())
    if len(runs) == 0:
        return "NOT_REUSABLE", None, ["no run directory"]
    if len(runs) > 1:
        return "AMBIGUOUS", None, [f"multiple candidate runs: {len(runs)}"]
    return "CANDIDATE", runs[0], []


def close_float(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def load_episode_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "record_type" in rows[0]:
        rows = [row for row in rows if row.get("record_type") == "episode" and row.get("split") == "test"]
    return rows


def first_existing(paths: Iterable[Path]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise AuditError("none of the expected files exists: " + ", ".join(str(path) for path in paths))


def test_csv_path(run: Path) -> Path:
    return first_existing((run / "test" / "evaluation_metrics.csv", run / "test" / "evaluation_episodes.csv"))


def recompute_metrics(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    if len(rows) != 1000:
        raise AuditError(f"expected 1000 test episode rows, found {len(rows)}")

    def number(row: Mapping[str, str], key: str) -> float:
        value = row.get(key)
        if value is None:
            value = row.get(key.upper())
        if value is None or value == "":
            raise AuditError(f"missing numeric field {key}")
        return float(value)

    success = [int(number(row, "success")) for row in rows]
    collision = [int(number(row, "collision")) for row in rows]
    timeout = [int(number(row, "timeout")) for row in rows]
    steps = [number(row, "steps") for row in rows]
    paths = [number(row, "path_length") for row in rows]
    spl = [number(row, "spl") for row in rows]
    efficiency = []
    rewards = []
    for row in rows:
        pe = row.get("path_efficiency")
        if pe not in (None, ""):
            efficiency.append(float(pe))
        elif int(number(row, "success")) == 1:
            optimal = number(row, "optimal_path_length")
            path_length = number(row, "path_length")
            efficiency.append(optimal / max(optimal, path_length))
        reward = row.get("episode_reward")
        if reward not in (None, ""):
            rewards.append(float(reward))
    payload: Dict[str, Any] = {
        "episode_count": len(rows),
        "success_count": sum(success),
        "collision_count": sum(collision),
        "timeout_count": sum(timeout),
        "success_rate": sum(success) / len(rows),
        "collision_rate": sum(collision) / len(rows),
        "timeout_rate": sum(timeout) / len(rows),
        "average_steps": statistics.fmean(steps),
        "average_path_length": statistics.fmean(paths),
        "spl": statistics.fmean(spl),
        "path_efficiency": statistics.fmean(efficiency),
        "path_efficiency_success_count": len(efficiency),
    }
    if rewards:
        payload["episode_reward"] = statistics.fmean(rewards)
    return payload


def compare_metrics(recomputed: Mapping[str, Any], reported: Mapping[str, Any]) -> List[str]:
    problems: List[str] = []
    for key in ("episode_count", "success_count", "collision_count", "timeout_count"):
        if int(recomputed[key]) != int(reported.get(key, -1)):
            problems.append(f"{key} recomputation mismatch")
    for key in ("success_rate", "collision_rate", "timeout_rate", "average_steps", "average_path_length", "spl", "path_efficiency"):
        if not close_float(recomputed[key], reported.get(key), 1e-9):
            problems.append(f"{key} recomputation mismatch")
    if "average_episode_reward" in reported and "episode_reward" in recomputed:
        if not close_float(recomputed["episode_reward"], reported["average_episode_reward"], 1e-9):
            problems.append("episode_reward recomputation mismatch")
    return problems


def test_task_ids() -> List[str]:
    payload = read_json(TASK_PATHS["test_tasks"])
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 1000:
        raise AuditError("frozen test task file does not contain 1000 tasks")
    return [str(task["task_id"]) for task in tasks]


def verify_test_order(rows: Sequence[Mapping[str, str]]) -> bool:
    return [str(row.get("task_id", "")) for row in rows] == test_task_ids()


def validation_interactions(run: Path) -> Tuple[int, ...]:
    root = run / "validation"
    interactions: List[int] = []
    if root.is_dir():
        for path in root.iterdir():
            if path.is_dir() and path.name.startswith("interaction_"):
                digits = "".join(ch for ch in path.name if ch.isdigit())
                if digits:
                    interactions.append(int(digits))
    return tuple(sorted(interactions))


def extract_external_table(docx_path: Path) -> Dict[str, List[float]]:
    """Extract the current external-baseline table using OOXML only."""
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(docx_path) as archive:
        tree = ET.fromstring(archive.read("word/document.xml"))
    for table in tree.findall(".//w:tbl", namespace):
        parsed: Dict[str, List[float]] = {}
        table_text = " ".join(node.text or "" for node in table.findall(".//w:t", namespace))
        if not all(name in table_text for name in METHODS):
            continue
        for row in table.findall("./w:tr", namespace):
            cells = []
            for cell in row.findall("./w:tc", namespace):
                cells.append("".join(node.text or "" for node in cell.findall(".//w:t", namespace)).strip())
            if not cells or cells[0] not in (*METHODS, "FULL"):
                continue
            values: List[float] = []
            for cell in cells[1:]:
                normalized = cell.replace("−", "-").replace("±", "+-").replace("��", "+-")
                mean = normalized.split("+-", 1)[0].strip()
                values.append(float(mean))
            parsed[cells[0]] = values
        if all(name in parsed for name in (*METHODS, "FULL")):
            return parsed
    raise AuditError("current manuscript external-baseline table not found")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def require_attempt_path(path: Path, method: str, seed: int) -> Path:
    resolved = path.resolve()
    expected_parent = (RUNS_ROOT / method / f"seed_{seed}").resolve()
    if resolved.parent != expected_parent or not resolved.name.startswith("run_"):
        raise AuditError(f"attempt path is outside the isolated revision layout: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
