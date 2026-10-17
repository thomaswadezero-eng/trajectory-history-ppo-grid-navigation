#!/usr/bin/env python
"""Shared, model-free Moving AI 64x64 candidate construction rules."""

from __future__ import print_function

import hashlib
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from protocol_utils import IntegrityError, sha256_file

SUBSETS = ("DAO", "Room", "Maze")
SPLITS = ("train", "validation", "test")
SYMBOL_MAPPING = {".": 0, "G": 0, "S": 0, "@": 1, "O": 1, "T": 1, "W": 1}
MOVE_DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CANDIDATE_HORIZONS = (180, 240, 300, 360, 400, 420, 450)
CROP_RULE_ID = "CANONICAL_64_V1_AXIS_LOW_HIGH_MIDPOINT_FLOOR_CLAMP"


def stable_hash(*parts):
    text = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def parse_map(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if len(lines) < 5 or not lines[1].startswith("height ") or not lines[2].startswith("width ") or lines[3].strip().lower() != "map":
        raise IntegrityError("invalid Moving AI map header: %s" % path)
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    rows = lines[4:]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise IntegrityError("Moving AI map dimensions do not match header: %s" % path)
    unknown = set("".join(rows)) - set(SYMBOL_MAPPING)
    if unknown:
        raise IntegrityError("unknown map symbols %r: %s" % (sorted(unknown), path))
    return {"path": Path(path).resolve(), "height": height, "width": width, "rows": rows, "sha256": sha256_file(path)}


def scenario_rows(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        if handle.readline().strip() != "version 1":
            raise IntegrityError("unsupported scenario header: %s" % path)
        for line_number, line in enumerate(handle, 2):
            fields = line.strip().split()
            if not fields:
                continue
            if len(fields) != 9:
                raise IntegrityError("scenario field count mismatch: %s:%d" % (path, line_number))
            yield line_number, fields


def axis_window(a, b, length, size):
    low = max(0, max(a, b) - (size - 1))
    high = min(min(a, b), length - size)
    if low > high:
        return None
    preferred = int(math.floor((a + b) / 2.0 - (size - 1) / 2.0))
    return min(max(preferred, low), high)


def canonical_crop(start_y, start_x, goal_y, goal_x, height, width, size):
    top = axis_window(start_y, goal_y, height, size)
    left = axis_window(start_x, goal_x, width, size)
    return None if top is None or left is None else (top, left)


def binary_crop(parsed, top, left, size):
    return np.asarray([[SYMBOL_MAPPING[symbol] for symbol in row[left:left + size]] for row in parsed["rows"][top:top + size]], dtype=np.uint8)


def shortest_path(grid, start, goal):
    if start == goal or grid[start] != 0 or grid[goal] != 0:
        return None
    height, width = grid.shape
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        current, distance = queue.popleft()
        if current == goal:
            return distance
        for dr, dc in MOVE_DELTAS:
            nxt = (current[0] + dr, current[1] + dc)
            if 0 <= nxt[0] < height and 0 <= nxt[1] < width and grid[nxt] == 0 and nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, distance + 1))
    return None


def collect_candidates(raw_root, construction_seed, horizon):
    by_subset_map = {subset: defaultdict(list) for subset in SUBSETS}
    inventory = {subset: [] for subset in SUBSETS}
    for subset in SUBSETS:
        map_dir = Path(raw_root) / subset / "maps"
        scenario_dir = Path(raw_root) / subset / "scenarios"
        for map_path in sorted(map_dir.glob("*.map")):
            scenario_path = scenario_dir / (map_path.name + ".scen")
            if not scenario_path.is_file():
                raise IntegrityError("missing paired scenario file: %s" % scenario_path)
            parsed = parse_map(map_path)
            scenario_sha = sha256_file(scenario_path)
            inventory[subset].append({
                "map_path": str(parsed["path"]), "map_sha256": parsed["sha256"],
                "scenario_path": str(scenario_path.resolve()), "scenario_sha256": scenario_sha,
                "height": parsed["height"], "width": parsed["width"],
            })
            if parsed["height"] < 64 or parsed["width"] < 64:
                continue
            signatures = set()
            for line_number, fields in scenario_rows(scenario_path):
                map_name = fields[1]
                width, height = int(fields[2]), int(fields[3])
                start_x, start_y, goal_x, goal_y = (int(fields[index]) for index in range(4, 8))
                if map_name != map_path.name or width != parsed["width"] or height != parsed["height"]:
                    raise IntegrityError("scenario/map identity mismatch: %s:%d" % (scenario_path, line_number))
                if not (0 <= start_x < width and 0 <= goal_x < width and 0 <= start_y < height and 0 <= goal_y < height):
                    raise IntegrityError("scenario endpoint outside source map: %s:%d" % (scenario_path, line_number))
                if start_x == goal_x and start_y == goal_y:
                    continue
                window = canonical_crop(start_y, start_x, goal_y, goal_x, height, width, 64)
                if window is None:
                    continue
                top, left = window
                start = (start_y - top, start_x - left)
                goal = (goal_y - top, goal_x - left)
                signature = (parsed["sha256"], top, left, start, goal)
                if signature in signatures:
                    continue
                signatures.add(signature)
                crop = binary_crop(parsed, top, left, 64)
                length = shortest_path(crop, start, goal)
                if length is None or length <= 0:
                    continue
                if horizon is not None and length > horizon:
                    continue
                by_subset_map[subset][map_path.name].append({
                    "subset": subset, "source_map_name": map_path.name, "source_map_path": str(parsed["path"]),
                    "source_map_sha256": parsed["sha256"], "source_scenario_path": str(scenario_path.resolve()),
                    "source_scenario_sha256": scenario_sha, "source_scenario_line": line_number,
                    "source_width": width, "source_height": height, "crop_top": top, "crop_left": left,
                    "start": list(start), "goal": list(goal), "shortest_path_length": int(length),
                    "ranking_sha256": stable_hash(construction_seed, subset, parsed["sha256"], line_number, top, left, start, goal),
                })
    return by_subset_map, inventory


def simulate_source_map_split(by_subset_map, protocol):
    seed = int(protocol["data"]["construction_seed"])
    quotas = protocol["data"]["draft_source_map_quotas"]
    assignment = {split: {subset: [] for subset in SUBSETS} for split in SPLITS}
    shortages = []
    for subset in SUBSETS:
        eligible = [name for name, rows in by_subset_map[subset].items() if rows]
        eligible.sort(key=lambda name: stable_hash(seed, "source-map-split", subset, by_subset_map[subset][name][0]["source_map_sha256"]))
        cursor = 0
        for split in SPLITS:
            requested = int(quotas[split][subset])
            chosen = eligible[cursor:cursor + requested]
            assignment[split][subset] = chosen
            if len(chosen) != requested:
                shortages.append({"split": split, "subset": subset, "requested": requested, "available": len(chosen)})
            cursor += requested
    return assignment, shortages


def split_source_maps(by_subset_map, protocol):
    assignment, shortages = simulate_source_map_split(by_subset_map, protocol)
    if shortages:
        item = shortages[0]
        raise IntegrityError("insufficient eligible %s source maps for %s: %d/%d" % (item["subset"], item["split"], item["available"], item["requested"]))
    return assignment


def select_tasks(by_subset_map, assignment, protocol):
    seed = int(protocol["data"]["construction_seed"])
    targets = protocol["data"]["draft_task_targets"]
    cap = int(protocol["data"]["maximum_tasks_per_source_map"])
    selected = {split: [] for split in SPLITS}
    for split in SPLITS:
        for subset in SUBSETS:
            pool = []
            for map_name in assignment[split][subset]:
                pool.extend(by_subset_map[subset][map_name])
            pool.sort(key=lambda row: stable_hash(seed, split, row["ranking_sha256"]))
            target = int(targets[split][subset])
            counts = defaultdict(int)
            chosen = []
            for row in pool:
                name = row["source_map_name"]
                if counts[name] >= cap:
                    continue
                chosen.append(dict(row))
                counts[name] += 1
                if len(chosen) == target:
                    break
            if len(chosen) != target:
                raise IntegrityError("insufficient deterministic candidates for %s/%s: %d/%d" % (split, subset, len(chosen), target))
            selected[split].extend(chosen)
    return selected
