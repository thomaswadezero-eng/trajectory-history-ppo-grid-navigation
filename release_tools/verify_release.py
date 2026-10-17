"""Read-only public-subset integrity and descriptive-summary verification. Python 3.8+."""
import argparse
import csv
from decimal import Decimal, localcontext, ROUND_HALF_EVEN
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import sys

SEEDS = {42, 1004, 6010, 68882, 19816, 74494, 81399, 31524}
METRICS = ("success_rate", "spl", "path_efficiency", "average_steps", "timeout_rate")
HEX = re.compile(r"^[0-9a-fA-F]{64}$")
MAX_FILE = 10 * 1024 * 1024
MAX_TOTAL = 20 * 1024 * 1024

class ReleaseError(Exception):
    pass

def digest(data):
    return hashlib.sha256(data).hexdigest().upper()

def safe_file(root, name):
    """Reject ambiguous names and links BEFORE reading any target bytes."""
    root = Path(root).absolute()
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ReleaseError("Invalid relative path")
    parts = name.split("/")
    if PureWindowsPath(name).is_absolute() or PurePosixPath(name).is_absolute():
        raise ReleaseError("Absolute path refused")
    if any(p in ("", ".", "..") or p.endswith((".", " ")) for p in parts):
        raise ReleaseError("Traversal/ambiguous component refused")
    if any(re.match(r"(?i)^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)", p) for p in parts):
        raise ReleaseError("Device path refused")
    target = root.joinpath(*parts)
    for q in [root] + [root.joinpath(*parts[:i]) for i in range(1, len(parts)+1)]:
        try:
            s = q.lstat()
        except OSError:
            raise ReleaseError("Missing/inaccessible file: " + name)
        if stat.S_ISLNK(s.st_mode) or getattr(s, "st_file_attributes", 0) & 1024:
            raise ReleaseError("Link/reparse component refused: " + name)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        raise ReleaseError("Repository boundary exceeded")
    except OSError:
        raise ReleaseError("Cannot resolve file boundary: " + name)
    if not target.is_file():
        raise ReleaseError("Not a regular file: " + name)
    return target

def parse_csv(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))

def verify_manifest(root):
    manifest = safe_file(root, "PACKAGE_MANIFEST.csv")
    if manifest.stat().st_size > MAX_FILE:
        raise ReleaseError("Manifest too large")
    rows = parse_csv(manifest.read_bytes())
    if not rows:
        raise ReleaseError("Empty manifest")
    names, total = {}, 0
    for r in rows:
        if set(r) != {"release_repo_path", "bytes", "sha256"}:
            raise ReleaseError("Invalid manifest structure")
        name = r["release_repo_path"]
        p = safe_file(root, name)
        if name == "PACKAGE_MANIFEST.csv" or name.casefold() in names:
            raise ReleaseError("Duplicate/self manifest path")
        if not HEX.fullmatch(r["sha256"] or ""):
            raise ReleaseError("Invalid SHA-256 field")
        if not re.fullmatch(r"[0-9]+", r["bytes"] or ""):
            raise ReleaseError("Invalid size field")
        size = p.stat().st_size
        total += size
        if size > MAX_FILE or total > MAX_TOTAL or size != int(r["bytes"]):
            raise ReleaseError("Size/budget mismatch: " + name)
        if digest(p.read_bytes()) != r["sha256"].upper():
            raise ReleaseError("SHA mismatch: " + name)
        names[name.casefold()] = r
    actual = set()
    for folder, dirs, files in os.walk(str(root), followlinks=False):
        for d in list(dirs):
            if d == ".git":
                dirs.remove(d)
                continue
            q = Path(folder) / d
            s = q.lstat()
            if stat.S_ISLNK(s.st_mode) or getattr(s, "st_file_attributes", 0) & 1024:
                raise ReleaseError("Link directory refused")
            if d == "__pycache__":
                dirs.remove(d)
        for n in files:
            rel = (Path(folder) / n).relative_to(root).as_posix()
            if rel != "PACKAGE_MANIFEST.csv":
                safe_file(root, rel)
                actual.add(rel)
    expected = {r["release_repo_path"] for r in rows}
    if actual != expected:
        raise ReleaseError("Manifest/file-set mismatch")
    return {r["release_repo_path"]: r for r in rows}

class Reader:
    def __init__(self, root, entries):
        self.root, self.entries = root, entries

    def data(self, rel):
        if rel not in self.entries:
            raise ReleaseError("Unlisted dependency: " + rel)
        raw = safe_file(self.root, rel).read_bytes()
        if digest(raw) != self.entries[rel]["sha256"].upper():
            raise ReleaseError("Changed listed dependency: " + rel)
        return raw

    def csv(self, rel):
        return parse_csv(self.data(rel))

    def json(self, rel):
        return json.loads(self.data(rel).decode("utf-8-sig"))

def check_sources(reader):
    materials = reader.csv("provenance/material_sources.csv")
    history = reader.csv("provenance/frozen_source_history.csv")
    origin = reader.json("provenance/origin.json")
    if not materials or not history:
        raise ReleaseError("Empty provenance")
    seen, source_map = set(), {}
    for r in materials:
        p = r["release_repo_path"]
        if p.casefold() in seen:
            raise ReleaseError("Duplicate material")
        seen.add(p.casefold())
        if not HEX.fullmatch(r["source_sha256"]) or not HEX.fullmatch(r["public_sha256"]):
            raise ReleaseError("Invalid material SHA")
        if digest(reader.data(p)) != r["public_sha256"].upper():
            raise ReleaseError("Public material identity mismatch")
        if r["transformation"] == "BYTE_IDENTICAL":
            if r["source_sha256"].upper() != r["public_sha256"].upper():
                raise ReleaseError("False byte-identity assertion")
        elif r["transformation"] not in ("PATH_REDACTED_NOT_FROZEN_IDENTITY", "PUBLIC_DOCUMENT_REVISION"):
            raise ReleaseError("Unknown transformation")
        source_map[p] = r
    seen, included = set(), 0
    for r in history:
        p = r["historical_repo_path"]
        if p.casefold() in seen or not HEX.fullmatch(r["historical_sha256"]):
            raise ReleaseError("Invalid historical identity")
        seen.add(p.casefold())
        if r["included"] == "YES":
            included += 1
            m = source_map.get(p)
            if not m or m["source_sha256"].upper() != r["historical_sha256"].upper():
                raise ReleaseError("Historical/source link mismatch")
            if r["public_sha256"].upper() != digest(reader.data(p)) or r["public_identity"] != m["transformation"]:
                raise ReleaseError("Historical/public link mismatch")
        elif r["included"] != "NO" or r["public_sha256"] or r["public_identity"] != "NOT_DISTRIBUTED":
            raise ReleaseError("Invalid omitted historical record")
    if len(history) != origin["historical_records"] or included != origin["included_frozen_reference_files"]:
        raise ReleaseError("Historical count mismatch")
    distributed = {p for p in reader.entries if p.startswith("reference/03_Code/")}
    declared = {r["historical_repo_path"] for r in history if r["included"] == "YES"}
    if distributed != declared:
        raise ReleaseError("Unaccounted distributed research file")
    return dict(historical_identities=len(history), distributed_research_files=included,
                privacy_redacted_materials=sum(r["transformation"] != "BYTE_IDENTICAL" for r in materials),
                original_full_gate="NOT_SATISFIED_BY_PUBLIC_SUBSET")

def number(s):
    try:
        d = Decimal(str(s))
    except Exception:
        raise ReleaseError("Invalid numerical observation")
    if not d.is_finite():
        raise ReleaseError("Non-finite numerical observation")
    return d

def describe(values):
    with localcontext() as ctx:
        ctx.prec = 40
        vals = [number(v) for v in values]
        if len(vals) < 2:
            raise ReleaseError("Sample SD needs at least two observations")
        mean = sum(vals) / Decimal(len(vals))
        sd = (sum((v-mean)**2 for v in vals) / Decimal(len(vals)-1)).sqrt()
        return mean, sd

def display(value, metric):
    places = 3 if metric in ("spl", "path_efficiency") else 1
    scale = Decimal(100) if metric in ("success_rate", "timeout_rate") else Decimal(1)
    return format((value * scale).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN), "f")

def matrix(rows, methods, seeds):
    seen = set()
    for r in rows:
        try:
            seed = int(r["seed"])
        except (KeyError, ValueError, TypeError):
            raise ReleaseError("Invalid seed")
        key = (r["method"], seed)
        if key in seen:
            raise ReleaseError("Duplicate method/seed")
        seen.add(key)
        for metric in METRICS:
            number(r[metric])
        if r.get("episode_count"):
            n = int(r["episode_count"])
            counts = [int(r[k]) for k in ("success_count", "collision_count", "timeout_count")]
            if n <= 0 or sum(counts) != n:
                raise ReleaseError("Episode accounting mismatch")
            for name, count in zip(("success_rate", "collision_rate", "timeout_rate"), counts):
                if name in r and not math.isclose(float(r[name]), count/n, abs_tol=1e-12):
                    raise ReleaseError("Rate/count mismatch")
    if seen != {(m, s) for m in methods for s in seeds}:
        raise ReleaseError("Formal method/seed matrix mismatch")

def equal_summary(raw, stored, moving=False):
    checked = 0
    for r in stored:
        method = r.get("method", "PPO_PBRS")
        if moving:
            pairs = [(k[:-5], r[k], r[k[:-5]+"_sample_sd"]) for k in r
                     if k.endswith("_mean") and k[:-5]+"_sample_sd" in r]
        else:
            pairs = [(r["metric"], r["mean"], r.get("sample_sd", r.get("sample_std")))]
        for metric, expected_mean, expected_sd in pairs:
            values = [x[metric] for x in raw if x["method"] == method and x.get(metric) not in ("", None)]
            if not values:
                empty_n = r.get("n", r.get("actual_n", r.get("n_seeds", "0")))
                if expected_mean or expected_sd or int(empty_n) != 0:
                    raise ReleaseError("Summary without observations: " + method + "/" + metric)
                continue
            mean, sd = describe(values)
            expected_n = r.get("n", r.get("actual_n", r.get("n_seeds")) )
            if expected_n and int(expected_n) != len(values):
                raise ReleaseError("Summary sample count mismatch")
            for a, b in ((mean, expected_mean), (sd, expected_sd)):
                if not math.isclose(float(a), float(number(b)), rel_tol=1e-10, abs_tol=1e-12):
                    raise ReleaseError("Descriptive summary mismatch: " + method + "/" + metric)
            checked += 1
    return checked

def check_results(reader):
    specs = [
        ("main", "per_seed_metrics.csv", {"PPO", "PPO_Astar", "PPO_Reward", "FULL"}, SEEDS),
        ("component_ablation", "per_seed_metrics.csv",
         {"PPO_Reward", "REWARD_WO_PROGRESS", "REWARD_WO_REVISIT", "REWARD_WO_OSCILLATION", "REWARD_WO_STAGNATION"}, SEEDS),
        ("external_baselines", "baseline_per_seed_metrics.csv", {"DDQN", "D3QN", "C51", "MaskablePPO"}, SEEDS),
        ("astar_branches", "per_seed_metrics.csv",
         {"REWARD_DISTILL_ONLY", "REWARD_PLANHEAD_ONLY", "PPO_Reward", "FULL"}, {42, 1004, 6010}),
        ("pbrs", "per_seed_metrics.csv", {"PPO_PBRS"}, SEEDS),
        ("progress_forward", "per_seed_metrics.csv", {"PPO_PROGRESS"}, SEEDS),
        ("movingai", "per_seed_metrics.csv", {"PPO", "PPO_Reward", "FULL"}, SEEDS)]
    counts, tables, summaries, groups = {}, [], 0, {}
    for group, filename, methods, seeds in specs:
        prefix = "results/public/" + (group if group == "movingai" else "barn/" + group) + "/"
        rows = reader.csv(prefix + filename)
        if group == "pbrs":
            rows = [dict(r, method="PPO_PBRS") for r in rows]
        matrix(rows, methods, seeds)
        counts[group] = len(rows)
        groups[group] = rows
        if group == "movingai":
            summaries += equal_summary(rows, reader.csv(prefix + "method_mean_sample_sd.csv"), True)
        elif group != "progress_forward":
            summary_inputs = rows
            if group == "external_baselines":
                summary_inputs = rows + [r for r in groups["main"] if r["method"] == "FULL"]
            summaries += equal_summary(summary_inputs, reader.csv(prefix + "mean_sample_sd.csv"))
        for method in sorted(methods):
            for metric in METRICS:
                values = [r[metric] for r in rows if r["method"] == method]
                mean, sd = describe(values)
                tables.append(dict(experiment=group, method=method, metric=metric, n=len(values),
                                   mean=str(mean), sample_sd=str(sd),
                                   mean_display=display(mean, metric), sd_display=display(sd, metric)))
    forward = groups["main"] + groups["progress_forward"]
    summaries += equal_summary(forward, reader.csv("results/public/barn/progress_forward/forward_ablation_aggregate.csv"))
    return dict(observation_views=counts, checked_descriptive_summaries=summaries,
                table_rows=len(tables), inferential_tests="NOT_RUN", models="NOT_RUN"), tables

def verify(root):
    root = Path(root).absolute()
    entries = verify_manifest(root)
    reader = Reader(root, entries)
    source = check_sources(reader)
    results, tables = check_results(reader)
    return dict(status="PASS", scope="PUBLIC_SUBSET_AND_RELEASED_DESCRIPTIVE_SUMMARIES",
                payload_files=len(entries), total_files=len(entries)+1,
                provenance=source, results=results,
                full_training_reproduction="NOT_VERIFIED"), tables

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", action="store_true", help="Print descriptive CSV to stdout; never write results")
    args = parser.parse_args()
    try:
        report, tables = verify(Path(__file__).absolute().parents[1])
        if args.tables:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(tables[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(tables)
        else:
            print(json.dumps(report, indent=2))
        return 0
    except (ReleaseError, OSError, ValueError, KeyError, TypeError, UnicodeError, csv.Error) as exc:
        print(json.dumps(dict(status="FAIL_RELEASE", error=str(exc))), file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())

