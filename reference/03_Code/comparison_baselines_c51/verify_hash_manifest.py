#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
from pathlib import Path


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent.parent
MANIFEST = SOURCE_DIR / "hash_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mismatches = []
    for record in payload["files"]:
        path = (PROJECT_ROOT / record["path"]).resolve()
        actual = sha256_file(path) if path.is_file() else None
        if actual != record["sha256"]:
            mismatches.append({"path": record["path"], "expected": record["sha256"], "actual": actual})
    print(json.dumps({"checked": len(payload["files"]), "mismatch_count": len(mismatches), "mismatches": mismatches}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
