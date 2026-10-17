"""Mock-file tests only; no research module imports."""
import csv
from decimal import Decimal
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).absolute().parents[1] / "release_tools" / "verify_release.py"
spec = importlib.util.spec_from_file_location("release_checker", str(SCRIPT))
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)

class ReleaseChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("SCI_TEST_TMP"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data.txt").write_bytes(b"mock")
        self.rows = [dict(release_repo_path="data.txt", bytes="4",
                          sha256=hashlib.sha256(b"mock").hexdigest())]

    def manifest(self, rows=None):
        with (self.root / "PACKAGE_MANIFEST.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["release_repo_path", "bytes", "sha256"])
            w.writeheader()
            w.writerows(self.rows if rows is None else rows)

    def rejected(self, rows=None):
        self.manifest(rows)
        with self.assertRaises(checker.ReleaseError):
            checker.verify_manifest(self.root)

    def test_valid_manifest(self):
        self.manifest()
        self.assertEqual(set(checker.verify_manifest(self.root)), {"data.txt"})

    def test_empty_manifest(self):
        self.rejected([])

    def test_duplicate_record(self):
        self.rejected(self.rows * 2)

    def test_case_duplicate(self):
        self.rejected(self.rows + [dict(self.rows[0], release_repo_path="DATA.txt")])

    def test_traversal_and_absolute_paths(self):
        for path in ("../data.txt", "/data.txt", "C:/data.txt", "C:data.txt",
                     "\\\\host\\share\\file", "reference\\..\\data.txt", "./data.txt",
                     "x//data.txt", "data.txt:stream", "CON", "data.txt."):
            with self.subTest(path=path):
                self.rejected([dict(self.rows[0], release_repo_path=path)])

    def test_malformed_sha(self):
        self.rejected([dict(self.rows[0], sha256="not-a-sha")])

    def test_missing_file(self):
        self.rejected([dict(self.rows[0], release_repo_path="missing.txt")])

    def test_tampered_bytes(self):
        (self.root / "data.txt").write_bytes(b"edit")
        self.rejected()

    def test_unlisted_file(self):
        (self.root / "extra.txt").write_bytes(b"x")
        self.rejected()

    def test_invalid_structure_and_size(self):
        self.rejected([dict(self.rows[0], bytes="-1")])
        (self.root / "PACKAGE_MANIFEST.csv").write_text("bad\nvalue\n", encoding="utf-8")
        with self.assertRaises(checker.ReleaseError):
            checker.verify_manifest(self.root)

    def test_reparse_boundary(self):
        original = Path.lstat
        def fake(p):
            if p == self.root / "data.txt":
                return types.SimpleNamespace(st_mode=0o100644, st_file_attributes=1024)
            return original(p)
        with patch.object(Path, "lstat", fake):
            with self.assertRaises(checker.ReleaseError):
                checker.safe_file(self.root, "data.txt")

    def test_real_outside_symlink(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("SCI_TEST_TMP")) as other:
            target = Path(other) / "outside.txt"
            target.write_bytes(b"mock")
            link = self.root / "link.txt"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("OS disallows unprivileged symlinks; reparse mock test is separate")
            self.rejected([dict(self.rows[0], release_repo_path="link.txt")])

    def test_unlisted_read_is_not_verified(self):
        with self.assertRaises(checker.ReleaseError):
            checker.Reader(self.root, {}).data("data.txt")

    def test_sample_sd_and_single_rounding(self):
        mean, sd = checker.describe(["1", "2", "3"])
        self.assertEqual(mean, Decimal(2))
        self.assertEqual(sd, Decimal(1))
        self.assertEqual(checker.display(Decimal("35.74975"), "average_steps"), "35.7")
        self.assertEqual(checker.display(Decimal("0.92725"), "success_rate"), "92.7")

    def test_invalid_numbers_and_seed_matrix(self):
        for value in ("NaN", "Infinity", ""):
            with self.assertRaises(checker.ReleaseError):
                checker.number(value)
        row = dict(method="mock", seed="42", **{m:"0" for m in checker.METRICS})
        checker.matrix([row], {"mock"}, {42})
        with self.assertRaises(checker.ReleaseError):
            checker.matrix([row, row], {"mock"}, {42})
        with self.assertRaises(checker.ReleaseError):
            checker.matrix([row], {"mock"}, {42, 1004})

if __name__ == "__main__":
    unittest.main()

