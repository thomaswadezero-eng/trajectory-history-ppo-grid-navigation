# Independent release check

`verify_release.py` is a read-only, standard-library checker for this selected public package, compatible with Python >=3.8. It checks a nonempty SHA-256 manifest, safe unique relative file names, regular files with no symlink/reparse components, exact distributed file coverage, historical/source-public SHA links, formal seed matrices and descriptive means/sample SDs.

It does not load research modules, models, datasets or private evidence and does not start subprocesses, use CUDA, write results or rerun inferential tests. `--tables` prints descriptive values to stdout only. Run the commands in the root README.

An omitted historical file is not counted as a verified current file. Some research copies are privacy-redacted; original experiment hash gates are unchanged and cannot be claimed satisfied by this subset. This tool is not an integrity-gate bypass or a portable training launcher.

Tests use isolated mock temporary files and deliberately exercise missing/tampered files, duplicate/unsafe paths, malformed hashes, empty manifests, links and descriptive rounding. No frozen experiment module is imported.
