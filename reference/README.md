# Research source identity

This directory contains selected implementations, configuration and protocol copies from the formal experiments, not a complete development archive. Original V7 and its eight-seed extension are both retained: the original is a real parent implementation, including for MovingAI, not an interchangeable obsolete version. The DDQN framework's A2C module/config are shared import/hash dependencies, not evidence of a reported A2C experiment.

## Three different identities

1. The historical full archive had 159 frozen files. All historical repository-relative names and SHA-256 identities are recorded in `../provenance/frozen_source_history.csv`. An omitted row is a historical identity, not a verified distributed file.
2. Selected public copies are listed in `../provenance/material_sources.csv`, with source SHA, public SHA and transformation. Unchanged copies are marked BYTE_IDENTICAL. Privacy-redacted copies are marked PATH_REDACTED_NOT_FROZEN_IDENTITY. Local machine root/home prefixes are replaced by explicit redaction labels; these are not valid runtime path settings.
3. `../PACKAGE_MANIFEST.csv` verifies actual distributed bytes, including newly written public documentation and tools. It is not a replacement for any original experiment's scientific integrity gate.

The root .gitattributes disables Git line-ending conversion so a checkout preserves these SHA-bound bytes on Windows and Unix. Do not apply automatic formatting or newline normalization to distributed research/results files.

No original source/config/task bytes were edited. Redacted copies retain original algorithmic expressions and numerical configuration values, but **do not satisfy original byte hashes** and have no demonstrated runtime equivalence. Self-hash fields and historical gate logic remain unchanged. Original hash manifests may also name nondistributed metadata and development checks. Thus the full historical integrity check cannot pass on this subset. Do not bypass it.

The dated status fields inside historical protocols describe their original freeze/run stages, not this repository's publication state. For current experiment coverage use the released per-seed results and public guides. Do not execute historical launchers or aggregation programs as a quick-start check: they can read/write historical locations or depend on unavailable identities. A safe runnable port would require a separately identified and validated implementation; none is claimed here.

The component-ablation hash_manifest.json is not distributed: its historical inventory includes internal project records unnecessary for public source inspection or released-summary recomputation. Its historical SHA remains recorded in the source-history index. Research programs still retain their original archive references; excluding this file does not make the historical launcher runnable, and no original gate or runtime PENDING state is changed.
