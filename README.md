# Trajectory-History-Conditioned PPO Grid Navigation

Research software and released numerical results for **Trajectory-History-Conditioned Dynamic Reward for PPO-Based Grid Navigation: A Controlled Ablation of a Training-Only A* Prior**.

Paper authors: Zhengsheng Jin, Lin Li, Shihao Zhang and Zhenhua Wang (corresponding author). Manuscript in preparation, 2026; unpublished, DOI not assigned. Software author and copyright holder: Zhengsheng Jin. Development used Codex assistance; software authorship is distinct from paper authorship.

## Method and evidence

The method combines trajectory-history-conditioned Progress, Revisit, Oscillation and Stagnation reward mechanisms with an optional training-only A* prior. A* must not be queried during validation, testing or deployment. Dynamic Reward is the main observed performance source; the BARN FULL-over-Reward contrast does not establish an independent A* benefit or equivalence. No curriculum or additional obsolete escape reward is introduced; the historical escape field accounts for Progress.

The release supports inspection of the research implementation and recomputation of descriptive tables from released per-seed observations. It does **not** provide a verified fresh-machine end-to-end training workflow.

## Safe quick start

From this directory, using Python 3.8 or newer and only the standard library:

```text
python -B release_tools/verify_release.py
python -B release_tools/verify_release.py --tables
python -B -m unittest discover -s tests -v
```

These commands read this release and use mock temporary files for tests. They do not import research modules, train, load models, access datasets, use a GPU or recompute inferential tests. The table command prints seed-level means and sample SDs to stdout. PASS verifies the **public subset**, not the historical full source archive or independent experimental replication.

## Contents

- BARN PPO, PPO+A*, PPO+Reward and FULL: four methods × eight seeds.
- Progress-only: eight seeds; forward comparisons and four leave-one-out variants with shared Reward controls.
- PBRS: eight seeds; DDQN, D3QN, C51 and MaskablePPO: 32 external baseline runs.
- A* distillation-only and planning-head-only: three seeds each, with matched three-seed controls.
- MovingAI PPO, PPO+Reward and FULL: three methods × eight seeds, 18,000 recorded test episodes; descriptive analysis only.
- Selected formal implementations, configurations, protocols, source identities and numerical provenance. Some shared framework modules are dependencies, not additional reported experiments.

`reference/03_Code` contains selected copies of historical research files, **not an executable portable port**. Some copies have local path prefixes redacted and are not byte-identical to the frozen originals. Original hash gates are neither replaced nor bypassed. The 159 historical source identities are retained in `provenance/frozen_source_history.csv`; only its included subset is distributed. See [source identity](reference/README.md).

## Reproduction scope

| Operation | Scope |
| --- | --- |
| Verify distributed files and recompute released descriptive summaries | Standard-library commands above |
| Inspect formal settings, algorithms and existing inferential outputs | Included, with provenance and statistical-version disclosures |
| Fresh-machine training or model evaluation | Not validated; data identities, paths, original hash gates and incomplete environment locks remain constraints |
| Exact Figures 1–5 regeneration | Not provided: complete final editable artwork, inputs and generation scripts are unavailable in this release |
| Full independent reproduction of original training outcomes | Not performed |

Read [environments](environment/README.md), [data and rights](data/README.md), [training/evaluation boundaries](docs/reproduction.md), [results and statistics](analysis/README.md), and [figures](docs/figures.md).

## License and citation

Original software is licensed under [MIT](LICENSE), Copyright (c) 2026 Zhengsheng Jin. Third-party libraries and datasets retain their own terms; no raw maps, task metadata, model weights, private evidence or development logs are distributed. See [third-party notices](THIRD_PARTY_NOTICES.md). Use [CITATION.cff](CITATION.cff) to cite the software; the associated unpublished manuscript is recorded separately. No software DOI or fixed publication tag is asserted.
