# Recompute released descriptive tables

From the repository root:

```text
python -B release_tools/verify_release.py
python -B release_tools/verify_release.py --tables
```

The second command prints one CSV row per method/metric: n, unrounded mean, sample SD and display values. Rates are displayed as percentages (one decimal), SPL/conditional PE with three decimals, and Average Steps with one decimal. Decimal values are rounded once, ties to even. No inferential statistics are computed. No result files are written.

## Data coverage

| Results subdirectory | Observations |
| --- | --- |
| barn/main | 4 methods × 8 paired seeds = 32 |
| barn/progress_forward | 8 Progress-only seeds; F1/F2 forward comparisons with shared existing controls |
| barn/component_ablation | 32 leave-one-out runs + 8 shared Reward controls = 40 records |
| barn/pbrs | 8 PBRS seeds |
| barn/external_baselines | DDQN, D3QN, C51 and MaskablePPO × 8 = 32, not 40 including FULL |
| barn/astar_branches | 2 branches × 3 seeds + 2 matched controls × 3 = 12 records |
| movingai | 3 methods × 8 = 24, 750 test episodes each |

These are overlapping result views, not a sum of independent runs. Source evidence hashes and transformation records accompany the numerical files. Evidence-path IDs and repository-relative original-results references are provenance labels, not included model/log files.

## Scientific interpretation

BARN H3 does not support an incremental FULL-over-Reward benefit; it also does not prove equivalence. C51's success-conditional PE advantage is retained. FULL Average Steps is 35.74975, sample SD 4.144583547923034: once-rounded **35.7 ± 4.1**, not 35.8 from double rounding.

Component analyses retain both the original joint twelve-test Holm scheme and the later four three-metric families. They use the same raw comparisons but different multiplicity groupings. The latter is revised/exploratory: available chronology does not establish prospective specification before observing all data. Separate families are not independent datasets or global twelve-test control. The detailed version relationship is preserved with the component results. No blanket assertion that no post-hoc family was added is made.

F1/F2 forward comparisons retain their own recorded three-metric family definitions. Do not interpret descriptive performance differences as causal shares of individual reward terms. PBRS and the three-seed A* branch comparisons remain descriptive. PPG has no confirmed formal completed result and is not substituted by a pilot.

MovingAI results and paired differences are **descriptive only**. No formal inferential tests were authorized or added (formal_inferential_tests=false). The released eight-seed table supersedes the earlier three-seed display. Figures are discussed in [the figure-source note](../docs/figures.md).

Hash checks establish the integrity of released observations, not independent validation of every original episode. Original inference/aggregation scripts are research references with unavailable historical dependencies and must not be invoked to reproduce the standard-library summary command.
