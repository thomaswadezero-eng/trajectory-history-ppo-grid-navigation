# Component correction versions: keep both

Seeds are independent trained policies, not episodes. No new inferential tests are calculated by the release checker. Numerical CSV strings from the original projections are retained; descriptive values use round-trip precision. Machine paths are represented by deterministic evidence-path IDs. Source hashes are ordinary full-byte SHA-256; a recorded runtime config self-hash is a different identity. Original private receipts are not distributed.

Both versions use the same 32 completed leave-one-out runs (12 reusable legacy plus 20 extension) and the same 8 Reward reference runs. The unit is the matched training seed. All 12 raw p values and paired input vectors agree. The change is multiplicity grouping, not a new experiment or a change to raw results.

## Original frozen scheme

The retained component_hypothesis_plan_v1.json records the plan dated 2026-09-18 with status FROZEN_BEFORE_NEW_FORMAL_RESULTS, and joint12_statistics.csv retains its output. Four components crossed with SR/SPL/Timeout form ONE 12-test Holm family. Progress and Oscillation each have adjusted p=0.09375 for all three metrics; Revisit SPL has 0.09375; all other rows have 1. None is below 0.05. This freeze-before-new-results statement is not proof of external preregistration before all legacy data existed.

## Subsequent revision

The subsequent audit was generated 2026-09-24T11:16:32+08:00. It uses FOUR families of three metrics, one per component. Progress/Oscillation each become 0.0234375 for SR/SPL/Timeout; Revisit SPL becomes 0.046875; the remaining values are 1. The same shared reference appears across comparisons; naming separate families does not make the observations statistically independent or establish global 12-test error control.

The source audit records an instruction superseding the old scheme but does not attach the original instruction/time or establish that grouping was chosen before seeing the data. The corresponding original analysis was recorded as REVISED_EXPLORATORY; the released four_family_statistics.csv is not an unchanged preregistered analysis. No new signed-rank/bootstrap calculation is made by the release checker.

Original comparisons/metrics did not change, but family grouping did. The chronology supports a subsequent revision, not proof of prospective grouping before all data were observed. Both versions are retained to make that distinction transparent.

Original frozen code: reference/03_Code/component_ablation_8seed_v1/aggregate_and_analyze.py. Later script identity: recalculate_component_statistics_4families.py, SHA 8E3C5C1EE1D62C701248BA485E87D435A0DF672D03D8FEFCF3C89B68424C6DAB (source ID E055). Its raw host-specific script is not newly distributed or executed. Descriptive output and existing statistics are inspectable; this is not a portable inference pipeline.
