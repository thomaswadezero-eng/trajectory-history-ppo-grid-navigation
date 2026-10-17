# Training and evaluation: implementation versus execution

## Formal coverage

BARN paired seeds: 42, 1004, 6010, 68882, 19816, 74494, 81399, 31524. Original single-branch comparisons use only 42, 1004, 6010. The formal per-run budget is 900,000 training interactions and 1,000 final BARN test episodes. MovingAI uses independently trained policies with the same eight seed IDs, 900,000 interactions and 750 test tasks per method/seed.

All entries below are relative to `reference/03_Code/`. This is a source-inspection index, **not a verified runnable launch workflow**.

| Experiment | Formal entry/config family |
| --- | --- |
| BARN four methods | final_two_factor_experiment_v7_8seed/final_two_factor_experiment_v7_8seed.py; experiment_config_v7_8seed.json and experiment_protocol_v7_8seed.json |
| Parent implementation | final_two_factor_experiment_v7/final_two_factor_experiment_v7.py; experiment_config.json and experiment_protocol.json |
| Progress-only | ppo_progress_8seed_v2/ppo_progress_8seed_v2.py; its experiment_config.json / experiment_protocol.json |
| Component leave-one-out | component_ablation_8seed_v1/run_component_ablation_8seeds.py; parent internal_ablation_v1 |
| A* single branches | internal_ablation_v1/internal_ablation_v1.py |
| PBRS | pbrs_baseline_v1/run_pbrs_8seeds.py; pbrs_config_v1.json / pbrs_protocol_v1.json |
| Four external baselines | external_baseline_8seed_v1/run_external_baselines_8seeds.py; C51, DDQN/D3QN framework and separate SB3 modules |
| MovingAI original | movingai_64x64_robustness_v1/run_movingai64_robustness.py; confirmed config/protocol |
| MovingAI extension | movingai_8seed_extension/run_movingai_8seed_extension.py; movingai_8seed_config.json / movingai_8seed_protocol.json |

BARN's actual main CLI has --config, --protocol, --method, --seed and --device; method choices are PPO, PPO_Astar, PPO_Reward and FULL. The main entry has no additional formal confirmation gate. Executing it is not a dry-run. Component/PBRS/external extension runners have --confirm-formal; MovingAI extension has --run and requires CUDA for its formal extension. Do not run research --help, preflights or aggregation entries as standard-library validation; their imports/side effects are not isolated.

The MovingAI extension reuses nine historical runs and adds fifteen. It is not a general executor that trains all 24 from zero. Some extension runners require reusable DONE/checkpoint identities and pristine historical output roots. Selected source copies have privacy redactions, unavailable hashed metadata, historical paths and possible omitted identity-only helpers. No output redirection or identity-gate bypass is provided. Windows directory mapping alone would not supply missing data, prevent historical output writes or establish numerical equivalence.

## Evaluation and metrics

V7 training includes periodic validation, checkpoint selection/reload and final deterministic masked testing. Validation, test and deployment must not query A*, update parameters or use training-only auxiliary targets. This describes the inspected formal implementation/evidence; no new model execution verified it during packaging.

- SR = successful episodes / all test episodes.
- SPL averages success × shortest length / max(shortest length, executed length) over all test episodes; failures contribute zero, with original boundary handling.
- Path Efficiency (PE) averages the shortest/executed ratio **conditional on success**. Across-seed results average per-seed PE, not pooled success-weighted PE.
- Average Steps and timeout/collision rates are all-episode final-test metrics, not learning convergence measures.
- Descriptive summaries use independent trained seeds, unweighted mean and **sample SD (n−1)**. Missing values remain unavailable.

There is no universal standalone weights-evaluation command across methods. Evaluation modes still need the exact checkpoint and data; weights are not distributed. Inspecting a mode's CLI is not proving runnable model evaluation. Do not change reward/A* rules, random seeds, task splits, model/loss, checkpoint selection or evaluation protocol to claim original-result reproduction.

## What has and has not been checked

The independent release checker validates distributed bytes and small released descriptive summaries, without importing research code. It does not verify upstream raw episodes, run original gates, establish import/runtime closure on a clean machine, reproduce training randomness, regenerate inferential tests or prove complete training replication. Environment and data limitations are documented separately. Exact portable training requires additional data identities and a separately validated port, not rewriting the frozen archive.
