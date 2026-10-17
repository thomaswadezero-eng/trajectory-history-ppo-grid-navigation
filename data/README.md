# Data acquisition and distribution boundary

Raw maps, scenarios, arrays and task JSON are not distributed. Thirteen task/split/inventory files from the larger local archive are excluded: three BARN task lists, three MovingAI map/task/test metadata files, six duplicated map/task splits in the formal parent directories, and one task-candidate audit report. This prevents relying on unconfirmed redistribution rights; it does not supply an equivalent route to the original frozen tasks.

## BARN

Use the [official BARN page](https://www.cs.utexas.edu/~xiao/BARN/BARN.html) and its Download section. Record archive URL, retrieval date, size and SHA yourself. The exact upstream archive SHA and complete original preprocessing chain are not established by the released material. The linked map generator is not a substitute for the study's original map/task bytes.

The formal split uses 1,000 training tasks on map IDs 0–19, 100 validation tasks on 20–29, and 1,000 test tasks on 130–229; 130 grid/cspace pairs are required. Do not invent tasks or edit a hashed list to claim exact replication. Explicit BARN map/derived-task redistribution permission was not established in the inspected evidence. Obtain applicable terms from the dataset provider.

## MovingAI

Acquire maps/scenarios from the official [DAO](https://www.movingai.com/benchmarks/dao/index.html), [Room](https://www.movingai.com/benchmarks/room/index.html) and [Maze](https://www.movingai.com/benchmarks/maze/index.html) benchmark pages. Consult the [grid index](https://www.movingai.com/benchmarks/grids.html) and [format specification](https://www.movingai.com/benchmarks/formats.html). Archive identity is not determined by a historical local directory label.

The retained generation/rules source documents 64×64 midpoint/clamped crops, map-disjoint splits, construction seed 20260920, at most ten tasks per source map, four-connected BFS ground truth, horizon 400 and train/validation/test counts 800/200/750. Native scenario distance conventions are not interchangeable with this four-connected protocol. The generator has historical prerequisites and output writes; it is not a verified portable one-command data preparation tool.

This research uses the MovingAI pathfinding benchmarks. The benchmark database is made available under [ODC Attribution 1.0](https://opendatacommons.org/licenses/by/1-0/). Preserve dataset attribution and applicable notices; database terms do not clear all independent map/content rights. Map-source permissions and exact derived task redistribution rights require material-specific review. Do not relicense upstream data as MIT.

## Consequences

Research settings and released summary results can be inspected without these datasets. Exact from-scratch training/model evaluation cannot be reproduced from this package alone: original task identities, map bytes and additional historical evidence are not supplied. Generated-task identity is provenance, not a direct input read by every launcher; the legacy scientific audit is a trusted-old-result identity dependency, not a PPO algorithm input. None of these exclusions authorizes changing task splits or bypassing integrity checks.
