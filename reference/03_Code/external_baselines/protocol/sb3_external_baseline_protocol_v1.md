# SB3-Contrib MaskablePPO 外部基线正式手动运行协议 v1

日期：2026-08-17  
状态：`READY_FOR_MANUAL_TRAINING_REVIEW`  
执行边界：只能由用户在 VS Code 中手动启动正式训练和测试。

## 冻结身份

- Python 3.10.11 独立便携环境；
- PyTorch 2.8.0+cu128；
- Stable-Baselines3 2.9.0；
- SB3-Contrib 2.9.0；
- Gymnasium 1.3.0；
- 正式入口：`sb3_external_baseline.py`；
- 正式配置：`config/sb3_external_baseline_config_v1.json`；
- official seeds：42、1004、6010；
- 每个 run 的 training interaction budget：900000。

## 已冻结的第二阶段选择

1. 外部 baseline 只使用 V7 基础 reward，不使用 Dynamic Reward。
2. 训练、validation 和 test 均不得调用 A*。
3. action mask 只由四邻域边界和 cspace occupancy 得到。
4. 使用 V7-compatible custom feature extractor 与官方 MaskablePPO 主体。
5. 采用精确 900000 interaction budget；在第 900000 步 callback 停止，不训练未完成的最终 rollout。
6. 任务 timeout 映射为 `terminated=True, truncated=False`，与 V7 的零 bootstrap 一致。
7. validation 在 100000 至 900000 的每个 100000 interaction 边界执行。
8. checkpoint 按 Success、SPL、Timeout、Average Steps、interaction 的冻结词典序选择。
9. test 只使用 validation 选中的 checkpoint，固定 1000 tasks、确定性 masked policy、参数更新为 0。

## 输出隔离

正式输出只能创建在：

`03_Code/external_baselines/outputs/MaskablePPO/seed_<seed>/run_<timestamp>/`

run 目录使用 `exist_ok=False`，不会复用或覆盖既有 run。每个 run 保存源代码/config/任务/cspace/环境身份、validation、checkpoints、test trajectories、formal result 和 manifest。

## 与 V7 的已知实现差异

- 官方 MaskablePPO 对 advantage 使用其自身标准化实现；不复制 V7 的 ±5 advantage clip。
- 固定 2048 rollout 与 900000 budget 不整除；本实现优先保持公平交互预算，丢弃未完成的最终 rollout，不执行 V7 的 928-transition 尾部更新。
- V7 的未使用 planning head 不加入外部 baseline。

这些差异已在运行 manifest 和论文解释边界中显式保留，不得根据结果重新选择。
