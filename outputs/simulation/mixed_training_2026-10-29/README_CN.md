# 混合地形联合模态选择仿真结果

本目录保存 2026-10-29 最新一版仿真训练结果，目标是联合选择 `导航路径 + LOW/MID/HIGH 模态 + 速度`。

## 结果状态

- 训练入口：masked independent path-mode-speed PPO。
- 初始候选路径长度：约 `6.5 m`、`7.6 m` 和 `18.0 m`。
- 最新续训：2 个 episode，每个 2500 步，PPO update `70 -> 72`。
- 两个 episode 均为 horizon 结束，没有碰撞、姿态、速度或 clearance 硬失败。
- 已观察到 `MID -> LOW` 联合模态行为；本轮尚未到达台阶区，因此不能作为 HIGH 越障完成证明。
- 当前结果是仿真研究中间 checkpoint，不是最终收敛模型，也不构成全局最优性证明。

## 文件说明

- `mixed_joint_ppo_gap_pathfix.pt`：最新 PPO checkpoint，包含模型和优化器状态。
- `mixed_joint_ppo_gap_pathfix.json`：完整训练与仿真执行记录。
- `mixed_geometry_gap_600.json`：600 步混合场景可行性 smoke，首决策选择正向短路与 LOW。
- `mixed_joint_ppo_gap_pathfix.sources.tar.gz`：该训练执行的源文件快照。
- `navigation_mixed_low_stairs_v1.json`：混合地形场景配置。

## 复现范围

这是 Isaac Sim/IsaacLab 仿真结果，不包含真实机器人运行。当前场景使用 `stair-start=3.5 m`，为低顶棚结束到台阶之间保留足够的模态切换窗口。
