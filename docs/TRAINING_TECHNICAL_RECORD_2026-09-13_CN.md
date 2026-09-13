# TITA-UC 安全残差强化学习训练技术记录

更新日期：2026-09-13

## 1. 任务范围

本项目在官方 8-DOF TITA 策略上训练关节目标残差控制器。官方策略保持为基础控制器，残差策略不选择 LOW/MID/HIGH 姿态，只根据上层姿态命令修正八维关节目标。所有结果均来自 Isaac Sim，不向实机发送运动指令。

控制链路为：

```text
official policy -> q_ref_base (8)
residual policy -> delta_q (8)
q_ref_raw = q_ref_base + posture_offset + delta_q
q_cmd = SafetyProject(q_ref_raw, q, q_dot, limits)
```

关节顺序固定为：

```text
joint_left_leg_1
joint_left_leg_2
joint_left_leg_3
joint_left_leg_4
joint_right_leg_1
joint_right_leg_2
joint_right_leg_3
joint_right_leg_4
```

## 2. 仿真与训练环境

```text
Isaac Sim 4.5.0
Isaac Lab 2.0.0
NVIDIA GeForce RTX 4070 12 GB
NVIDIA driver 580.178.04
physics dt 0.005 s
control dt 0.020 s
control decimation 4
```

残差阶段使用冻结的离线 P95 归一化：

```text
route  0.0007925402
energy 1.9832050
mode   0.6782783
smooth 0.2076107264
```

机械能耗定义为：

```text
E_mech = integral(sum(abs(tau_i * qdot_i))) dt
COT_mech = E_mech / (M * g * D)
```

该量只表示仿真机械能耗，不等同于真实电池能耗。

## 3. LOW 随机狗洞训练结果

### 3.1 已选择课程

| 课程 | 净高 | 净宽 | 长度 | 多种子确定性结果 |
| --- | --- | --- | --- | --- |
| Bridge | 0.46-0.50 m | 0.66-0.74 m | 0.45-0.70 m | 102 次通过 / 0 碰撞 |
| Bridge-low | 0.44-0.48 m | 0.64-0.72 m | 0.50-0.75 m | 79 次通过 / 10 碰撞 |
| Bridge-lower | 0.42-0.46 m | 0.62-0.70 m | 0.55-0.82 m | 52 次通过 / 28 碰撞 |

Bridge-low 和 Bridge-lower 都采用：

```text
16 vector environments
192 rollout steps per PPO iteration
learning rate 5e-6
policy log_std -4.5, frozen
residual filter alpha 0.10
acceleration projector 500 rad/s^2
command speed 0.15 m/s
```

Bridge-lower selected 在上一档 0.44-0.48 m 范围的保留测试中仍为 28 次通过 / 2 碰撞，没有观察到灾难性遗忘。

### 3.2 当前限制

直接进入净高 0.40-0.44 m、净宽 0.58-0.65 m、长度 0.60-0.90 m 时，训练前 seed 131 结果为 0 次通过 / 32 次碰撞。该分布没有成功轨迹，因此没有直接执行 PPO 更新。

LOW 狗洞当前结论是“中等净空课程已跑通，但最低目标课程未完成”。下一档应使用 0.41-0.45 m 的重叠范围，而不是简单增加现有检查点的训练次数。

### 3.3 LOW 检查点

```text
flat protected:
  rl_runs/phase_b_residual/phase_b5_floor_np3o_iter020_selected/checkpoints/residual_phase_b_selected.pt
  sha256 9049f50a1a3d64915464ad463e4598afaeeb7666ef7960d93eabf862dfa00281

bridge-low selected:
  rl_runs/phase_b_residual/phase_b4_tunnel_low_bridge_low_lownoise_001/checkpoints/residual_phase_b_selected.pt
  sha256 eed4394f83430be00c2f20486e8831650e79b0fdd9731f62e62f85e9e70b0cba

bridge-lower selected:
  rl_runs/phase_b_residual/phase_b4_tunnel_low_bridge_lower_lownoise_001/checkpoints/residual_phase_b_selected.pt
  sha256 eafa1e20b548a024b02648aef5e33d30fe4509a1618363074f779bec69e5d353
```

## 4. HIGH 随机楼梯阶段

### 4.1 已实现内容

新增了独立的楼梯课程，不复用狗洞完成条件：

- 每个向量环境独立采样台阶高度、踏面长度和宽度。
- 生成三阶实体碰撞几何和末端平台。
- HIGH 姿态作为固定上层命令，策略不能自行改写姿态。
- 楼梯立面进入局部路线预览，但作为可通行地形，不触发非通行障碍的制动 HOLD。
- 增加楼梯完成奖励、正向高度增量奖励和累计完成统计。
- 保留关节位置、速度、力矩、加速度、碰撞、机身高度和姿态硬终止。

第一档 smoke 参数来自官方 0.61 m 踏面参考，但台阶高度保持保守：

```text
step height 0.025-0.045 m
tread       0.55-0.67 m
width       0.90-1.20 m
steps       3
start x     0.75 m
landing     0.80 m
posture     HIGH
```

配置文件：

```text
configs/phase_b5_stairs_high_smoke.json
```

楼梯阶段从平地 selected 热启动，不从 LOW-only 狗洞模型热启动，避免把 LOW 专项残差错误迁移到 HIGH 未见状态。

### 4.2 当前验证状态

楼梯采样器、参数校验及原有核心逻辑共 26 项单元测试通过，相关 Python 文件编译检查通过。重启后的楼梯物理 smoke 已完成，但当前仍没有楼梯 selected，也不能声明已经会爬目标楼梯。

2026-09-13 的首次 smoke 被系统级 CUDA 故障阻断：

```text
torch.cuda.is_available() = False
CUDA error 999 / CUDA-capable device is busy or unavailable
nvidia-smi can enumerate RTX 4070, but CUDA tensor creation fails
Isaac Sim reports that the system should be restarted
```

一个从 2026-09-11 持续运行的残留 pytest GPU 进程已通过 SIGTERM 停止。系统重启后，`torch.cuda.is_available()` 已恢复为 `True`，Isaac Sim Python 可以正常创建 `cuda:0` 张量。

2026-09-13 重启后的 seed-163 物理结果：

| 课程 | 速度 | 1000 步结果 | 判定 |
| --- | --- | --- | --- |
| 原始浅楼梯 0.025-0.045 m | 0.20 m/s | 0 次通过 / 56 次机身碰撞 | 起点过难，不直接训练 |
| 超浅 bootstrap 0.005-0.015 m | 0.15 m/s | 4 次通过 / 5 次机身碰撞 | 存在成功轨迹，可开始短训练 |

原始浅楼梯的失败均为 `collision`，不是关节速度、力矩、加速度或投影不可行。单环境诊断显示机器人接近第一阶后机身高度从约 0.35 m 降到 0.216 m 并翻倒。为保持 on-policy 训练有正样本，新增：

```text
configs/phase_b5_stairs_high_bootstrap.json
```

bootstrap 仍为三阶随机实体楼梯，净高改为 0.005-0.015 m，踏面 0.60-0.75 m，宽度 1.00-1.30 m。通过后再逐级升高，不能把 bootstrap 结果外推为真实楼梯能力。

### 4.3 第一轮 bootstrap 训练与修正

第一轮使用 16 环境、192 步、6 次迭代。训练轨迹累计记录 14 次完成和 2 次碰撞，但独立 seed-167 评估未满足晋级门槛：

| 模型 | seed 163 | seed 167 |
| --- | --- | --- |
| 训练前平地 selected | 4 次通过 / 5 碰撞 | 3 次通过 / 6 碰撞 |
| iter 1 | 未筛选 | 2 次通过 / 3 碰撞 |
| iter 3 | 6 次通过 / 0 碰撞 | 2 次通过 / 6 碰撞 |
| iter 4 | 未筛选 | 1 次通过 / 6 碰撞 |
| iter 6 | 4 次通过 / 3 碰撞 | 未筛选 |

因此第一轮不创建 selected。检查发现两个结构问题：旧的 50 维观测只有距下一级台阶的距离，没有台阶高度；旧完成判定只检查 x 坐标，可能把侧向绕过误计为完成。

v2 修正为：

- 观测追加归一化台阶高度，维度由 50 增至 51；旧检查点第一层新增列以 0 初始化，保持旧输出不变。
- 完成必须同时满足到达末端、位于实体楼梯宽度内、机身相对进场高度真实上升。
- 每通过一个经过高度验证的台阶给一次中间进度奖励。
- 进场高度取进入第一阶前的最后高度，而非接近过程的瞬时最高点；已验证级数在单个 episode 内只增不减，避免重复领取进度奖励。
- 楼梯路线成本的横向走廊与完成判定统一为 `楼梯半宽 - 机身半宽`，消除成本允许区比成功允许区更宽的不一致。
- 训练环境增至 64，学习率降到 2e-6，探索 log_std 调整为 -3.8。

v2 配置：

```text
configs/phase_b5_stairs_high_bootstrap_v2.json
```

### 4.4 v3 严格重训与 v4 高度口径修正

v3 从平地 selected 重新训练，使用 64 环境、192 步 rollout、4 次迭代、
2e-6 学习率和冻结 log_std=-3.8。第 2 个 rollout 有 8/64 个环境达到经过
高度验证的 level 3，但严格完成数仍为 0；第 3、4 个 rollout 明显退化。
独立 seed-163 评估为 0 次完成、5 次硬终止，因此没有创建 selected。

进一步检查发现两个口径问题：

- `J_mode` 原先使用世界坐标机身高度；登阶时正确抬升会被错误计为 HIGH 姿态误差。
- 配置的 `episode_length_s` 原先未传给环境；最长随机三阶终点约 3.2 m，
  0.15 m/s 命令在 20 s 内理论上无法稳定到达。

v4 将姿态高度改为相对当前楼梯支撑面计算，训练脚本正式读取 episode 时长，
加入经过高度验证的级数保持奖励，并将 episode 延长到 30 s、学习率降至 5e-7。
训练第 7 个 rollout 出现 1 次严格完成，但该轨迹来自随机探索。对应确定性候选
在 seed-163 的 30 s 独立评估中仍为 0 次完成、7 次硬终止；修正后基线为
0 次完成、11 次硬终止。v4 改善了终止数，但没有复现完整通过，因此仍未入选。

### 4.5 单阶与两阶课程

为避免在三阶稀疏奖励上继续退化，课程拆成单阶、两阶、三阶：

```text
configs/phase_b5_stairs_high_single_step_v1.json
configs/phase_b5_stairs_high_two_step_v1.json
configs/phase_b5_stairs_high_bootstrap_v4.json
```

单阶 0.005-0.015 m 基线在 seed-163、8 环境、18 s 条件下得到 19 次严格完成、
0 次硬终止，证明资产和官方基础策略具备单阶物理通过能力，不需要额外训练。

两阶基线存在正样本，因此使用 64 环境、192 步 rollout、6 次迭代和 5e-7
学习率训练。第 2 个 rollout 记录 4 次严格完成；由于 rollout N 在更新并保存
checkpoint N 之前采集，这组轨迹实际对应 `iter 1`。独立配对结果为：

| seed | 平地 selected 基线 | 两阶 iter 1 |
| --- | --- | --- |
| 163 | 2 完成 / 3 碰撞 | 3 完成 / 3 碰撞 |
| 167 | 0 完成 / 6 碰撞 | 2 完成 / 6 碰撞 |
| 173 | 0 完成 / 3 碰撞 | 0 完成 / 3 碰撞 |
| 合计 | 2 完成 / 12 碰撞 | 5 完成 / 12 碰撞 |

候选在每个种子上完成数不降、碰撞不增；HIGH 平地 1000 步保留测试为 0 硬终止、
0 关节速度违规。因此两阶仿真候选已保存为：

```text
rl_runs/phase_b_residual/phase_b5_stairs_high_two_step_v1_001/checkpoints/residual_phase_b_selected.pt
```

后续校准审计发现，虽然 `posture_height` 属性已改为支撑面相对高度，actor 观测和
`J_mode` 奖励仍直接读取世界高度，实际绕过了该属性。现已修复这两个引用；由于
楼梯观测语义发生变化，上述 selected 暂标记为 provisional，必须按相同三种子
重新验证后才能作为三阶 warm start。

修正后的 seed-163 复核结果为：基线 2 完成 / 4 硬终止，旧候选 0 完成 /
4 硬终止。候选完成数下降，首个种子即未通过，因此停止后续种子评估，并将文件
重命名为 `residual_phase_b_rejected_old_observation.pt`。当前不存在可用于三阶的
两阶 selected，必须在完整支撑面相对观测与 `J_mode` 下重新训练。

该 provisional selected 只表示旧观测口径下 HIGH 姿态两阶 0.005-0.015 m 超浅楼梯仿真课程通过，
不代表三阶、常规楼梯、域随机化或实机能力。训练脚本已增加初始 `iter 0`
检查点，并在后续指标中记录 rollout 使用的检查点编号，避免再次发生选模错一位。

将两阶 selected 零训练迁移到三阶的 seed-163 结果为 0 完成 / 7 硬终止，
优于平地 selected 三阶基线的 0 完成 / 11 硬终止，因此下一轮三阶训练应从
两阶 selected 热启动。但三阶训练后期 `C_route` 均值接近归一化裁剪上限 10，
说明沿用平地 P95 会让路线约束乘子压过登阶奖励。评估脚本已增加四项原始成本
均值与 P95 的结构化输出；下一轮必须先采集三阶离线基线并冻结楼梯 P95，不能
使用在线 min-max，也不能在未校准情况下继续堆叠训练迭代。

## 5. 重启后的执行门槛

### 5.1 CUDA 检查

```bash
cd /home/ky/tita_uc_residual_rl
env PYTHONUNBUFFERED=1 /home/ky/isaacsim/python.sh -c 'import torch; print(torch.cuda.is_available()); print(torch.zeros(1, device="cuda:0"))'
```

只有输出 `True` 且成功创建 CUDA 张量后，才能启动 Isaac Sim。

### 5.2 HIGH 楼梯 smoke

```bash
cd /home/ky/tita_uc_residual_rl
env PYTHONUNBUFFERED=1 PYTHONPATH=. TERM=xterm \
./IsaacLab/isaaclab.sh -p scripts/run_tita_residual_lab.py \
  --asset ./tita_rl_official/resources/tita/urdf/tita_description.urdf \
  --checkpoint ./tita_rl_official/tita_example_10000.pt \
  --residual-checkpoint ./rl_runs/phase_b_residual/phase_b5_floor_np3o_iter020_selected/checkpoints/residual_phase_b_selected.pt \
  --num_envs 8 --steps 1000 --command-x 0.20 --posture HIGH \
  --metric-warmup-steps 100 --acceleration-limit 500 \
  --residual-filter-alpha 0.10 --enforce-acceleration-limit \
  --seed 163 --terrain stairs \
  --stair-height-range 0.025 0.045 \
  --stair-tread-range 0.55 0.67 \
  --stair-width-range 0.90 1.20 --stair-num-steps 3 \
  --headless --device cuda:0
```

Smoke 必须确认：随机尺寸正确、观测有限、八维动作顺序正确、接触发生、完成统计有效、安全投影无速度违规。若 1000 步内完全没有接近或登上第一阶，先调整课程或奖励，不直接长时间训练。

### 5.3 楼梯训练晋级规则

1. 先保存训练输入模型的确定性 seed-163 基线。
2. 使用 64 环境、192 步 rollout、保守学习率和冻结 log_std 做短训练，逐级从单阶推进到两阶、三阶。
3. 逐个评估检查点，不默认选择最后一个。
4. 至少三个种子中完成数不得下降、碰撞不得增加。
5. 对平地 HIGH 姿态做保留测试。
6. 只有全部门槛通过才复制为 `residual_phase_b_selected.pt`。

## 6. 总体完成度判断

```text
8-DOF 官方策略接入和安全残差链路：完成
固定 P95 统一目标与 NP3O 训练：完成第一版
LOW 狗洞中等净空课程：完成
LOW 狗洞 0.40-0.44 m 目标课程：未完成
HIGH 楼梯环境及物理 smoke：完成
HIGH 单阶 0.005-0.015 m 仿真课程：通过
HIGH 两阶 0.005-0.015 m 超浅楼梯：短几何与中等踏面课程已通过仿真筛选
HIGH 三阶 0.005-0.015 m 仿真课程：训练有探索成功，确定性评估未通过
HIGH 0.025-0.045 m 浅楼梯训练：未开始
域随机化完整消融：未完成
ONNX/TensorRT 一致性：未完成
ROS2 shadow：未完成
实机验证：未开始
```

当前模型只适合继续仿真研发，不满足实机部署标准。

## 7. 2026-09-13 重启后续训记录

主机重启后，RTX 4070、NVIDIA 580.178.04、CUDA 和 Isaac Sim 4.5 均正常。
受限终端内的 `nvidia-smi` 失败只是设备隔离现象；主机环境能正常加载 NVIDIA
内核模块并创建 CUDA 仿真，不需要为每次训练重新启动电脑。Isaac 任务进程均可
在训练或评估结束后自动关闭。

### 7.1 两阶 v2 独立评估

使用完整的支撑面相对高度观测，对 v2 `iter 1` 进行 seed-163、8 环境、1200 步、
24 秒 episode 的确定性评估：

```text
严格完成：0
硬终止：3（全部碰撞）
验证级数：7/8 环境曾到达第二级区域
结论：未达到基线 2 次完成，拒绝
```

到达第二级区域不等于严格完成。严格完成还要求越过第二级踏面和 0.2 m 落地区域，
因此没有放宽成功判定。

### 7.2 两阶 v3 固定楼梯 P95

v2 仍使用平地路线代价 P95 `0.0007925402`，导致训练中的路线成本接近饱和。
v3 使用修正后楼梯基线离线采样并冻结：

```text
route  = 5.4187655
energy = 2.5330608
mode   = 0.6782783
smooth = 0.2076107
```

该配置不使用在线 min-max。学习率降为 `1e-7`，其余课程保持 64 环境、192 步、
3 次更新、HIGH、0.15 m/s、两级 0.005-0.015 m。归一化路线成本在关键 rollout
中降至 `0.046`，说明饱和问题已解除。

训练 rollout 中，checkpoint 1 和 checkpoint 2 均出现 2 次随机探索完成。但独立
确定性评估结果为：

```text
checkpoint iter 1：0 完成 / 3 碰撞
checkpoint iter 2：0 完成 / 4 硬终止（3 碰撞、1 姿态）
基线：              2 完成 / 4 硬终止（3 碰撞、1 姿态）
```

两个候选均未达到“完成数不少于基线且硬终止不增加”的门槛，未创建 selected。
这表明随机探索能找到完成轨迹，但 PPO 更新后的确定性策略均值没有稳定保留这些
动作。下一轮应先采集成功探索轨迹并蒸馏到确定性残差均值，再进行小学习率 NP3O
微调；暂不推进三阶或真实高度楼梯。

后续审计确认 v3 训练日志中的 `stair_success_total` 是累计计数：第 2 个 rollout
新增 2 次完成，第 3 个 rollout 计数仍为 2，实际新增完成数为 0。训练脚本已改为
每个 rollout 开始前保存计数快照，后续只报告本轮新增成功数。该修正不影响已保存
模型和独立确定性评估结论。

### 7.3 成功轨迹蒸馏实验

从 v3 随机策略收集严格成功轨迹：seed 163 得到 2 个 episode、1003 个样本，
seed 167 得到 8 个 episode、3667 个样本；默认剔除经过安全投影的动作后，共有
3796 个训练样本。蒸馏只更新 actor，reward critic、cost critic 和 `log_std`
保持冻结，动作 MSE 从 `0.00090` 降至 `0.00026`。

三组蒸馏候选的 seed-163 确定性结果均为 0 次完成：保守单种子候选 3 次碰撞，
多种子 anchor=3 候选 5 次硬终止，多种子 anchor=0.25 候选 6 次硬终止。因此
简单成功轨迹行为克隆未能稳定复现通过动作，不创建 selected，也不继续枚举蒸馏
超参数。

### 7.4 两阶 v4 完整时域训练

原两阶路线约 1.95 m，0.15 m/s 下理论行驶时间约 13 s，而 v3 的 192 步 rollout
只有 3.84 s。v4 将 rollout 扩展为完整 24 s（1200 步），使用 32 个环境、2 次
更新和 `1e-7` 学习率。修正后的逐 rollout 数据为：第 1 轮新增 1 次严格完成、
7 次硬终止；第 2 轮新增 1 次严格完成、7 次硬终止。

独立 seed-163 确定性评估中，iter 1 为 0 完成/4 硬终止，iter 2 为 0 完成/
3 次碰撞，均低于平地 selected 基线的 2 完成/4 硬终止，因此 v4 被拒绝且没有
selected。完整时域解决了终点奖励不可达的问题，但正样本密度仍过低。

下一课程固定为短接近、短踏面两阶：`start_x=0.35 m`、`tread=0.25-0.35 m`、
`landing=0.40 m`、高度 `0.005-0.015 m`、宽度 `1.00-1.30 m`。先运行当前可靠
平地 selected 的确定性基线；只有成功密度提高后才开始 v5 训练。

### 7.5 两阶 v5 短几何与 v6 中等踏面续训

v5 短几何课程使用 `start_x=0.35 m`、踏面 `0.25-0.35 m`、落地段 `0.40 m`，
在 32 环境、12 s episode 下训练 3 次更新。三个 checkpoint 在 seed-163
确定性评估均为 34-35 次完成、0 次硬终止，说明成功密度已经足够进入下一档。

随后从 v5 iter 3 热启动 v6 中等踏面课程：

```text
start_x       0.45 m
tread         0.35-0.45 m
landing       0.50 m
episode       18 s
```

v6 使用 32 环境、900 步 rollout、3 次更新、学习率 `1e-7`。选中的 iter 2
在三种子上的配对结果为：

| seed | 平地 selected 基线 | v6 iter 2 | 硬终止 |
| --- | ---: | ---: | ---: |
| 163 | 23 | 23 | 0 |
| 167 | 16 | 19 | 0 |
| 173 | 15 | 18 | 0 |

候选没有降低任何种子的完成数，也没有增加硬终止；HIGH 平地保留测试 1000 步为
0 次硬终止、0 次关节速度违规。因此保存为：

```text
rl_runs/phase_b_residual/phase_b5_stairs_high_two_step_v6_medium_001/checkpoints/residual_phase_b_selected.pt
```

该 selected 仅覆盖仿真 HIGH 两阶、净高 `0.005-0.015 m`、踏面 `0.35-0.45 m`。

## 现实踏面过渡训练（v7/v8）

用户实测楼梯尺寸为单级高度 `0.17 m`、踏面深度 `0.32 m`。为避免从毫米级课程直接跳到现实高度，使用可视化 Isaac Sim 做了两轮独立过渡训练，均从 v6 selected 热启动，未覆盖原 selected：

| 阶段 | 台阶高度 | 踏面 | 级数/环境 | 结果 |
|---|---:|---:|---:|---|
| v7 real-tread | 0.02-0.03 m | 0.31-0.33 m | 2 / 16 | 8 轮，0 完成，碰撞主导，最大前进约 0.65 m |
| v8 height-bridge | 0.02-0.03 m | 0.50-0.60 m | 2 / 16 | 8 轮，0 完成，碰撞主导，最大前进约 0.66 m |

v8 已经放宽踏面仍不能越过 2-3 cm 台阶，说明当前瓶颈是 HIGH 姿态下基础动作与台阶接触/抬腿策略，而不是实测 `0.32 m` 踏面本身。两轮均没有生成 selected checkpoint，也不能据此宣称能够爬 `0.17 m` 现实楼梯。下一步应先检查足端/机身碰撞几何、台阶接触奖励和专门的抬腿动作，再恢复高度课程；不应继续无条件增加 PPO 迭代。

## 现实十级楼梯目标场景

根据用户补充信息，目标场景为总计 10 级，扶手作为碰撞物，楼梯净宽暂不定。因此新增目标配置：

```text
configs/phase_b5_stairs_high_real_10step_target.json
```

其中单级高度固定为 `0.17 m`、踏面固定为 `0.32 m`，净宽暂以 `0.90-1.20 m` 做逐环境随机化；两侧扶手采用 `0.90 m` 高、`0.05 m` 厚的静态碰撞几何，并在每级设置立柱。该配置仅用于目标场景和后续验证，不代表当前策略已经具备十级楼梯能力，也不会覆盖 v6 selected。

楼梯生成器已支持 `handrail_enabled`、扶手高度/厚度/侧向偏移/立柱间距等参数，训练脚本会从 `stair_curriculum` 读取这些字段。语法检查通过，核心单元测试 `28 passed`。

目标场景已用 Isaac Sim 图形窗口做 1 环境 smoke：场景成功生成，日志确认
`steps=10`、`height=0.17 m`、`tread=0.32 m`，无场景创建错误。该 smoke 只运行了
40 个控制步，机器人尚未到达楼梯，因此不能作为攀爬成功评估。

随后启动 10 级带扶手的浅台阶 bootstrap 可视化训练：使用 `0.005-0.015 m` 高度、
`0.31-0.33 m` 踏面、`0.90-1.20 m` 净宽，16 环境、6 轮更新。部分环境的最高验证级数
达到第 5 级，但 6 轮均为 `0` 次完整通过，单轮硬终止约 23-30 次，因此未创建 selected。
该结果说明十级路线的信用分配已有信号，但仍需先稳定连续通过，再提升台阶高度。

随后使用浅台阶 bootstrap 的 iter 6 直接热启动真实目标，进行 10 级、`0.17 m` 高、
`0.32 m` 踏面、带扶手的 8 环境可视化诊断训练（4 轮）。4 轮均为 0 次完成、0 级验证，
硬终止 16-20 次/轮，最大前进约 `0.82 m`，即在第一级前发生碰撞。该实验确认当前策略
不能从浅台阶直接迁移到真实高度；结果保存在：

```text
rl_runs/phase_b_residual/phase_b5_stairs_high_real_10step_direct_001/metrics.jsonl
```
下一步仍应逐级推进三阶或更高楼梯，并重新执行多种子门槛；不能外推为普通楼梯、
域随机化或实机能力。

## 修正后的跳升可达性与接触审计

为排除碰撞后自动复位造成的高度假象，`scripts/probe_stair_jump_feasibility.py`
在扫描期间将 `reset_stabilization_steps` 设为极大值，保持每个候选动作在同一条
连续物理轨迹中。18 组候选（触发距离 `0.08/0.12/0.16 m`、速度
`0.10/0.20/0.30 m/s`、下蹲 `6/10` 步）中有 2 组越过了几何上的第一级：

| 触发/速度 | 根部最大上升 | 验证级数 | 原始机身碰撞步数 | 峰值机身接触力 |
| --- | ---: | ---: | ---: | ---: |
| `0.08 m / 0.20 m/s`, 下蹲 6 步 | `0.105 m` | 1 | 341 | `5.75 kN` |
| `0.12 m / 0.30 m/s`, 下蹲 6 步 | `0.106 m` | 1 | 404 | `4.28 kN` |

这两条轨迹并非无碰撞攀爬；越级依赖持续机身接触，不能作为训练成功或实机动作
先验。其余候选最大上升不足 `0.071 m`。因此当前 URDF、执行器和机身碰撞几何下，
还没有得到可接受的 `17 cm` 单级越级动作。

## 真实单级跳升先验训练

将扫描中的髋/膝同步脉冲（下蹲 6 步、伸展 12 步）实现为可复位的仿真先验，使用
8 环境、HIGH 姿态、单级 `0.17 m`、踏面 `0.32 m`、扶手碰撞和速度命令
`0.20 m/s` 做 6 轮可视化 PPO。6 轮均为 `0/8` 完成，验证级数为 0，硬终止主要
是机身碰撞（每轮约 `27-34` 次），未生成 selected checkpoint。实验目录为：

```text
rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_timed_jump_001
```

结论仍是：不能继续无条件增加 PPO 迭代。下一步应检查机身碰撞盒、腿/轮接触模型、
关节方向和力矩接口；在得到无碰撞单级动作前，不应扩展到十级楼梯或实机。

## 8. 现实单级严格精炼与交接状态（2026-09-13）

### 8.1 训练输入与实际进度

现实单级精炼配置固定为 HIGH 姿态和实测楼梯几何：

```text
step height       0.17 m
tread             0.32 m
width             0.90-1.20 m (per environment)
steps             1
handrail          enabled, static collision
collision_terminates false (training only)
collision_penalty 200.0
level_hold_reward 10.0
num_envs          16
rollout_steps     500
learning rate     2e-6
residual alpha    0.10
fixed posture     HIGH
seed              211
```

训练目录：

```text
rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001
```

第一段 60 轮从原有残差检查点热启动，第二段再次从第一段的 `iter_00060` 热启动并完成
60 轮。启动器的迭代编号在第二段重新从 1 写到 60，因此当前文件名仍为
`residual_phase_b_iter_00060.pt`；按训练更新次数计，累计完成约 120 轮。最新检查点：

```text
rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001/checkpoints/residual_phase_b_iter_00060.pt
```

训练状态为 `completed`，最新训练结束时间为 `2026-09-13 17:57:51 +08:00`。训练日志中
HIGH 姿态有效，单轮通常有 11-15/16 个环境达到验证平台；这些是训练期统计，不能替代
严格零碰撞评估。

### 8.2 最新严格复测

严格复测文件：

```text
rl_runs/diagnostics/trained_policy_seed211.json
rl_runs/diagnostics/trained_policy_seed211.log
```

复测使用官方策略 + 上述残差检查点、环境内部 HIGH 跳升先验、单级 `0.17/0.32 m` 楼梯，
扫描 250 个控制步。最新 seed-211 结果：

| 指标 | 结果 |
| --- | ---: |
| 到达平台 | 1/1 |
| 最大验证级数 | 1 |
| 原始 `base_link` 碰撞步数 | 20 |
| 到达平台前碰撞步数 | 20 |
| 到达平台后碰撞步数 | 0 |
| 最大稳定保持步数 | 0 |
| 严格成功 | `false` |

机器人已经能够越过单级平台，且到达平台后没有继续发生机身碰撞；但仍在台阶前发生
机身碰撞，姿态/速度也尚未满足稳定保持条件。因此当前模型只能称为“可碰撞越级”，
不能称为无碰撞爬梯成功，也没有达到实机部署或十级楼梯验证标准。

### 8.3 验证脚本修正

`scripts/run_authorized_stair_scan.sh` 已删除 `--stop-after-level`。保留该参数会在首次
进入平台后停止移动，导致平台稳定保持统计失真；现在扫描会继续运行完整时域，才能检查
所需的 25 步稳定保持。训练脚本和环境逻辑未因该验证修正而改变。

### 8.4 下一位同事的继续步骤

1. 在具备 NVIDIA GPU 和可用 X/Isaac Sim 图形环境的终端确认训练状态和 checkpoint：

   ```bash
   cd /home/ky/tita_uc_residual_rl
   sed -n '1,20p' rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001/training.status
   ls -l rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001/checkpoints/residual_phase_b_iter_00060.pt
   ```

2. 先用当前检查点做严格单级扫描（不要把训练日志中的 `stair_success_total` 当成严格成功）：

   ```bash
   ./scripts/run_authorized_stair_scan.sh --worker
   ```

   判定依据是 `rl_runs/diagnostics/trained_policy_seed211.json` 中的：
   `strict_success=true`、`raw_collision_steps=0`、`max_stable_hold_steps>=25`。

3. 在 seed-211 通过之前，优先处理台阶前 20 个 `base_link` 碰撞，检查机身碰撞盒、足端/腿
   接触几何、HIGH 姿态的髋膝伸展时序和碰撞惩罚；不要直接推进 10 级楼梯。

4. 单级 seed-211 严格通过后，至少用两个额外确定性种子（建议 163、307）重复验证，
   再执行多种子聚合。只有多种子严格候选数大于 0，才可创建或标记新的 selected checkpoint。

5. 完成单级多种子后，再逐级推进两级/三阶，最后才进入目标 10 级配置
   `configs/phase_b5_stairs_high_real_10step_target.json`。该目标配置固定高度 `0.17 m`、
   踏面 `0.32 m`、10 级和碰撞扶手；当前策略尚未通过该目标。

### 8.5 环境注意事项

Codex 受限会话无法稳定创建 Isaac Sim 的 GPU/X 窗口；如果启动器在受限会话中只留下旧的
`training.status` 或报 `No device could be created`，应在用户本机带 RTX 4070 的终端运行
上述 `--worker` 命令。不要删除已有 checkpoint 或诊断文件；它们是下一轮训练和审计的输入。

### 8.6 参数调整原则

最新严格复测的 20 个 `base_link` 碰撞全部发生在到达平台前，平台后碰撞为 0；因此下一轮
应优先调整跳升动作和台阶接触时序，而不是只增加训练轮数或平台保持奖励。

建议一次只改一个参数组，并保留当前 checkpoint 作为基线：

| 参数组 | 当前值 | 建议首轮扫描范围 | 作用 |
| --- | ---: | ---: | --- |
| `jump_assist_trigger_distance_m` | 0.245 m | 0.18/0.22/0.27 m | 提前或延后开始下蹲 |
| 脉冲速度 | 0.45 m/s | 0.25/0.35/0.45 m/s | 接近台阶的水平速度 |
| `jump_assist_crouch_steps` | 8 | 6/8/10 | 下蹲持续时间 |
| `jump_assist_extension_steps` | 12 | 10/12/14/16 | 伸展持续时间 |
| 髋/膝伸展角 | 0.60/0.90 rad | 髋 0.45-0.70、膝 0.80-1.10 | 抬腿和伸展幅度 |
| 伸展延迟 | 0 步 | 0/2/4 步 | 髋膝错开伸展时序 |

扫描阶段优先使用 `scripts/probe_stair_jump_feasibility.py` 的多候选参数，比较
`raw_collision_steps`、`first_collision_step`、`max_base_contact_force_n` 和
`max_stable_hold_steps`。只有出现碰撞明显下降的候选，才用它热启动下一轮 PPO。

训练奖励参数可按以下顺序微调：先保持 `collision_terminates=false`，将
`collision_penalty` 从 200 小步提高到 300-400；确认碰撞下降后，再增加显式的落台姿态/速度
稳定奖励。`level_hold_reward_scale` 只按已验证级数给奖励，并不等于实际稳定保持奖励，不能
单独依靠把它调大来解决 `max_stable_hold_steps=0`。

以下参数在没有明确实验依据前不要大幅修改：学习率 `2e-6`、`residual_limit=0.25`、
冻结探索 `policy_log_std=-3`、残差滤波 `alpha=0.10`、安全投影和机身碰撞盒尺寸。碰撞盒
只有在和真实/URDF 尺寸核对后才能修改，不能为了得到 `strict_success` 而缩小碰撞体。

调参流程应为：复制配置和新建 run-id -> 运行 30-60 轮短训练 -> seed-211 严格扫描 ->
记录指标 -> 只保留碰撞下降且不破坏平地 HIGH 保留测试的候选。单级零碰撞并保持 25 步前，
不要推进到 10 级楼梯。

### 8.7 下一轮实验已准备

已保留当前现实单级 checkpoint，并创建独立的碰撞惩罚消融实验：

```text
config:
  configs/phase_b5_stairs_high_real_single_step_collision350.json
run-dir:
  rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_collision350_001
initial checkpoint:
  rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001/checkpoints/residual_phase_b_iter_00060.pt
changed parameter:
  stair_curriculum.collision_penalty: 200.0 -> 350.0
```

其余 HIGH 姿态、真实单级几何、跳升时序、学习率和安全投影参数保持不变。该实验尚未在
Codex 受限会话中启动；应在用户本机的 GPU/图形终端执行交接消息中的 IsaacLab 命令，
完成后再运行严格扫描并与 seed-211 基线的 20 个碰撞步进行比较。
