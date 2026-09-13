# TITA 楼梯强化学习入门与操作指南

更新日期：2026-09-13

这份文档面向第一次接触本项目强化学习的同事。它解释当前代码实际做什么，如何启动训练，如何判断训练是否有效，以及如何记录一次实验。

## 0. 先记住三件事

1. 当前系统只在 Isaac Sim 中训练和验证，不会向真实机器人发送动作。
2. 官方 TITA 策略被冻结；强化学习只学习一个 8 维关节目标残差。
3. 训练日志中的“到达楼梯”不等于成功。严格成功必须是到达平台、`base_link` 零碰撞，并在平台稳定保持至少 25 个控制步。

当前现实楼梯目标是：单级高度 `0.17 m`、踏步深度 `0.32 m`、总共 10 级、扶手为碰撞物。当前研发阶段只训练和验证 1 级，不应把 1 级结果外推为 10 级能力。

## 1. 强化学习是什么

可以把机器人训练理解为一个循环：

~~~
观察当前状态
  -> 策略给出动作
  -> Isaac Sim 推进一步物理
  -> 计算奖励和代价
  -> 用这些数据更新策略
  -> 重复
~~~

在本项目中：

- 状态（observation）：机器人姿态、速度、关节状态、楼梯距离和楼梯高度等。
- 动作（action）：8 个关节目标的残差，不是直接力矩。
- 奖励（reward）：前进、速度跟踪、上台阶和成功会增加奖励。
- 代价（cost）：路线偏离、能耗、姿态变化、动作突变和接触会增加代价。
- episode：一个机器人从出生到超时、失败或完成任务的一段轨迹。
- policy：根据状态产生动作的神经网络。
- critic：估计未来奖励/代价，帮助 policy 判断哪些动作值得保留。

## 2. 当前控制链路

~~~
官方策略（冻结）
  输入：33 维本体信息 + 10 帧历史
  输出：8 维基础关节目标
              |
              v
残差策略（训练）
  输入：约 51 维环境观测
  输出：8 维 residual，限制在约 +/-0.25 rad
              |
              v
HIGH 姿态偏置 + 楼梯跳升辅助
              |
              v
安全投影：关节位置/速度/力矩/加速度限制
              |
              v
PD/轮子力矩控制器 -> Isaac Sim
~~~

最终目标大致是：

~~~
q_raw = q_base + posture_offset + residual + stair_jump_assist
q_cmd = SafetyProject(q_raw)
~~~

对应实现见 [isaaclab_tita_env.py](/home/ky/tita_uc_residual_rl/tita_uc_residual_rl/isaaclab_tita_env.py:735)。

## 3. 当前残差策略

文件：[residual_policy.py](/home/ky/tita_uc_residual_rl/tita_uc_residual_rl/residual_policy.py:12)

主要逻辑：

~~~
residual = tanh(network(observation)) * residual_limit
~~~

- 网络结构：输入观测 -> `256` -> `128` -> `8`。
- 最后一层初始化为零，所以训练第 0 轮等价于官方基础策略。
- 训练时从 Gaussian 分布采样动作，当前 `log_std=-3`，标准差约为 `0.05`。
- 评估时使用策略均值，不使用探索噪声。
- 策略不会自己选择 LOW/MID/HIGH；本实验由命令固定为 HIGH。

## 4. 当前奖励和代价

奖励实现见 [isaaclab_tita_env.py](/home/ky/tita_uc_residual_rl/tita_uc_residual_rl/isaaclab_tita_env.py:1356)。

基础奖励：

~~~
tracking_reward = exp(-线速度误差^2 / 0.25)
                + 0.5 * exp(-偏航速度误差^2 / 0.25)

基础奖励 = 2.0 * tracking_reward
         + 2.0 * 正向位移
~~~

四个软代价：

1. route：绕路、前进不足、靠近台阶风险、曲率和走廊偏离。
2. energy：`sum(abs(torque * joint_velocity)) * dt`。
3. mode：机身高度变化过快，或实际高度偏离 HIGH 目标高度。
4. smooth：关节目标变化、残差变化、加速度比例、腿部软接触。

代价先除以冻结的离线 P95，再进入目标函数。当前配置中的权重约为：

~~~
-0.40 * route
-0.35 * energy
-0.25 * mode
-0.05 * smooth
~~~

现实单级楼梯额外使用：

~~~
+200 * 通过楼梯
+12  * 正向高度增加
+40  * 接近台阶时抬升
+16  * 验证通过一级
+10  * 已通过平台保持奖励
-350 * 机身碰撞（collision350 实验）
~~~

原始基线配置的碰撞惩罚是 `200`；`collision350` 是当前准备的下一轮消融实验。

## 5. 当前训练算法

文件：[phase_b_learner.py](/home/ky/tita_uc_residual_rl/tita_uc_residual_rl/phase_b_learner.py:103)

这是项目自己写的紧凑 PPO/NP3O 风格实现，不是官方 NP3O trainer 的逐行复制。

每次更新：

1. 并行运行 16 个环境，每个环境采样 500 个控制步。
2. 记录观测、动作、旧策略概率、奖励、4 个代价和 done 标志。
3. 用 reward critic 和 4 个 cost critic 计算 GAE 优势。
4. 对 reward advantage 和 cost advantage 做归一化。
5. 用 PPO 的概率比裁剪限制策略单次变化。
6. 用四个拉格朗日乘子惩罚超过预算的代价。
7. 用 Adam 更新 residual policy、reward critic 和 cost critic。
8. 每轮写入 `metrics.jsonl`，并保存 checkpoint。

核心超参数：

~~~
gamma            0.99
GAE lambda       0.95
PPO clip ratio   0.20
update epochs    4
mini-batches     4
gradient clip    1.0
learning rate    当前楼梯实验 2e-6
~~~

## 6. 启动前检查

训练必须在能使用 NVIDIA GPU 和 Isaac Sim 图形窗口的主机终端运行。Codex 受限会话可能看不到 GPU/X 显示，这不是训练代码本身的错误。

~~~
cd /home/ky/tita_uc_residual_rl
nvidia-smi
~~~

如果要确认 Isaac Lab 能创建仿真，可运行一个很短的 headless smoke：

~~~
PYTHONUNBUFFERED=1 PYTHONPATH=. TERM=xterm \
./IsaacLab/isaaclab.sh -p scripts/run_tita_residual_lab.py \
  --asset ./tita_rl_official/resources/tita/urdf/tita_description.urdf \
  --headless --device cuda:0 --num_envs 1 --steps 10
~~~

看到 `cuda:0`、环境初始化和 physics step 后再开始长训练。

## 7. 继续当前现实单级实验

当前最新基线 checkpoint：

~~~
rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001/checkpoints/residual_phase_b_iter_00060.pt
~~~

已经准备好的下一轮配置：

~~~
configs/phase_b5_stairs_high_real_single_step_collision350.json
rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_collision350_001
~~~

它只把 `collision_penalty` 从 `200` 改成 `350`，其余参数保持相同。

在用户本机、希望看到 Isaac Sim 界面的终端运行：

~~~
cd /home/ky/tita_uc_residual_rl
./IsaacLab/isaaclab.sh -p scripts/train_phase_b.py \
  --asset ./tita_rl_official/resources/tita/urdf/tita_description.urdf \
  --checkpoint ./tita_rl_official/tita_example_10000.pt \
  --initial-residual-checkpoint \
  ./rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_strict_visual_001/checkpoints/residual_phase_b_iter_00060.pt \
  --run-dir ./rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_collision350_001 \
  --num-envs 16 \
  --iterations 60 \
  --rollout-steps 500 \
  --command-x 0.45 \
  --seed 211 \
  --acceleration-limit 500 \
  --residual-filter-alpha 0.10 \
  --terrain stairs \
  --fixed-posture HIGH \
  --device cuda:0
~~~

不要同时启动训练和扫描。训练时另开一个终端监控：

~~~
tail -f rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_collision350_001/training.log
~~~

状态文件：

~~~
cat rl_runs/phase_b_residual/phase_b5_stairs_high_real_single_step_collision350_001/training.status
~~~

## 8. 严格评估

训练完成后，必须运行：

~~~
cd /home/ky/tita_uc_residual_rl
./scripts/run_authorized_stair_scan.sh --worker
~~~

扫描脚本已经删除 `--stop-after-level`，会继续观察平台保持阶段。

结果文件：

~~~
rl_runs/diagnostics/trained_policy_seed211.json
~~~

严格成功必须同时满足：

~~~
verified_level_max >= 1
raw_collision_steps == 0
max_stable_hold_steps >= 25
strict_success == true
~~~

训练期的 `stair_success_total` 只是奖励统计，不能替代这些条件。

## 9. 调参顺序

一次只改一个参数组。当前失败是台阶前碰撞，所以下一轮优先级如下。

### 9.1 动作时序

~~~
jump_assist_trigger_distance_m: 0.18 / 0.22 / 0.27
pulse speed:                    0.25 / 0.35 / 0.45 m/s
jump_assist_crouch_steps:       6 / 8 / 10
jump_assist_extension_steps:    10 / 12 / 14 / 16
hip extension:                  0.45-0.70 rad
knee extension:                 0.80-1.10 rad
extension delay:                0 / 2 / 4 steps
~~~

这些参数可以先用 `scripts/probe_stair_jump_feasibility.py` 扫描，不要每种组合都训练 60 轮。先找碰撞明显少的候选，再用该候选热启动 PPO。

### 9.2 奖励参数

如果动作时序已经有改善，再小步尝试：

~~~
collision_penalty: 200 -> 300 -> 350 -> 400
~~~

平台保持奖励不能单独解决稳定问题，因为当前 `level_hold_reward_scale` 主要按已验证级数给奖励，不是完整的速度/姿态稳定奖励。

### 9.3 暂时不要动的参数

没有明确实验依据时保持：

~~~
learning_rate       2e-6
residual_limit      0.25
policy_log_std      -3.0
freeze_policy_std   true
residual_filter     0.10
acceleration_limit  500
~~~

不要为了得到成功而缩小 `base_link` 碰撞盒；碰撞盒只能依据 URDF/实物尺寸核对后修改。

## 10. 新实验的标准流程

~~~
复制配置
  -> 改一个参数组
  -> prepare_rl_run 创建新 run-id
  -> 训练 30-60 轮
  -> 严格扫描 seed-211
  -> 与基线比较
  -> 记录结论
~~~

创建新实验目录：

~~~
python3 scripts/prepare_rl_run.py \
  --config configs/my_stair_experiment.json \
  --stage phase_b_residual \
  --run-id my_stair_experiment_001
~~~

不要直接删除旧目录，也不要覆盖原始 selected checkpoint。

## 11. 如何记录一次实验

每次训练结束后，在实验目录保留以下信息：

~~~
日期：
实验名称：
配置文件：
run-dir：
初始 checkpoint：
seed：
num_envs：
iterations：
rollout_steps：
修改的参数：
训练是否完成：
最新 checkpoint：
严格评估文件：
verified_level_max：
raw_collision_steps：
post_level_collision_steps：
max_stable_hold_steps：
strict_success：
结论：保留 / 拒绝 / 需要继续调参
下一步：
~~~

建议同时保存训练日志和严格评估日志，不要只截图 Isaac Sim 窗口。

## 12. 启动脚本逐行说明

### 12.1 `run_authorized_stair_training.sh`

文件：[run_authorized_stair_training.sh](/home/ky/tita_uc_residual_rl/scripts/run_authorized_stair_training.sh:1)

| 行 | 含义 |
|---:|---|
| 1 | 告诉系统使用 bash 执行。 |
| 2 | 开启未定义变量和管道错误检查。 |
| 4 | 定义项目根目录。 |
| 5 | 定义当前训练输出目录。 |
| 6-9 | 定义训练日志、状态、PID 和后台启动日志路径。 |
| 11 | 切换到项目根目录，保证相对路径有效。 |
| 13-18 | 不带 `--worker` 时启动后台 worker；带 `--worker` 时执行本体。 |
| 20-22 | 设置实时 Python 输出、模块路径和终端类型。 |
| 24-26 | 清理旧状态并写入 `state=running`、PID 和开始时间。 |
| 28-40 | 定义退出处理函数，写入 `completed` 或 `failed`。 |
| 42 | 启动 Isaac Lab 的训练 Python。 |
| 43-46 | 指定 URDF、官方策略、初始残差 checkpoint 和输出目录。 |
| 47-49 | 设置并行环境数、更新轮数和每轮采样步数。 |
| 50-53 | 设置速度命令、随机种子、加速度限制和残差滤波。 |
| 54-56 | 选择楼梯、HIGH 姿态和 `cuda:0`，并写入日志。 |
| 58 | 返回 Isaac Lab 的真实退出码。 |

### 12.2 `run_authorized_stair_scan.sh`

文件：[run_authorized_stair_scan.sh](/home/ky/tita_uc_residual_rl/scripts/run_authorized_stair_scan.sh:1)

| 行 | 含义 |
|---:|---|
| 1-2 | 使用 bash，并启用严格检查。 |
| 4-10 | 定义诊断目录、结果、日志、状态、PID 和后台日志路径。 |
| 12-13 | 创建诊断目录并进入项目根目录。 |
| 15-20 | 不带 `--worker` 时启动后台扫描；带 `--worker` 时执行本体。 |
| 22-24 | 设置实时输出、模块路径和终端类型。 |
| 26-28 | 删除旧结果并写入 `state=running`。 |
| 30-41 | 退出时根据结果文件和退出码写入最终状态。 |
| 43-48 | 启动扫描、指定 URDF、checkpoint、步数和 JSON 输出。 |
| 49-62 | 指定 seed、触发距离、速度、下蹲/伸展时序和髋膝幅度。 |
| 63 | 使用环境内部跳升辅助，避免重复叠加。 |
| 64-65 | 输出摘要、使用 `cuda:0` 并保存日志。 |
| 67 | 返回 Isaac Lab 的真实退出码。 |

对于需要观察 GUI 的本机，建议直接执行带 `--worker` 的命令，避免后台脱离终端后看不到启动错误。

## 13. 常见问题

### Isaac Sim 报 `No device could be created`

说明当前终端没有可用 GPU/X 显示。换到主机图形终端运行，检查 `nvidia-smi` 和 `DISPLAY`，不要删除 checkpoint。

### 状态文件一直是 `running`

查看 `training.log` 的最后时间和 checkpoint 是否仍在更新。若没有更新且没有 Isaac 进程，通常是窗口被关闭或进程异常退出；查看 launcher log 的最后 50 行。

### 训练显示很多 `stair_success_total`

这只是训练期奖励统计。必须运行严格扫描，检查零碰撞和稳定保持。

### 修改后结果变差

不要立刻继续增加轮数。先恢复上一份 checkpoint，确认只改了一个参数组，然后比较 `raw_collision_steps`、`first_collision_step`、`max_base_contact_force_n` 和 `max_stable_hold_steps`。

## 14. 当前交接结论

当前模型能越过一个现实高度单级台阶，但 seed-211 严格复测仍有 20 个台阶前机身碰撞，平台后碰撞为 0，稳定保持为 0，严格成功为 `false`。下一位同事应先完成单级零碰撞和 25 步稳定保持，再进行多种子验证，最后才推进 10 级楼梯。

完整历史记录见：[TRAINING_TECHNICAL_RECORD_2026-09-13_CN.md](/home/ky/tita_uc_residual_rl/docs/TRAINING_TECHNICAL_RECORD_2026-09-13_CN.md:521)。
