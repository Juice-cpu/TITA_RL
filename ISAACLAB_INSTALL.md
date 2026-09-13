# Isaac Lab installation record

Installed in `work/IsaacLab`:

- Isaac Lab repository: tag `v2.0.0`, commit `157a19e`
- Isaac Sim: existing `/home/ky/isaacsim`, version 4.5.0
- `isaaclab`: 0.33.13
- `isaaclab_tasks`: 0.10.22
- `isaaclab_rl`: 0.1.0
- `rl-games`: 1.6.1

The repository uses `_isaac_sim -> /home/ky/isaacsim`.  The Isaac Sim Python
runtime was restored after installation to the compatible versions
`numpy==1.26.4`, `protobuf==4.25.9`, and `opencv-python==4.10.0.84`.

The official 8-DOF TITA asset is available at
`work/tita_rl_official/resources/tita/urdf/tita_description.urdf`; its joint
order is `joint_left_leg_1..4, joint_right_leg_1..4`.

After reboot, the host GPU stack is healthy:

- GPU: NVIDIA GeForce RTX 4070 (12,282 MiB)
- Driver: 580.178.04
- Graphics API: Vulkan
- `torch.cuda.is_available()`: `True`

The Isaac Lab TITA task was run with the official URDF in headless mode on
`cuda:0`.  The 128-environment, 100-step smoke completed with finite rewards,
`terminated=0`, and policy observations of shape `(128, 40)` at every report.
The one-environment smoke also verified actual initialization and one physics
step.  Isaac Sim may print non-fatal warnings for missing masses on the
camera/accessory links in the official URDF; these are importer warnings and
do not change the validated eight-joint articulation contract.

On a GPU-enabled host, verify the installation with:

```bash
cd work/IsaacLab
TERM=xterm ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
```

Run the TITA task smoke from this package directory:

```bash
cd /home/ky/tita_uc_residual_rl
PYTHONUNBUFFERED=1 PYTHONPATH=. TERM=xterm \
  ./IsaacLab/isaaclab.sh -p scripts/run_tita_residual_lab.py \
  --asset ./tita_rl_official/resources/tita/urdf/tita_description.urdf \
  --headless --device cuda:0 --num_envs 128 --steps 100
```
