# G1 + BrainCo Revo2 LeRobot 训练与推理教程

本文档给出一条不依赖 OpenPI 源码的 LeRobot 路线：

```text
G1 + BrainCo .h5 数据
        |
        v
convert_unitree_h5_to_lerobot.py
        |
        v
LeRobotDataset v3
        |
        v
LeRobot pi0 fine-tune
        |
        v
LeRobot policy checkpoint
        |
        v
dataset dry-run / G1 + BrainCo 实机推理
```

默认硬件假设：

- 机器人：Unitree G1 29DoF 双臂
- 灵巧手：BrainCo Revo2，左右手各 6 维
- 数据：`.h5`
- 训练机：NVIDIA RTX A6000 48GB
- 推理机：可以是训练机、机器人端工控机，或同局域网 GPU 主机
- LeRobot 源码：使用本仓库内的 `unitree_lerobot/lerobot/src`

本文档命令默认从仓库根目录运行：

```bash
cd /path/to/unitree_lerobot
```

如果你没有执行 `pip install -e .`，按本文档统一设置：

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/unitree_lerobot/lerobot/src:$PYTHONPATH
export HF_LEROBOT_HOME=/data/lerobot
export HF_HOME=/data/huggingface
```

`HF_LEROBOT_HOME` 用于保存转换后的 LeRobot 数据集，例如：

```text
/data/lerobot/local/g1_brainco_press_green_button
```

---

## 1. 数据格式约定

每条 episode 的 H5 至少包含：

```text
actual_joint_positions_rad      (T, 14) float
ik_joint_positions_rad          (T, 14) float
left_finger_actual_angles       (T, 6)  float
right_finger_actual_angles      (T, 6)  float
left_finger_target_angles       (T, 6)  float
right_finger_target_angles      (T, 6)  float
head_rgb                        (T, H, W, 3) uint8
left_rgb                        (T, H, W, 3) uint8
right_rgb                       (T, H, W, 3) uint8
timestamps                      (T,)
```

训练使用的 state/action 顺序固定为：

```text
state  = actual_joint_positions_rad(14)
       + left_finger_actual_angles(6)
       + right_finger_actual_angles(6)
       = 26D

action = ik_joint_positions_rad(14)
       + left_finger_target_angles(6)
       + right_finger_target_angles(6)
       = 26D
```

action 维度约定：

```text
0:14   -> 双臂目标关节角，单位 rad
14:20  -> 左 BrainCo 手 6 维目标
20:26  -> 右 BrainCo 手 6 维目标
```

相机映射：

```text
head_rgb  -> observation.images.cam_left_high
left_rgb  -> observation.images.cam_left_wrist
right_rgb -> observation.images.cam_right_wrist
```

注意：

- `cam_left_high` 只是沿用 Unitree/LeRobot 的命名。这里只有一路头部 RGB，不使用 `cam_right_high`。
- 图像必须在训练和推理时保持同一方向、同一颜色通道约定。建议统一为 `HWC uint8 RGB`。
- BrainCo 手指值可以和手臂 rad 不同尺度，LeRobot 会根据 dataset stats 做归一化；关键是训练和推理的 26 维顺序必须完全一致。

---

## 2. A6000 训练环境

建议使用 Python 3.10：

```bash
conda create -y -n g1_lerobot python=3.10
conda activate g1_lerobot
conda install -y ffmpeg=7.1.1 -c conda-forge
```

安装依赖。这里不是 editable install，后续命令通过 `PYTHONPATH` 使用源码：

```bash
cd /path/to/unitree_lerobot/unitree_lerobot/lerobot
pip install ".[pi]"

cd /path/to/unitree_lerobot
pip install tyro h5py opencv-python-headless tqdm matplotlib meshcat logging_mp
```

检查源码导入：

```bash
cd /path/to/unitree_lerobot
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/unitree_lerobot/lerobot/src:$PYTHONPATH

python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from unitree_lerobot.utils.constants import ROBOT_CONFIGS
print("lerobot import ok")
print(len(ROBOT_CONFIGS["Unitree_G1_Brainco"].motors))
PY
```

期望输出包含：

```text
lerobot import ok
26
```

---

## 3. 检查 H5 数据

假设原始数据目录：

```bash
export RAW_DIR=/data/raw/press_green_button
```

先检查第一条 episode：

```bash
python - <<'PY'
import h5py
from pathlib import Path

raw_dir = Path("/data/raw/press_green_button")
h5_path = sorted(list(raw_dir.glob("*.h5")) + list(raw_dir.glob("*.hdf5")))[0]
print("file:", h5_path)

with h5py.File(h5_path, "r") as f:
    print("attrs:", dict(f.attrs))
    f.visititems(lambda n, o: print(n, getattr(o, "shape", None), getattr(o, "dtype", None)))
PY
```

再检查 state/action 拼接结果：

```bash
python - <<'PY'
import h5py
import numpy as np
from pathlib import Path

h5_path = sorted(list(Path("/data/raw/press_green_button").glob("*.h5")))[0]
with h5py.File(h5_path, "r") as f:
    state = np.concatenate([
        f["actual_joint_positions_rad"][:],
        f["left_finger_actual_angles"][:],
        f["right_finger_actual_angles"][:],
    ], axis=1)
    action = np.concatenate([
        f["ik_joint_positions_rad"][:],
        f["left_finger_target_angles"][:],
        f["right_finger_target_angles"][:],
    ], axis=1)

    print("state", state.shape, state.dtype, np.isfinite(state).all())
    print("action", action.shape, action.dtype, np.isfinite(action).all())
    for k in ["head_rgb", "left_rgb", "right_rgb"]:
        x = f[k]
        print(k, x.shape, x.dtype)
PY
```

正确结果应为：

```text
state  (T, 26)
action (T, 26)
head_rgb/left_rgb/right_rgb (T, H, W, 3) uint8
```

---

## 4. 转换为 LeRobotDataset

转换命令：

```bash
cd /path/to/unitree_lerobot
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/unitree_lerobot/lerobot/src:$PYTHONPATH
export HF_LEROBOT_HOME=/home/ps/lzk/manipulation/data/lerobot

python unitree_lerobot/utils/convert_unitree_h5_to_lerobot.py \
  --raw-dir /data/raw/press_green_button \
  --repo-id local/g1_brainco_press_green_button \
  --robot-type Unitree_G1_Brainco \
  --task "press the green button on the electrical cabinet" \
  --push-to-hub false \
  --mode video \
  --strict-dims true
```

输出目录：

```text
/data/lerobot/local/g1_brainco_press_green_button
```

转换脚本检查结论：

- G1 + BrainCo flat H5 的 state/action 拼接逻辑正确。
- BrainCo target 支持 6D，也兼容旧版 7D `[event_flag + 6 targets]`。
- 三路相机映射为 `cam_left_high/cam_left_wrist/cam_right_wrist`。
- 当前脚本已经改为 LeRobot v3 的 `dataset.finalize()`，不再调用旧接口 `consolidate()`。
- 当前脚本已经加入本地源码路径 bootstrap，没有 `pip install -e .` 时也能找到本仓库的 `unitree_lerobot` 和 vendored `lerobot` 源码。

---

## 5. 验证转换结果

加载 dataset：

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/g1_brainco_press_green_button")
print(ds)
print(ds.meta.features)
print("frames:", len(ds))
print("episodes:", ds.meta.total_episodes)

sample = ds[0]
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(k, tuple(v.shape), getattr(v, "dtype", None))
    else:
        print(k, v)
PY
```

重点确认：

```text
observation.state                 -> 26
action                            -> 26
observation.images.cam_left_high  -> exists
observation.images.cam_left_wrist -> exists
observation.images.cam_right_wrist-> exists
task                              -> correct text
```

可视化：

```bash
python unitree_lerobot/lerobot/src/lerobot/scripts/lerobot_dataset_viz.py \
  --repo-id local/g1_brainco_press_green_button \
  --episode-index 0
```

如果颜色明显异常，先确认 H5 中的图像到底是 RGB 还是 OpenCV BGR。训练和推理必须一致。

---

## 6. LeRobot pi0 训练

LeRobot 的 pi0 内部会把 26D state/action padding 到 32D，与 pi0 base checkpoint 兼容；推理时会裁回 dataset 定义的 26D action。

### 6.1 小规模 overfit 测试

先用少量 step 验证数据、模型、显存、checkpoint 写入都正常：

```bash
cd /path/to/unitree_lerobot
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/unitree_lerobot/lerobot/src:$PYTHONPATH
export HF_LEROBOT_HOME=/home/ps/lzk/manipulation/data/lerobot
export HF_HOME=/home/ps/lzk/manipulation/data/huggingface

python unitree_lerobot/lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=local/g1_brainco_press_green_button \
  --policy.type=pi0 \
  --policy.pretrained_path=/home/ps/lzk/manipulation/models/pi0_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --batch_size=4 \
  --steps=1000 \
  --save_freq=500 \
  --output_dir=outputs/train/pi0_g1_brainco_overfit \
  --job_name=pi0_g1_brainco_overfit \
  --wandb.enable=false
```

A6000 上建议先从 `batch_size=4` 开始。稳定后可以试：

```text
batch_size=8
policy.chunk_size=10 或 20
```

如果 episode 较短，先保持 `chunk_size=10`。`chunk_size=50` 会要求每个训练样本有更长未来 action window。

### 6.2 正式训练

```bash
python unitree_lerobot/lerobot/src/lerobot/scripts/lerobot_train.py \
  --dataset.repo_id=local/g1_brainco_press_green_button \
  --policy.type=pi0 \
  --policy.pretrained_path=/home/ps/lzk/manipulation/models/pi0_base \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=false \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --batch_size=8 \
  --steps=30000 \
  --save_freq=5000 \
  --output_dir=outputs/train/pi0_g1_brainco_press_green_button \
  --job_name=pi0_g1_brainco_press_green_button \
  --wandb.enable=false
```

checkpoint 路径类似：

```text
outputs/train/pi0_g1_brainco_press_green_button/checkpoints/005000/pretrained_model
outputs/train/pi0_g1_brainco_press_green_button/checkpoints/010000/pretrained_model
outputs/train/pi0_g1_brainco_press_green_button/checkpoints/last/pretrained_model
```

如果显存不足：

```text
1. batch_size 改为 4 或 2
2. 保持 gradient_checkpointing=true
3. policy.compile_model=false
4. chunk_size 保持 10
```

如果训练 loss 很快变成 NaN：

```text
1. 检查 H5 state/action 是否有 NaN/Inf
2. 检查 BrainCo target 是否有异常尖峰
3. 先用 5-10 条高质量 episode overfit
4. 确认 action 26 维顺序没有和推理端不一致
```

---

## 7. 训练后离线推理检查

先在 dataset 上 dry-run，不发给机器人：

```bash
export POLICY_DIR=outputs/train/pi0_g1_brainco_press_green_button/checkpoints/last/pretrained_model

python unitree_lerobot/eval_robot/eval_g1_dataset.py \
  --policy.path=$POLICY_DIR \
  --repo_id=local/g1_brainco_press_green_button \
  --frequency=30 \
  --arm=G1_29 \
  --ee=brainco \
  --visualization=true \
  --send_real_robot=false
```

这个步骤用于确认：

- checkpoint 能加载
- dataset stats 能加载
- preprocessor/postprocessor 能创建
- policy 输出 action 是 26D
- action 切分后可以对应 `arm 14D + left hand 6D + right hand 6D`

也可以用最小 Python 检查 action shape：

```bash
python - <<'PY'
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.utils import get_safe_torch_device
from unitree_lerobot.eval_robot.utils.utils import extract_observation, predict_action

repo_id = "local/g1_brainco_press_green_button"
policy_dir = "outputs/train/pi0_g1_brainco_press_green_button/checkpoints/last/pretrained_model"

ds = LeRobotDataset(repo_id)
cfg = PreTrainedConfig.from_pretrained(policy_dir)
cfg.pretrained_path = policy_dir
cfg.device = "cuda"

policy = make_policy(cfg=cfg, ds_meta=ds.meta)
policy.eval()

pre, post = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=policy_dir,
    dataset_stats=rename_stats(ds.meta.stats, {}),
    preprocessor_overrides={"device_processor": {"device": cfg.device}},
)

step = ds[0]
obs = extract_observation(step)
action = predict_action(
    obs,
    policy,
    get_safe_torch_device(cfg.device),
    pre,
    post,
    cfg.use_amp,
    step["task"],
    use_dataset=True,
)
print(action.shape, action.dtype)
print(action[:14])
print(action[14:20])
print(action[20:26])
PY
```

期望：

```text
torch.Size([26])
```

---

## 8. 实机推理部署

### 8.1 部署前准备

推理机需要：

- 能访问 G1 DDS 网络
- 已安装 `unitree_sdk2_python`
- 已配置 Unitree 网络接口
- 能启动相机 image server
- 能读取 G1 双臂当前关节角
- 能读取 BrainCo 左右手当前 6D state
- 能发送 G1 双臂目标和 BrainCo 左右手 6D target

安装 Unitree SDK：

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

推理机也建议设置同样的源码路径：

```bash
cd /path/to/unitree_lerobot
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/unitree_lerobot/lerobot/src:$PYTHONPATH
export HF_LEROBOT_HOME=/data/lerobot
```

把训练好的 checkpoint 和转换后的 dataset meta/stats 同步到推理机。最简单做法是同步整个：

```text
/data/lerobot/local/g1_brainco_press_green_button
outputs/train/pi0_g1_brainco_press_green_button/checkpoints/last/pretrained_model
```

### 8.2 使用 eval_g1.py 实机推理

确认机器人处于安全姿态，急停可用，周围无人员和障碍物。然后运行：

```bash
export POLICY_DIR=outputs/train/pi0_g1_brainco_press_green_button/checkpoints/last/pretrained_model

python unitree_lerobot/eval_robot/eval_g1.py \
  --policy.path=$POLICY_DIR \
  --repo_id=local/g1_brainco_press_green_button \
  --frequency=30 \
  --arm=G1_29 \
  --ee=brainco \
  --visualization=true
```

脚本会读取 dataset 第 0 帧作为初始化参考姿态，并提示输入：

```text
Enter 's' to initialize the robot and start the evaluation:
```

只有确认安全后再输入 `s`。

实机 observation 与训练必须一致：

```python
observation = {
    "observation.images.cam_left_high": head_rgb,         # HWC uint8 RGB
    "observation.images.cam_left_wrist": left_wrist_rgb,  # HWC uint8 RGB
    "observation.images.cam_right_wrist": right_wrist_rgb,# HWC uint8 RGB
    "observation.state": np.concatenate([
        actual_joint_positions_rad,  # 14
        left_finger_actual_angles,   # 6
        right_finger_actual_angles,  # 6
    ]).astype("float32"),
}
```

policy 输出：

```python
action = policy_output  # 26D
arm_action = action[:14]
left_hand_action = action[14:20]
right_hand_action = action[20:26]
```

发送规则：

```text
arm_action        -> G1 双臂目标关节角
left_hand_action  -> 左 BrainCo 6 维目标
right_hand_action -> 右 BrainCo 6 维目标
```

### 8.3 A6000 远程推理部署

如果 G1 机器人端不适合直接跑 pi0，推荐使用远程推理：

```text
G1/机器人侧
  - TeleImager 图像服务
  - G1 + BrainCo 控制
  - eval_g1_remote.py 采集 observation、执行 action

A6000 工作站
  - remote_policy_server.py 加载 policy
  - 接收 observation
  - 返回 26D action
```

机器人侧和 A6000 不在同一个局域网时，可以用 Tailscale 互联。关键是 G1 端能访问：

```text
http://<A6000_TAILSCALE_IP>:8088/predict
```

A6000 不需要访问 G1 的图像服务；图像由 G1 端采集后主动发给 A6000。

#### 8.3.1 启动 TeleImager 图像服务

你的图像服务使用 `teleimager/cam_config_server.yaml`，当前配置是三路 RealSense：

```text
head_camera        -> ZMQ 55555
left_wrist_camera  -> ZMQ 55556
right_wrist_camera -> ZMQ 55557
config request     -> ZMQ 60000
image_shape        -> 480x640
fps                -> 30
```

因为 `type: realsense`，启动图像服务时必须加 `--rs`。

在 G1/机器人侧运行：

```bash
cd /path/to/teleimager
conda activate teleimager

python -m teleimager.image_server --rs
```

如果已经安装了 console script，也可以运行：

```bash
teleimager-server --rs
```

如果只在源码目录运行，没有安装 editable package：

```bash
cd /path/to/teleimager
export PYTHONPATH=$PWD/src:$PYTHONPATH
python -m teleimager.image_server --rs
```

另开一个终端验证图像客户端：

```bash
cd /path/to/teleimager
python -m teleimager.image_client --host 192.168.123.164
```

如果你的图像服务 IP 不是 `192.168.123.164`，后面的 `--image_host` 要改成实际 IP。

#### 8.3.2 启动 A6000 policy server

在 A6000 工作站运行：

```bash
cd /path/to/unitree_lerobot
conda activate g1_lerobot

export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/unitree_lerobot/lerobot/src:$PYTHONPATH
export HF_LEROBOT_HOME=/data/lerobot

export POLICY_DIR=outputs/train/pi0_g1_brainco_press_green_button/checkpoints/last/pretrained_model

python unitree_lerobot/eval_robot/remote_policy_server.py \
  --policy.path=$POLICY_DIR \
  --repo_id=local/g1_brainco_press_green_button \
  --host=0.0.0.0 \
  --port=8088 \
  --device=cuda \
  --task "press the green button on the electrical cabinet" \
  --robot_type=Unitree_G1_Brainco
```

如果走 Tailscale，`--host=0.0.0.0` 保持不变，让服务监听所有网卡。G1 侧连接时使用 A6000 的 Tailscale IP。

在 G1 侧可以先验证服务健康状态：

```bash
curl http://<A6000_TAILSCALE_IP>:8088/health
```

应返回类似：

```json
{"status": "ok", "policy_path": "...", "repo_id": "local/g1_brainco_press_green_button", "device": "cuda:0"}
```

#### 8.3.3 启动 G1 远程推理客户端

先 dry-run，不发真实机器人，只确认图像、状态、远程推理、动作范围和延迟：

```bash
cd /path/to/unitree_lerobot
conda activate g1_lerobot

export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/unitree_lerobot/lerobot/src:$PYTHONPATH
export HF_LEROBOT_HOME=/data/lerobot

python unitree_lerobot/eval_robot/eval_g1_remote.py \
  --server_host=<A6000_TAILSCALE_IP> \
  --server_port=8088 \
  --image_host=192.168.123.164 \
  --repo_id=local/g1_brainco_press_green_button \
  --frequency=30 \
  --arm=G1_29 \
  --ee=brainco \
  --task "press the green button on the electrical cabinet" \
  --send_real_robot=false \
  --visualization=true \
  --max_steps=300
```

终端提示后输入：

```text
s
```

这时脚本会循环：

```text
1. 从 TeleImager 读取三路图像
2. 从 G1/BrainCo 读取 26D state
3. 将图像和 state 发给 A6000
4. A6000 返回 26D action
5. send_real_robot=false 时只打印和可视化，不执行
```

确认正常后再真实执行：

```bash
python unitree_lerobot/eval_robot/eval_g1_remote.py \
  --server_host=<A6000_TAILSCALE_IP> \
  --server_port=8088 \
  --image_host=192.168.123.164 \
  --repo_id=local/g1_brainco_press_green_button \
  --frequency=30 \
  --arm=G1_29 \
  --ee=brainco \
  --task "press the green button on the electrical cabinet" \
  --send_real_robot=true \
  --visualization=true \
  --max_steps=300 \
  --ready_pose_source=dataset \
  --ready_move_duration=4.0
```

`--max_steps=300` 在 30Hz 下约为 10 秒。第一次真实执行建议先用更短时间：

```bash
--max_steps=150
```

#### 8.3.4 指令是如何给模型的

远程推理不是在某一帧额外发送“现在去按按钮”的控制信号。任务指令在启动时通过 `--task` 固定传入：

```bash
--task "press the green button on the electrical cabinet"
```

每一帧 A6000 policy server 都会收到：

```text
task text
三路图像
当前 26D state
```

模型根据这些输入输出下一步动作。真正让机器人开始执行的是：

```text
1. G1 端脚本启动后输入 s
2. 设置 --send_real_robot=true
```

如果 `--send_real_robot=false`，模型仍会推理动作，但不会发给机器人。

#### 8.3.5 ready pose 与推理前后回位

推理必须从和采集数据一致的初始姿态开始。你的任务采集时小臂是水平上抬，因此推理前也应回到同一个水平上抬姿态。

`eval_g1_remote.py` 默认使用：

```bash
--ready_pose_source=dataset
--repo_id=local/g1_brainco_press_green_button
```

含义是读取 LeRobot 数据集第一帧：

```python
ready_arm_q = dataset[0]["observation.state"][:14]
```

真实执行时流程为：

```text
输入 s
  -> 先插值移动到 ready_arm_q
  -> 开始远程 policy 推理
  -> max_steps 到达后回到 ready_arm_q
```

如果想确认第一帧机械臂关节角，运行：

```bash
python - <<'PY'
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("local/g1_brainco_press_green_button")
state = ds[0]["observation.state"]
if hasattr(state, "cpu"):
    state = state.cpu().numpy()

arm_q = state[:14]
print(np.array2string(arm_q, precision=6, separator=", "))
print("max abs:", float(np.max(np.abs(arm_q))))
print("mean abs:", float(np.mean(np.abs(arm_q))))
PY
```

如果第一帧本来就接近 G1 的 14 维零位，也可以手动使用零位：

```bash
--ready_pose_source=manual \
--ready_arm_q="0,0,0,0,0,0,0,0,0,0,0,0,0,0"
```

如果机器人侧没有 LeRobot 数据集，可以把 14 维 ready pose 写成 JSON：

```json
{
  "ready_arm_q": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}
```

然后运行：

```bash
--ready_pose_source=file \
--ready_arm_q_file=/path/to/ready_arm_q.json
```

默认 Ctrl+C 不会自动回位，因为 Ctrl+C 通常表示紧急中断，不应继续发动作。如果希望 Ctrl+C 后也回位，加：

```bash
--return_to_ready_on_interrupt=true
```

#### 8.3.6 推荐安全流程

第一次真实部署按这个顺序：

```text
1. TeleImager image_server --rs 正常出图
2. A6000 remote_policy_server.py 正常加载 checkpoint
3. G1 eval_g1_remote.py 使用 send_real_robot=false 跑 300 步
4. 检查 Rerun 图像、action 范围和 roundtrip latency
5. send_real_robot=true，但 max_steps=100~150
6. 确认按按钮方向正确后，再增加到 max_steps=300
```

如果动作异常，先停在 dry-run，检查：

```text
1. task 是否和训练时一致
2. 图像方向和 RGB/BGR 是否一致
3. head/left_wrist/right_wrist 是否接反
4. state/action 26D 顺序是否一致
5. ready pose 是否和采集第一帧一致
6. Tailscale 延迟是否过高或抖动过大
```

### 8.4 自定义 bridge 的最小结构

如果不用 `eval_g1.py` 或 `eval_g1_remote.py`，可以按下面结构接入你自己的相机、G1、BrainCo 控制代码：

```python
import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.utils import get_safe_torch_device
from unitree_lerobot.eval_robot.utils.utils import predict_action

REPO_ID = "local/g1_brainco_press_green_button"
POLICY_DIR = "outputs/train/pi0_g1_brainco_press_green_button/checkpoints/last/pretrained_model"
TASK = "press the green button on the electrical cabinet"

dataset = LeRobotDataset(REPO_ID)
cfg = PreTrainedConfig.from_pretrained(POLICY_DIR)
cfg.pretrained_path = POLICY_DIR
cfg.device = "cuda"

policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
policy.eval()

preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=POLICY_DIR,
    dataset_stats=rename_stats(dataset.meta.stats, {}),
    preprocessor_overrides={"device_processor": {"device": cfg.device}},
)

device = get_safe_torch_device(cfg.device)

while True:
    head_bgr = get_head_bgr()
    left_bgr = get_left_wrist_bgr()
    right_bgr = get_right_wrist_bgr()

    head_rgb = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    arm_q = get_g1_dual_arm_q()                  # 14
    left_hand_state = get_left_brainco_state()   # 6
    right_hand_state = get_right_brainco_state() # 6

    state = np.concatenate([arm_q, left_hand_state, right_hand_state]).astype(np.float32)

    observation = {
        "observation.images.cam_left_high": torch.from_numpy(head_rgb),
        "observation.images.cam_left_wrist": torch.from_numpy(left_rgb),
        "observation.images.cam_right_wrist": torch.from_numpy(right_rgb),
        "observation.state": torch.from_numpy(state).float(),
    }

    action = predict_action(
        observation,
        policy,
        device,
        preprocessor,
        postprocessor,
        cfg.use_amp,
        TASK,
        use_dataset=False,
    ).numpy()

    arm_action = action[:14]
    left_hand_action = action[14:20]
    right_hand_action = action[20:26]

    send_g1_dual_arm_target(arm_action)
    send_left_brainco_target(left_hand_action)
    send_right_brainco_target(right_hand_action)
```

第一版建议只执行单步 action；稳定后再引入 action queue。LeRobot 的 `policy.select_action()` 内部已经有 `n_action_steps` 队列逻辑，`predict_action()` 每次调用会返回当前队列中的下一步 action。

---

## 9. 推荐调参顺序

先保证数据闭环正确：

```text
5-10 条 episode -> convert -> dataset dry-run -> pi0 overfit 1000 steps -> dataset inference -> 实机低速测试
```

再扩数据：

```text
50+ 条 episode -> 正式训练 30k steps -> 多 checkpoint 对比 -> 实机测试
```

A6000 初始配置建议：

```text
policy.type=pi0
policy.pretrained_path=lerobot/pi0_base
policy.dtype=bfloat16
policy.gradient_checkpointing=true
policy.chunk_size=10
policy.n_action_steps=10
batch_size=4 或 8
steps=30000
```

如果任务动作很慢、轨迹平滑，可以尝试：

```text
policy.chunk_size=20
policy.n_action_steps=20
```

如果任务要求快速响应，先保持：

```text
policy.chunk_size=10
policy.n_action_steps=10
frequency=30
```

---

## 10. 常见问题

### 10.1 `ModuleNotFoundError: No module named 'lerobot'`

没有设置 `PYTHONPATH`。从仓库根目录运行：

```bash
export PROJECT_ROOT=$PWD
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/unitree_lerobot/lerobot/src:$PYTHONPATH
```

### 10.2 `ModuleNotFoundError: No module named 'unitree_lerobot'`

同样是 `PYTHONPATH` 没包含仓库根目录。确认：

```bash
echo $PYTHONPATH
```

里面应包含：

```text
/path/to/unitree_lerobot
/path/to/unitree_lerobot/unitree_lerobot/lerobot/src
```

### 10.3 action 输出不是 26D

检查训练用 dataset：

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/g1_brainco_press_green_button")
print(ds.meta.features["action"])
PY
```

应为：

```text
shape: [26]
```

如果 dataset action 是 32D，说明转换阶段已经错了；不要在推理端硬切，先修 dataset。

### 10.4 实机动作方向不对

优先检查：

```text
1. 训练和推理 state/action 26 维顺序是否一致
2. left/right wrist 相机是否反了
3. 图像是否 RGB/BGR 反了
4. BrainCo 左右手 target 是否反了
5. 推理频率是否和训练 fps 差太多
```

### 10.5 手指动作幅度异常

检查 BrainCo target 的范围是否和采集时一致。不要在推理端临时归一化或缩放 action；LeRobot postprocessor 已经根据 dataset stats 反归一化。

### 10.6 pi0 下载模型失败

训练机需要能访问 Hugging Face，并可能需要登录：

```bash
huggingface-cli login
```

如果部署机不联网，把训练机上的 `HF_HOME` 缓存和 checkpoint 一起同步过去。

### 10.7 `relative_actions_processor` not found

如果训练时报：

```text
Processor step 'relative_actions_processor' not found in registry
```

说明 `--policy.pretrained_path` 指向的 pi0 base 目录里带了旧版 processor JSON，当前 LeRobot 源码无法加载。训练时应该只从 base 目录加载权重，processor 应按当前 dataset stats 新建。

修改 `unitree_lerobot/lerobot/src/lerobot/scripts/lerobot_train.py`：

```python
processor_pretrained_path = cfg.policy.pretrained_path if cfg.resume else None
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg.policy,
    pretrained_path=processor_pretrained_path,
    **processor_kwargs,
    **postprocessor_kwargs,
)
```

也就是把原来的：

```python
pretrained_path=cfg.policy.pretrained_path,
```

改成：

```python
pretrained_path=processor_pretrained_path,
```

这样非 resume 的 fine-tune 会加载 pi0 base 权重，但不会加载 base model 自带的旧 processor。
