# G1 + BrainCo Revo2 跑通 pi0 路线 A 文档

本文档面向当前项目结构：

```text
manipulation/
  code/
    openpi/
    unitree_lerobot-main/unitree_lerobot-main/
    xr_teleoperate/
```

目标是用已经采集好的 `.h5` 数据跑通 pi0：

1. 检查 H5 数据格式
2. 转成 LeRobot dataset
3. 在 openpi 中增加 G1 + BrainCo 的数据映射和训练配置
4. 计算归一化统计量
5. 训练 pi0
6. 启动 policy server
7. 部署到 G1 + BrainCo Revo2

当前采集数据的约定：

```text
state  = actual_joint_positions_rad(14)
       + left_finger_actual_angles(6)
       + right_finger_actual_angles(6)
       = 26D

action = ik_joint_positions_rad(14)
       + left_finger_target_angles(6)
       + right_finger_target_angles(6)
       = 26D

camera = head_rgb  -> cam_left_high
       = left_rgb  -> cam_left_wrist
       = right_rgb -> cam_right_wrist
```

注意：这里的 `cam_left_high` 只是沿用 Unitree/LeRobot 的命名。你的头部 D435i 只有一路 RGB，所以没有 `cam_right_high`。

---

## 0. 最终数据格式要求

每条 episode 的 H5 至少应包含：

```text
actual_joint_positions_rad      (T, 14) float
ik_joint_positions_rad          (T, 14) float
left_finger_actual_angles       (T, 6)  float
right_finger_actual_angles      (T, 6)  float
left_finger_target_angles       (T, 6)  float
right_finger_target_angles      (T, 6)  float
head_rgb                        (T, 480, 640, 3) uint8
left_rgb                        (T, 480, 640, 3) uint8
right_rgb                       (T, 480, 640, 3) uint8
timestamps                      (T,)
```

`left_hand_event_flag` / `right_hand_event_flag` 可以保留，但本路线不使用它们训练 pi0。

采集后先检查：

```bash
python - <<'PY'
import h5py
f = h5py.File("./episode_1.h5", "r")
print(dict(f.attrs))
f.visititems(lambda n, o: print(n, getattr(o, "shape", None), getattr(o, "dtype", None)))
PY
```

再检查 state/action：

```bash
python - <<'PY'
import h5py, numpy as np
f = h5py.File("./episode_1.h5", "r")
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
print("action min/max", action.min(), action.max())
for k in ["head_rgb", "left_rgb", "right_rgb"]:
    x = f[k][:]
    print(k, x.shape, x.dtype, x.min(), x.max())
PY
```

正确结果应接近：

```text
state  (T, 26)
action (T, 26)
head_rgb/right_rgb/left_rgb (T, 480, 640, 3) uint8, min/max in [0,255]
```

---

## 1. 数据采集建议

为了先跑通 pi0，建议：

- 每条 episode 从相近的 ready pose 开始
- 头部相机俯仰角固定
- 三路相机均输出 `[480, 640]`
- 图像方向在训练和部署时保持完全一致
- 不要混用不同图像旋转逻辑采集的数据
- 一开始可以用 5 到 10 条 episode 验证管线，正式训练建议至少几十条，最好上百条

当前 H5 里手臂和手指 action 尺度不同：

```text
arm action:  rad, 大约 -几到几
hand action: BrainCo 手指目标值, 例如 0 到 600/1000
```

这不是格式错误。openpi 会根据训练集计算 state/action normalization，但训练和部署时 26 维 action 的顺序必须完全一致：

```text
0:14   -> 双臂目标关节角 ik_joint_positions_rad
14:20  -> 左手 6 维目标
20:26  -> 右手 6 维目标
```

---

## 2. H5 转 LeRobot dataset

进入 `unitree_lerobot-main`：

```bash
cd ~/airs/manipulation/code/unitree_lerobot-main/unitree_lerobot-main
```

如果当前环境还没有安装本仓库：

```bash
pip install -e .
```

假设 H5 数据目录是：

```text
/media/nvidia/KINGSTON/press_green_button
```

执行转换：

```bash
python unitree_lerobot/utils/convert_unitree_h5_to_lerobot.py \
  --raw-dir /media/nvidia/KINGSTON/press_green_button \
  --repo-id local/g1_brainco_press_green_button \
  --robot-type Unitree_G1_Brainco \
  --task "press the green button on the electrical cabinet" \
  --push-to-hub false \
  --mode video \
  --strict-dims true
```

转换脚本会做如下映射：

```text
head_rgb  -> observation.images.cam_left_high
left_rgb  -> observation.images.cam_left_wrist
right_rgb -> observation.images.cam_right_wrist
state     -> observation.state, 26D
action    -> action, 26D
task      -> LeRobot episode task
```

LeRobot 数据通常会写入：

```text
~/.cache/huggingface/lerobot/local/g1_brainco_press_green_button
```

检查转换结果：

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("local/g1_brainco_press_green_button")
print(ds)
print(ds.meta.features)
print("num frames:", len(ds))
sample = ds[0]
for k, v in sample.items():
    if hasattr(v, "shape"):
        print(k, v.shape, getattr(v, "dtype", None))
    else:
        print(k, v)
PY
```

重点确认：

```text
observation.state shape = 26
action shape = 26
observation.images.cam_left_high exists
observation.images.cam_left_wrist exists
observation.images.cam_right_wrist exists
```

---

## 3. openpi 环境准备

进入 openpi：

```bash
cd ~/airs/manipulation/code/openpi
```

推荐使用 openpi 自带的 `uv` 环境：

```bash
uv sync
```

如果你已经在 `(openpi)` conda 环境中，也要保证 openpi 是 editable 安装：

```bash
pip install -e .
```

测试 openpi 是否能导入：

```bash
python - <<'PY'
from openpi.training import config
print("openpi import ok")
print(config.get_config("pi0_libero").name)
PY
```

---

## 4. 在 openpi 中增加 G1 + BrainCo policy transform

openpi 的 pi0 模型最终需要三个视觉输入槽：

```text
base_0_rgb
left_wrist_0_rgb
right_wrist_0_rgb
```

你的三路相机对应关系建议设为：

```text
cam_left_high   -> base_0_rgb
cam_left_wrist  -> left_wrist_0_rgb
cam_right_wrist -> right_wrist_0_rgb
```

在 openpi 中新增文件：

```text
src/openpi/policies/g1_brainco_policy.py
```

内容：

```python
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class G1BraincoInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        images_in = data["images"]

        base_image = _parse_image(images_in["cam_left_high"])
        left_wrist = _parse_image(images_in["cam_left_wrist"])
        right_wrist = _parse_image(images_in["cam_right_wrist"])

        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class G1BraincoOutputs(transforms.DataTransformFn):
    action_dim: int = 26

    def __call__(self, data: dict) -> dict:
        # Model action_dim is normally 32 for pi0 base. Our robot uses only the first 26 dims.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim], dtype=np.float32)}
```

---

## 5. 在 openpi 中增加训练 config

修改：

```text
src/openpi/training/config.py
```

在 import 区域增加：

```python
import openpi.policies.g1_brainco_policy as g1_brainco_policy
```

在 `_CONFIGS = [` 里面增加一个 config：

```python
TrainConfig(
    name="pi0_g1_brainco_press_button",
    model=pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=10,
        max_token_len=48,
    ),
    data=SimpleDataConfig(
        repo_id="local/g1_brainco_press_green_button",
        data_transforms=lambda model: _transforms.Group(
            inputs=[
                g1_brainco_policy.G1BraincoInputs(model_type=model.model_type),
            ],
            outputs=[
                g1_brainco_policy.G1BraincoOutputs(action_dim=26),
            ],
        ).push(
            # Your actions are absolute targets:
            # first 14 dims are arm joint targets, last 12 dims are BrainCo finger targets.
            # Train pi0 on delta arm actions, but keep finger targets absolute.
            inputs=[_transforms.DeltaActions(_transforms.make_bool_mask(14, -12))],
            outputs=[_transforms.AbsoluteActions(_transforms.make_bool_mask(14, -12))],
        ),
        model_transforms=ModelTransformFactory(
            default_prompt="press the green button on the electrical cabinet"
        ),
        base_config=DataConfig(
            prompt_from_task=True,
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi0_base/params"
    ),
    batch_size=8,
    num_workers=2,
    num_train_steps=10000,
    save_interval=1000,
    keep_period=5000,
)
```

说明：

- `action_dim=32`：保持 pi0 base checkpoint 的 action projection 维度，不建议改成 26。
- `PadStatesAndActions` 会在 model transform 中把 26D state/action 补零到 32D。
- `G1BraincoOutputs(action_dim=26)` 会在推理输出时只取前 26D。
- `action_horizon=10`：先跑通用 10，约等于 30Hz 下 0.33s action chunk。后面可尝试 20 或 50。
- `DeltaActions(make_bool_mask(14, -12))`：只把手臂 14 维转成相对当前 state 的 delta，手指 12 维保持绝对目标值。

如果你发现机械臂动作过小或恢复绝对动作不符合预期，可以先去掉 `DeltaActions/AbsoluteActions`，让 26 维 action 全部按绝对值训练。但第一版建议保留上面的配置。

检查 config 是否能被 openpi 找到：

```bash
cd ~/airs/manipulation/code/openpi
uv run python - <<'PY'
from openpi.training import config
cfg = config.get_config("pi0_g1_brainco_press_button")
print(cfg)
PY
```

---

## 6. 计算 normalization stats

openpi 训练前必须先算 state/action 的归一化统计量：

```bash
cd ~/airs/manipulation/code/openpi

uv run scripts/compute_norm_stats.py \
  --config-name pi0_g1_brainco_press_button
```

如果数据很多，可以限制帧数：

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi0_g1_brainco_press_button \
  --max-frames 50000
```

成功后会写入：

```text
openpi/assets/pi0_g1_brainco_press_button/local/g1_brainco_press_green_button/
```

如果这里报 `num_batches=0` 或 dataloader 为空，通常是：

- episode 太短
- `batch_size` 太大
- `action_horizon` 太长

先把 config 里的：

```python
batch_size=4
action_horizon=10
```

然后重试。

---

## 7. 训练 pi0

先做一个小规模 overfit 测试，目的是验证整条管线能跑：

```bash
cd ~/airs/manipulation/code/openpi

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_g1_brainco_press_button \
  --exp-name=overfit_test \
  --overwrite \
  --num-train-steps=1000 \
  --batch-size=4 \
  --wandb-enabled=false
```

确认没有报错后，再正式训练：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_g1_brainco_press_button \
  --exp-name=press_green_button_v1 \
  --overwrite \
  --wandb-enabled=false
```

默认 checkpoint 路径：

```text
openpi/checkpoints/pi0_g1_brainco_press_button/press_green_button_v1/
```

里面会有按 step 保存的目录，例如：

```text
1000/
2000/
...
10000/
```

如果显存不足，优先改：

```bash
--batch-size=2
```

或在 config 里改成 LoRA：

```python
model=pi0_config.Pi0Config(
    action_dim=32,
    action_horizon=10,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)
freeze_filter=pi0_config.Pi0Config(
    action_dim=32,
    action_horizon=10,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
).get_freeze_filter()
ema_decay=None
```

---

## 8. 启动 openpi policy server

选择一个 checkpoint，例如 `10000`：

```bash
cd ~/airs/manipulation/code/openpi

uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_g1_brainco_press_button \
  --policy.dir=checkpoints/pi0_g1_brainco_press_button/press_green_button_v1/10000 \
  --port=8000
```

服务启动后会监听：

```text
0.0.0.0:8000
```

---

## 9. 不上机器人，先做一次客户端推理测试

新开一个终端：

```bash
cd ~/airs/manipulation/code/openpi
```

运行：

```bash
uv run python - <<'PY'
import numpy as np
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8000)

obs = {
    "images": {
        "cam_left_high": np.zeros((480, 640, 3), dtype=np.uint8),
        "cam_left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        "cam_right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
    },
    "state": np.zeros(26, dtype=np.float32),
    "prompt": "press the green button on the electrical cabinet",
}

out = client.infer(obs)
print(out.keys())
print(out["actions"].shape, out["actions"].dtype)
print(out["actions"][0])
PY
```

期望：

```text
actions shape = (action_horizon, 26)
```

例如：

```text
(10, 26)
```

如果输出是 `(10, 32)`，说明 `G1BraincoOutputs(action_dim=26)` 没有生效，检查 config 里的 output transform。

---

## 10. 部署到 G1 + BrainCo

### 10.1 推荐部署方式

Route A 推荐用 openpi 的 websocket policy server，然后写一个 G1 bridge：

```text
G1 cameras + robot state
        |
        v
build openpi observation dict
        |
        v
websocket_client_policy.infer()
        |
        v
action_chunk (10, 26)
        |
        v
split action -> arm 14D + left hand 6D + right hand 6D
        |
        v
send to G1_29_ArmController + BrainCo controller
```

发送给 openpi 的 observation 必须和 `G1BraincoInputs` 对齐：

```python
obs = {
    "images": {
        "cam_left_high": head_rgb,        # HWC uint8 RGB
        "cam_left_wrist": left_rgb,       # HWC uint8 RGB
        "cam_right_wrist": right_rgb,     # HWC uint8 RGB
    },
    "state": np.concatenate([
        actual_joint_positions_rad,       # 14
        left_finger_actual_angles,        # 6
        right_finger_actual_angles,       # 6
    ]).astype(np.float32),
    "prompt": "press the green button on the electrical cabinet",
}
```

注意图像格式：

- openpi 端建议输入 RGB
- 如果从 `ImageClient` 拿到的是 OpenCV BGR，要先 `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
- 训练时是什么方向，部署时必须是什么方向

### 10.2 action 解析

policy 返回：

```python
action_chunk = out["actions"]   # shape: (10, 26)
```

最简单的跑通方式是每次只执行第 0 个 action：

```python
action = action_chunk[0]

arm_action = action[:14]
left_hand_action = action[14:20]
right_hand_action = action[20:26]
```

执行：

```python
tau = arm_ik.solve_tau(arm_action)
arm_ctrl.ctrl_dual_arm(arm_action, tau)

brainco_left_target[:] = left_hand_action
brainco_right_target[:] = right_hand_action
```

更平滑的方式是维护一个 action queue：

```text
每次 policy infer 得到 10 步 action
然后按 30Hz 逐步执行
queue 快空时再请求下一次 infer
```

第一版建议先用 `action_chunk[0]` 跑通闭环，再加 action queue。

### 10.3 不建议直接用 eval_g1.py 跑 openpi

`unitree_lerobot/eval_robot/eval_g1.py` 当前是 LeRobot PyTorch policy 的评估入口：

```python
from lerobot.policies.factory import make_policy
```

它不是 openpi checkpoint 的 websocket 客户端。要用 pi0，建议先走 openpi：

```text
openpi/scripts/serve_policy.py
        +
自定义 G1 bridge client
```

后续可以把 G1 bridge 整合进 `unitree_lerobot/eval_robot`，但不要一开始混在一起，否则问题来源会很难定位。

---

## 11. 最小 G1 bridge 伪代码

下面是结构示意，不建议直接复制运行，需要按你现有 `teleop_hand_and_arm0626.py` 的 ImageClient、G1_29_ArmController、BrainCo 控制器接口补齐。

```python
import cv2
import numpy as np
from openpi_client import websocket_client_policy

client = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8000)

while True:
    # 1. Read images from image server.
    head_bgr = get_head_bgr()
    left_bgr = get_left_wrist_bgr()
    right_bgr = get_right_wrist_bgr()

    head_rgb = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2RGB)
    left_rgb = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right_rgb = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)

    # 2. Read robot state.
    arm_q = arm_ctrl.get_current_dual_arm_q()          # 14
    left_hand_state = get_left_brainco_state()         # 6
    right_hand_state = get_right_brainco_state()       # 6

    state = np.concatenate([arm_q, left_hand_state, right_hand_state]).astype(np.float32)

    # 3. Query policy.
    obs = {
        "images": {
            "cam_left_high": head_rgb,
            "cam_left_wrist": left_rgb,
            "cam_right_wrist": right_rgb,
        },
        "state": state,
        "prompt": "press the green button on the electrical cabinet",
    }
    action_chunk = client.infer(obs)["actions"]
    action = action_chunk[0]

    # 4. Execute first action.
    arm_action = action[:14]
    left_hand_action = action[14:20]
    right_hand_action = action[20:26]

    tau = arm_ik.solve_tau(arm_action)
    arm_ctrl.ctrl_dual_arm(arm_action, tau)
    send_brainco_targets(left_hand_action, right_hand_action)
```

---

## 12. 调试顺序

不要直接上机器人测试完整闭环。建议按这个顺序：

### 阶段 A：数据检查

```text
H5 shape 正确
state/action 26D 正确
图像 RGB 正确
episode 起止帧正确
```

### 阶段 B：转换检查

```text
LeRobot dataset 能加载
sample 中有三路 observation.images
state/action 是 26D
task 字段存在
```

### 阶段 C：openpi dataloader 检查

```bash
uv run scripts/compute_norm_stats.py --config-name pi0_g1_brainco_press_button
```

能跑通说明 repack/data transforms 基本正确。

### 阶段 D：overfit

用少量 episode 训练 1000 step，看 loss 是否下降。

### 阶段 E：policy server

用全黑图 + 零 state 先测试能否返回 `(10, 26)`。

### 阶段 F：真实观测但不控制机器人

读取真实相机和真实 state，调用 policy，只打印 action：

```text
arm_action min/max
left_hand_action min/max
right_hand_action min/max
```

确认数值范围合理后再控制机器人。

### 阶段 G：低速闭环

先不要让机器人接近电柜按钮：

- 限制手臂 action 变化速度
- 限制手指目标范围
- 保留急停
- 人站在安全位置

---

## 13. 常见问题

### 13.1 H5 的 episode 编号不连续有影响吗？

没有影响。转换脚本会按文件名排序读取 `episode_*.h5`，编号跳过不影响训练。

### 13.2 三个相机必须同 shape 吗？

pi0 最终会 resize 到 224x224。理论上原始 shape 可以不同，但你的路线中建议统一：

```text
480 x 640 x 3
```

这样转换、可视化、部署更稳。

### 13.3 left/right wrist 图像方向不水平怎么办？

方向不一定要“人眼水平”，但必须训练和部署一致。不要训练时旋转，部署时不旋转，或者相反。

### 13.4 为什么 action_dim 用 32，不用 26？

pi0 base checkpoint 的 action projection 默认是 32D。为了加载 `pi0_base` 权重，模型 action_dim 保持 32D，然后把你的 26D state/action 补零到 32D，输出时再切回 26D。

### 13.5 为什么只对前 14 维做 delta？

前 14 维是机械臂目标关节角，适合转成相对当前 state 的 delta。后 12 维是 BrainCo 手指绝对目标值，保持绝对值更直接。

### 13.6 训练数据少能不能跑？

可以跑通流程，但不代表能成功按按钮。建议：

- 跑通管线：5 到 10 条 episode
- 初步过拟合测试：10 到 20 条 episode
- 有实际泛化：至少几十到上百条，且覆盖按钮位置、光照、起始姿态的小变化

---

## 14. 当前路线的最小命令汇总

```bash
# 1. Convert H5 to LeRobot
cd ~/airs/manipulation/code/unitree_lerobot-main/unitree_lerobot-main
python unitree_lerobot/utils/convert_unitree_h5_to_lerobot.py \
  --raw-dir /media/nvidia/KINGSTON/press_green_button \
  --repo-id local/g1_brainco_press_green_button \
  --robot-type Unitree_G1_Brainco \
  --task "press the green button on the electrical cabinet" \
  --push-to-hub false \
  --mode video \
  --strict-dims true

# 2. Compute norm stats
cd ~/airs/manipulation/code/openpi
uv run scripts/compute_norm_stats.py \
  --config-name pi0_g1_brainco_press_button

# 3. Overfit test
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_g1_brainco_press_button \
  --exp-name=overfit_test \
  --overwrite \
  --num-train-steps=1000 \
  --batch-size=4 \
  --wandb-enabled=false

# 4. Full training
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi0_g1_brainco_press_button \
  --exp-name=press_green_button_v1 \
  --overwrite \
  --wandb-enabled=false

# 5. Serve policy
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi0_g1_brainco_press_button \
  --policy.dir=checkpoints/pi0_g1_brainco_press_button/press_green_button_v1/10000 \
  --port=8000
```

---

## 15. 完成标准

认为 pi0 路线 A 跑通，需要满足：

```text
1. H5 -> LeRobot 转换成功
2. LeRobot dataset 可加载，三路图像/state/action 正确
3. openpi config 可加载
4. compute_norm_stats 成功
5. train.py 能保存 checkpoint
6. serve_policy.py 能启动
7. websocket client 能拿到 (action_horizon, 26) 的 actions
8. G1 bridge 能正确拆分并发送 arm/hand action
```

前 7 步完成后，说明软件管线跑通。第 8 步才是真机闭环部署。
