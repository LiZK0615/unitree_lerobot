# 会话修改记录

本文档记录本次 G1 + BrainCo LeRobot/OpenPI 调试与部署会话中修改过的代码和文档。

## 已修改文件

### `unitree_lerobot/utils/convert_unitree_h5_to_lerobot.py`

用途：让 H5 到 LeRobot 格式的转换脚本可以在源码仓库中稳定运行。

修改内容：

- 增加本地仓库 `sys.path` 初始化逻辑，使脚本在没有执行 `pip install -e .` 的情况下，也能找到本地 `unitree_lerobot` 和仓库内置的 `lerobot/src`。
- 将旧的数据集结束写入调用替换为当前 LeRobot 的 `dataset.finalize()` API。
- 确认 G1 + BrainCo 的扁平 H5 映射关系：
  - `state = actual_joint_positions_rad(14) + left_finger_actual_angles(6) + right_finger_actual_angles(6)`
  - `action = ik_joint_positions_rad(14) + left_finger_target_angles(6) + right_finger_target_angles(6)`
  - `head_rgb -> observation.images.cam_left_high`
  - `left_rgb -> observation.images.cam_left_wrist`
  - `right_rgb -> observation.images.cam_right_wrist`

### `LEROBOT_G1_BRAINCO_TRAIN_INFER_GUIDE.md`

用途：提供面向 G1 + BrainCo 的 LeRobot 训练与推理教程。

修改内容：

- 增加训练机器上将 H5 转换为 LeRobot 数据集格式的流程。
- 记录所需的 state/action/image 字段约定。
- 将已废弃的 `LEROBOT_HOME` 用法替换为 `HF_LEROBOT_HOME`。
- 增加 PI0 全量微调命令示例。
- 增加 OOM 处理、processor 不兼容、输出目录和 checkpoint 位置说明。
- 增加推理和机器人部署说明，包括如何将 26 维动作拆分为机械臂、左手和右手。

### `unitree_lerobot/lerobot/src/lerobot/scripts/lerobot_train.py`

用途：避免从旧版或本地 PI0 预训练 checkpoint 微调时加载不兼容的 processor。

修改内容：

- 修改 processor 加载逻辑：从 `cfg.policy.pretrained_path` 微调时，除非 `cfg.resume=True`，否则不盲目加载旧 checkpoint 中的 processor。
- normalizer/unnormalizer processor 继续使用当前 LeRobot 数据集统计量。
- 修复旧 processor JSON 中包含 `relative_actions_processor` 等当前 LeRobot 代码不存在的条目时导致的失败。

### `unitree_lerobot/eval_robot/eval_g1_dataset.py`

用途：让离线数据集评估能够更有效地衡量模型训练效果。

修改内容：

- 增加 GT action、predicted action 和 action error 计算。
- 增加离线 MAE 指标：
  - 全部动作维度
  - 机械臂维度 `0:14`
  - 左手维度 `14:20`
  - 右手维度 `20:26`
- 增加每个动作分段的 GT、预测值和误差的 min/max/mean 统计。
- 将 Matplotlib 切换到非交互式 `Agg` 后端，避免训练/评估机器上的 Tk GUI 崩溃。
- 将单张 26 维动作图替换为三张更小的保存图：
  - `figure_arm.png`
  - `figure_left_hand.png`
  - `figure_right_hand.png`
- 为两种手型的按钮任务增加右手二值形态评估：
  - 将 `right_hand[20:26]` 分类为 `open` 或 `press`
  - 输出整体手型分类准确率
  - 输出 open/press 各自准确率
  - 输出 GT 和预测的 open-to-press 切换帧
  - 输出切换偏差，单位包括帧数和秒数

### `unitree_lerobot/eval_robot/utils/rerun_visualizer.py`

用途：让 Rerun 可视化能够显示更有用的评估信号。

修改内容：

- 增加对 torch tensor、NumPy array、list 和 tuple 的日志记录支持。
- 增加启动状态和自动检测状态文本日志。
- 泛化 scalar/vector 日志逻辑，使新增 key 可以自动可视化。
- 增加以下内容的可视化支持：
  - `ground_truth_action`
  - `predicted_action`
  - `action_error`
  - `observation.state`
  - 自动检测到的图像 observation
- 将 `rr.Scalar` 替换为 `rr.Scalars`，以兼容 A6000 机器上安装的 Rerun 版本。

## 生成的评估输出

运行 `eval_g1_dataset.py` 后，脚本现在会保存：

- `figure_arm.png`
- `figure_left_hand.png`
- `figure_right_hand.png`

同时会记录以下 Rerun 数据流：

- 数据集相机图像
- observation state
- predicted action
- ground truth action
- action error

## 当前结果解读记录

根据用户运行得到的离线评估指标：

- `arm MAE ~= 0.0014`，说明在被评估 episode 上机械臂轨迹拟合较好。
- `left_hand MAE = 0`，这是预期现象，因为该任务中左手保持不动。
- 右手 MAE 较大，主要因为该任务只使用两种右手形态，逐帧 MAE 对 open/press 切换时刻非常敏感。

对于这个任务，右手二值手型准确率和切换帧偏差比单独看原始右手 MAE 更有意义。

## 2026-07-01 远程推理脚本

### `unitree_lerobot/eval_robot/remote_policy_server.py`

用途：在 A6000 工作站上运行训练好的 LeRobot policy，并通过 HTTP 提供远程推理服务。

修改内容：

- 增加 `GET /health` 和 `POST /predict` 接口。
- 加载 LeRobot 数据集 metadata/statistics，并从 `--policy.path` 加载训练好的 policy checkpoint。
- 接收 base64 JPEG 相机图像和扁平机器人状态，重建 LeRobot observation，并返回后处理后的动作。
- 使用单 policy 推理锁，避免多个 HTTP 请求并发访问有状态 policy。
- 仅使用 Python 标准库 HTTP 服务和工程已有依赖，不额外引入 FastAPI/uvicorn。

### `unitree_lerobot/eval_robot/eval_g1_remote.py`

用途：在机器人侧进行数据采集和动作执行，同时把 policy 推理委托给 A6000 服务器。

修改内容：

- 增加机器人侧远程推理循环，复用现有 `make_robot.py` 中的相机和控制初始化逻辑。
- 将 `cam_left_high`、`cam_left_wrist`、`cam_right_wrist` 和 26 维 G1 + BrainCo 状态发送到远程服务器。
- 支持 JPEG 质量设置、图像 resize、请求超时、Rerun 可视化、dry-run 模式和限定步数运行。
- 增加基础部署安全控制：
  - 默认 `--send_real_robot=false`
  - 单步机械臂关节 delta 限幅
  - BrainCo 手部命令 clamp
  - 可选动作平滑

## 2026-07-01 远程推理 ready pose 回位

### `unitree_lerobot/eval_robot/eval_g1_remote.py`

用途：让真实推理前后都回到采集时的水平上抬初始姿态，避免从错误手臂构型开始或推理结束后停在按钮附近。

修改内容：

- 增加 `--repo_id` 参数，默认从 `local/g1_brainco_press_green_button` 的第一帧 `observation.state[:14]` 读取机械臂 ready pose。
- 增加 `--ready_pose_source` 参数，支持：
  - `dataset`：从 LeRobot 数据集第一帧读取 ready pose
  - `manual`：通过 `--ready_arm_q` 传入 14 维关节角
  - `file`：通过 `--ready_arm_q_file` 读取 14 维关节角
  - `current`：以脚本启动时当前机械臂姿态作为 ready pose
  - `none`：禁用 ready pose 回位
- 增加推理前回位：用户输入 `s` 后，先插值移动到 ready pose，再开始远程 policy 推理。
- 增加推理后回位：`--max_steps` 到达或正常退出时，插值回到同一个 ready pose。
- 增加 `--ready_move_duration` 和 `--ready_tolerance`，控制回位时长和到位误差告警。
- 默认 Ctrl+C 不自动回位，避免紧急中断时脚本继续发动作；如需 Ctrl+C 后也回位，可设置 `--return_to_ready_on_interrupt=true`。

## 2026-07-01 远程推理教程补充

### `LEROBOT_G1_BRAINCO_TRAIN_INFER_GUIDE.md`

用途：把 A6000 远程推理的完整启动流程记录到 LeRobot 训练与推理教程中。

修改内容：

- 在第 8 章新增 `8.3 A6000 远程推理部署`。
- 记录 G1/机器人侧和 A6000 工作站的职责划分。
- 增加 TeleImager 图像服务启动方式，说明 `cam_config_server.yaml` 中三路 RealSense 对应的 ZMQ 端口。
- 增加 A6000 `remote_policy_server.py` 启动命令和 `/health` 检查命令。
- 增加 G1 侧 `eval_g1_remote.py` dry-run 和真实执行命令。
- 说明 `--task` 如何作为语言指令传给模型，以及 `--send_real_robot` 和输入 `s` 才是真实执行开关。
- 增加 ready pose 与推理前后回位说明，包括从数据集第一帧查看 14 维机械臂初始关节角的方法。
- 增加 Tailscale 远程互联和首次实机部署安全流程。
