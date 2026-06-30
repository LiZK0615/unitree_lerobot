# Session Modification Log

This document records the code and documentation changes made during the G1 + BrainCo LeRobot/OpenPI debugging session.

## Modified Files

### `unitree_lerobot/utils/convert_unitree_h5_to_lerobot.py`

Purpose: make H5 to LeRobot conversion work reliably from the repository checkout.

Changes:

- Added local repository bootstrap for `sys.path`, so the script can find local `unitree_lerobot` and vendored `lerobot/src` without requiring `pip install -e .`.
- Replaced the old dataset finalization call with the current LeRobot `dataset.finalize()` API.
- Confirmed the G1 + BrainCo flat H5 mapping:
  - `state = actual_joint_positions_rad(14) + left_finger_actual_angles(6) + right_finger_actual_angles(6)`
  - `action = ik_joint_positions_rad(14) + left_finger_target_angles(6) + right_finger_target_angles(6)`
  - `head_rgb -> observation.images.cam_left_high`
  - `left_rgb -> observation.images.cam_left_wrist`
  - `right_rgb -> observation.images.cam_right_wrist`

### `LEROBOT_G1_BRAINCO_TRAIN_INFER_GUIDE.md`

Purpose: provide a LeRobot training and inference guide for G1 + BrainCo.

Changes:

- Added training-machine workflow for converting H5 to LeRobot format.
- Documented the required state/action/image conventions.
- Replaced deprecated `LEROBOT_HOME` usage with `HF_LEROBOT_HOME`.
- Added PI0 full fine-tuning command examples.
- Added notes for OOM handling, processor incompatibility, and output/checkpoint locations.
- Added inference and robot deployment notes for splitting 26D actions into arm, left hand, and right hand.

### `unitree_lerobot/lerobot/src/lerobot/scripts/lerobot_train.py`

Purpose: avoid incompatible processor loading when fine-tuning PI0 from an older/local pretrained checkpoint.

Changes:

- Changed processor loading so fine-tuning from `cfg.policy.pretrained_path` does not blindly load the old checkpoint processors unless `cfg.resume=True`.
- Keeps using dataset statistics from the current LeRobot dataset for normalizer/unnormalizer processors.
- Fixes failures caused by old processor JSON entries such as `relative_actions_processor` not existing in the current LeRobot code.

### `unitree_lerobot/eval_robot/eval_g1_dataset.py`

Purpose: make offline dataset evaluation useful for measuring model quality.

Changes:

- Added GT action, predicted action, and action error computation.
- Added offline MAE metrics:
  - all action dims
  - arm dims `0:14`
  - left hand dims `14:20`
  - right hand dims `20:26`
- Added min/max/mean statistics for GT, prediction, and error per action segment.
- Switched Matplotlib to non-interactive `Agg` backend to avoid Tk GUI crashes on the training/eval machine.
- Replaced the single 26D plot with three smaller saved figures:
  - `figure_arm.png`
  - `figure_left_hand.png`
  - `figure_right_hand.png`
- Added right-hand binary shape evaluation for the two-shape button task:
  - classifies `right_hand[20:26]` as `open` or `press`
  - reports overall shape classification accuracy
  - reports open/press accuracy
  - reports GT and predicted open-to-press switch frames
  - reports switch offset in frames and seconds

### `unitree_lerobot/eval_robot/utils/rerun_visualizer.py`

Purpose: make Rerun visualization display useful evaluation signals.

Changes:

- Added support for logging torch tensors, NumPy arrays, lists, and tuples.
- Added startup and auto-detection status text logs.
- Generalized scalar/vector logging so additional keys can be visualized automatically.
- Added support for visualizing:
  - `ground_truth_action`
  - `predicted_action`
  - `action_error`
  - `observation.state`
  - any detected image observations
- Replaced `rr.Scalar` with `rr.Scalars` for compatibility with the installed Rerun version on the A6000 machine.

## Generated Evaluation Outputs

After running `eval_g1_dataset.py`, the script now saves:

- `figure_arm.png`
- `figure_left_hand.png`
- `figure_right_hand.png`

It also logs Rerun streams for:

- dataset camera images
- observation state
- predicted action
- ground truth action
- action error

## Current Interpretation Notes

Observed offline metrics from the user's run:

- `arm MAE ~= 0.0014`, indicating the arm trajectory is closely fitted on the evaluated episode.
- `left_hand MAE = 0`, expected because the left hand stays fixed during this task.
- right hand MAE is larger because the task uses two right-hand shapes and frame-wise MAE is sensitive to open/press switching time.

For this task, right-hand binary shape accuracy and switch-frame offset are more meaningful than raw right-hand MAE alone.

## 2026-07-01 Remote Inference Scripts

### `unitree_lerobot/eval_robot/remote_policy_server.py`

Purpose: run the trained LeRobot policy on the A6000 workstation and expose inference through HTTP.

Changes:

- Added `GET /health` and `POST /predict` endpoints.
- Loads LeRobot dataset metadata/statistics and the trained policy checkpoint from `--policy.path`.
- Accepts base64 JPEG camera images plus flat robot state, reconstructs LeRobot observations, and returns the postprocessed action.
- Uses a per-policy inference lock so concurrent HTTP requests do not race through a stateful policy.
- Uses only Python standard-library HTTP serving plus existing project dependencies.

### `unitree_lerobot/eval_robot/eval_g1_remote.py`

Purpose: run robot-side data acquisition and action execution while delegating policy inference to the A6000 server.

Changes:

- Added robot-side remote inference loop using existing `make_robot.py` camera and control setup.
- Sends `cam_left_high`, `cam_left_wrist`, `cam_right_wrist`, and 26D G1 + BrainCo state to the remote server.
- Supports JPEG quality, image resize, request timeout, Rerun visualization, dry-run mode, and bounded execution.
- Adds basic deployment safety controls:
  - default `--send_real_robot=false`
  - per-step arm joint delta limit
  - BrainCo hand command clamp
  - optional action smoothing
