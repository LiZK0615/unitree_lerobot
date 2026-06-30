"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import torch
import tqdm
import logging
import time
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pprint import pformat
from typing import Any
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)

from unitree_lerobot.eval_robot.utils.utils import (
    extract_observation,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data


import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def _save_action_comparison_plot(
    ground_truth_actions: np.ndarray,
    predicted_actions: np.ndarray,
    dim_start: int,
    dim_end: int,
    title: str,
    output_path: str,
) -> None:
    if dim_start >= ground_truth_actions.shape[1]:
        return

    dim_end = min(dim_end, ground_truth_actions.shape[1])
    n_dims = dim_end - dim_start
    fig, axes = plt.subplots(n_dims, 1, figsize=(12, max(3, 2.0 * n_dims)), sharex=True, squeeze=False)
    fig.suptitle(title)

    for row, dim_idx in enumerate(range(dim_start, dim_end)):
        ax = axes[row][0]
        ax.plot(ground_truth_actions[:, dim_idx], label="Ground Truth", color="blue")
        ax.plot(predicted_actions[:, dim_idx], label="Predicted", color="red", linestyle="--")
        ax.set_ylabel(f"Dim {dim_idx}")
        ax.legend(loc="upper right")

    axes[-1][0].set_xlabel("Timestep")
    plt.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def _log_action_segment_stats(name: str, ground_truth: np.ndarray, predicted: np.ndarray) -> None:
    error = predicted - ground_truth
    logger_mp.info(
        "%s stats: gt[min=%.6f max=%.6f mean=%.6f] pred[min=%.6f max=%.6f mean=%.6f] err[min=%.6f max=%.6f mean=%.6f]",
        name,
        float(ground_truth.min()),
        float(ground_truth.max()),
        float(ground_truth.mean()),
        float(predicted.min()),
        float(predicted.max()),
        float(predicted.mean()),
        float(error.min()),
        float(error.max()),
        float(error.mean()),
    )


def _classify_by_nearest_shape(actions: np.ndarray, open_shape: np.ndarray, press_shape: np.ndarray) -> np.ndarray:
    open_dist = np.linalg.norm(actions - open_shape[None, :], axis=1)
    press_dist = np.linalg.norm(actions - press_shape[None, :], axis=1)
    return (press_dist < open_dist).astype(np.int64)


def _first_switch_frame(labels: np.ndarray, src: int, dst: int) -> int | None:
    switches = np.flatnonzero((labels[:-1] == src) & (labels[1:] == dst)) + 1
    if len(switches) == 0:
        return None
    return int(switches[0])


def _log_right_hand_binary_shape_metrics(
    ground_truth_actions: np.ndarray,
    predicted_actions: np.ndarray,
    frequency: float,
) -> None:
    gt_right = ground_truth_actions[:, 20:26]
    pred_right = predicted_actions[:, 20:26]
    if gt_right.shape[1] == 0:
        logger_mp.warning("Right hand binary shape metrics skipped: action dim is smaller than 26.")
        return

    # The task uses two right-hand shapes. Use the first GT frame as the open prototype,
    # then pick the farthest GT frame as the press prototype and refine both prototypes once.
    open_shape = gt_right[0]
    press_shape = gt_right[np.argmax(np.linalg.norm(gt_right - open_shape[None, :], axis=1))]
    if np.allclose(open_shape, press_shape):
        logger_mp.warning("Right hand binary shape metrics skipped: only one GT hand shape was detected.")
        return

    gt_labels = _classify_by_nearest_shape(gt_right, open_shape, press_shape)
    if np.any(gt_labels == 0):
        open_shape = np.median(gt_right[gt_labels == 0], axis=0)
    if np.any(gt_labels == 1):
        press_shape = np.median(gt_right[gt_labels == 1], axis=0)

    gt_labels = _classify_by_nearest_shape(gt_right, open_shape, press_shape)
    pred_labels = _classify_by_nearest_shape(pred_right, open_shape, press_shape)

    accuracy = float((pred_labels == gt_labels).mean())
    open_mask = gt_labels == 0
    press_mask = gt_labels == 1
    open_accuracy = float((pred_labels[open_mask] == 0).mean()) if np.any(open_mask) else float("nan")
    press_accuracy = float((pred_labels[press_mask] == 1).mean()) if np.any(press_mask) else float("nan")

    gt_open_to_press = _first_switch_frame(gt_labels, 0, 1)
    pred_open_to_press = _first_switch_frame(pred_labels, 0, 1)
    if gt_open_to_press is not None and pred_open_to_press is not None:
        switch_offset_frames = pred_open_to_press - gt_open_to_press
        switch_offset_s = switch_offset_frames / frequency
        switch_text = (
            f"gt_open_to_press={gt_open_to_press} pred_open_to_press={pred_open_to_press} "
            f"offset={switch_offset_frames} frames ({switch_offset_s:.3f}s)"
        )
    else:
        switch_text = f"gt_open_to_press={gt_open_to_press} pred_open_to_press={pred_open_to_press}"

    logger_mp.info(
        "Right hand binary shape: accuracy=%.4f open_accuracy=%.4f press_accuracy=%.4f "
        "gt_counts[open=%d press=%d] pred_counts[open=%d press=%d] %s",
        accuracy,
        open_accuracy,
        press_accuracy,
        int((gt_labels == 0).sum()),
        int((gt_labels == 1).sum()),
        int((pred_labels == 0).sum()),
        int((pred_labels == 1).sum()),
        switch_text,
    )
    logger_mp.info("Right hand open prototype: %s", np.array2string(open_shape, precision=3))
    logger_mp.info("Right hand press prototype: %s", np.array2string(press_shape, precision=3))


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    # init pose  dataset.meta.episodes["dataset_from_index"][episode_index]
    from_idx = dataset.meta.episodes["dataset_from_index"][0]
    step = dataset[from_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][0]

    ground_truth_actions = []
    predicted_actions = []

    if cfg.send_real_robot:
        from unitree_lerobot.eval_robot.make_robot import setup_robot_interface

        robot_interface = setup_robot_interface(cfg)
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

    # ===============init robot=====================
    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):")
    if user_input.lower() == "s":
        if cfg.send_real_robot:
            # Initialize robot to starting pose
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)

            time.sleep(1)

        for step_idx in tqdm.tqdm(range(from_idx, to_idx)):
            loop_start_time = time.perf_counter()

            step = dataset[step_idx]
            observation = extract_observation(step)

            action = predict_action(
                observation,
                policy,
                get_safe_torch_device(policy.config.device),
                preprocessor,
                postprocessor,
                policy.config.use_amp,
                step["task"],
                use_dataset=True,
                robot_type=None,
            )
            action_np = action.cpu().numpy()
            gt_action_np = step["action"].numpy()
            action_error_np = action_np - gt_action_np

            ground_truth_actions.append(gt_action_np)
            predicted_actions.append(action_np)

            if cfg.send_real_robot:
                # Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)
                # logger_mp.info(f"Arm Action: {arm_action}")

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

            if cfg.visualization:
                visualization_data(
                    step_idx,
                    observation,
                    observation["observation.state"],
                    action_np,
                    rerun_logger,
                    extra={
                        "ground_truth_action": gt_action_np,
                        "predicted_action": action_np,
                        "action_error": action_error_np,
                    },
                )

            # Maintain frequency
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))

        ground_truth_actions = np.array(ground_truth_actions)
        predicted_actions = np.array(predicted_actions)
        action_errors = predicted_actions - ground_truth_actions
        arm_mae = np.abs(action_errors[:, :14]).mean()
        left_hand_mae = np.abs(action_errors[:, 14:20]).mean()
        right_hand_mae = np.abs(action_errors[:, 20:26]).mean()
        all_mae = np.abs(action_errors).mean()
        logger_mp.info(
            "Offline action MAE: all=%.6f arm=%.6f left_hand=%.6f right_hand=%.6f",
            all_mae,
            arm_mae,
            left_hand_mae,
            right_hand_mae,
        )
        _log_action_segment_stats("arm[0:14]", ground_truth_actions[:, :14], predicted_actions[:, :14])
        _log_action_segment_stats("left_hand[14:20]", ground_truth_actions[:, 14:20], predicted_actions[:, 14:20])
        _log_action_segment_stats("right_hand[20:26]", ground_truth_actions[:, 20:26], predicted_actions[:, 20:26])
        _log_right_hand_binary_shape_metrics(ground_truth_actions, predicted_actions, cfg.frequency)

        _save_action_comparison_plot(
            ground_truth_actions, predicted_actions, 0, 14, "Arm Actions", "figure_arm.png"
        )
        _save_action_comparison_plot(
            ground_truth_actions, predicted_actions, 14, 20, "Left Hand Actions", "figure_left_hand.png"
        )
        _save_action_comparison_plot(
            ground_truth_actions, predicted_actions, 20, 26, "Right Hand Actions", "figure_right_hand.png"
        )


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
