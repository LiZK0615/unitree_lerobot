#!/usr/bin/env python3
"""
Robot-side remote inference client for Unitree G1 + BrainCo.

This script keeps camera acquisition and robot control on the G1 side, while
policy inference runs on a remote GPU server started by remote_policy_server.py.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
import urllib.error
import urllib.request
from multiprocessing.sharedctypes import SynchronizedArray
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging
from unitree_lerobot.eval_robot.make_robot import (
    process_images_and_observations,
    setup_image_client,
    setup_robot_interface,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data
from unitree_lerobot.eval_robot.utils.utils import to_list, to_scalar


LOGGER = logging.getLogger(__name__)

MODEL_IMAGE_KEYS = (
    "observation.images.cam_left_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robot-side client for remote LeRobot PI0 inference on Unitree G1 + BrainCo."
    )
    parser.add_argument("--server_host", required=True, help="Remote policy server host or Tailscale IP.")
    parser.add_argument("--server_port", type=int, default=8088, help="Remote policy server port.")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP request timeout in seconds.")
    parser.add_argument("--task", default="press the green button on the electrical cabinet")
    parser.add_argument("--robot_type", default="Unitree_G1_Brainco")
    parser.add_argument(
        "--repo_id",
        default="local/g1_brainco_press_green_button",
        help="LeRobot dataset repo id used to read the first-frame ready pose.",
    )

    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--arm", default="G1_29", choices=["G1_29", "G1_23"])
    parser.add_argument("--ee", default="brainco")
    parser.add_argument("--motion", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--sim", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--base_type", default="legs")
    parser.add_argument("--image_host", default="192.168.123.164", help="Robot image server host.")

    parser.add_argument("--visualization", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--send_real_robot", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run until Ctrl+C.")
    parser.add_argument(
        "--ready_pose_source",
        choices=["dataset", "manual", "file", "current", "none"],
        default="dataset",
        help="Where to load the arm ready pose from before/after inference.",
    )
    parser.add_argument(
        "--ready_arm_q",
        default="",
        help="Comma-separated 14D arm ready pose. Used when --ready_pose_source=manual.",
    )
    parser.add_argument(
        "--ready_arm_q_file",
        default="",
        help="JSON or plain-text file containing a 14D arm ready pose. Used when --ready_pose_source=file.",
    )
    parser.add_argument("--move_to_ready_on_start", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--return_to_ready_on_exit", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument(
        "--return_to_ready_on_interrupt",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="If false, Ctrl+C stops without commanding a return motion.",
    )
    parser.add_argument(
        "--ready_move_duration",
        type=float,
        default=4.0,
        help="Seconds used for interpolated motion to the ready pose.",
    )
    parser.add_argument(
        "--ready_tolerance",
        type=float,
        default=0.08,
        help="Warn if final arm joint error to ready pose is above this value.",
    )

    parser.add_argument("--jpeg_quality", type=int, default=80)
    parser.add_argument(
        "--resize",
        type=int,
        default=224,
        help="Resize images to square resolution before sending. Use 0 to send original size.",
    )
    parser.add_argument(
        "--arm_max_delta",
        type=float,
        default=0.05,
        help="Per-step max arm joint delta in radians. Use <=0 to disable.",
    )
    parser.add_argument("--hand_min", type=float, default=0.0)
    parser.add_argument("--hand_max", type=float, default=1000.0)
    parser.add_argument(
        "--action_smoothing_alpha",
        type=float,
        default=1.0,
        help="1.0 disables smoothing; smaller values blend with previous executed action.",
    )
    return parser.parse_args()


def _parse_ready_pose_text(text: str) -> np.ndarray:
    values = [float(item.strip()) for item in text.replace("\n", ",").split(",") if item.strip()]
    return np.asarray(values, dtype=np.float32)


def _load_ready_pose_file(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            for key in ("ready_arm_q", "arm_q", "q"):
                if key in payload:
                    return np.asarray(payload[key], dtype=np.float32).reshape(-1)
            raise ValueError(f"JSON ready pose file must contain one of: ready_arm_q, arm_q, q")
        return np.asarray(payload, dtype=np.float32).reshape(-1)
    except json.JSONDecodeError:
        return _parse_ready_pose_text(text)


def _load_dataset_ready_pose(repo_id: str, arm_dof: int) -> np.ndarray:
    dataset = LeRobotDataset(repo_id=repo_id)
    from_idx = int(dataset.meta.episodes["dataset_from_index"][0])
    step = dataset[from_idx]
    ready_q = step["observation.state"][:arm_dof]
    if isinstance(ready_q, torch.Tensor):
        ready_q = ready_q.detach().cpu().numpy()
    return np.asarray(ready_q, dtype=np.float32).reshape(-1)


def resolve_ready_pose(args: argparse.Namespace, robot_interface: dict[str, Any], arm_dof: int) -> np.ndarray | None:
    source = args.ready_pose_source
    if source == "none":
        return None
    if source == "current":
        return np.asarray(robot_interface["arm_ctrl"].get_current_dual_arm_q(), dtype=np.float32).reshape(-1)
    if source == "manual":
        ready_q = _parse_ready_pose_text(args.ready_arm_q)
    elif source == "file":
        ready_q = _load_ready_pose_file(args.ready_arm_q_file)
    elif source == "dataset":
        ready_q = _load_dataset_ready_pose(args.repo_id, arm_dof)
    else:
        raise ValueError(f"Unsupported ready_pose_source: {source}")

    if ready_q.shape[0] != arm_dof:
        raise ValueError(f"Ready pose dim is {ready_q.shape[0]}, expected {arm_dof}")
    if not np.all(np.isfinite(ready_q)):
        raise ValueError("Ready pose contains NaN or Inf")
    return ready_q.astype(np.float32)


def move_arm_to_ready_pose(
    robot_interface: dict[str, Any],
    ready_arm_q: np.ndarray,
    duration: float,
    frequency: float,
    tolerance: float,
    send_real_robot: bool,
    label: str,
) -> None:
    if ready_arm_q is None:
        return
    if not send_real_robot:
        LOGGER.info("%s skipped because --send_real_robot=false. ready_arm_q=%s", label, ready_arm_q.tolist())
        return

    arm_ctrl = robot_interface["arm_ctrl"]
    arm_ik = robot_interface["arm_ik"]
    current_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float32).reshape(-1)
    if current_q.shape != ready_arm_q.shape:
        raise ValueError(f"Current arm q shape {current_q.shape} does not match ready pose {ready_arm_q.shape}")

    steps = max(1, int(max(duration, 0.1) * max(frequency, 1.0)))
    LOGGER.info("%s: moving arm to ready pose over %.2fs (%s steps)", label, duration, steps)
    for alpha in np.linspace(0.0, 1.0, steps):
        q_target = (1.0 - alpha) * current_q + alpha * ready_arm_q
        tau = arm_ik.solve_tau(q_target)
        arm_ctrl.ctrl_dual_arm(q_target, tau)
        time.sleep(max(0.0, duration / steps))

    tau = arm_ik.solve_tau(ready_arm_q)
    arm_ctrl.ctrl_dual_arm(ready_arm_q, tau)
    time.sleep(0.3)
    final_q = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=np.float32).reshape(-1)
    max_err = float(np.max(np.abs(final_q - ready_arm_q)))
    if max_err > tolerance:
        LOGGER.warning("%s: ready pose max joint error %.4f > tolerance %.4f", label, max_err, tolerance)
    else:
        LOGGER.info("%s: ready pose reached, max joint error %.4f", label, max_err)


def _as_numpy_hwc_rgb(image: Any) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = np.asarray(image)

    if image_np.ndim != 3:
        raise ValueError(f"Expected HWC or CHW image, got shape {image_np.shape}")
    if image_np.shape[0] in (1, 3, 4) and image_np.shape[-1] not in (1, 3, 4):
        image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.shape[-1] == 1:
        image_np = np.repeat(image_np, 3, axis=-1)
    if image_np.shape[-1] == 4:
        image_np = image_np[..., :3]
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image_np)


def encode_image_jpeg(image: Any, resize: int, quality: int) -> dict[str, Any]:
    rgb = _as_numpy_hwc_rgb(image)
    if resize > 0:
        rgb = cv2.resize(rgb, (resize, resize), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return {
        "encoding": "jpg",
        "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
        "shape": list(rgb.shape),
    }


class RemotePolicyClient:
    def __init__(self, host: str, port: int, timeout: float, resize: int, jpeg_quality: int) -> None:
        self.url = f"http://{host}:{port}/predict"
        self.timeout = timeout
        self.resize = resize
        self.jpeg_quality = jpeg_quality

    def predict(self, observation: dict[str, Any], task: str, robot_type: str) -> tuple[np.ndarray, dict[str, Any]]:
        if "observation.state" not in observation:
            raise ValueError("observation.state is required")

        state = observation["observation.state"]
        if isinstance(state, torch.Tensor):
            state_list = state.detach().cpu().numpy().astype(np.float32).reshape(-1).tolist()
        else:
            state_list = np.asarray(state, dtype=np.float32).reshape(-1).tolist()

        images = {}
        for key in MODEL_IMAGE_KEYS:
            value = observation.get(key)
            if value is not None:
                images[key] = encode_image_jpeg(value, resize=self.resize, quality=self.jpeg_quality)

        payload = {
            "task": task,
            "robot_type": robot_type,
            "state": state_list,
            "images": images,
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Remote policy HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Remote policy request failed: {exc}") from exc

        decoded = json.loads(raw.decode("utf-8"))
        if "action" not in decoded:
            raise RuntimeError(f"Remote response missing action: {decoded}")
        decoded["roundtrip_latency_ms"] = (time.perf_counter() - started) * 1000.0
        return np.asarray(decoded["action"], dtype=np.float32).reshape(-1), decoded


def _get_ee_state(ee_shared_mem: dict[str, Any], ee_dof: int) -> tuple[np.ndarray, np.ndarray]:
    if ee_dof <= 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    with ee_shared_mem["lock"]:
        full_state = np.asarray(ee_shared_mem["state"][:], dtype=np.float32)
    return full_state[:ee_dof], full_state[ee_dof : ee_dof * 2]


def _limit_action(
    raw_action: np.ndarray,
    current_arm_q: np.ndarray,
    arm_dof: int,
    ee_dof: int,
    arm_max_delta: float,
    hand_min: float,
    hand_max: float,
) -> np.ndarray:
    expected_dim = arm_dof + 2 * ee_dof
    if raw_action.shape[0] < expected_dim:
        raise ValueError(f"Action dim {raw_action.shape[0]} is smaller than expected {expected_dim}")
    if not np.all(np.isfinite(raw_action[:expected_dim])):
        raise ValueError("Action contains NaN or Inf")

    action = raw_action[:expected_dim].astype(np.float32).copy()
    arm_action = action[:arm_dof]
    if arm_max_delta > 0:
        current_arm_q = np.asarray(current_arm_q, dtype=np.float32).reshape(-1)
        action[:arm_dof] = current_arm_q + np.clip(arm_action - current_arm_q, -arm_max_delta, arm_max_delta)

    if ee_dof > 0:
        action[arm_dof : arm_dof + 2 * ee_dof] = np.clip(
            action[arm_dof : arm_dof + 2 * ee_dof],
            hand_min,
            hand_max,
        )
    return action


def _smooth_action(action: np.ndarray, previous: np.ndarray | None, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if previous is None or alpha >= 1.0:
        return action
    return alpha * action + (1.0 - alpha) * previous


def _send_action(
    action: np.ndarray,
    robot_interface: dict[str, Any],
    arm_dof: int,
    ee_dof: int,
) -> None:
    arm_action = action[:arm_dof]
    tau = robot_interface["arm_ik"].solve_tau(arm_action)
    robot_interface["arm_ctrl"].ctrl_dual_arm(arm_action, tau)

    if ee_dof <= 0 or not robot_interface.get("ee_shared_mem"):
        return

    ee_shared_mem = robot_interface["ee_shared_mem"]
    left_ee_action = action[arm_dof : arm_dof + ee_dof]
    right_ee_action = action[arm_dof + ee_dof : arm_dof + 2 * ee_dof]

    if isinstance(ee_shared_mem["left"], SynchronizedArray):
        ee_shared_mem["left"][:] = to_list(left_ee_action)
        ee_shared_mem["right"][:] = to_list(right_ee_action)
    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
        ee_shared_mem["left"].value = to_scalar(left_ee_action)
        ee_shared_mem["right"].value = to_scalar(right_ee_action)


def main() -> None:
    args = parse_args()
    init_logging()

    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg_quality must be in [1, 100]")
    if args.frequency <= 0:
        raise ValueError("--frequency must be positive")

    client = RemotePolicyClient(
        host=args.server_host,
        port=args.server_port,
        timeout=args.timeout,
        resize=args.resize,
        jpeg_quality=args.jpeg_quality,
    )

    LOGGER.info("Connecting image client to %s", args.image_host)
    img_client, camera_config = setup_image_client(args)
    LOGGER.info("Initializing robot interface arm=%s ee=%s sim=%s", args.arm, args.ee, args.sim)
    robot_interface = setup_robot_interface(args)

    arm_ctrl = robot_interface["arm_ctrl"]
    arm_dof = int(robot_interface["arm_dof"])
    ee_dof = int(robot_interface["ee_dof"])
    ee_shared_mem = robot_interface["ee_shared_mem"]
    ready_arm_q = resolve_ready_pose(args, robot_interface, arm_dof)
    if ready_arm_q is not None:
        LOGGER.info(
            "Resolved ready pose from %s: min=%.4f max=%.4f mean=%.4f",
            args.ready_pose_source,
            float(ready_arm_q.min()),
            float(ready_arm_q.max()),
            float(ready_arm_q.mean()),
        )

    rerun_logger = RerunLogger() if args.visualization else None
    last_action: np.ndarray | None = None

    user_input = input("Please enter the start signal (enter 's' to start the remote policy loop):")
    if user_input.strip().lower() != "s":
        LOGGER.info("Start cancelled.")
        return

    if args.move_to_ready_on_start and ready_arm_q is not None:
        move_arm_to_ready_pose(
            robot_interface,
            ready_arm_q,
            duration=args.ready_move_duration,
            frequency=args.frequency,
            tolerance=args.ready_tolerance,
            send_real_robot=args.send_real_robot,
            label="Before inference",
        )

    LOGGER.info(
        "Starting remote eval loop: server=%s:%s frequency=%.1fHz send_real_robot=%s",
        args.server_host,
        args.server_port,
        args.frequency,
        args.send_real_robot,
    )

    idx = 0
    interrupted = False
    try:
        while args.max_steps <= 0 or idx < args.max_steps:
            loop_start = time.perf_counter()

            observation, current_arm_q = process_images_and_observations(img_client, camera_config, arm_ctrl)
            if current_arm_q is None:
                LOGGER.warning("Skipping frame %s because arm state is unavailable", idx)
                time.sleep(max(0.0, (1.0 / args.frequency) - (time.perf_counter() - loop_start)))
                continue

            missing_images = [key for key in MODEL_IMAGE_KEYS if observation.get(key) is None]
            if missing_images:
                LOGGER.warning("Skipping frame %s because images are missing: %s", idx, missing_images)
                time.sleep(max(0.0, (1.0 / args.frequency) - (time.perf_counter() - loop_start)))
                continue

            left_ee_state, right_ee_state = _get_ee_state(ee_shared_mem, ee_dof)
            state = np.concatenate(
                (
                    np.asarray(current_arm_q, dtype=np.float32).reshape(-1),
                    left_ee_state.reshape(-1),
                    right_ee_state.reshape(-1),
                ),
                axis=0,
            )
            observation["observation.state"] = torch.from_numpy(state).float()

            raw_action, response_meta = client.predict(observation, task=args.task, robot_type=args.robot_type)
            action = _limit_action(
                raw_action,
                current_arm_q=np.asarray(current_arm_q, dtype=np.float32),
                arm_dof=arm_dof,
                ee_dof=ee_dof,
                arm_max_delta=args.arm_max_delta,
                hand_min=args.hand_min,
                hand_max=args.hand_max,
            )
            action = _smooth_action(action, last_action, args.action_smoothing_alpha)
            last_action = action.copy()

            if args.send_real_robot:
                _send_action(action, robot_interface, arm_dof=arm_dof, ee_dof=ee_dof)

            if rerun_logger is not None:
                visualization_data(
                    idx,
                    observation,
                    state,
                    action,
                    rerun_logger,
                    extra={
                        "raw_action": raw_action[: action.shape[0]],
                        "remote_server_latency_ms": np.asarray([response_meta.get("server_latency_ms", np.nan)]),
                        "remote_roundtrip_latency_ms": np.asarray([response_meta.get("roundtrip_latency_ms", np.nan)]),
                    },
                )

            if idx % 10 == 0:
                right_start = arm_dof + ee_dof
                right_end = arm_dof + 2 * ee_dof
                right_action = action[right_start:right_end] if ee_dof else np.asarray([np.nan])
                LOGGER.info(
                    "step=%s server_ms=%.1f roundtrip_ms=%.1f arm[min=%.3f max=%.3f] "
                    "right_hand[min=%.1f max=%.1f]",
                    idx,
                    response_meta.get("server_latency_ms", -1.0),
                    response_meta.get("roundtrip_latency_ms", -1.0),
                    float(action[:arm_dof].min()),
                    float(action[:arm_dof].max()),
                    float(right_action.min()),
                    float(right_action.max()),
                )

            idx += 1
            time.sleep(max(0.0, (1.0 / args.frequency) - (time.perf_counter() - loop_start)))
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.info("Interrupted by user.")
    finally:
        should_return = (
            args.return_to_ready_on_exit
            and ready_arm_q is not None
            and (not interrupted or args.return_to_ready_on_interrupt)
        )
        if should_return:
            move_arm_to_ready_pose(
                robot_interface,
                ready_arm_q,
                duration=args.ready_move_duration,
                frequency=args.frequency,
                tolerance=args.ready_tolerance,
                send_real_robot=args.send_real_robot,
                label="After inference",
            )
        elif interrupted and args.return_to_ready_on_exit and ready_arm_q is not None:
            LOGGER.info(
                "Return-to-ready skipped after Ctrl+C. Set --return_to_ready_on_interrupt=true to enable it."
            )


if __name__ == "__main__":
    main()
