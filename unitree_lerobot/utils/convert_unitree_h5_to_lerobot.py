"""
Convert Unitree HDF5 episodes to LeRobot format.

Example:
python unitree_lerobot/utils/convert_unitree_h5_to_lerobot.py \
    --raw-dir $HOME/datasets/Voltage_tester2 \
    --repo-id local/g1_brainco_voltage_tester \
    --robot-type Unitree_G1_Brainco \
    --task "press the button on the electrical cabinet" \
    --push-to-hub false

For flat HDF5 files produced by G1 + BrainCo Revo2 teleoperation, the default mapping is:
- state:  actual_joint_positions_rad + left_finger_actual_angles + right_finger_actual_angles
- action: ik_joint_positions_rad + 6D left_finger_target_angles + 6D right_finger_target_angles
- cameras: head_rgb -> cam_left_high, left_rgb -> cam_left_wrist, right_rgb -> cam_right_wrist
"""

import dataclasses
from pathlib import Path
import shutil
import sys
from typing import Literal

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_LEROBOT_SRC = _REPO_ROOT / "unitree_lerobot" / "lerobot" / "src"
for _path in (_REPO_ROOT, _LOCAL_LEROBOT_SRC):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cv2
import h5py
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME
import numpy as np
import tqdm
import tyro

from unitree_lerobot.utils.constants import ROBOT_CONFIGS


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_text(value.item())
        if len(value) > 0:
            return _decode_text(value[0])
    return str(value)


def _read_task(ep: h5py.File, default_task: str) -> str:
    if "language_raw" in ep:
        return _decode_text(ep["language_raw"][()])
    if "substep_reasonings" in ep and len(ep["substep_reasonings"]) > 0:
        return _decode_text(ep["substep_reasonings"][0])
    return default_task


def _has_unitree_nested_layout(ep: h5py.File) -> bool:
    return "/observations/qpos" in ep and "/action" in ep and "/observations/images" in ep


def _has_g1_brainco_flat_layout(ep: h5py.File) -> bool:
    required_keys = (
        "actual_joint_positions_rad",
        "ik_joint_positions_rad",
        "left_finger_actual_angles",
        "right_finger_actual_angles",
        "left_finger_target_angles",
        "right_finger_target_angles",
        "head_rgb",
    )
    return all(key in ep for key in required_keys)


def _decode_image_frame(frame: np.ndarray | np.void | bytes) -> np.ndarray:
    if isinstance(frame, np.void):
        frame = frame.tobytes()

    if isinstance(frame, bytes):
        encoded = np.frombuffer(frame, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode compressed image frame.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.asarray(frame)
    if image.ndim == 1:
        decoded = cv2.imdecode(image.astype(np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("Failed to decode compressed image frame.")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if image.ndim != 3:
        raise ValueError(f"Expected image frame with 3 dimensions, got shape {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return image.astype(np.uint8)


def _read_camera_episode(ep: h5py.File, camera: str) -> list[np.ndarray]:
    dataset = ep[f"/observations/images/{camera}"]
    return [_decode_image_frame(dataset[i]) for i in range(len(dataset))]


def _read_flat_camera_episode(ep: h5py.File, camera_key: str) -> list[np.ndarray]:
    dataset = ep[camera_key]
    return [_decode_image_frame(dataset[i]) for i in range(len(dataset))]


def _read_state_action(ep: h5py.File) -> tuple[np.ndarray, np.ndarray]:
    if _has_unitree_nested_layout(ep):
        state = np.asarray(ep["/observations/qpos"][:], dtype=np.float32)
        action = np.asarray(ep["/action"][:], dtype=np.float32)
        return state, action

    if _has_g1_brainco_flat_layout(ep):
        arm_state = np.asarray(ep["actual_joint_positions_rad"][:], dtype=np.float32)
        left_hand_state = np.asarray(ep["left_finger_actual_angles"][:], dtype=np.float32)
        right_hand_state = np.asarray(ep["right_finger_actual_angles"][:], dtype=np.float32)

        arm_action = np.asarray(ep["ik_joint_positions_rad"][:], dtype=np.float32)
        left_hand_action = _read_hand_target(ep["left_finger_target_angles"])
        right_hand_action = _read_hand_target(ep["right_finger_target_angles"])

        state = np.concatenate([arm_state, left_hand_state, right_hand_state], axis=1)
        action = np.concatenate([arm_action, left_hand_action, right_hand_action], axis=1)
        return state, action

    raise ValueError("Unsupported HDF5 layout. Expected Unitree nested layout or G1 BrainCo flat layout.")


def _read_hand_target(dataset: h5py.Dataset) -> np.ndarray:
    values = np.asarray(dataset[:], dtype=np.float32)
    if values.shape[1] == 6:
        return values
    if values.shape[1] == 7:
        # Backward compatibility with old files that stored [event_flag, 6 finger targets].
        return values[:, 1:7]
    raise ValueError(f"Expected 6D hand targets, or legacy 7D [flag + 6 targets], got shape {values.shape}.")


def _read_cameras(ep: h5py.File, robot_type: str) -> dict[str, list[np.ndarray]]:
    if _has_unitree_nested_layout(ep):
        return {
            camera: _read_camera_episode(ep, camera)
            for camera in ROBOT_CONFIGS[robot_type].cameras
            if f"/observations/images/{camera}" in ep
        }

    if _has_g1_brainco_flat_layout(ep):
        camera_map = {
            "head_rgb": "cam_left_high",
            "left_rgb": "cam_left_wrist",
            "right_rgb": "cam_right_wrist",
        }
        return {
            camera_name: _read_flat_camera_episode(ep, h5_key)
            for h5_key, camera_name in camera_map.items()
            if h5_key in ep
        }

    raise ValueError("Unsupported HDF5 layout. Expected Unitree nested layout or G1 BrainCo flat layout.")


def _discover_h5_files(raw_dir: Path) -> list[Path]:
    h5_files = sorted(raw_dir.glob("episode_*.h5")) + sorted(raw_dir.glob("episode_*.hdf5"))
    if not h5_files:
        h5_files = sorted(raw_dir.glob("*.h5")) + sorted(raw_dir.glob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 or .hdf5 files found under {raw_dir}")
    return h5_files


def _inspect_first_episode(h5_path: Path, robot_type: str) -> tuple[int, int, dict[str, tuple[int, int, int]]]:
    with h5py.File(h5_path, "r") as ep:
        state, action = _read_state_action(ep)
        state_dim = int(state.shape[1])
        action_dim = int(action.shape[1])
        image_shapes = {camera: images[0].shape for camera, images in _read_cameras(ep, robot_type).items()}
    if not image_shapes:
        raise ValueError(f"No known cameras for {robot_type} found in {h5_path}")
    return state_dim, action_dim, image_shapes


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    state_dim: int,
    action_dim: int,
    image_shapes: dict[str, tuple[int, int, int]],
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = ROBOT_CONFIGS[robot_type].motors

    state_names = motors if state_dim == len(motors) else [f"state_{i}" for i in range(state_dim)]
    action_names = motors if action_dim == len(motors) else [f"action_{i}" for i in range(action_dim)]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [state_names],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [action_names],
        },
    }

    for camera, shape in image_shapes.items():
        features[f"observation.images.{camera}"] = {
            "dtype": mode,
            "shape": shape,
            "names": ["height", "width", "channel"],
        }

    dataset_path = HF_LEROBOT_HOME / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def populate_dataset(
    dataset: LeRobotDataset,
    h5_files: list[Path],
    robot_type: str,
    task: str,
    *,
    strict_dims: bool = True,
) -> LeRobotDataset:
    expected_dim = len(ROBOT_CONFIGS[robot_type].motors)

    for h5_path in tqdm.tqdm(h5_files, desc="Converting HDF5 episodes"):
        with h5py.File(h5_path, "r") as ep:
            state, action = _read_state_action(ep)

            if strict_dims and (state.shape[1] != expected_dim or action.shape[1] != expected_dim):
                raise ValueError(
                    f"{h5_path} has state/action dims {state.shape[1]}/{action.shape[1]}, "
                    f"but {robot_type} expects {expected_dim}. Use --strict-dims false only after verifying order."
                )

            episode_task = _read_task(ep, task)
            cameras = _read_cameras(ep, robot_type)

        num_frames = min([len(state), len(action), *(len(images) for images in cameras.values())])
        for frame_idx in range(num_frames):
            frame = {
                "observation.state": state[frame_idx],
                "action": action[frame_idx],
                "task": episode_task,
            }
            for camera, images in cameras.items():
                frame[f"observation.images.{camera}"] = images[frame_idx]
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def h5_to_lerobot(
    raw_dir: Path,
    repo_id: str,
    robot_type: str,
    task: str = "",
    *,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    strict_dims: bool = True,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> None:
    h5_files = _discover_h5_files(raw_dir)
    state_dim, action_dim, image_shapes = _inspect_first_episode(h5_files[0], robot_type)
    dataset = create_empty_dataset(
        repo_id,
        robot_type=robot_type,
        state_dim=state_dim,
        action_dim=action_dim,
        image_shapes=image_shapes,
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(dataset, h5_files, robot_type=robot_type, task=task, strict_dims=strict_dims)
    dataset.finalize()

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)


if __name__ == "__main__":
    tyro.cli(h5_to_lerobot)
