#!/usr/bin/env python3
"""
Run a LeRobot policy on a GPU workstation and expose it over HTTP.

The robot-side script sends one observation to POST /predict. The server returns
one unnormalized action in the same action space as the training dataset.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import time
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.utils import get_safe_torch_device, init_logging
from unitree_lerobot.eval_robot.utils.utils import predict_action


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-side remote policy server for Unitree G1 + BrainCo evaluation."
    )
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Path to a trained LeRobot policy.")
    parser.add_argument("--repo_id", required=True, help="LeRobot dataset repo id used for metadata and stats.")
    parser.add_argument("--host", default="0.0.0.0", help="Listen address. Use 0.0.0.0 for remote clients.")
    parser.add_argument("--port", type=int, default=8088, help="Listen port.")
    parser.add_argument("--device", default="cuda", help="Policy device, usually cuda on the A6000 machine.")
    parser.add_argument("--task", default="", help="Default task text when the client does not send one.")
    parser.add_argument("--robot_type", default="", help="Optional robot_type string passed to the policy.")
    parser.add_argument(
        "--rename_map_json",
        default="{}",
        help='JSON dict for observation renaming, for example \'{"old_key":"new_key"}\'.',
    )
    parser.add_argument("--request_limit_mb", type=float, default=32.0, help="Maximum accepted JSON request size.")
    return parser.parse_args()


def _decode_image(value: Any) -> torch.Tensor:
    if isinstance(value, str):
        encoded = value
    elif isinstance(value, dict):
        encoded = value.get("data", "")
    else:
        raise TypeError(f"Unsupported image payload type: {type(value)!r}")

    raw = base64.b64decode(encoded)
    array = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode returned None")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb)


def _build_observation(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "state" not in payload:
        raise ValueError("Request missing required field: state")

    observation: dict[str, torch.Tensor] = {
        "observation.state": torch.as_tensor(payload["state"], dtype=torch.float32)
    }

    images = payload.get("images", {})
    if not isinstance(images, dict):
        raise ValueError("Field 'images' must be a dict keyed by observation image name")

    for key, value in images.items():
        if not key.startswith("observation.images."):
            raise ValueError(f"Unexpected image key: {key}")
        observation[key] = _decode_image(value)

    return observation


def _json_response(handler: BaseHTTPRequestHandler, status: int, body: dict[str, Any]) -> None:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


class PolicyServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.rename_map = json.loads(args.rename_map_json)
        if not isinstance(self.rename_map, dict):
            raise ValueError("--rename_map_json must decode to a dict")

        self.device = get_safe_torch_device(args.device, log=True)
        LOGGER.info("Loading dataset metadata: %s", args.repo_id)
        self.dataset = LeRobotDataset(repo_id=args.repo_id)

        LOGGER.info("Loading policy config: %s", args.policy_path)
        self.policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        self.policy_cfg.pretrained_path = args.policy_path
        self.policy_cfg.device = str(self.device)

        LOGGER.info("Creating policy")
        self.policy = make_policy(cfg=self.policy_cfg, ds_meta=self.dataset.meta)
        self.policy.eval()

        LOGGER.info("Creating pre/post processors")
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=self.policy_cfg.pretrained_path,
            dataset_stats=rename_stats(self.dataset.meta.stats, self.rename_map),
            preprocessor_overrides={
                "device_processor": {"device": str(self.device)},
                "rename_observations_processor": {"rename_map": self.rename_map},
            },
        )
        self.lock = Lock()
        self.request_limit_bytes = int(args.request_limit_mb * 1024 * 1024)

        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if hasattr(self.preprocessor, "reset"):
            self.preprocessor.reset()
        if hasattr(self.postprocessor, "reset"):
            self.postprocessor.reset()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = _build_observation(payload)
        task = payload.get("task") or self.args.task
        robot_type = payload.get("robot_type") or self.args.robot_type

        started = time.perf_counter()
        with self.lock:
            action = predict_action(
                observation=observation,
                policy=self.policy,
                device=self.device,
                preprocessor=self.preprocessor,
                postprocessor=self.postprocessor,
                use_amp=bool(getattr(self.policy_cfg, "use_amp", False)),
                task=task,
                use_dataset=False,
                robot_type=robot_type,
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        action_np = action.detach().cpu().numpy().astype(np.float32)
        return {
            "action": action_np.reshape(-1).tolist(),
            "action_shape": list(action_np.shape),
            "server_latency_ms": latency_ms,
        }

    def handler_class(self) -> type[BaseHTTPRequestHandler]:
        server_state = self

        class RequestHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: Any) -> None:
                LOGGER.info("%s - %s", self.address_string(), fmt % args)

            def do_GET(self) -> None:
                if self.path != "/health":
                    _json_response(self, 404, {"error": "not found"})
                    return
                _json_response(
                    self,
                    200,
                    {
                        "status": "ok",
                        "policy_path": server_state.args.policy_path,
                        "repo_id": server_state.args.repo_id,
                        "device": str(server_state.device),
                    },
                )

            def do_POST(self) -> None:
                if self.path != "/predict":
                    _json_response(self, 404, {"error": "not found"})
                    return

                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if content_length <= 0:
                        raise ValueError("Empty request body")
                    if content_length > server_state.request_limit_bytes:
                        raise ValueError(
                            f"Request is {content_length} bytes, above limit {server_state.request_limit_bytes}"
                        )

                    raw = self.rfile.read(content_length)
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("JSON request body must be an object")
                    response = server_state.predict(payload)
                    _json_response(self, 200, response)
                except Exception as exc:  # noqa: BLE001 - return JSON to the robot client.
                    LOGGER.exception("Prediction request failed")
                    _json_response(self, 500, {"error": str(exc)})

        return RequestHandler


def main() -> None:
    args = parse_args()
    init_logging()

    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    policy_server = PolicyServer(args)
    httpd = ThreadingHTTPServer((args.host, args.port), policy_server.handler_class())
    LOGGER.info("Remote policy server listening on http://%s:%s", args.host, args.port)
    LOGGER.info("Health check: GET /health, inference: POST /predict")

    context = (
        torch.autocast(device_type=policy_server.device.type)
        if policy_server.device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), context:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
