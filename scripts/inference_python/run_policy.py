"""Unified SDK real-robot client: sync / RTC / asynchronous B-spline.

--mock exercises scheduling and recording without importing or enabling hardware.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from openpi_async_bspline import AsyncBSplinePolicy
from openpi_async_bspline import BSplineConfig
from openpi_client.base_policy import BasePolicy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.aloha_sim.benchmark_policies import RTCPolicy  # noqa: E402
from examples.aloha_sim.benchmark_policies import SyncPolicy  # noqa: E402


class ModelAdapter(BasePolicy):
    def __init__(self, raw, dimension, horizon, rtc):
        self.raw, self.dimension, self.horizon, self.rtc = raw, dimension, horizon, rtc

    def infer(self, obs):
        result = self.raw.infer({**obs, "enable_rtc": self.rtc})
        actions = np.asarray(result["actions"], dtype=float)
        if actions.shape != (self.horizon, self.dimension) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid model actions {actions.shape}; expected {(self.horizon, self.dimension)}")
        return {**result, "actions": actions}

    def reset(self):
        self.raw.reset()


class MockRobot:
    def __init__(self, dimension, cameras):
        self.state = np.zeros(dimension)
        self.cameras = cameras

    def set_up(self):
        pass

    def get_observation(self):
        return {"state": self.state.copy(), "images": {name: np.zeros((3, 48, 64), np.uint8) for name in self.cameras}}

    def step(self, action):
        self.state[:] = action

    def stop(self):
        pass


class MockModel(BasePolicy):
    def __init__(self, horizon):
        self.horizon = horizon

    def infer(self, obs):
        return {"actions": np.tile(obs["state"], (self.horizon, 1))}


def make_policy(raw, args, config):
    dimension = 7 if args.single_arm else 14
    model = ModelAdapter(raw, dimension, args.horizon, args.mode == "rtc")
    if args.mode == "bspline":
        return AsyncBSplinePolicy(model, config)
    if args.mode == "rtc":
        return RTCPolicy(
            model,
            interval=args.interval,
            horizon=args.horizon,
            hz=args.hz,
            guidance=args.guidance,
            action_dim=dimension,
        )
    return SyncPolicy(model, interval=args.interval, horizon=args.horizon, action_dim=dimension)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sync", "rtc", "bspline"), default="bspline")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hz", type=float, default=30)
    parser.add_argument("--interval", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--guidance", type=float, default=0.3)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--single-arm", action="store_true")
    parser.add_argument("--camera-type", choices=("v4l2", "orbbec"), default="v4l2")
    parser.add_argument("--camera-names", nargs="+", default=["cam_high", "cam_left_wrist", "cam_right_wrist"])
    parser.add_argument(
        "--hardware-config", type=Path, default=ROOT / "scripts/inference_python/configs/y1_hardware.yaml"
    )
    parser.add_argument(
        "--bspline-config", type=Path, default=ROOT / "packages/openpi-async_bspline/configs/y1_real.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--record-trajectories", action="store_true")
    parser.add_argument("--plot-trajectories", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)
    if (
        not np.isfinite(args.hz)
        or args.hz <= 0
        or args.steps < 1
        or not 1 <= args.interval < args.horizon
        or not np.isfinite(args.guidance)
        or args.guidance < 0
        or not args.prompt.strip()
    ):
        parser.error("Invalid rate, steps, interval, horizon, guidance or prompt")
    return args


def main(argv=None):
    args = parse_args(argv)
    dimension = 7 if args.single_arm else 14
    config = BSplineConfig.from_yaml(args.bspline_config) if args.mode == "bspline" else None
    if config is not None:
        if config.fps != args.hz or config.replan_interval + config.effective_lead >= args.horizon - 1:
            raise ValueError("YAML fps must equal --hz; interval + offset must be < horizon - 1")
        if any(i >= dimension for i in config.passthrough_indices):
            raise ValueError("Single arm requires passthrough_indices: [6], dual arm: [6,13]")
    output = args.output / args.mode
    if output.exists():
        raise FileExistsError(output)
    # All imports/configuration and server checks precede hardware initialization.
    if args.mock:
        raw = MockModel(args.horizon)
        env = MockRobot(dimension, args.camera_names)
    else:
        hardware = yaml.safe_load(args.hardware_config.read_text())
        raw = WebsocketClientPolicy(args.host, args.port)
        metadata = raw.get_server_metadata()
        if metadata.get("robot") != "y1" or metadata.get("action_dim") != dimension:
            raise ValueError("Server must advertise robot=y1 and matching action_dim; do not use simulation weights")
        if metadata.get("action_horizon") != args.horizon:
            raise ValueError("Server action_horizon mismatch")
        if args.mode == "rtc" and (not metadata.get("rtc") or metadata.get("rtc_prefix_space") != "environment"):
            raise ValueError("RTC server must support environment-space prefixes")
        import real_robot_env as hardware_env  # noqa: PLC0415

        hardware_env.ARM_CAN_DEV = hardware["arm_can"]
        hardware_env.V4l2_CAMERA_DEV = hardware["v4l2"]
        hardware_env.Orbbec_CAMERA_SERIALS = hardware["orbbec"]
        for name in args.camera_names:
            if name not in hardware[args.camera_type]:
                raise ValueError(f"Camera missing from hardware YAML: {name}")
        env = hardware_env.RealRobotEnv(args.single_arm, args.camera_names, camera_type=args.camera_type)
    recorder = None
    if args.record_trajectories or args.plot_trajectories:
        from openpi_trajectory_recorder import TrajectoryRecorder  # noqa: PLC0415

        recorder = TrajectoryRecorder(capacity=args.steps, action_dim=dimension, hz=args.hz)
    if args.video:
        import imageio.v2 as imageio  # noqa: PLC0415

    policy = make_policy(raw, args, config)
    frames = []  # Bounded by --steps; encode only after control stops.
    error = None
    completed = 0
    output.mkdir(parents=True)
    (output / "config.json").write_text(json.dumps(vars(args), default=str, indent=2))
    try:
        env.set_up()
        deadline = time.monotonic() + 5.0
        first_obs = env.get_observation()
        while first_obs is None and time.monotonic() < deadline:
            time.sleep(0.02)
            first_obs = env.get_observation()
        if first_obs is None:
            raise TimeoutError("No initial camera/robot observation within 5 seconds")
        for _ in range(args.steps):
            began = time.perf_counter()
            obs = env.get_observation()
            if obs is None:
                raise RuntimeError("Missing camera/robot observation; stopped sending actions")
            state = np.asarray(obs["state"], dtype=float).copy()
            if state.shape != (dimension,) or not np.isfinite(state).all():
                raise ValueError("Invalid robot state")
            result = policy.infer({**obs, "prompt": args.prompt})
            elapsed = time.perf_counter() - began
            action = np.asarray(result["actions"], dtype=float)
            if action.shape != (dimension,) or not np.isfinite(action).all():
                raise ValueError("Invalid control action")
            # No extra interpolation: the B-spline worker already samples at --hz.
            env.step(action)
            completed += 1
            if recorder is not None:
                recorder.append(action, state, timestamp=began, infer_seconds=elapsed)
            if args.video:
                frames.append(np.transpose(obs["images"][args.camera_names[0]], (1, 2, 0)).copy())
            time.sleep(max(0, 1 / args.hz - (time.perf_counter() - began)))
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        # Stop reading devices before waiting for outstanding network requests.
        try:
            env.stop()
        finally:
            policy.close()
        (output / "summary.json").write_text(
            json.dumps({"mock": args.mock, "steps": completed, "error": error}, indent=2)
        )
        if recorder is not None:
            recording = recorder.save(output / "trajectory.npz", metadata={"mode": args.mode, "mock": args.mock})
            if args.plot_trajectories and recorder.count >= 3:
                from openpi_trajectory_recorder.analysis import analyze  # noqa: PLC0415

                analyze([recording], output / "plots", labels=[args.mode])
        if frames:
            imageio.mimwrite(output / "camera.mp4", frames, fps=args.hz)
        if config is not None and config.diagnostics_enabled:
            policy.save_diagnostics(output / "bspline_diagnostics")


if __name__ == "__main__":
    main()
