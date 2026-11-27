# ruff: noqa: PLC0415
# Optional simulator/video/plugin dependencies are deliberately imported lazily.
"""Compare sync, RTC and async B-spline control using the same ALOHA checkpoint.

Physics advances once per returned command. Wall-clock latency is reported
separately; this is not a simulation of a robot moving while the RPC blocks.
"""

import argparse
import json
import logging
from pathlib import Path
import time

from benchmark_policies import MeasuredPolicy
from benchmark_policies import RTCPolicy
from benchmark_policies import SyncPolicy
import numpy as np
from openpi_client.base_policy import BasePolicy

JOINTS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


class MockPolicy(BasePolicy):
    """Hold the observed pose; exercises plumbing, never a learned task policy."""

    def infer(self, obs):
        return {"actions": np.repeat(np.asarray(obs["state"])[None, :], 50, axis=0)}


def make_policy(mode, raw_policy, args):
    measured = MeasuredPolicy(raw_policy, delay_ms=args.delay_ms, rtc=mode == "rtc")
    if mode == "sync":
        policy = SyncPolicy(measured, interval=args.interval)
    elif mode == "rtc":
        policy = RTCPolicy(measured, interval=args.interval, hz=args.hz, guidance=args.guidance)
    else:
        from openpi_async_bspline import AsyncBSplinePolicy
        from openpi_async_bspline import BSplineConfig

        policy = AsyncBSplinePolicy(
            measured,
            BSplineConfig.from_dict(args.bspline_settings)
            if getattr(args, "bspline_settings", None) is not None
            else BSplineConfig(
                replan_interval=args.interval,
                blend_lead_steps=args.blend_lead,
                passthrough_indices=(6, 13),
            ),
        )
    return policy, measured


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if np.size(x) else None


def summarize_trace(actions, states, infer_s, cycle_s, hz):
    actions = np.asarray(actions).reshape(-1, 14)
    states = np.asarray(states).reshape(-1, 14)
    joint_delta = np.diff(actions[:, JOINTS], axis=0)
    return {
        "steps": len(actions),
        "control_call_p95_ms": float(np.percentile(infer_s, 95) * 1000) if len(infer_s) else None,
        "control_overrun_fraction": float(np.mean(np.asarray(cycle_s) > 1 / hz)) if len(cycle_s) else None,
        "command_joint_step_max_rad": float(np.max(np.abs(joint_delta))) if joint_delta.size else None,
        "command_joint_velocity_rms": rms(joint_delta * hz),
        "command_joint_acceleration_rms": rms(np.diff(joint_delta, axis=0) * hz**2),
        "command_joint_jerk_rms": rms(np.diff(joint_delta, n=2, axis=0) * hz**3),
        "state_joint_velocity_rms": rms(np.diff(states[:, JOINTS], axis=0) * hz),
    }


def run_episode(env, policy, measured, args, seed, output):
    policy.reset()
    env.reset(seed=seed)
    recorded = None
    active_policy = policy
    if getattr(args, "record_trajectories", False):
        from openpi_trajectory_recorder import RecordedPolicy

        recorded = RecordedPolicy(policy, capacity=args.steps, action_dim=14, hz=args.hz)
        active_policy = recorded
    actions, states, infer_s, cycle_s, frames = [], [], [], [], []
    error = None
    bootstrap_ms = None
    started = None
    try:
        # Bootstrap is excluded from sustained control latency for all modes.
        obs = env.get_observation()
        bootstrap_start = time.perf_counter()
        first_action = active_policy.infer(obs)
        bootstrap_ms = (time.perf_counter() - bootstrap_start) * 1000
        started = time.perf_counter()
        for step in range(args.steps):
            tick = time.perf_counter()
            obs = env.get_observation()
            call_start = time.perf_counter()
            result = first_action if step == 0 else active_policy.infer(obs)
            elapsed = time.perf_counter() - call_start
            action = np.asarray(result["actions"])
            if action.shape != (14,) or not np.isfinite(action).all():
                raise ValueError("Invalid control action")
            if recorded is None:
                actions.append(action.copy())
                states.append(np.asarray(obs["state"]).copy())
            if args.video:
                frames.append(np.transpose(obs["images"]["cam_high"], (1, 2, 0)).copy())
            env.apply_action({"actions": action})
            work_s = time.perf_counter() - tick
            if step > 0:
                infer_s.append(elapsed)
                cycle_s.append(work_s)
            if env.is_episode_complete():
                break
            time.sleep(max(0, 1 / args.hz - work_s))
    except Exception as exc:
        logging.exception("Episode seed=%s failed", seed)
        error = f"{type(exc).__name__}: {exc}"
    wall_s = time.perf_counter() - started if started is not None else 0.0
    if recorded is not None:
        actions = recorded.recorder.actions[: recorded.recorder.count]
        states = recorded.recorder.states[: recorded.recorder.count]
    # Drain pending RPCs before reading metrics; reset clears the collector, so
    # capture its list first. The next episode is never contaminated by old work.
    completed_latencies = measured.latencies
    try:
        policy.reset()
    except Exception as exc:
        error = error or f"reset: {type(exc).__name__}: {exc}"
    rpc_latencies = list(completed_latencies)
    record = {
        "seed": seed,
        "success": error is None and env.episode_reward >= args.success_reward,
        "max_reward": env.episode_reward,
        "error": error,
        "bootstrap_ms": bootstrap_ms,
        "wall_seconds_excluding_bootstrap": wall_s,
        "rpc_p95_ms": float(np.percentile(rpc_latencies, 95) * 1000) if rpc_latencies else None,
        "rpc_count": len(rpc_latencies),
        **summarize_trace(actions, states, infer_s, cycle_s, args.hz),
    }
    np.savez_compressed(
        output / f"seed_{seed}.npz",
        actions=np.asarray(actions),
        states=np.asarray(states),
        infer_seconds=np.asarray(infer_s),
        cycle_work_seconds=np.asarray(cycle_s),
    )
    (output / f"seed_{seed}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    if recorded is not None:
        recording_path = recorded.save(
            output / "recordings" / f"seed_{seed}.npz", metadata={"mode": args.mode, "seed": seed}
        )
        if getattr(args, "plot_trajectories", False) and recorded.recorder.count >= 3:
            from openpi_trajectory_recorder.analysis import analyze

            analyze([recording_path], output / "recordings" / f"seed_{seed}_plots", labels=[args.mode])
    if frames:
        import imageio.v2 as imageio

        imageio.mimwrite(output / f"seed_{seed}.mp4", frames, fps=args.hz)
    return record


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sync", "rtc", "bspline"), required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--hz", type=float, default=50)
    parser.add_argument("--interval", type=int)
    parser.add_argument("--blend-lead", type=int)
    parser.add_argument("--bspline-config", type=Path, help="Plugin YAML; do not combine with --interval/--blend-lead")
    parser.add_argument("--guidance", type=float, default=0.5)
    parser.add_argument("--delay-ms", type=float, default=0)
    parser.add_argument("--success-reward", type=float, default=4)
    parser.add_argument("--output", type=Path, default=Path("data/aloha_sim/benchmark"))
    parser.add_argument("--mock", action="store_true", help="Hold-pose mock; never a learned policy result")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--record-trajectories", action="store_true", help="Record vectors in preallocated RAM")
    parser.add_argument("--plot-trajectories", action="store_true", help="Record, then plot after each episode ends")
    args = parser.parse_args()
    args.bspline_settings = None
    if args.bspline_config is not None:
        from openpi_async_bspline import BSplineConfig

        if args.interval is not None or args.blend_lead is not None:
            parser.error("Set scheduling parameters in YAML when using --bspline-config")
        config = BSplineConfig.from_yaml(args.bspline_config)
        if config.fps != args.hz:
            parser.error("YAML fps must match --hz so derivative units match the control rate")
        args.bspline_settings = config.to_dict()
        args.interval = config.replan_interval
        args.blend_lead = config.effective_lead
    else:
        args.interval = 15 if args.interval is None else args.interval
        args.blend_lead = 3 if args.blend_lead is None else args.blend_lead
    args.record_trajectories = args.record_trajectories or args.plot_trajectories
    if (
        args.episodes < 1
        or args.steps < 1
        or args.hz != 50
        or args.delay_ms < 0
        or args.interval < 1
        or args.blend_lead < 0
        or args.interval + args.blend_lead >= 49
        or not np.isfinite(args.delay_ms)
        or not np.isfinite(args.guidance)
        or args.guidance < 0
    ):
        parser.error(
            "Use positive counts, hz=50 (ALOHA physics rate), finite nonnegative delays/guidance and interval+lead<49"
        )
    return args


def main():
    args = parse_args()
    # Lazy import allows --help and scheduler tests without MuJoCo installed.
    from env import AlohaSimEnvironment

    output = args.output / args.mode
    # Refuse to silently replace a prior run with the same seeds.
    output.mkdir(parents=True, exist_ok=False)
    env = AlohaSimEnvironment(task="gym_aloha/AlohaTransferCube-v0", seed=args.seed)
    policy = None
    try:
        env.reset(seed=args.seed)
        if args.mock:
            raw_policy = MockPolicy()
            metadata = {"mock": True, "description": "hold-pose plumbing test, no RTC guidance"}
        else:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy

            raw_policy = WebsocketClientPolicy(host=args.host, port=args.port)
            metadata = raw_policy.get_server_metadata()
            if metadata.get("benchmark") != "aloha_sim" or metadata.get("action_horizon") != 50:
                raise ValueError("Use the pi0_aloha_sim_rtc server config for the comparison")
            if not metadata.get("rtc") or metadata.get("rtc_prefix_space") != "environment":
                raise ValueError("RTC requires the pi0_aloha_sim_rtc server with environment-prefix support")
        # Warm up both JIT signatures, outside episode metrics. Use the same
        # number of warmup requests for each mode. Baselines disable guidance.
        obs = env.get_observation()
        bootstrap = {**obs, "enable_rtc": args.mode == "rtc"}
        if args.mode == "rtc":
            # Match RTCPolicy's first request, including keyword arguments, so
            # its no-prefix JIT signature is compiled before episode metrics.
            bootstrap["max_guidance_weight"] = args.guidance
        result = raw_policy.infer(bootstrap)
        warmup = {**obs, "enable_rtc": args.mode == "rtc"}
        if args.mode == "rtc":
            warmup.update(
                prev_action_chunk=result["actions"],
                rtc_prefix_space="environment",
                inference_delay=1,
                execute_horizon=args.interval,
                max_guidance_weight=args.guidance,
            )
        raw_policy.infer(warmup)
        policy, measured = make_policy(args.mode, raw_policy, args)
        records = []
        config = vars(args).copy()
        config["output"] = str(args.output)
        config["bspline_config"] = str(args.bspline_config) if args.bspline_config is not None else None
        (output / "config.json").write_text(json.dumps({"args": config, "server": metadata}, indent=2))
        for seed in range(args.seed, args.seed + args.episodes):
            record = run_episode(env, policy, measured, args, seed, output)
            records.append(record)
            print(json.dumps(record))
            summary = {
                "mode": args.mode,
                "mock": args.mock,
                "physics": "pause_during_blocking_inference",
                "episodes": len(records),
                "success_rate": sum(r["success"] for r in records) / len(records),
                "failed_episodes": sum(r["error"] is not None for r in records),
                "results": records,
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
    finally:
        if policy is not None:
            policy.close()
        env.close()
        if args.mode == "bspline" and policy is not None and policy.config.diagnostics_enabled:
            policy.save_diagnostics()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
