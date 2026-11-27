import argparse
import threading

import benchmark
from benchmark_policies import MeasuredPolicy
from benchmark_policies import RTCPolicy
from benchmark_policies import SyncPolicy
import numpy as np
from openpi_client.base_policy import BasePolicy
import pytest


class ChunkPolicy(BasePolicy):
    def __init__(self):
        self.calls = []
        self.release = threading.Event()
        self.started = threading.Event()
        self.resets = 0

    def infer(self, obs):
        self.calls.append(obs)
        if len(self.calls) > 1:
            self.started.set()
            assert self.release.wait(5)
        values = np.arange(obs["tick"], obs["tick"] + 50)
        return {"actions": np.repeat(values[:, None], 14, axis=1)}

    def reset(self):
        self.resets += 1


def test_sync_replan_cadence():
    inner = ChunkPolicy()
    inner.release.set()
    policy = SyncPolicy(inner, interval=15)
    for tick in range(40):
        np.testing.assert_array_equal(policy.infer({"tick": tick})["actions"], np.full(14, tick))
    assert [obs["tick"] for obs in inner.calls] == [0, 15, 30]


def test_rtc_aligns_prefix_and_discards_elapsed_output_steps():
    inner = ChunkPolicy()
    policy = RTCPolicy(inner)
    try:
        for tick in range(16):
            policy.infer({"tick": tick})
        assert inner.started.wait(2)
        for tick in range(16, 20):
            np.testing.assert_array_equal(policy.infer({"tick": tick})["actions"], np.full(14, tick))
        request = inner.calls[-1]
        assert request["rtc_prefix_space"] == "environment"
        assert request["prev_action_chunk"].shape == (50, 14)
        np.testing.assert_array_equal(request["prev_action_chunk"][:35, 0], np.arange(15, 50))
        assert request["execute_horizon"] <= 35
        inner.release.set()
        policy.future.result(timeout=5)
        np.testing.assert_array_equal(policy.infer({"tick": 20})["actions"], np.full(14, 20))
        policy.reset()
        assert policy.actions is None
        assert policy.step == 0
    finally:
        inner.release.set()
        policy.close()


def test_rtc_exhaustion_is_explicit():
    inner = ChunkPolicy()
    policy = RTCPolicy(inner)
    try:
        for tick in range(50):
            policy.infer({"tick": tick})
        with pytest.raises(TimeoutError, match="exhausted"):
            policy.infer({"tick": 50})
        with pytest.raises(RuntimeError, match="reset"):
            policy.infer({"tick": 51})
    finally:
        inner.release.set()
        policy.close()


def test_metrics_use_joint_columns_and_physical_timestep():
    actions = np.zeros((5, 14))
    actions[:, 0] = np.arange(5) * 0.01
    actions[:, 6] = np.arange(5) * 100  # gripper units must not contaminate joint metrics
    result = benchmark.summarize_trace(actions, actions, [0.01, 0.03], [0.015, 0.035], 50)
    assert result["command_joint_step_max_rad"] == pytest.approx(0.01)
    assert result["control_overrun_fraction"] == 0.5
    assert result["command_joint_acceleration_rms"] == pytest.approx(0, abs=1e-12)


class FakeEnv:
    episode_reward = 0

    def reset(self, *, seed):
        self.seed = seed
        self.step = 0

    def get_observation(self):
        return {"state": np.zeros(14)}

    def apply_action(self, action):
        assert action["actions"].shape == (14,)
        self.step += 1

    def is_episode_complete(self):
        return self.step == 5


@pytest.mark.parametrize("mode", ["sync", "rtc", "bspline"])
def test_episode_records_all_modes(mode, tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark.time, "sleep", lambda duration: None)
    args = argparse.Namespace(
        interval=15,
        hz=50,
        guidance=0.5,
        delay_ms=0,
        blend_lead=3,
        steps=20,
        video=False,
        success_reward=4,
    )
    policy, measured = benchmark.make_policy(mode, benchmark.MockPolicy(), args)
    try:
        result = benchmark.run_episode(FakeEnv(), policy, measured, args, 123, tmp_path)
        assert result["steps"] == 5
        assert result["error"] is None
        assert result["rpc_count"] == 1  # reset must not erase the recorded latencies
        assert not result["success"]
        assert (tmp_path / "seed_123.json").exists()
        assert np.load(tmp_path / "seed_123.npz")["actions"].shape == (5, 14)
    finally:
        policy.close()


def test_measured_policy_captures_delay_and_sets_mode():
    inner = ChunkPolicy()
    measured = MeasuredPolicy(inner, rtc=True)
    measured.infer({"tick": 0})
    old_latencies = measured.latencies
    measured.reset()
    assert len(old_latencies) == 1
    assert measured.latencies == []
    assert inner.calls[0]["enable_rtc"] is True
