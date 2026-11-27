import threading

import numpy as np
from openpi_client.base_policy import BasePolicy
import pytest

from openpi_async_bspline import AsyncBSplinePolicy
from openpi_async_bspline import BSplineConfig
from openpi_async_bspline import build_trajectory


def chunk(start, n=50):
    t = np.arange(start, start + n)
    return np.column_stack((np.sin(t / 30), t / 50, (t % 10 < 5).astype(float)))


def test_windows_preserve_prefix_and_bound_memory():
    config = BSplineConfig(passthrough_indices=(2,))
    previous = build_trajectory(chunk(0), 0, config)
    for start in range(15, 150, 15):
        anchor = start + 5
        current = build_trajectory(chunk(start), start, config, previous, anchor)
        for step in range(start, anchor + 1):
            np.testing.assert_array_equal(current.sample(step), previous.sample(step))
        assert current.actions.shape == (50, 3)
        assert np.isfinite(current.actions).all()
        np.testing.assert_array_equal(current.actions[6:, 2], chunk(start)[6:, 2])
        previous = current


def test_linear_fit_and_copy():
    config = BSplineConfig()
    data = np.arange(50.0)[:, None]
    trajectory = build_trajectory(data, 0, config)
    np.testing.assert_allclose(trajectory.actions, data, atol=1e-12)
    trajectory.sample(0)[:] = 123
    assert abs(trajectory.sample(0)[0]) < 1e-12
    with pytest.raises(TimeoutError):
        trajectory.sample(50)


@pytest.mark.parametrize("data", [np.zeros(50), np.zeros((3, 2)), np.full((50, 2), np.nan)])
def test_invalid_chunk(data):
    with pytest.raises(ValueError, match="finite"):
        build_trajectory(data, 0, BSplineConfig())


class ControlledPolicy(BasePolicy):
    def __init__(self):
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.resets = 0
        self.fail = False

    def infer(self, obs):
        self.calls += 1
        if self.calls > 1:
            self.started.set()
            assert self.release.wait(5), "test worker was not released"
        if self.fail:
            raise ValueError("worker failure")
        return {"actions": chunk(obs["step"]), "server": "test"}

    def reset(self):
        self.resets += 1
        self.calls = 0


def test_async_worker_does_not_block_control_and_installs_merge():
    inner = ControlledPolicy()
    with AsyncBSplinePolicy(inner) as policy:
        for step in range(16):
            result = policy.infer({"step": step})
            assert result["actions"].shape == (3,)
        assert inner.started.wait(2)
        # Worker waits on an Event; control must still return without releasing it.
        for step in range(16, 20):
            policy.infer({"step": step})
        inner.release.set()
        policy._future.result(timeout=5)
        result = policy.infer({"step": 20})
        assert result["server"] == "test"
        assert policy._trajectory.end == 64
        policy.reset()
        assert inner.resets == 1
        np.testing.assert_allclose(
            policy.infer({"step": 0})["actions"], build_trajectory(chunk(0), 0, policy.config).sample(0)
        )


def test_exhaustion_fails_without_replaying_actions():
    inner = ControlledPolicy()
    try:
        with AsyncBSplinePolicy(inner) as policy:
            for step in range(50):
                policy.infer({"step": step})
            with pytest.raises(TimeoutError):
                policy.infer({"step": 50})
            with pytest.raises(RuntimeError, match="reset"):
                policy.infer({"step": 51})
            inner.release.set()
    finally:
        inner.release.set()


def test_worker_error_propagates_and_reset_recovers():
    inner = ControlledPolicy()
    inner.release.set()
    with AsyncBSplinePolicy(inner) as policy:
        for step in range(15):
            policy.infer({"step": step})
        inner.fail = True
        policy.infer({"step": 15})
        with pytest.raises(ValueError, match="worker failure"):
            policy._future.result(timeout=5)
        with pytest.raises(ValueError, match="worker failure"):
            policy.infer({"step": 16})
        inner.fail = False
        policy.reset()
        assert policy.infer({"step": 0})["actions"].shape == (3,)


def test_reset_waits_for_worker_and_discards_old_result():
    inner = ControlledPolicy()
    with AsyncBSplinePolicy(inner) as policy:
        for step in range(16):
            policy.infer({"step": step})
        assert inner.started.wait(2)
        done = threading.Event()
        thread = threading.Thread(target=lambda: (policy.reset(), done.set()))
        thread.start()
        assert not done.wait(0.05)
        inner.release.set()
        thread.join(5)
        assert done.is_set()
        assert policy._trajectory is None
        assert policy.infer({"step": 0})["actions"].shape == (3,)
    with pytest.raises(RuntimeError, match="closed"):
        policy.infer({})


def test_missed_deadline_rejects_result():
    inner = ControlledPolicy()
    inner.release.set()
    with AsyncBSplinePolicy(inner) as policy:
        for step in range(16):
            policy.infer({"step": step})
        _, anchor, _ = policy._future.result(timeout=5)
        policy._step = anchor + 1
        with pytest.raises(TimeoutError, match="deadline"):
            policy.infer({"step": anchor + 1})


def test_missing_overlap_and_dimension_change_are_explicit():
    config = BSplineConfig()
    previous = build_trajectory(chunk(0), 0, config)
    with pytest.raises(TimeoutError, match="overlapping"):
        build_trajectory(chunk(45), 45, config, previous, 50)
    with pytest.raises(ValueError, match="dimension changed"):
        build_trajectory(chunk(15)[:, :2], 15, config, previous, 20)


@pytest.mark.parametrize(
    "kwargs",
    [{"replan_interval": 0}, {"blend_lead_steps": -1}, {"knots_interior": 1.5}, {"passthrough_indices": (-1,)}],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError, match=r"Invalid|nonnegative"):
        BSplineConfig(**kwargs)
