import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from openpi_trajectory_recorder import RecordedPolicy
from openpi_trajectory_recorder import TrajectoryRecorder
from openpi_trajectory_recorder.analysis import kinematics


class Policy:
    def __init__(self):
        self.result = {"actions": np.array([1.0, 2.0]), "extra": object()}
        self.resets = 0

    def infer(self, obs, **kwargs):
        obs["state"][:] = -1
        return self.result

    def reset(self):
        self.resets += 1


def test_memory_only_snapshot_and_result_identity(monkeypatch):
    policy = Policy()
    wrapped = RecordedPolicy(policy, capacity=3, action_dim=2, hz=50)

    def forbidden(*args, **kwargs):
        raise AssertionError("I/O during infer")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(np, "savez_compressed", forbidden)
    for i in range(3):
        obs = {"state": np.array([float(i), 3.0]), "images": object()}
        assert wrapped.infer(obs) is policy.result
    policy.result["actions"][:] = 9
    np.testing.assert_array_equal(wrapped.recorder.actions[:3], [[1, 2]] * 3)
    np.testing.assert_array_equal(wrapped.recorder.states[:, 0], [0, 1, 2])
    wrapped.reset()
    assert policy.resets == 1
    assert wrapped.recorder.count == 3


def test_overflow_invalid_and_failed_inference_preserve_policy_behavior():
    policy = Policy()
    wrapped = RecordedPolicy(policy, capacity=1, action_dim=2, hz=50)
    assert wrapped.infer({"state": np.zeros(3)}) is policy.result
    assert wrapped.recorder.invalid == 1
    wrapped.recorder.clear()
    wrapped.infer({"state": np.zeros(2)})
    assert wrapped.infer({"state": np.zeros(2)}) is policy.result
    assert wrapped.recorder.dropped == 1
    assert wrapped.recorder.count == 1
    wrapped.recorder.clear()
    error = RuntimeError("model failed")

    def fail(obs):
        raise error

    policy.infer = fail
    with pytest.raises(RuntimeError, match="model failed") as raised:
        wrapped.infer({"state": np.zeros(2)})
    assert raised.value is error
    assert wrapped.recorder.count == 0
    assert wrapped.recorder.failed_inferences == 1


def test_offline_save_and_no_overwrite(tmp_path):
    recorder = TrajectoryRecorder(capacity=5, action_dim=2, hz=50)
    recorder.append([1, 2], [3, 4], timestamp=42.0, infer_seconds=0.1)
    recorder.append([2, 3], [4, 5], timestamp=42.02, infer_seconds=0.2)
    path = recorder.save(tmp_path / "record.npz", metadata={"mode": "test"})
    with np.load(path, allow_pickle=False) as saved:
        assert saved["actions"].shape == (2, 2)
        np.testing.assert_allclose(saved["timestamps"], [0, 0.02])
        assert json.loads(str(saved["recording_metadata"]))["metadata"]["mode"] == "test"
    with pytest.raises(FileExistsError):
        recorder.save(path)


def test_no_matplotlib_import_in_runtime_package():
    subprocess.run(
        [sys.executable, "-c", "import sys; import openpi_trajectory_recorder; assert 'matplotlib' not in sys.modules"],
        check=True,
    )


def test_derivatives_of_quadratic():
    t = np.arange(10) / 50
    curves = kinematics((3 * t * t + 2 * t)[:, None], 50)
    np.testing.assert_allclose(curves["velocity"][1][:, 0], 6 * curves["velocity"][0] + 2)
    np.testing.assert_allclose(curves["acceleration"][1][:, 0], 6)
