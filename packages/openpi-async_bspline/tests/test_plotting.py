import json

import numpy as np
import pytest

from openpi_async_bspline.plotting import _load
from openpi_async_bspline.plotting import differentiate


def test_quadratic_derivatives_and_time_alignment():
    time = np.arange(8) / 50
    position = np.column_stack((3 * time**2 + 2 * time + 1, -time))
    curves = differentiate(position, 50)
    tv, velocity = curves["velocity"]
    ta, acceleration = curves["acceleration"]
    np.testing.assert_allclose(velocity[:, 0], 6 * tv + 2)
    np.testing.assert_allclose(velocity[:, 1], -1)
    np.testing.assert_allclose(acceleration[:, 0], 6)
    np.testing.assert_allclose(acceleration[:, 1], 0, atol=1e-12)
    np.testing.assert_allclose(ta, time[1:-1])
    np.testing.assert_array_equal(curves["position"][1], position)


@pytest.mark.parametrize(
    ("values", "hz"),
    [(np.zeros((2, 14)), 50), (np.full((3, 14), np.nan), 50), (np.zeros((3, 14)), 0), (np.zeros((3, 14)), np.inf)],
)
def test_invalid_trace_rejected(values, hz):
    with pytest.raises(ValueError, match="finite"):
        differentiate(values, hz)


def test_missing_pairs_and_rate_mismatch_rejected(tmp_path):
    for mode in ["sync", "rtc"]:
        directory = tmp_path / mode
        directory.mkdir()
        (directory / "config.json").write_text(json.dumps({"args": {"hz": 50}}))
        np.savez(directory / "seed_0.npz", actions=np.zeros((4, 14)), states=np.zeros((4, 14)))
    np.savez(tmp_path / "sync/seed_1.npz", actions=np.zeros((3, 14)), states=np.zeros((3, 14)))
    with pytest.raises(ValueError, match="Missing seed 1"):
        _load(tmp_path, ("sync", "rtc"), None, None)
    seeds, hz, traces = _load(tmp_path, ("sync", "rtc"), [0], None)
    assert seeds == [0]
    assert hz == 50
    assert len(traces) == 4
    (tmp_path / "rtc/config.json").write_text(json.dumps({"args": {"hz": 25}}))
    with pytest.raises(ValueError, match="same sampling rate"):
        _load(tmp_path, ("sync", "rtc"), [0], None)
