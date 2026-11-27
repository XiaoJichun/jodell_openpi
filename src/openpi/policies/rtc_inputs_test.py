import numpy as np
import pytest

from openpi.policies.rtc_inputs import transform_with_rtc_prefix


def test_environment_prefix_uses_action_transform_and_padding():
    original = np.arange(8, dtype=float).reshape(4, 2)
    obs = {"state": np.array([1.0, 2.0]), "prev_action_chunk": original, "rtc_prefix_space": "environment"}

    def transform(data):
        assert "rtc_prefix_space" not in data
        assert "prev_action_chunk" not in data
        # Stand in for delta-action conversion, normalization and model padding.
        actions = (data["actions"] - data["state"]) / 2
        return {"state": data["state"], "actions": np.pad(actions, ((0, 0), (0, 1)))}

    inputs, prefix = transform_with_rtc_prefix(obs, transform, action_horizon=4, action_dim=3)
    np.testing.assert_array_equal(prefix[:, :2], (original - obs["state"]) / 2)
    np.testing.assert_array_equal(prefix[:, 2], 0)
    assert "actions" not in inputs
    assert "actions" not in obs
    np.testing.assert_array_equal(obs["prev_action_chunk"], original)


def test_legacy_model_space_is_unchanged():
    prefix = np.ones((4, 3))
    _, actual = transform_with_rtc_prefix(
        {"prev_action_chunk": prefix}, lambda data: data, action_horizon=4, action_dim=3
    )
    np.testing.assert_array_equal(prefix, actual)


@pytest.mark.parametrize("prefix", [np.zeros((3, 2)), np.zeros((4, 5)), np.full((4, 2), np.nan)])
def test_bad_environment_prefix_rejected(prefix):
    with pytest.raises(ValueError, match="finite"):
        transform_with_rtc_prefix(
            {"rtc_prefix_space": "environment", "prev_action_chunk": prefix},
            lambda data: data,
            action_horizon=4,
            action_dim=3,
        )
