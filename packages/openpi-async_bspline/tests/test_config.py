import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from openpi_async_bspline import AsyncBSplinePolicy
from openpi_async_bspline import BSplineConfig
from openpi_async_bspline import build_trajectory


@pytest.mark.parametrize("layout", ["flat", "bspline", "env"])
def test_yaml_layouts(tmp_path, layout):
    values = {
        "degree": 4,
        "max_deriv": 2,
        "knots_interior": 2,
        "knots_interior_merge": 1,
        "merge_input_mode": "refit",
        "passthrough_indices": [1],
        "replan_interval": 10,
    }
    content = (
        values
        if layout == "flat"
        else ({"bspline": values} if layout == "bspline" else {"env": {"fps": 50, "bspline": values}, "robot": {}})
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(content))
    config = BSplineConfig.from_yaml(path)
    assert config == BSplineConfig.from_dict(values)
    assert config.passthrough_indices == (1,)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("bspline:\n  knots_interor: 3", "Unsupported"),
        ("bspline:\n  publish_velocity: 1", "boolean"),
        ("bspline:\n  degree: 3\n  degree: 4", "Duplicate"),
        ("bspline:\n  max_deriv: 3\n  degree: 3", "max_deriv"),
        ("bspline:\n  degree: true", "degree"),
        ("bspline:\n  merge_input_mode: invalid", "merge_input_mode"),
        ("bspline:\n  passthrough_indices: [1, 1]", "duplicates"),
        ("[]", "mapping"),
        ("bspline: {}\nenv:\n  bspline: {}", "Ambiguous"),
    ],
)
def test_invalid_yaml_rejected(tmp_path, content, message):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        BSplineConfig.from_yaml(path)


def test_unsafe_yaml_rejected(tmp_path):
    path = tmp_path / "unsafe.yaml"
    path.write_text("!!python/object/apply:builtins.str [1]")
    with pytest.raises(yaml.constructor.ConstructorError):
        BSplineConfig.from_yaml(path)


def test_degree_changes_fit_and_refit_knots_affect_merge():
    t = np.arange(50.0) / 49
    raw = (t**5)[:, None]
    low = build_trajectory(raw, 0, BSplineConfig(degree=3, knots_interior=0))
    high = build_trajectory(raw, 0, BSplineConfig(degree=5, knots_interior=0))
    np.testing.assert_allclose(high.actions, raw, atol=1e-12)
    assert np.max(np.abs(low.actions - raw)) > 1e-3
    old = (np.sin(np.arange(50) / 5))[:, None]
    new = (np.cos(np.arange(15, 65) / 6))[:, None]
    merged = []
    for knots in (0, 3):
        config = BSplineConfig(merge_input_mode="refit", knots_interior_merge=knots)
        previous = build_trajectory(old, 0, config)
        current = build_trajectory(new, 15, config, previous, 20)
        np.testing.assert_array_equal(current.actions[:6], previous.actions[15:21])
        assert np.isfinite(current.actions).all()
        merged.append(current.actions)
    assert not np.allclose(merged[0][6:], merged[1][6:])


def test_policy_reads_yaml_only_at_construction(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("bspline:\n  knots_interior: 1")

    class Policy:
        def infer(self, obs):
            return {"actions": np.arange(50.0)[:, None]}

    with AsyncBSplinePolicy(Policy(), path) as policy:
        path.unlink()
        assert policy.config.knots_interior == 1
        for _ in range(3):
            assert policy.infer({})["actions"].shape == (1,)


def test_derivatives_cache_and_fractional_splice():
    t = np.arange(50.0)
    config = BSplineConfig(
        publish_velocity=True, publish_acceleration=True, fps=10, offset_timestep=3.5, knots_interior=0
    )
    assert BSplineConfig.from_dict(config.to_dict()) == config
    old = build_trajectory((t**2)[:, None], 0, config)
    np.testing.assert_allclose(old.velocities[:, 0], 2 * t * 10, atol=1e-9)
    np.testing.assert_allclose(old.accelerations[:, 0], 200, atol=1e-9)
    new = build_trajectory(((t + 15) ** 2 + 2)[:, None], 15, config, old, 20.5)
    for name in ("actions", "velocities", "accelerations"):
        np.testing.assert_array_equal(getattr(new, name)[:6], getattr(old, name)[15:21])
        assert np.isfinite(getattr(new, name)).all()
    cached = build_trajectory((t**2)[:, None], 0, BSplineConfig(interpolate_factor=3))
    assert not np.array_equal(old.actions, cached.actions)
    np.testing.assert_allclose(old.actions, cached.actions, atol=0.03)


def test_complete_yaml_template_and_env_fps(tmp_path):
    template = Path(__file__).parents[1] / "configs/aloha_sim.yaml"
    values = yaml.safe_load(template.read_text())["bspline"]
    original = {
        "degree",
        "max_deriv",
        "knots_interior",
        "knots_interior_merge",
        "merge_input_mode",
        "interpolate_factor",
        "offset_timestep",
        "publish_position",
        "publish_velocity",
        "publish_acceleration",
        "bspline_infer_plot",
        "plots_dir",
        "bspline_publish_plot",
        "timing_print",
        "joint_col",
        "plot_ncol",
        "plot_joint_indices",
        "plot_point_indices",
        "plot_merge_steps",
        "plot_raw",
        "plot_scatter",
        "plot_c2_merge",
        "plot_final",
        "plot_overall",
    }
    assert original <= values.keys()
    values.pop("fps")
    path = tmp_path / "original.yaml"
    path.write_text(yaml.safe_dump({"env": {"fps": 80, "bspline": values}}))
    assert BSplineConfig.from_yaml(path).fps == 80
    with pytest.raises(ValueError, match="conflict"):
        BSplineConfig.from_dict({"offset_timestep": 4, "blend_lead_steps": 3})


def test_diagnostics_bounded_and_offline(tmp_path, capsys):
    class Policy:
        def infer(self, obs):
            return {"actions": np.column_stack((np.arange(50.0) ** 2, np.arange(50.0)))}

        def reset(self):
            pass

    destination = tmp_path / "diagnostics"
    config = BSplineConfig(
        bspline_infer_plot=True,
        bspline_publish_plot=True,
        timing_print=True,
        plots_dir=str(destination),
        diagnostic_capacity=1,
        publish_capacity=4,
        publish_position=False,
        publish_velocity=True,
        publish_acceleration=True,
        plot_joint_indices="0",
        plot_point_indices="0,1,2",
        plot_merge_steps="1",
        plot_scatter=True,
        plot_c2_merge=True,
        plot_overall=True,
    )
    with AsyncBSplinePolicy(Policy(), config) as policy:
        for _ in range(3):
            result = policy.infer({})
            assert "positions" not in result
            assert result["velocities"].shape == (2,)
        policy.reset()
        for _ in range(3):
            policy.infer({})
        assert not destination.exists()
        assert capsys.readouterr().out == ""
        with pytest.raises(RuntimeError, match="close"):
            policy.save_diagnostics()
    policy.save_diagnostics()
    summary = json.loads((destination / "config.json").read_text())
    assert summary["windows"] == 1
    assert summary["windows_dropped"] == 1
    assert summary["published"] == 4
    assert summary["published_dropped"] == 2
    assert len(list(destination.glob("*.pdf"))) == 5
    with np.load(destination / "window_0001_episode_0.npz") as data:
        assert data["final"].shape == (3, 2, 3)
    assert "B-spline worker" in capsys.readouterr().out
