"""Hardware-free integration of the real-client scheduling path."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["sync", "rtc", "bspline"])
def test_mock_real_pipeline(mode, tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/inference_python/run_policy.py"),
            "--mock",
            "--mode",
            mode,
            "--steps",
            "35",
            "--prompt",
            "test",
            "--output",
            str(tmp_path),
            "--record-trajectories",
        ],
        check=True,
        timeout=30,
    )
    with np.load(tmp_path / mode / "trajectory.npz") as data:
        assert data["actions"].shape == (35, 14)
        assert np.isfinite(data["actions"]).all()


def test_bad_single_arm_config_fails_before_hardware_import(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/inference_python/run_policy.py"),
            "--single-arm",
            "--mode",
            "bspline",
            "--prompt",
            "test",
            "--output",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert "Single arm requires" in result.stderr
    assert "y1_sdk" not in result.stderr


def test_single_arm_bspline_with_matching_yaml(tmp_path):
    import yaml  # noqa: PLC0415

    config = tmp_path / "single.yaml"
    config.write_text(yaml.safe_dump({"fps": 30, "replan_interval": 8, "passthrough_indices": [6]}))
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/inference_python/run_policy.py"),
            "--mock",
            "--single-arm",
            "--steps",
            "30",
            "--prompt",
            "test",
            "--output",
            str(tmp_path / "run"),
            "--bspline-config",
            str(config),
            "--record-trajectories",
        ],
        check=True,
        timeout=30,
    )
    with np.load(tmp_path / "run/bspline/trajectory.npz") as data:
        assert data["actions"].shape == (30, 7)
