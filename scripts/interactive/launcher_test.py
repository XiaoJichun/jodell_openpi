"""Exercise the terminal entry points without starting GPU workloads."""

import os
from pathlib import Path
import subprocess
import sys

from openpi_async_bspline import BSplineConfig
import yaml

ROOT = Path(__file__).resolve().parents[2]


def run_launcher(role, commands, cwd):
    env = {**os.environ, f"OPENPI_{role.upper()}_PYTHON": sys.executable}
    result = subprocess.run(
        ["bash", str(ROOT / f"scripts/start_{role}.sh"), "--dry-run"],
        input="\n".join(commands) + "\n",
        text=True,
        capture_output=True,
        cwd=cwd,
        env=env,
        timeout=20,
        check=True,
    )
    assert not result.stderr
    return result.stdout


def test_server_rejects_bad_port_and_preserves_path_as_argument(tmp_path):
    checkpoint = tmp_path / "checkpoint space $(test)"
    checkpoint.mkdir()
    output = run_launcher("server", ["2", "99999", "s", "2", "8010", "4", str(checkpoint), "s"], tmp_path)
    assert "端口必须" in output
    assert "--port 8010" in output
    assert "'" + str(checkpoint) + "'" in output
    assert "未启动推理进程" in output


def test_client_edits_yaml_and_saves_separate_snapshot(tmp_path):
    original = ROOT / "packages/openpi-async_bspline/configs/aloha_sim.yaml"
    before = original.read_bytes()
    values = BSplineConfig.from_yaml(original).to_dict()
    offset_index = list(values).index("offset_timestep") + 1
    passthrough_index = list(values).index("passthrough_indices") + 1
    destination = tmp_path / "results with spaces"
    output = run_launcher(
        "client",
        ["y", str(passthrough_index), "[bad", str(offset_index), "4.5", "b", "16", str(destination), "s"],
        tmp_path,
    )
    assert "列表格式错误" in output
    assert "--mode bspline" in output
    assert "--plot-trajectories" in output
    (snapshot,) = destination.glob("bspline_config_*.yaml")
    config = BSplineConfig.from_yaml(snapshot)
    assert config.offset_timestep == 4.5
    assert config.plots_dir == str(destination / "bspline_diagnostics")
    assert original.read_bytes() == before
    assert not (destination / "bspline").exists()
    assert yaml.safe_load(snapshot.read_text())["bspline"]["fps"] == 50


def test_sync_does_not_pass_bspline_config(tmp_path):
    output = run_launcher("client", ["1", "sync", "16", str(tmp_path / "sync"), "s"], tmp_path)
    assert "--mode sync" in output
    assert "--interval 15" in output
    assert "--bspline-config" not in output


class FakeUI:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.options = []

    def select(self, title, options, default=None):
        self.options.append(options)
        return next(self.answers)

    def checkbox(self, title, options, defaults):
        self.options.append(options)
        return next(self.answers)

    def confirm(self, title, default):
        return next(self.answers)


def test_arrow_mode_choice_and_cancel(monkeypatch):
    import launcher  # noqa: PLC0415

    ui = FakeUI(["mode", "rtc", "mode", None, None])
    monkeypatch.setattr(launcher, "UI", ui)
    values = {"mode": "bspline"}
    assert launcher.arrow_menu("客户端", values, {}, [("q", "退出")]) == "edited"
    assert values["mode"] == "rtc"
    assert [key for key, _ in ui.options[1]] == ["sync", "rtc", "bspline"]
    launcher.arrow_menu("客户端", values, {}, [])
    assert values["mode"] == "rtc"
    assert launcher.arrow_menu("客户端", values, {}, []) == "q"


def test_checkbox_enables_record_for_plot_and_can_clear_all(monkeypatch):
    import launcher  # noqa: PLC0415

    monkeypatch.setattr(launcher, "UI", FakeUI(["record_options", ["plot", "video"], "record_options", []]))
    values = {"record": False, "plot": False, "video": False}
    launcher.arrow_menu("客户端", values, {}, [])
    assert values == {"record": True, "plot": True, "video": True}
    launcher.arrow_menu("客户端", values, {}, [])
    assert values == {"record": False, "plot": False, "video": False}


def test_yaml_groups_cover_every_editable_field_and_cancel_discards(monkeypatch):
    import launcher  # noqa: PLC0415

    config = BSplineConfig()
    fields = set(config.to_dict()) - {"blend_lead_steps"}
    fields.add("offset_timestep")
    grouped = [key for group in launcher.BSPLINE_GROUPS.values() for key in group]
    assert set(grouped) == fields
    assert len(grouped) == len(fields)
    ui = FakeUI(["异步调度与输出", "publish_velocity", True, "back", "q"])
    monkeypatch.setattr(launcher, "UI", ui)
    monkeypatch.setattr(launcher, "USE_TUI", True)
    assert launcher.edit_bspline(config) == config


def test_real_switch_routes_to_sdk_client(tmp_path):
    destination = tmp_path / "real"
    output = run_launcher("client", ["17", "real", "16", str(destination), "13", "true", "s"], tmp_path)
    assert "scripts/inference_python/run_policy.py" in output
    assert "--hz 30.0" in output
    assert "--mock" in output
    (snapshot,) = destination.glob("bspline_config_*.yaml")
    assert BSplineConfig.from_yaml(snapshot).fps == 30
    assert "benchmark.py" not in output


def test_real_server_switch_presets(tmp_path):
    checkpoint = tmp_path / "real-checkpoint"
    checkpoint.mkdir()
    output = run_launcher("server", ["6", "real", "4", str(checkpoint), "s"], tmp_path)
    assert "--policy.config pi05_rtc_inference_dual_arm" in output


def test_custom_launcher_yaml_and_target(tmp_path):
    config = yaml.safe_load((ROOT / "scripts/inference_python/configs/client.yaml").read_text())
    config["default_target"] = "real"
    config["profiles"]["real"].update(host="192.168.1.20", steps=90, output=str(tmp_path / "run"), video=True)
    path = tmp_path / "client.yaml"
    path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/start_client.sh"), "--config", str(path), "--dry-run"],
        input="s\n",
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "OPENPI_CLIENT_PYTHON": sys.executable},
        timeout=20,
    )
    assert "--host 192.168.1.20" in result.stdout
    assert "--steps 90" in result.stdout
    assert "--video" in result.stdout
    assert "run_policy.py" in result.stdout
    (snapshot,) = (tmp_path / "run").glob("launcher_*.yaml")
    assert yaml.safe_load(snapshot.read_text())["settings"]["target"] == "real"


def test_launcher_rejects_unknown_or_wrong_type(tmp_path):
    from configuration import load_profiles  # noqa: PLC0415
    import pytest  # noqa: PLC0415

    config = yaml.safe_load((ROOT / "scripts/inference_python/configs/client.yaml").read_text())
    config["common"]["steps"] = "400"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="steps must be int"):
        load_profiles("client", path)
    config["common"]["steps"] = 400
    config["common"]["typo"] = True
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="unknown"):
        load_profiles("client", path)


def test_start_scripts_use_activated_environment(tmp_path):
    env_dir = tmp_path / "activated environment"
    bin_dir = env_dir / "bin"
    bin_dir.mkdir(parents=True)
    interpreter = bin_dir / "python"
    interpreter.write_text('#!/bin/sh\nprintf "activated-python %s\\n" "$@"\n')
    interpreter.chmod(0o755)
    for role in ("server", "client"):
        env = {k: v for k, v in os.environ.items() if k != f"OPENPI_{role.upper()}_PYTHON"}
        env["VIRTUAL_ENV"] = str(env_dir)
        result = subprocess.run(
            ["bash", str(ROOT / f"scripts/start_{role}.sh"), "--dry-run"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "activated-python --dry-run" in result.stdout
        assert f"activated-python {role}" in result.stdout


def test_start_scripts_require_activation():
    for role in ("server", "client"):
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", f"OPENPI_{role.upper()}_PYTHON")}
        result = subprocess.run(
            ["bash", str(ROOT / f"scripts/start_{role}.sh"), "--dry-run"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1
        assert "source .venv-" in result.stderr
