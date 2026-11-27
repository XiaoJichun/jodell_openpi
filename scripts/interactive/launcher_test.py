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
