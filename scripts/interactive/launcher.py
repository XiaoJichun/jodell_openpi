"""Interactive launchers for the local ALOHA inference benchmark."""

# Chinese terminal labels intentionally use full-width punctuation.
# ruff: noqa: RUF001

import argparse
from datetime import UTC
from datetime import datetime
import math
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
USE_TUI = False
UI = None

ENUMS = {
    "target": [("sim", "仿真 — ALOHA"), ("real", "真机 — Y1 SDK / CAN")],
    "camera_type": [("v4l2", "V4L2 USB 相机"), ("orbbec", "Orbbec SDK 相机")],
    "mode": [("sync", "同步 — 阻塞式推理"), ("rtc", "RTC — 实时动作分块"), ("bspline", "B 样条 — 异步轨迹优化")],
    "merge_input_mode": [("crop", "crop — 裁剪已有拟合"), ("refit", "refit — 重新拟合")],
    "render": [("egl", "EGL — GPU 无窗口渲染"), ("osmesa", "OSMesa — CPU 无窗口渲染"), ("glfw", "GLFW — 窗口渲染")],
}
RECORD_OPTIONS = [("record", "内存记录轨迹"), ("plot", "结束后绘图（自动开启记录）"), ("video", "保存录像")]
BSPLINE_GROUPS = {
    "拟合与融合": (
        "degree",
        "max_deriv",
        "knots_interior",
        "knots_interior_merge",
        "merge_input_mode",
        "interpolate_factor",
    ),
    "异步调度与输出": (
        "replan_interval",
        "offset_timestep",
        "fps",
        "passthrough_indices",
        "publish_position",
        "publish_velocity",
        "publish_acceleration",
    ),
    "诊断记录与容量": (
        "bspline_infer_plot",
        "bspline_publish_plot",
        "timing_print",
        "plots_dir",
        "diagnostic_capacity",
        "publish_capacity",
    ),
    "绘图内容与筛选": (
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
    ),
}


def edit_field(values, key, labels):
    current = values[key]
    label = labels.get(key, key)
    if key in ENUMS:
        result = UI.select(label, ENUMS[key], default=current)
    elif isinstance(current, bool):
        result = UI.confirm(label, current)
    else:

        def validate(raw):
            try:
                convert(raw, current)
                return True
            except ValueError as exc:
                return str(exc)

        if key in ("checkpoint", "output", "plots_dir", "hardware_config"):
            result = UI.path(label, str(current))
        else:
            default = str(list(current)) if isinstance(current, tuple) else str(current)
            result = UI.text(label, default, validate=validate)
        if result is not None:
            result = convert(result, current)
    if result is not None:
        values[key] = result


def arrow_menu(title, values, labels, actions):
    if title.startswith("B 样条"):
        options = [(key, key) for key in BSPLINE_GROUPS]
    else:
        hidden = {"record", "plot", "video"}
        if "mode" in values:
            if values.get("target", "sim") == "real":
                hidden.update(("episodes", "seed", "delay_ms", "render"))
            else:
                hidden.update(
                    ("hz", "prompt", "single_arm", "camera_type", "camera_names", "hardware_config", "horizon")
                )
        options = [(key, f"{labels.get(key, key)}：{value}") for key, value in values.items() if key not in hidden]
        if "record" in values:
            enabled = " / ".join(label for key, label in RECORD_OPTIONS if values[key]) or "全部关闭"
            options.append(("record_options", f"记录 / 绘图 / 录像：{enabled}"))
    choice = UI.select(title, options + actions)
    if choice is None:
        return "q"
    if choice in BSPLINE_GROUPS:
        while True:
            field = UI.select(
                choice, [(key, f"{key}：{values[key]}") for key in BSPLINE_GROUPS[choice]] + [("back", "返回参数分类")]
            )
            if field in (None, "back"):
                break
            edit_field(values, field, labels)
        return "edited"
    if choice == "record_options":
        selected = UI.checkbox("记录 / 绘图 / 录像", RECORD_OPTIONS, [key for key, _ in RECORD_OPTIONS if values[key]])
        if selected is not None:
            for key, _ in RECORD_OPTIONS:
                values[key] = key in selected
            if values["plot"]:
                values["record"] = True
        return "edited"
    if choice in values:
        edit_field(values, choice, labels)
        return "edited"
    return choice


def convert(text, current):
    if isinstance(current, bool):
        if text.lower() not in ("true", "false", "1", "0", "yes", "no", "y", "n"):
            raise ValueError("请输入 true/false 或 y/n")
        return text.lower() in ("true", "1", "yes", "y")
    if isinstance(current, int):
        return int(text)
    if isinstance(current, float):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError("必须为有限数值")
        return value
    if isinstance(current, (tuple, list)) or current is None:
        import yaml  # noqa: PLC0415

        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError("列表格式错误，请使用例如 [6,13]") from exc
    return "" if text == '""' else text


def menu(title, values, labels, *, extra="", actions=()):
    if USE_TUI:
        return arrow_menu(title, values, labels, list(actions))
    print(f"\n{title}（输入编号修改；回车保留当前值）")
    keys = list(values)
    for index, key in enumerate(keys, 1):
        print(f"  {index:2}. {labels.get(key, key)} [{key}] = {values[key]}")
    print(extra)
    answer = input("> ").strip().lower()
    if answer.isdigit() and 1 <= int(answer) <= len(keys):
        key = keys[int(answer) - 1]
        entered = input(f"{key} = ").strip()
        if entered:
            try:
                values[key] = convert(entered, values[key])
            except ValueError as exc:
                print(f"输入无效：{exc}")
        return "edited"
    return answer


def positive(values, *keys):
    for key in keys:
        if values[key] < 1:
            raise ValueError(f"{key} 必须大于 0")


def port_check(port):
    if not 1 <= port <= 65535:
        raise ValueError("端口必须在 1–65535")


def launch(command, environment, *, dry_run):
    print("\n启动命令：\n" + shlex.join(command), flush=True)
    print("环境变量：" + shlex.join([f"{k}={v}" for k, v in environment.items()]), flush=True)
    if dry_run:
        print("预览完成，未启动推理进程。")
        return
    os.chdir(ROOT)
    os.execvpe(command[0], command, {**os.environ, **environment})


def server(dry_run):
    values = {
        "gpu": "0",
        "port": 8000,
        "config": "pi0_aloha_sim_rtc",
        "checkpoint": "checkpoints/pi0_aloha_sim",
        "memory_fraction": 0.60,
        "target": "sim",
    }
    labels = {
        "gpu": "GPU 编号（可逗号分隔）",
        "port": "服务端口",
        "config": "模型配置（三模式对比用 pi0_aloha_sim_rtc）",
        "checkpoint": "模型路径或 gs:// 地址",
        "memory_fraction": "JAX 显存预分配比例",
    }
    active_target = "sim"
    labels["target"] = "运行环境"
    while True:
        choice = menu(
            "模型服务端", values, labels, extra="s 启动 | q 退出", actions=[("s", "启动服务端"), ("q", "退出")]
        )
        if values["target"] != active_target:
            if values["target"] not in ("sim", "real"):
                values["target"] = active_target
                print("运行环境应为 sim / real")
                continue
            active_target = values["target"]
            values["config"] = "pi05_rtc_inference_dual_arm" if active_target == "real" else "pi0_aloha_sim_rtc"
            values["checkpoint"] = (
                "checkpoints/pi05_base_full_dual_arm/bi_y1_test_20260410/30000"
                if active_target == "real"
                else "checkpoints/pi0_aloha_sim"
            )
            print("已切换模型配置；请确认 checkpoint 路径。")
        if choice == "q":
            return
        if choice != "s":
            continue
        try:
            port_check(values["port"])
            if not 0 < values["memory_fraction"] <= 1:
                raise ValueError("显存比例必须在 (0, 1]")
            checkpoint = values["checkpoint"]
            if not checkpoint.startswith("gs://") and not (ROOT / checkpoint).is_dir():
                raise ValueError(f"模型目录不存在：{checkpoint}")
            if not values["config"].strip():
                raise ValueError("模型配置不能为空")
            command = [
                sys.executable,
                str(ROOT / "scripts/serve_policy.py"),
                "--port",
                str(values["port"]),
                "policy:checkpoint",
                "--policy.config",
                values["config"],
                "--policy.dir",
                checkpoint,
            ]
            environment = {
                "CUDA_VISIBLE_DEVICES": values["gpu"],
                "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
                "OPENPI_DATA_HOME": str(ROOT / ".cache/openpi"),
                "XLA_PYTHON_CLIENT_MEM_FRACTION": str(values["memory_fraction"]),
            }
            launch(command, environment, dry_run=dry_run)
            return
        except ValueError as exc:
            print(f"配置错误：{exc}")


def edit_bspline(config):
    from openpi_async_bspline import BSplineConfig  # noqa: PLC0415

    values = config.to_dict()
    # Expose the original offset name, not two competing aliases.
    values["offset_timestep"] = float(config.effective_lead)
    values.pop("blend_lead_steps", None)
    while True:
        choice = menu(
            "B 样条全部 YAML 参数",
            values,
            {},
            actions=[("b", "校验并应用参数"), ("q", "放弃本次修改并返回")],
            extra='布尔值输入 true/false；列表如 [6,13]；空字符串输入 ""。\nb 校验并返回 | q 放弃本次修改',
        )
        if choice == "q":
            return config
        if choice == "b":
            try:
                return BSplineConfig.from_dict(values)
            except ValueError as exc:
                print(f"配置错误：{exc}")


def client(dry_run):
    from openpi_async_bspline import BSplineConfig  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    config = BSplineConfig.from_yaml(ROOT / "packages/openpi-async_bspline/configs/aloha_sim.yaml")
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S_%f")
    values = {
        "mode": "bspline",
        "host": "localhost",
        "port": 8000,
        "episodes": 1,
        "steps": 400,
        "seed": 0,
        "interval": 15,
        "guidance": 0.5,
        "delay_ms": 0.0,
        "record": True,
        "plot": True,
        "video": False,
        "mock": False,
        "render": "egl",
        "gpu": "0",
        "output": f"data/aloha_sim/interactive_{stamp}",
        "target": "sim",
        "hz": 50.0,
        "prompt": "Grab the black electrical tape from the left plane to the right plane",
        "single_arm": False,
        "camera_type": "v4l2",
        "camera_names": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "hardware_config": "scripts/inference_python/configs/y1_hardware.yaml",
        "horizon": 50,
    }
    labels = {
        "target": "运行环境",
        "hz": "控制频率 Hz（仿真固定 50）",
        "prompt": "真机任务提示词",
        "single_arm": "单臂（右臂；需匹配的单臂模型）",
        "camera_type": "相机接口",
        "camera_names": "相机名称列表",
        "hardware_config": "CAN / 相机设备 YAML",
        "horizon": "模型动作窗口长度",
        "mode": "模式：sync / rtc / bspline",
        "host": "服务器地址",
        "port": "服务端口",
        "episodes": "回合数",
        "steps": "每回合步数上限",
        "seed": "起始随机种子",
        "interval": "同步/RTC 请求间隔（B 样条使用 YAML）",
        "guidance": "RTC 引导强度",
        "delay_ms": "额外模拟延迟 ms",
        "record": "内存记录轨迹",
        "plot": "回合结束绘图（自动启用记录）",
        "video": "保存视频",
        "mock": "无模型测试",
        "render": "渲染后端：egl / osmesa / glfw",
        "gpu": "客户端 GPU 编号",
        "output": "结果目录",
    }
    active_target = "sim"
    active_single = False
    while True:
        choice = menu(
            "真机客户端（Y1 SDK）" if active_target == "real" else "仿真客户端（固定 50 Hz）",
            values,
            labels,
            extra="y 编辑全部 B 样条参数 | l 加载另一份 YAML | s 启动 | q 退出",
            actions=[("y", "B 样条参数设置"), ("l", "加载 YAML 配置"), ("s", "启动客户端"), ("q", "退出")],
        )
        if values["target"] != active_target:
            if values["target"] not in ("sim", "real"):
                values["target"] = active_target
                print("运行环境应为 sim / real")
                continue
            active_target = values["target"]
            active_single = not values["single_arm"]
            name = "y1_real.yaml" if active_target == "real" else "aloha_sim.yaml"
            config = BSplineConfig.from_yaml(ROOT / "packages/openpi-async_bspline/configs" / name)
            values["hz"] = config.fps
            values["interval"] = config.replan_interval
            values["output"] = f"data/{'real_robot' if active_target == 'real' else 'aloha_sim'}/interactive_{stamp}"
            print(f"已加载 {name}；真机模式每次启动执行一次任务。")
        if active_target == "real" and active_single != values["single_arm"]:
            active_single = values["single_arm"]
            settings = config.to_dict()
            settings["passthrough_indices"] = [6] if active_single else [6, 13]
            config = BSplineConfig.from_dict(settings)
        if choice == "q":
            return
        if choice == "y":
            config = edit_bspline(config)
        elif choice == "l":
            try:
                path = (
                    UI.path("YAML 配置路径", str(ROOT / "packages/openpi-async_bspline/configs/aloha_sim.yaml"))
                    if USE_TUI
                    else input("YAML 路径：").strip()
                )
                if path is None:
                    continue
                config = BSplineConfig.from_yaml(ROOT / Path(path).expanduser())
                print("YAML 已加载到内存。")
            except (OSError, ValueError, yaml.YAMLError) as exc:
                print(f"加载失败：{exc}")
        elif choice == "s":
            try:
                if active_target == "real":
                    command = real_command(values, config, stamp)
                    launch(command, {"CUDA_VISIBLE_DEVICES": values["gpu"]}, dry_run=dry_run)
                    return
                positive(values, "episodes", "steps", "interval")
                port_check(values["port"])
                if values["mode"] not in ("sync", "rtc", "bspline"):
                    raise ValueError("模式应为 sync / rtc / bspline")
                if values["render"] not in ("egl", "osmesa", "glfw"):
                    raise ValueError("渲染后端应为 egl / osmesa / glfw")
                if values["seed"] < 0 or values["delay_ms"] < 0 or values["guidance"] < 0:
                    raise ValueError("种子、延迟及引导强度不能为负")
                if not values["host"].strip() or not values["output"].strip():
                    raise ValueError("地址和输出目录不能为空")
                output = ROOT / values["output"]
                if (output / values["mode"]).exists():
                    raise ValueError("该模式的结果目录已存在，请更换输出目录")
                command = [sys.executable, str(ROOT / "examples/aloha_sim/benchmark.py")]
                for key in ("mode", "host", "port", "episodes", "steps", "seed", "guidance", "delay_ms", "output"):
                    command.extend(["--" + key.replace("_", "-"), str(values[key])])
                if values["mode"] == "bspline":
                    if config.fps != 50 or config.replan_interval + config.effective_lead >= 49:
                        raise ValueError("ALOHA 要求 fps=50 且 replan_interval + offset_timestep < 49")
                    # Preserve the user's source YAML; each run gets its own resolved snapshot.
                    snapshot = output / f"bspline_config_{stamp}.yaml"
                    settings = config.to_dict()
                    settings["plots_dir"] = str(output / "bspline_diagnostics")
                    output.mkdir(parents=True, exist_ok=True)
                    with snapshot.open("x", encoding="utf-8") as stream:
                        yaml.safe_dump({"bspline": settings}, stream, sort_keys=False, allow_unicode=True)
                    command.extend(["--bspline-config", str(snapshot)])
                    print(f"本次配置：{snapshot}\n内部诊断输出：{settings['plots_dir']}")
                else:
                    if values["interval"] + 3 >= 49:
                        raise ValueError("请求间隔必须小于 46")
                    command.extend(["--interval", str(values["interval"])])
                for key, flag in (
                    ("record", "record-trajectories"),
                    ("plot", "plot-trajectories"),
                    ("video", "video"),
                    ("mock", "mock"),
                ):
                    if values[key]:
                        command.append("--" + flag)
                launch(command, {"MUJOCO_GL": values["render"], "CUDA_VISIBLE_DEVICES": values["gpu"]}, dry_run=dry_run)
                return
            except (ValueError, OSError) as exc:
                print(f"配置错误：{exc}")


def real_command(values, config, stamp):
    """Build a real client command without importing any hardware modules."""
    from openpi_async_bspline import BSplineConfig  # noqa: PLC0415
    import yaml  # noqa: PLC0415

    positive(values, "steps", "interval", "horizon")
    port_check(values["port"])
    if values["mode"] not in ("sync", "rtc", "bspline"):
        raise ValueError("Invalid mode")
    if not math.isfinite(values["hz"]) or values["hz"] <= 0 or values["interval"] >= values["horizon"]:
        raise ValueError("Invalid frequency / interval / horizon")
    if values["camera_type"] not in ("v4l2", "orbbec"):
        raise ValueError("Invalid camera type")
    if (
        not isinstance(values["camera_names"], list)
        or not values["camera_names"]
        or any(not isinstance(name, str) or not name for name in values["camera_names"])
    ):
        raise ValueError("camera_names must be a nonempty list of strings")
    if not values["prompt"].strip() or not values["output"].strip():
        raise ValueError("Prompt and output must not be empty")
    output = ROOT / values["output"]
    if (output / values["mode"]).exists():
        raise ValueError("结果已存在，请更换输出目录")
    command = [sys.executable, str(ROOT / "scripts/inference_python/run_policy.py")]
    for key in (
        "mode",
        "host",
        "port",
        "steps",
        "interval",
        "horizon",
        "hz",
        "guidance",
        "prompt",
        "camera_type",
        "hardware_config",
        "output",
    ):
        command.extend(["--" + key.replace("_", "-"), str(values[key])])
    command.extend(["--camera-names", *values["camera_names"]])
    if values["mode"] == "bspline":
        if config.fps != values["hz"] or config.replan_interval + config.effective_lead >= values["horizon"] - 1:
            raise ValueError("YAML fps 必须匹配控制频率，interval + offset < horizon - 1")
        settings = config.to_dict()
        settings["plots_dir"] = str(output / "bspline" / "bspline_diagnostics")
        BSplineConfig.from_dict(settings)
        output.mkdir(parents=True, exist_ok=True)
        snapshot = output / f"bspline_config_{stamp}.yaml"
        with snapshot.open("x", encoding="utf-8") as stream:
            yaml.safe_dump({"bspline": settings}, stream, sort_keys=False)
        command.extend(["--bspline-config", str(snapshot)])
    for key, flag in (
        ("single_arm", "single-arm"),
        ("mock", "mock"),
        ("record", "record-trajectories"),
        ("plot", "plot-trajectories"),
        ("video", "video"),
    ):
        if values[key]:
            command.append("--" + flag)
    return command


def main():
    global USE_TUI, UI  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("--dry-run", action="store_true", help="交互配置后仅保存配置和显示命令，不启动推理")
    parser.add_argument("--plain", action="store_true", help="使用兼容的编号菜单；非 TTY 输入也自动使用此模式")
    args = parser.parse_args()
    USE_TUI = not args.plain and sys.stdin.isatty() and sys.stdout.isatty()
    if USE_TUI:
        try:
            import tui  # noqa: PLC0415
        except ImportError:
            parser.error(
                f"缺少终端依赖，请运行 {sys.executable} -m pip install -r {ROOT / 'scripts/interactive/requirements.txt'}"
            )
        UI = tui
        UI.banner("OpenPI 模型服务端" if args.role == "server" else "OpenPI 推理客户端")
    os.chdir(ROOT)
    try:
        (server if args.role == "server" else client)(args.dry_run)
    except (KeyboardInterrupt, EOFError):
        print("\n已退出。")


if __name__ == "__main__":
    main()
