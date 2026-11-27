"""Offline ALOHA trajectory comparison; matplotlib is an optional dependency."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

MODES = ("sync", "rtc", "bspline")
COLORS = ("#2563eb", "#e67e22", "#08916e")
METRICS = ("position", "velocity", "acceleration")
SOURCES = ("actions", "states")
JOINTS = tuple(
    f"{arm} {joint}"
    for arm in ("Left", "Right")
    for joint in ("waist", "shoulder", "elbow", "forearm roll", "wrist angle", "wrist rotate", "gripper")
)


def differentiate(positions, hz):
    """Return (time, value) pairs, without filtering or extrapolating endpoints.

    First differences live at sample midpoints; second differences at interior
    position timestamps. These are discrete estimates, not analytic derivatives.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] < 3 or not np.isfinite(positions).all():
        raise ValueError("Positions must be a finite (N >= 3, D) array")
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be finite and positive")
    time = np.arange(len(positions)) / hz
    velocity = np.diff(positions, axis=0) * hz
    acceleration = np.diff(positions, n=2, axis=0) * hz**2
    return {
        "position": (time, positions.copy()),
        "velocity": ((time[:-1] + time[1:]) / 2, velocity),
        "acceleration": (time[1:-1], acceleration),
    }


def _load(root, modes, seeds, hz):
    available = {}
    rates = []
    for mode in modes:
        directory = root / mode
        available[mode] = {int(p.stem.removeprefix("seed_")) for p in directory.glob("seed_*.npz")}
        if hz is None:
            config = json.loads((directory / "config.json").read_text())
            rates.append(float(config["args"]["hz"]))
    if hz is None:
        if len(set(rates)) != 1:
            raise ValueError("Modes must use the same sampling rate")
        hz = rates[0]
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be finite and positive")
    selected = sorted(set.union(*available.values())) if seeds is None else sorted(set(seeds))
    if not selected:
        raise ValueError("No seed_N.npz traces found")
    traces = {}
    for seed in selected:
        for mode in modes:
            if seed not in available[mode]:
                raise ValueError(f"Missing seed {seed} in {mode}; select shared seeds explicitly")
            with np.load(root / mode / f"seed_{seed}.npz", allow_pickle=False) as archive:
                for source in SOURCES:
                    values = np.asarray(archive[source], dtype=float)
                    if values.ndim != 2 or values.shape[1] != 14:
                        raise ValueError(f"{mode}/{seed}/{source}: expected ALOHA (N, 14) positions")
                    differentiate(values, hz)  # Validate before creating output files.
                    traces[seed, mode, source] = values.copy()
            if len(traces[seed, mode, "actions"]) != len(traces[seed, mode, "states"]):
                raise ValueError(f"{mode}/{seed}: action/state lengths differ")
    return selected, hz, traces


def _unit(joint, metric):
    base = "normalized" if joint in (6, 13) else "rad"
    return base + {"position": "", "velocity": "/s", "acceleration": "/s²"}[metric]


def _viewer(output, seeds):
    options = "".join(f'<option value="{seed}">Seed {seed}</option>' for seed in seeds)
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>轨迹对比 · OpenPI</title><style>
body{font:16px system-ui;margin:24px auto;max-width:1400px;padding:0 20px;background:#f5f7fb;color:#172338}
select,a{margin:6px;padding:8px}img{width:100%;background:white;border-radius:12px}
p{line-height:1.7}nav{position:sticky;top:0;background:#f5f7fb;padding:12px 0}
</style><h1>同步 / RTC / 异步 B 样条：全关节曲线</h1>
<p>每张图包含双臂 12 个关节和 2 个夹爪。速度与加速度来自未滤波位置差分，
并非直接记录的传感器速度/加速度。横轴为名义仿真时间；阻塞推理时物理仿真暂停。
不同模式按种子和起始时刻对齐，不进行任务阶段对齐，也不补齐提前结束的轨迹。
观测位置记录于动作施加前；夹爪为归一化开合量。</p>
<nav><label>回合 <select id="seed">SEED_OPTIONS</select></label>
<label>数据 <select id="source"><option value="actions">动作指令</option><option value="states">观测位置及其差分</option></select></label>
<label>曲线 <select id="metric"><option value="position">位置</option><option value="velocity">速度</option><option value="acceleration">加速度</option></select></label>
<a id="png">下载 PNG</a><a id="pdf">下载本回合 PDF</a><a href="metrics.csv">指标 CSV</a>
<a href="manifest.json">数据说明</a></nav><img id="plot" alt="全部14个维度的三模式对比曲线">
<p>指标 CSV 仅使用同一种子三个模式共有的时间前缀，逐关节报告 RMS 和绝对峰值。
位置 RMS 不是控制效果评分。原始时间序列及差分值可在各回合目录的 NPZ 中读取。</p>
<script>
const seed=document.getElementById('seed'), source=document.getElementById('source'), metric=document.getElementById('metric');
function update(){const base=`seed_${seed.value}/${source.value}`;
 document.getElementById('plot').src=`${base}_${metric.value}.png`;
 document.getElementById('png').href=`${base}_${metric.value}.png`;
 document.getElementById('pdf').href=`${base}.pdf`;}
[seed,source,metric].forEach(el=>el.addEventListener('change',update));update();
</script></html>"""
    (output / "index.html").write_text(page.replace("SEED_OPTIONS", options), encoding="utf-8")


def build_report(input_dir, output_dir, *, modes=MODES, seeds=None, hz=None, dpi=130):
    """Create all 14-DOF plots, PDFs, derived NPZ arrays and common-prefix metrics.

    Expects mode/config.json and mode/seed_N.npz (actions and states). Refuses to
    overwrite a report. Returns the output directory; does not require a GPU.
    """
    # Keep plotting dependencies off the inference import path.
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    root, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    modes = tuple(modes)
    if len(modes) < 2 or len(set(modes)) != len(modes):
        raise ValueError("Provide at least two distinct modes")
    if dpi < 50:
        raise ValueError("dpi must be at least 50")
    seeds, hz, traces = _load(root, modes, seeds, hz)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    metadata = []
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    for seed in seeds:
        directory = output / f"seed_{seed}"
        directory.mkdir()
        common = min(len(traces[seed, mode, "actions"]) for mode in modes)
        metadata.append(
            {
                "seed": seed,
                "common_samples": common,
                "samples": {mode: len(traces[seed, mode, "actions"]) for mode in modes},
            }
        )
        for source in SOURCES:
            curves = {mode: differentiate(traces[seed, mode, source], hz) for mode in modes}
            for mode in modes:
                arrays = {
                    f"{metric}_{suffix}": values
                    for metric, pair in curves[mode].items()
                    for suffix, values in zip(("time", "values"), pair, strict=True)
                }
                np.savez_compressed(directory / f"{source}_{mode}.npz", **arrays)
                for metric, (_time, values) in differentiate(traces[seed, mode, source][:common], hz).items():
                    for joint, name in enumerate(JOINTS):
                        rows.append(
                            {
                                "seed": seed,
                                "mode": mode,
                                "source": source,
                                "joint": name,
                                "metric": metric,
                                "unit": _unit(joint, metric),
                                "common_end_s": (common - 1) / hz,
                                "rms": float(np.sqrt(np.mean(values[:, joint] ** 2))),
                                "max_abs": float(np.max(np.abs(values[:, joint]))),
                            }
                        )
            with PdfPages(directory / f"{source}.pdf") as pdf:
                for metric in METRICS:
                    fig, axes = plt.subplots(7, 2, figsize=(15, 19), sharex=True, layout="constrained")
                    # Left arm in left column, right arm in right column.
                    for joint in range(14):
                        ax = axes[joint % 7, joint // 7]
                        for index, mode in enumerate(modes):
                            time, values = curves[mode][metric]
                            ax.plot(
                                time,
                                values[:, joint],
                                label=mode,
                                color=COLORS[index % len(COLORS)],
                                linewidth=1.0,
                                alpha=0.9,
                            )
                        ax.set_title(JOINTS[joint], loc="left", fontweight="bold")
                        ax.set_ylabel(_unit(joint, metric))
                        ax.grid(alpha=0.2)
                        ax.set_xlim(0, max(curves[mode]["position"][0][-1] for mode in modes))
                        if joint % 7 == 6:
                            ax.set_xlabel("Nominal simulation time (s)")
                    axes[0, 0].legend(loc="best", ncol=min(3, len(modes)))
                    kind = "Command" if source == "actions" else "Observed position / finite differences"
                    fig.suptitle(
                        f"Seed {seed} | {kind} | {metric.capitalize()} | {hz:g} Hz\n"
                        "Raw differences, no smoothing; each trace stops at its own episode end",
                        fontsize=15,
                    )
                    try:
                        fig.savefig(directory / f"{source}_{metric}.png", dpi=dpi)
                        pdf.savefig(fig)
                    finally:
                        plt.close(fig)
        print(f"Plotted seed {seed}: all 14 dimensions, commands and observations", flush=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "input": str(root),
        "hz": hz,
        "modes": modes,
        "seeds": seeds,
        "episodes": metadata,
        "derivatives": "Unfiltered forward differences; velocity at midpoints, acceleration at interior timestamps",
        "time_axis": "Nominal simulation time; NOT wall-clock time",
        "states": "Pre-action positions",
        "grippers": "Indices 6 and 13 are normalized; other joints are radians",
        "metrics": "Common prefix per seed; no extrapolation, phase alignment or seed averaging",
        "png_count": len(seeds) * 6,
        "pdf_count": len(seeds) * 2,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _viewer(output, seeds)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directory containing sync/, rtc/, bspline/ traces")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=MODES)
    parser.add_argument("--seeds", nargs="+", type=int, help="Default: all seeds; missing pairs raise an error")
    parser.add_argument("--hz", type=float, help="Explicit rate override; default: read each config.json")
    parser.add_argument("--dpi", type=int, default=130)
    args = parser.parse_args()
    output = build_report(args.input, args.output, modes=args.modes, seeds=args.seeds, hz=args.hz, dpi=args.dpi)
    print(f"Report: {output / 'index.html'}")


if __name__ == "__main__":
    main()
