"""Explicit offline export; this module is never imported during control."""

# Internal implementation of AsyncBSplinePolicy.save_diagnostics.
# ruff: noqa: SLF001

import csv
import json
from pathlib import Path

import numpy as np


def save_diagnostics(policy, path=None):
    config = policy.config
    destination = Path(path or config.plots_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": config.to_dict(),
        "windows": len(policy._diagnostics),
        "windows_dropped": policy._diagnostics_dropped,
        "published": policy._published_count,
        "published_dropped": policy._published_dropped,
    }
    (destination / "config.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (destination / "timings.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window", "episode", "start_tick", "inference_and_smoothing_seconds"])
        for index, (episode, start, elapsed, _) in enumerate(policy._diagnostics, 1):
            writer.writerow([index, episode, start, elapsed])
    if config.timing_print and policy._diagnostics:
        elapsed = [entry[2] for entry in policy._diagnostics]
        print(f"B-spline worker: {len(elapsed)} windows, mean {np.mean(elapsed) * 1000:.3f} ms")
    if not (config.bspline_infer_plot or config.bspline_publish_plot):
        return destination
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = [part.strip() for part in config.joint_col.split(",")]

    def plot_curves(stem, times, series):
        dimension = next(iter(series.values())).shape[1]
        joints = config.indices("plot_joint_indices") or tuple(range(dimension))
        if any(j >= dimension for j in joints):
            raise ValueError("plot_joint_indices exceeds action dimension")
        columns = min(config.plot_ncol, len(joints))
        with PdfPages(destination / f"{stem}.pdf") as pdf:
            for order, quantity in enumerate(("position", "velocity", "acceleration")):
                fig, axes = plt.subplots(
                    int(np.ceil(len(joints) / columns)),
                    columns,
                    figsize=(6 * columns, 3 * int(np.ceil(len(joints) / columns))),
                    squeeze=False,
                )
                for axis, joint in zip(axes.flat, joints, strict=False):
                    for name, data in series.items():
                        axis.plot(times / config.fps, data[:, joint, order], label=name)
                    axis.set_title(labels[joint] if joint < len(labels) and labels[joint] else f"Joint {joint}")
                    axis.set_xlabel("Time (s)")
                    axis.set_ylabel(quantity + ("" if order == 0 else f" / s^{order}"))
                    axis.grid(alpha=0.3)
                    axis.legend()
                for axis in list(axes.flat)[len(joints) :]:
                    axis.set_visible(False)
                fig.tight_layout()
                fig.savefig(destination / f"{stem}_{quantity}.png", dpi=130)
                pdf.savefig(fig)
                plt.close(fig)

    def differences(values):
        result = [values]
        for _ in range(2):
            result.append(
                np.gradient(result[-1], 1 / config.fps, axis=0, edge_order=2)
                if len(values) >= 3
                else np.full_like(values, np.nan)
            )
        return np.stack(result, axis=-1)

    selected = config.indices("plot_merge_steps")
    for index, (episode, _start, _, data) in enumerate(policy._diagnostics, 1):
        if data is None or (selected and index not in selected):
            continue
        times = data["times"]
        # Merged curves are only applicable after the splice anchor.
        mask = np.ones(len(times), dtype=bool) if data["anchor"] is None else times > data["anchor"]
        series = {}
        if config.plot_raw:
            series["raw (finite differences)"] = differences(data["raw"])[mask]
        for enabled, key in ((config.plot_scatter, "fits"), (config.plot_c2_merge, "c2"), (config.plot_final, "final")):
            if enabled:
                series[key] = np.stack(
                    [
                        np.column_stack(
                            [
                                (curve.derivative(order) if order else curve)(times[mask]) * config.fps**order
                                for curve in data[key]
                            ]
                        )
                        for order in range(3)
                    ],
                    axis=-1,
                )
                for joint in config.passthrough_indices:
                    series[key][:, joint] = differences(data["raw"])[mask, joint]
        points = config.indices("plot_point_indices")
        if points:
            # Select original chunk point indices, consistently across all stages.
            chosen = np.isin(np.flatnonzero(mask), points)
            series = {name: values[chosen] for name, values in series.items()}
            mask_indices = np.flatnonzero(mask)[chosen]
        else:
            mask_indices = np.flatnonzero(mask)
        stem = f"window_{index:04d}_episode_{episode}"
        np.savez_compressed(
            destination / f"{stem}.npz", times=times, raw=data["raw"], plot_times=times[mask_indices], **series
        )
        if series and len(mask_indices):
            plot_curves(stem, times[mask_indices], series)
    if config.bspline_publish_plot and policy._published is not None:
        published = policy._published[: policy._published_count]
        np.savez_compressed(destination / "published.npz", samples=published)
        for episode in np.unique(published[:, 0]):
            rows = published[published[:, 0] == episode]
            series = {"executed (finite differences)": differences(rows[:, 2:])}
            plot_curves(f"published_episode_{int(episode)}", rows[:, 1], series)
            if config.plot_overall:
                fig, axes = plt.subplots(3, 1, figsize=(12, 9))
                joints = config.indices("plot_joint_indices") or tuple(range(rows.shape[1] - 2))
                for order, axis in enumerate(axes):
                    for joint in joints:
                        axis.plot(
                            rows[:, 1] / config.fps,
                            series["executed (finite differences)"][:, joint, order],
                            label=f"Joint {joint}",
                        )
                    axis.set_ylabel(("Position", "Velocity", "Acceleration")[order])
                    axis.legend()
                axes[-1].set_xlabel("Time (s)")
                fig.tight_layout()
                fig.savefig(destination / f"overall_episode_{int(episode)}.pdf")
                plt.close(fig)
    return destination
