"""Offline plotting of one or more saved recordings. Never imported by infer()."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def kinematics(positions, hz):
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or len(positions) < 3 or not np.isfinite(positions).all():
        raise ValueError("Need at least three finite position vectors")
    if not np.isfinite(hz) or hz <= 0:
        raise ValueError("hz must be finite and positive")
    time = np.arange(len(positions)) / hz
    return {
        "position": (time, positions),
        "velocity": ((time[:-1] + time[1:]) / 2, np.diff(positions, axis=0) * hz),
        "acceleration": (time[1:-1], np.diff(positions, n=2, axis=0) * hz**2),
    }


def analyze(paths, output, *, labels=None, hz=None):
    """Compare full traces; compute metrics only over their common time prefix.

    Accepts recorder NPZs or legacy benchmark NPZs (the latter require hz).
    For D=14 assumes ALOHA: grippers 6/13 normalized, other positions radians.
    Other dimensions use generic position units. No filtering or extrapolation.
    """
    paths = [Path(p) for p in paths]
    labels = list(labels) if labels is not None else [f"trace_{i + 1}" for i in range(len(paths))]
    if not paths or len(labels) != len(paths) or len(set(labels)) != len(labels):
        raise ValueError("Provide one unique label per recording")
    records, rates, info = [], [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            rate = float(saved["hz"]) if hz is None else float(hz)
            metadata = json.loads(str(saved["recording_metadata"])) if "recording_metadata" in saved else {}
            if metadata.get("invalid", 0) or metadata.get("failed_inferences", 0):
                raise ValueError(f"{path}: missing/invalid samples; uniform-time derivatives would be misleading")
            values = {source: saved[source].copy() for source in ("actions", "states")}
        for v in values.values():
            kinematics(v, rate)
        records.append(values)
        rates.append(rate)
        info.append({"path": str(path.resolve()), **metadata})
    if len(set(rates)) != 1:
        raise ValueError("Recordings must have the same nominal frequency")
    for source in ("actions", "states"):
        if len({r[source].shape[1] for r in records}) != 1:
            raise ValueError("Compared vectors must have matching dimensions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    rows = []
    with PdfPages(output / "curves.pdf") as pdf:
        for source in ("actions", "states"):
            curves = [kinematics(r[source], rates[0]) for r in records]
            common = min(len(r[source]) for r in records)
            dimensions = records[0][source].shape[1]
            for metric in ("position", "velocity", "acceleration"):
                fig, axes = plt.subplots(
                    (dimensions + 1) // 2,
                    2,
                    squeeze=False,
                    figsize=(14, max(4, 2.3 * ((dimensions + 1) // 2))),
                    sharex=True,
                    layout="constrained",
                )
                for joint, ax in enumerate(axes.flat):
                    if joint >= dimensions:
                        ax.set_visible(False)
                        continue
                    unit = (
                        "normalized"
                        if dimensions == 14 and joint in (6, 13)
                        else ("rad" if dimensions == 14 else "position unit")
                    )
                    unit += {"position": "", "velocity": "/s", "acceleration": "/s²"}[metric]
                    for label, curve in zip(labels, curves, strict=True):
                        time, values = curve[metric]
                        ax.plot(time, values[:, joint], label=label, linewidth=1)
                    ax.set_title(f"DOF {joint}" + (" (gripper)" if dimensions == 14 and joint in (6, 13) else ""))
                    ax.set_ylabel(unit)
                    ax.set_xlabel("Nominal time (s)")
                    ax.grid(alpha=0.2)
                    if joint == 0:
                        ax.legend()
                    for label, record in zip(labels, records, strict=True):
                        _time, values = kinematics(record[source][:common], rates[0])[metric]
                        rows.append(
                            {
                                "label": label,
                                "source": source,
                                "joint": joint,
                                "metric": metric,
                                "unit": unit,
                                "common_end_s": (common - 1) / rates[0],
                                "rms": float(np.sqrt(np.mean(values[:, joint] ** 2))),
                                "peak_abs": float(np.max(np.abs(values[:, joint]))),
                            }
                        )
                fig.suptitle(
                    f"{source} | {metric} | {rates[0]:g} Hz\n"
                    "Unfiltered differences; state is pre-action; traces end independently"
                )
                try:
                    fig.savefig(output / f"{source}_{metric}.png", dpi=130)
                    pdf.savefig(fig)
                finally:
                    plt.close(fig)
    with (output / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "analysis.json").write_text(
        json.dumps(
            {
                "recordings": info,
                "labels": labels,
                "hz": rates[0],
                "notes": [
                    "Velocity/acceleration are position differences, not measured velocity/acceleration.",
                    "Time is nominal, not wall-clock. Metrics use the common time prefix.",
                    "dropped > 0 means the saved trace ends at recorder capacity.",
                    "No phase alignment or smoothing; position RMS is not a task score.",
                ],
            },
            indent=2,
        )
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--hz", type=float, help="Nominal rate override, required for legacy benchmark files")
    args = parser.parse_args()
    print(analyze(args.recordings, args.output, labels=args.labels, hz=args.hz))


if __name__ == "__main__":
    main()
