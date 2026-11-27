"""Immutable trajectory windows adapted from ominf's TrajectoryBuffer pipeline."""

from dataclasses import dataclass

import numpy as np

from .bspline_c2_merge import fit_lsq_bspline
from .bspline_c2_merge import merge_window_trajectories
from .config import BSplineConfig


@dataclass(frozen=True)
class Trajectory:
    start: int
    actions: np.ndarray
    times: np.ndarray
    raw: np.ndarray
    fits: tuple

    velocities: np.ndarray | None = None
    accelerations: np.ndarray | None = None
    diagnostics: dict | None = None

    @property
    def end(self):
        return self.start + len(self.actions) - 1

    def sample(self, step: int) -> np.ndarray:
        if not self.start <= step <= self.end:
            raise TimeoutError("B-spline trajectory exhausted; stop control and reset the policy")
        return self.actions[step - self.start].copy()


def build_trajectory(actions, start: int, config: BSplineConfig, previous=None, anchor=None) -> Trajectory:
    """Fit a chunk, merge at a future anchor and preserve its already planned prefix.

    Time is measured in control ticks. The original crop/C2/must-pass algorithm
    is retained; the switch from the old curve guarantees position matching,
    but does not impose equality of old/new velocity and acceleration.
    """
    raw = np.asarray(actions, dtype=float).copy()
    if raw.ndim != 2 or raw.shape[0] < max(4, config.degree + 1) or raw.shape[1] < 1 or not np.isfinite(raw).all():
        raise ValueError("actions must be finite with horizon >= max(4, degree + 1) and action_dim >= 1")
    if any(i >= raw.shape[1] for i in config.passthrough_indices):
        raise ValueError("passthrough index exceeds action dimension")
    if config.replan_interval + config.effective_lead >= len(raw) - 1:
        raise ValueError("horizon must exceed replan_interval + blend_lead_steps + 1")
    times = np.arange(start, start + len(raw), dtype=float)
    knots = min(config.knots_interior, len(raw) - config.degree - 1)
    fits = tuple(fit_lsq_bspline(times, raw[:, d], knots, degree=config.degree) for d in range(raw.shape[1]))
    curves = list(fits)
    c2_curves = list(fits)
    if previous is not None:
        if raw.shape[1] != previous.raw.shape[1]:
            raise ValueError("action dimension changed between chunks")
        if anchor is None or not max(start, previous.start) < anchor < min(times[-1], previous.end):
            raise TimeoutError("Insufficient overlapping trajectory for B-spline merge")
        previous_times = np.arange(previous.start, previous.end + 1)
        for d, fit in enumerate(fits):
            if d in config.passthrough_indices:
                continue
            stages = merge_window_trajectories(
                previous.times,
                previous.raw[:, d],
                times,
                raw[:, d],
                float(anchor),
                float(np.interp(anchor, previous_times, previous.actions[:, d])),
                knots_interior_merge=config.knots_interior_merge,
                knots_interior=knots,
                degree=config.degree,
                max_deriv=config.max_deriv,
                merge_input_mode=config.merge_input_mode,
                spl1_lsq_fit=previous.fits[d] if config.merge_input_mode == "crop" else None,
                spl2_lsq_fit=fit,
            )
            c2_curves[d], curves[d] = stages[-2:]

    def evaluate(order):
        values = np.zeros_like(raw)
        tail = np.ones(len(times), dtype=bool) if previous is None else times > anchor
        for d, curve in enumerate(curves):
            if d in config.passthrough_indices:
                value = raw[:, d]
                for _ in range(order):
                    value = np.gradient(value, edge_order=2)
                values[:, d] = value
            elif np.any(tail):
                derivative = curve.derivative(order) if order else curve
                if config.interpolate_factor:
                    count = max(2, int(np.ceil(len(raw) * config.interpolate_factor)))
                    grid = np.linspace(times[0] if previous is None else anchor, times[-1], count)
                    values[tail, d] = np.interp(times[tail], grid, derivative(grid, extrapolate=False))
                else:
                    values[tail, d] = derivative(times[tail], extrapolate=False)
        values *= config.fps**order
        if previous is not None:
            old = (previous.actions, previous.velocities, previous.accelerations)[order]
            values[~tail] = old[(times[~tail] - previous.start).astype(int)]
        if not np.isfinite(values).all():
            raise ValueError("B-spline merge produced invalid samples")
        return values

    sampled = evaluate(0)
    velocities = evaluate(1) if config.publish_velocity else None
    accelerations = evaluate(2) if config.publish_acceleration else None
    diagnostics = None
    if config.bspline_infer_plot:
        diagnostics = {
            "times": times,
            "raw": raw,
            "fits": fits,
            "c2": tuple(c2_curves),
            "final": tuple(curves),
            "anchor": anchor,
        }
    return Trajectory(start, sampled, times, raw, fits, velocities, accelerations, diagnostics)
