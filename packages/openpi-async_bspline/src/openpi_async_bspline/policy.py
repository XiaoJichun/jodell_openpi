"""Single-worker inference and smoothing with nonblocking control ticks."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from os import PathLike
import threading
import time

import numpy as np
from openpi_client.base_policy import BasePolicy

from .trajectory import BSplineConfig
from .trajectory import build_trajectory


class AsyncBSplinePolicy(BasePolicy):
    """Replace ActionChunkBroker with this wrapper (do not stack the two).

    Each infer() consumes one logical control tick. First infer() blocks to
    bootstrap; subsequent calls only poll the worker and sample a prepared array.
    Call infer/reset/close from one control thread. The wrapped policy is owned
    by this wrapper and must not be accessed concurrently elsewhere.
    """

    def __init__(self, policy: BasePolicy, config: BSplineConfig | str | PathLike | None = None):
        self._policy = policy
        self.config = (
            BSplineConfig.from_yaml(config) if isinstance(config, (str, PathLike)) else (config or BSplineConfig())
        )
        if not isinstance(self.config, BSplineConfig):
            raise TypeError("config must be BSplineConfig or a YAML path")
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="openpi-async_bspline")
        self._lock = threading.Lock()
        self._step = 0
        self._trajectory = None
        self._future = None
        self._last_request = 0
        self._metadata = {}
        self._closed = False
        self._error = None

        self._episode = 0
        self._diagnostics = []
        self._diagnostics_dropped = 0
        self._published = None
        self._published_count = 0
        self._published_dropped = 0

    def _compute(self, obs, start, previous):
        started = time.perf_counter() if self.config.diagnostics_enabled else 0
        result = self._policy.infer(obs)
        with self._lock:
            anchor = self._step + self.config.effective_lead if previous is not None else None
        trajectory = build_trajectory(result["actions"], start, self.config, previous, anchor)
        if self.config.diagnostics_enabled:
            if len(self._diagnostics) < self.config.diagnostic_capacity:
                self._diagnostics.append((self._episode, start, time.perf_counter() - started, trajectory.diagnostics))
            else:
                self._diagnostics_dropped += 1
        if self.config.bspline_publish_plot and self._published is None:
            self._published = np.zeros((self.config.publish_capacity, 2 + trajectory.actions.shape[1]))
        metadata = {k: v for k, v in result.items() if k != "actions"}
        return trajectory, anchor, metadata

    def _submit(self, obs):
        self._last_request = self._step
        self._future = self._executor.submit(self._compute, deepcopy(obs), self._step, self._trajectory)

    def infer(self, obs: dict) -> dict:
        if self._closed:
            raise RuntimeError("Policy is closed")
        if self._error is not None:
            raise RuntimeError("Asynchronous policy failed; reset before reuse") from self._error
        try:
            if self._trajectory is None and self._future is None:
                self._submit(obs)
            if self._future is not None and (self._trajectory is None or self._future.done()):
                trajectory, anchor, metadata = self._future.result()
                self._future = None
                if anchor is not None and self._step > anchor:
                    raise TimeoutError("B-spline computation missed its splice deadline")
                self._trajectory = trajectory
                self._metadata = metadata
            action = self._trajectory.sample(self._step)
            if self._future is None and self._step - self._last_request >= self.config.replan_interval:
                self._submit(obs)
            output = {**self._metadata, "actions": action}
            if self.config.publish_position:
                output["positions"] = action.copy()
            index = self._step - self._trajectory.start
            if self.config.publish_velocity:
                output["velocities"] = self._trajectory.velocities[index].copy()
            if self.config.publish_acceleration:
                output["accelerations"] = self._trajectory.accelerations[index].copy()
            if self.config.bspline_publish_plot:
                if self._published_count < len(self._published):
                    row = self._published[self._published_count]
                    row[:2] = self._episode, self._step
                    row[2:] = action
                    self._published_count += 1
                else:
                    self._published_dropped += 1
            with self._lock:
                self._step += 1
            return output
        except Exception as exc:
            self._error = exc
            raise

    def reset(self) -> None:
        """Wait for outstanding work before resetting the underlying policy."""
        if self._closed:
            raise RuntimeError("Policy is closed")
        # A running Python thread cannot be cancelled. Queue reset behind it so
        # websocket calls never overlap, and an old result cannot enter a new episode.
        self._executor.submit(self._policy.reset).result()
        self._episode += 1
        self._future = None
        self._trajectory = None
        self._metadata = {}
        self._step = 0
        self._last_request = 0
        self._error = None

    def close(self) -> None:
        """Join the worker; transport ownership remains with the caller."""
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)

    def save_diagnostics(self, path=None):
        """Explicit post-run export. close() must finish all worker writes first."""
        if not self._closed:
            raise RuntimeError("Call close() before save_diagnostics()")
        from .diagnostics import save_diagnostics  # noqa: PLC0415

        return save_diagnostics(self, path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
