"""Single-control-thread, bounded recording. Save only after control has stopped."""

import json
from pathlib import Path
import time

import numpy as np
from openpi_client.base_policy import BasePolicy


class TrajectoryRecorder:
    """Preallocated vectors only; no images, filtering, growing lists or I/O.

    Overflow drops new samples (not old samples), with an observable counter.
    append/clear/save must run on the same control thread. Saved states are
    pre-action positions, not post-action measurements.
    """

    def __init__(self, *, capacity: int, action_dim: int, hz: float, state_dim: int | None = None):
        state_dim = action_dim if state_dim is None else state_dim
        if capacity < 1 or action_dim < 1 or state_dim < 1 or not np.isfinite(hz) or hz <= 0:
            raise ValueError("Dimensions/capacity and finite hz must be positive")
        self.hz = float(hz)
        self.capacity = capacity
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actions = np.empty((capacity, action_dim), dtype=np.float64)
        self.states = np.empty((capacity, state_dim), dtype=np.float64)
        self.infer_seconds = np.empty(capacity, dtype=np.float64)
        self.timestamps = np.empty(capacity, dtype=np.float64)
        # Touch all pages before control starts, rather than on the first write.
        for array in (self.actions, self.states, self.infer_seconds, self.timestamps):
            array.fill(0)
        self.clear()

    def clear(self):
        """Explicitly discard recorded samples; never called by policy.reset()."""
        self.count = 0
        self.dropped = 0
        self.invalid = 0
        self.failed_inferences = 0
        self.last_error = None
        self._origin = None

    @property
    def allocated_bytes(self):
        return sum(a.nbytes for a in (self.actions, self.states, self.infer_seconds, self.timestamps))

    def append(self, actions, states, *, timestamp: float, infer_seconds: float = 0.0):
        """Copy a single tick. Returns False on overflow or invalid vectors.

        Validation of finite values and derivatives happens offline. Conversion
        overhead is lowest for existing NumPy vectors. No automatic CUDA copies.
        """
        if self.count == self.capacity:
            self.dropped += 1
            return False
        try:
            if np.shape(actions) != (self.action_dim,) or np.shape(states) != (self.state_dim,):
                raise ValueError("Expected one action/state vector; wrap outside the action chunk scheduler")
            i = self.count
            self.actions[i] = actions
            self.states[i] = states
            self.infer_seconds[i] = infer_seconds
            if self._origin is None:
                self._origin = timestamp
            self.timestamps[i] = timestamp - self._origin
        except (TypeError, ValueError, RuntimeError) as exc:
            self.invalid += 1
            self.last_error = str(exc)
            return False
        self.count += 1
        return True

    def save(self, path, *, metadata=None):
        """Explicit offline NPZ save; refuses replacement and does no plotting."""
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Use a .npz output path")
        info = {
            "schema_version": 1,
            "hz": self.hz,
            "count": self.count,
            "capacity": self.capacity,
            "dropped": self.dropped,
            "invalid": self.invalid,
            "failed_inferences": self.failed_inferences,
            "last_error": self.last_error,
            "state_semantics": "pre-action positions",
            "timestamp_semantics": "infer call start, monotonic seconds relative to first recorded call",
            "metadata": metadata or {},
        }
        encoded = json.dumps(info)  # Validate metadata before creating a file.
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                actions=self.actions[: self.count],
                states=self.states[: self.count],
                infer_seconds=self.infer_seconds[: self.count],
                timestamps=self.timestamps[: self.count],
                hz=np.asarray(self.hz),
                recording_metadata=np.asarray(encoded),
            )
        return path


class RecordedPolicy(BasePolicy):
    """Wrap the *final per-step* policy, preserving its result and exceptions.

    reset() delegates but retains recorded data. Use a fresh recorder per episode,
    or explicitly clear it after saving. Does not own/close the wrapped policy.
    """

    def __init__(
        self, policy, *, capacity=None, action_dim=None, hz=None, recorder=None, state_key="state", action_key="actions"
    ):
        if recorder is None:
            if capacity is None or action_dim is None or hz is None:
                raise ValueError("Provide recorder or capacity, action_dim and hz")
            recorder = TrajectoryRecorder(capacity=capacity, action_dim=action_dim, hz=hz)
        self.policy = policy
        self.recorder = recorder
        self.state_key = state_key
        self.action_key = action_key
        self._state = np.empty(recorder.state_dim, dtype=np.float64)
        self._state.fill(0)

    def infer(self, obs, **kwargs):
        if self.recorder.count == self.recorder.capacity:
            result = self.policy.infer(obs, **kwargs)
            self.recorder.dropped += 1
            return result
        # Snapshot only the state before infer, since some policies mutate obs.
        valid = True
        try:
            state = obs[self.state_key]
            if np.shape(state) != self._state.shape:
                raise ValueError("State shape does not match recorder")
            self._state[:] = state
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            valid = False
            self.recorder.invalid += 1
            self.recorder.last_error = str(exc)
        start = time.perf_counter()
        try:
            result = self.policy.infer(obs, **kwargs)
        except Exception:
            self.recorder.failed_inferences += 1
            raise
        elapsed = time.perf_counter() - start
        if valid:
            try:
                action = result[self.action_key]
            except (KeyError, TypeError) as exc:
                self.recorder.invalid += 1
                self.recorder.last_error = str(exc)
            else:
                self.recorder.append(action, self._state, timestamp=start, infer_seconds=elapsed)
        return result

    def reset(self):
        self.policy.reset()

    def get_server_metadata(self):
        return self.policy.get_server_metadata()

    def save(self, path, *, metadata=None):
        return self.recorder.save(path, metadata=metadata)
