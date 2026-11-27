"""Client-side scheduling for the ALOHA simulation comparison."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import math
import time

import numpy as np
from openpi_client.base_policy import BasePolicy


class MeasuredPolicy(BasePolicy):
    """Record completed RPC latency; optionally inject reproducible extra latency."""

    def __init__(self, policy, *, delay_ms=0.0, rtc=False):
        self.policy = policy
        self.delay_ms = delay_ms
        self.rtc = rtc
        self.latencies = []

    def infer(self, obs):
        payload = dict(obs)
        payload["enable_rtc"] = self.rtc
        start = time.perf_counter()
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000)
        result = self.policy.infer(payload)
        self.latencies.append(time.perf_counter() - start)
        return result

    def reset(self):
        self.policy.reset()
        self.latencies = []


def checked_actions(result, horizon, action_dim=14):
    actions = np.asarray(result["actions"], dtype=float)
    if actions.shape != (horizon, action_dim) or not np.isfinite(actions).all():
        raise ValueError(f"Expected finite ({horizon}, {action_dim}) action chunk, got {actions.shape}")
    return actions.copy()


class SyncPolicy(BasePolicy):
    def __init__(self, policy, *, interval=15, horizon=50, action_dim=14):
        self.policy = policy
        self.interval = interval
        self.horizon = horizon
        self.action_dim = action_dim
        self.actions = None
        self.index = 0

    def infer(self, obs):
        if self.actions is None or self.index >= self.interval:
            self.actions = checked_actions(self.policy.infer(obs), self.horizon, self.action_dim)
            self.index = 0
        action = self.actions[self.index].copy()
        self.index += 1
        return {"actions": action}

    def reset(self):
        self.policy.reset()
        self.actions = None
        self.index = 0

    def close(self):
        pass


class RTCPolicy(BasePolicy):
    """One outstanding RPC, time-aligned remaining prefix and stale-result checks.

    Prefixes are environment-space absolute actions. The server must advertise
    rtc_prefix_space=environment and apply its training transforms before guidance.
    """

    def __init__(self, policy, *, interval=15, horizon=50, hz=50.0, guidance=0.5, action_dim=14):
        self.policy = policy
        self.interval = interval
        self.horizon = horizon
        self.action_dim = action_dim
        self.hz = hz
        self.guidance = guidance
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aloha-rtc")
        self.future = None
        self.actions = None
        self.start = 0
        self.step = 0
        self.last_request = 0
        self.delay_steps = 1
        self.error = None
        self.closed = False

    def _request(self, payload, start):
        began = time.perf_counter()
        actions = checked_actions(self.policy.infer(payload), self.horizon, self.action_dim)
        return actions, start, time.perf_counter() - began

    def _submit(self, obs):
        payload = deepcopy(obs)
        payload.update(enable_rtc=True, max_guidance_weight=self.guidance)
        if self.actions is not None:
            offset = self.step - self.start
            remaining = self.actions[offset:]
            if not len(remaining):
                raise TimeoutError("RTC trajectory exhausted")
            # Fixed shape avoids recompilation; repeated tail receives zero weight
            # because execute_horizon is limited to the genuinely available prefix.
            prefix = np.concatenate((remaining, np.repeat(remaining[-1:], offset, axis=0)))
            delay = min(self.delay_steps, len(remaining) - 1)
            payload.update(
                prev_action_chunk=prefix,
                rtc_prefix_space="environment",
                inference_delay=delay,
                execute_horizon=min(len(remaining), max(self.interval, delay + 1)),
            )
        self.last_request = self.step
        self.future = self.executor.submit(self._request, payload, self.step)

    def infer(self, obs):
        if self.closed:
            raise RuntimeError("RTC policy is closed")
        if self.error is not None:
            raise RuntimeError("RTC policy failed; reset before reuse") from self.error
        try:
            if self.actions is None and self.future is None:
                self._submit(obs)
            if self.future is not None and (self.actions is None or self.future.done()):
                actions, start, elapsed = self.future.result()
                self.future = None
                if self.step >= start + self.horizon:
                    raise TimeoutError("RTC response arrived after its horizon")
                self.actions, self.start = actions, start
                self.delay_steps = max(1, math.ceil(elapsed * self.hz))
            index = self.step - self.start
            if index >= self.horizon:
                raise TimeoutError("RTC trajectory exhausted")
            action = self.actions[index].copy()
            if self.future is None and self.step - self.last_request >= self.interval:
                self._submit(obs)
            self.step += 1
            return {"actions": action}
        except Exception as exc:
            self.error = exc
            raise

    def reset(self):
        if self.closed:
            raise RuntimeError("RTC policy is closed")
        self.executor.submit(self.policy.reset).result()
        self.future = None
        self.actions = None
        self.start = self.step = self.last_request = 0
        self.delay_steps = 1
        self.error = None

    def close(self):
        if not self.closed:
            self.closed = True
            self.executor.shutdown(wait=True, cancel_futures=True)
