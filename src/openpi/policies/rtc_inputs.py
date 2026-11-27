"""Explicit opt-in conversion of RTC prefixes from environment to model space."""

from copy import deepcopy

import numpy as np


def transform_with_rtc_prefix(obs, transform, *, action_horizon: int, action_dim: int):
    """Transform a prefix through the same action pipeline used for training.

    Older clients keep passing already-normalized model-space prefixes. New
    clients request environment-space conversion explicitly; no heuristics are
    used to infer action units. The caller's observations remain unchanged.
    """
    space = obs.get("rtc_prefix_space", "model")
    if space not in ("model", "environment"):
        raise ValueError("rtc_prefix_space must be 'model' or 'environment'")
    inputs = deepcopy(obs)
    prefix = obs.get("prev_action_chunk")
    convert = space == "environment" and prefix is not None
    if convert:
        prefix = np.asarray(prefix, dtype=np.float32)
        if (
            prefix.ndim != 2
            or prefix.shape[0] != action_horizon
            or not 0 < prefix.shape[1] <= action_dim
            or not np.isfinite(prefix).all()
        ):
            raise ValueError(
                "Environment RTC prefix must be finite with shape (action_horizon, action_dim <= model dim)"
            )
        inputs["actions"] = prefix.copy()
    # These are sampling options, not observation tensors. In particular the
    # string-valued space marker must never reach jnp.asarray in Policy.infer.
    inputs.pop("rtc_prefix_space", None)
    inputs.pop("prev_action_chunk", None)
    inputs = transform(inputs)
    if convert:
        prefix = np.asarray(inputs.pop("actions"), dtype=np.float32)
        if prefix.shape != (action_horizon, action_dim) or not np.isfinite(prefix).all():
            raise ValueError("Transformed RTC prefix must match the model's full action shape")
    return inputs, prefix
