"""Validated launcher defaults, separate from robot hardware and algorithm settings."""

from copy import deepcopy
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SERVER = {"gpu": str, "port": int, "config": str, "checkpoint": str, "memory_fraction": float, "target": str}
CLIENT = {
    "mode": str,
    "host": str,
    "port": int,
    "episodes": int,
    "steps": int,
    "seed": int,
    "interval": int,
    "guidance": float,
    "delay_ms": float,
    "record": bool,
    "plot": bool,
    "video": bool,
    "mock": bool,
    "render": str,
    "gpu": str,
    "output": str,
    "target": str,
    "hz": float,
    "prompt": str,
    "single_arm": bool,
    "camera_type": str,
    "camera_names": list,
    "hardware_config": str,
    "horizon": int,
    "bspline_config": str,
}


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Non-string or duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def load_profiles(role, path=None):
    path = ROOT / (path or f"scripts/inference_python/configs/{role}.yaml")
    try:
        data = yaml.load(path.read_text(), Loader=UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {path}: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"default_target", "common", "profiles"}:
        raise ValueError("Launcher YAML requires exactly default_target, common, profiles")
    if data["default_target"] not in ("sim", "real"):
        raise ValueError("default_target must be sim or real")
    if (
        not isinstance(data["common"], dict)
        or not isinstance(data["profiles"], dict)
        or set(data["profiles"]) != {"sim", "real"}
    ):
        raise ValueError("common must be a mapping; profiles requires sim and real")
    schema = SERVER if role == "server" else CLIENT
    for target in ("sim", "real"):
        if not isinstance(data["profiles"][target], dict):
            raise ValueError(f"profiles.{target} must be a mapping")
        if "target" in data["common"] or "target" in data["profiles"][target]:
            raise ValueError("Use default_target, not a target field in common/profiles")
        merged = {**data["common"], **data["profiles"][target], "target": target}
        if set(merged) != set(schema):
            raise ValueError(
                f"Invalid {role}/{target} fields: missing={set(schema) - set(merged)}, unknown={set(merged) - set(schema)}"
            )
        for key, kind in schema.items():
            value = merged[key]
            valid = type(value) is kind
            if kind is float:
                valid = type(value) in (int, float) and math.isfinite(value)
            if not valid:
                raise ValueError(f"{role}/{target}.{key} must be {kind.__name__}")
        if role == "client" and (
            not merged["camera_names"] or any(not isinstance(n, str) for n in merged["camera_names"])
        ):
            raise ValueError("camera_names must be a nonempty string list")
    return {**data, "_role": role}


def profile_values(data, target, stamp=""):
    schema = SERVER if data["_role"] == "server" else CLIENT
    merged = {**data["common"], **data["profiles"][target], "target": target}
    values = {key: deepcopy(merged[key]) for key in schema}
    for key, kind in schema.items():
        if kind is float:
            values[key] = float(values[key])
    if "output" in values:
        values["output"] = values["output"].replace("{timestamp}", stamp)
    return values
