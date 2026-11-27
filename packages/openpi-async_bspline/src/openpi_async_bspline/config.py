"""Validated B-spline settings. YAML is read once, before the worker starts."""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
import math
from pathlib import Path


@dataclass(frozen=True)
class BSplineConfig:
    replan_interval: int = 15
    blend_lead_steps: int = 3
    knots_interior: int = 6
    passthrough_indices: tuple[int, ...] = ()
    degree: int = 3
    max_deriv: int = 2
    knots_interior_merge: int = 3
    merge_input_mode: str = "crop"

    # Original ominf fields; plotting/printing is deferred until save_diagnostics().
    interpolate_factor: float = 0.0
    offset_timestep: float | None = None
    publish_position: bool = True
    publish_velocity: bool = False
    publish_acceleration: bool = False
    bspline_infer_plot: bool = False
    plots_dir: str = "bspline_plots"
    bspline_publish_plot: bool = False
    timing_print: bool = False
    joint_col: str = ""
    plot_ncol: int = 2
    plot_joint_indices: str = ""
    plot_point_indices: str = ""
    plot_merge_steps: str = ""
    plot_raw: bool = True
    plot_scatter: bool = False
    plot_c2_merge: bool = False
    plot_final: bool = True
    plot_overall: bool = False
    # Plugin-specific units and bounded diagnostic storage.
    fps: float = 50.0
    diagnostic_capacity: int = 128
    publish_capacity: int = 30000

    @property
    def effective_lead(self):
        return self.blend_lead_steps if self.offset_timestep is None else self.offset_timestep

    @property
    def diagnostics_enabled(self):
        return self.bspline_infer_plot or self.bspline_publish_plot or self.timing_print

    def indices(self, name):
        value = getattr(self, name)
        return tuple(int(part.strip()) for part in value.split(",") if part.strip())

    def __post_init__(self):
        for name in (
            "replan_interval",
            "blend_lead_steps",
            "knots_interior",
            "degree",
            "max_deriv",
            "knots_interior_merge",
            "plot_ncol",
            "diagnostic_capacity",
            "publish_capacity",
        ):
            value = getattr(self, name)
            minimum = 0 if name in ("knots_interior", "knots_interior_merge") else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid {name}: {value}")
        for name in ("fps", "interpolate_factor", "offset_timestep"):
            value = getattr(self, name)
            if value is None and name == "offset_timestep":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0 or (name == "fps" and value == 0):
                raise ValueError(f"Invalid {name}: {value}")
        for name in (
            "publish_position",
            "publish_velocity",
            "publish_acceleration",
            "bspline_infer_plot",
            "bspline_publish_plot",
            "timing_print",
            "plot_raw",
            "plot_scatter",
            "plot_c2_merge",
            "plot_final",
            "plot_overall",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("plots_dir", "joint_col", "plot_joint_indices", "plot_point_indices", "plot_merge_steps"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if not self.plots_dir.strip():
            raise ValueError("plots_dir must not be empty")
        for name in ("plot_joint_indices", "plot_point_indices", "plot_merge_steps"):
            try:
                indices = self.indices(name)
            except ValueError as exc:
                raise ValueError(f"{name} must contain comma-separated integers") from exc
            if any(i < (1 if name == "plot_merge_steps" else 0) for i in indices) or len(set(indices)) != len(indices):
                raise ValueError(f"Invalid {name}: {indices}")
        if self.degree > 5 or self.degree < self.max_deriv + 1:
            raise ValueError("Require max_deriv + 1 <= degree <= 5")
        if self.merge_input_mode not in ("crop", "refit"):
            raise ValueError("merge_input_mode must be crop or refit")
        if not isinstance(self.passthrough_indices, (list, tuple)) or any(
            isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in self.passthrough_indices
        ):
            raise ValueError("passthrough_indices must contain nonnegative integers")
        if len(set(self.passthrough_indices)) != len(self.passthrough_indices):
            raise ValueError("passthrough_indices must not contain duplicates")
        object.__setattr__(self, "passthrough_indices", tuple(self.passthrough_indices))

    def to_dict(self):
        values = asdict(self)
        if self.offset_timestep is not None:
            values.pop("blend_lead_steps")
        return values

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("B-spline configuration must be a mapping")
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unsupported B-spline keys: {sorted(unknown, key=str)}")
        if (
            values.get("offset_timestep") is not None
            and "blend_lead_steps" in values
            and values["offset_timestep"] != values["blend_lead_steps"]
        ):
            raise ValueError("offset_timestep and blend_lead_steps conflict; specify only one")
        return cls(**values)

    @classmethod
    def from_yaml(cls, path, *, section=None):
        """Load flat settings, bspline: settings, or env.bspline (ominf layout).

        Unknown fields in the selected B-spline section, duplicate YAML keys and
        unsafe YAML tags are rejected. Unrelated top-level robot/model sections
        are not consumed. An explicit dotted section overrides auto-detection.
        """
        import yaml  # noqa: PLC0415

        class UniqueSafeLoader(yaml.SafeLoader):
            pass

        def mapping(loader, node, *, deep=False):
            loader.flatten_mapping(node)
            result = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if not isinstance(key, str):
                    raise ValueError("YAML configuration keys must be strings")
                if key in result:
                    raise ValueError(f"Duplicate YAML key: {key}")
                result[key] = loader.construct_object(value_node, deep=deep)
            return result

        UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
        with Path(path).open(encoding="utf-8") as stream:
            values = yaml.load(stream, Loader=UniqueSafeLoader)
        if not isinstance(values, dict):
            raise ValueError("YAML configuration must contain a mapping")
        if section is None:
            has_root = "bspline" in values
            has_env = isinstance(values.get("env"), dict) and "bspline" in values["env"]
            if has_root and has_env:
                raise ValueError("Ambiguous bspline sections; specify section explicitly")
            section = "bspline" if has_root else ("env.bspline" if has_env else "")
        env = values.get("env")
        env_fps = env.get("fps") if section == "env.bspline" and isinstance(env, dict) else None
        for key in section.split(".") if section else ():
            if not isinstance(values, dict) or key not in values:
                raise ValueError(f"Missing YAML section: {section}")
            values = values[key]
        if env_fps is not None and isinstance(values, dict):
            values = {"fps": env_fps, **values}
        return cls.from_dict(values)
