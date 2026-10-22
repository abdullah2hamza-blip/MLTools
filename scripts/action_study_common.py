"""Bounded development inputs/source identities for the action-only study runner."""

import copy
import hashlib
import json
from pathlib import Path

from normalization_profiles import PROFILES, require, validate
from prepare_act_candidates import tensors

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "action_designs.py",
    "query_designs.py",
    "normalization_profiles.py",
    "normalization_certificate.py",
    "prepare_act_candidates.py",
    "action_study_common.py",
    "action_study_worker.py",
    "run_action_study.py",
)
CONDITIONS = (
    "numeric_control",
    "offset_intervention",
    "slope_intervention",
    "fixed_offset_slope_composition",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def sources():
    paths = [*sorted((ROOT / "src/mltools").glob("*.py")), ROOT / "uv.lock"]
    paths.extend(ROOT / "scripts" / n for n in SCRIPTS)
    paths.extend(ROOT / "integrations/openpi" / n for n in ("requirements-cpu.lock", "frameworks.txt"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def interventions(spec):
    """Recipes depend on metadata alone; no execution outcomes enter selection."""
    validate(spec)
    stats = spec["statistics"]
    mean_std = spec["profile"] == "lerobot_mean_std_f64_v1"
    widths = stats["std"] if mean_std else [b - a for a, b in zip(stats["q01"], stats["q99"], strict=True)]
    selected = next((i for i, v in enumerate(widths) if v > 0), None)
    require(selected is not None, "a nondegenerate metadata coordinate for the declared intervention")
    shift = max(1.0, widths[selected] / 2)
    result = {}
    for name in CONDITIONS:
        candidate = copy.deepcopy(spec)
        data = candidate["statistics"]
        if name in ("slope_intervention", "fixed_offset_slope_composition"):
            if mean_std:
                data["std"][selected] *= 2
            else:
                center = stats["q01"][selected] / 2 + stats["q99"][selected] / 2
                data["q01"][selected] = center - widths[selected]
                data["q99"][selected] = center + widths[selected]
        if name in ("offset_intervention", "fixed_offset_slope_composition"):
            for field in ("mean",) if mean_std else ("q01", "q99"):
                data[field][selected] += shift
        validate(candidate)
        if name != "numeric_control":
            require(candidate != spec, "representable deployment-statistics intervention")
        result[name] = {
            "specification": candidate,
            "annotation": "intended numeric control"
            if name == "numeric_control"
            else "explicit injected fault",
            "selected_coordinate": selected,
            "offset_change": shift if "offset" in name else 0.0,
            "slope_factor": 2 if name in ("slope_intervention", "fixed_offset_slope_composition") else 1,
        }
    return result


def development_inputs():
    values = {}
    for profile in PROFILES:
        data = (
            {"mean": [1.0, 5.0], "std": [2.0, 0.0]}
            if "mean_std" in profile
            else {"q01": [-2.0, 4.0], "q99": [2.0, 4.0]}
        )
        values["designed_" + profile] = {
            "framework": "openpi" if profile.startswith("openpi") else "lerobot",
            "profile": profile,
            "statistics": data,
            "source_artifact": None,
            "authority": "Independently authored numerical fixture; includes a degenerate coordinate",
            "lineage": "designed arithmetic, not a published checkpoint",
        }
    relative = "examples/probes/act/policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    path = ROOT / relative
    require(
        sha(path) == "9155f3cb55dc35771a721dfd14d63e0ed7e785519e36b2c4794a72f556ac02dc",
        "already executed ACT move-cube development statistics identity",
    )
    decoded = tensors(path)
    values["published_act_move_cube_development"] = {
        "framework": "lerobot",
        "profile": "lerobot_mean_std_f64_v1",
        "statistics": {k: decoded["action." + k]["values"] for k in ("mean", "std")},
        "source_artifact": {"path": relative, "sha256": sha(path), "format": "safetensors"},
        "authority": "Pinned already executed publisher numerical metadata; no physical authority",
        "lineage": "yutofujita-so101-move-cube; reused development cluster",
    }
    for cell in values.values():
        cell["intended"] = {
            "schema_version": 1,
            "profile": cell.pop("profile"),
            "statistics": cell.pop("statistics"),
        }
        cell["conditions"] = interventions(cell["intended"])
        cell["checkpoint_physical_status"] = "UNKNOWN"
    return values
