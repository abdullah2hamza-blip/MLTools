"""Shared, independently grounded development inputs for old/new stateful comparison."""

import copy
import hashlib
import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import mltools
from mltools.io import digest, read_json
from mltools.obligations import Reference, Suite

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "examples/probes/lerobot/good.json"
BASE = ROOT / "examples/so101/good.json"
STATS = ROOT / "examples/probes/lerobot/stats.json"
EXPECTED = {
    "control": "PASS",
    "moving_anchor": "FAIL",
    "wrong_stats": "FAIL",
    "relative_gripper": "FAIL",
    "absolute_body": "FAIL",
    "missing_authority": "UNKNOWN",
    "missing_controller_units": "UNKNOWN",
    "wrong_controller_units": "FAIL",
}
NUMERIC_FAULTS = ("moving_anchor", "wrong_stats", "relative_gripper", "absolute_body")


def require(condition, message):
    if not condition:
        raise ValueError(f"Stateful comparison: {message}")


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode(value))


def sources():
    modules = sorted((ROOT / "src/mltools").glob("*.py"))
    installed = Path(mltools.__file__).parent
    require(all((installed / p.name).read_bytes() == p.read_bytes() for p in modules), "installed core bytes")
    paths = [
        *modules,
        ROOT / "uv.lock",
        *(
            ROOT / "scripts" / name
            for name in (
                "stateful_fixtures.py",
                "stateful_comparison_worker.py",
                "compare_stateful_development.py",
            )
        ),
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def spec():
    data = read_json(CASE)
    return Suite.from_dict(data["suite"]), Reference(**data["reference"])


def intended_reference(name):
    _suite, reference = spec()
    return replace(reference, authority=None) if name == "missing_authority" else reference


def scenario(suite, names):
    events = []
    for point in suite.probes:
        values = {(a.port, a.name): v for a, v in zip(suite.axes, point.values, strict=True)}
        origin, current = (
            {name: values[(port, str(i))] for i, name in enumerate(names)} for port in ("origin", "current")
        )
        action = [values[("action", str(i))] for i in range(len(names))]
        events.extend(
            [
                {"kind": "reset"},
                {"kind": "tick", "state": origin, "cameras": {}, "prediction": [action, action[:]]},
                {"kind": "tick", "state": current, "cameras": {}},
            ]
        )
    return events


def inputs():
    suite, reference = spec()
    base, stats = read_json(BASE), read_json(STATS)
    joints = copy.deepcopy(base["contract"]["joints"])
    names = [j["name"] for j in joints]
    require(list(suite.outputs) == [f"{name}.pos" for name in names], "independent named-port agreement")
    axes = {(a.port, a.name): a for a in suite.axes}
    for i, joint in enumerate(joints):
        require(
            reference.output_units[i] == joint["unit"] == base["adapter"]["units"][joint["name"]],
            "independent unit agreement",
        )
        expected = []
        offset = Fraction(stats["mean"][i])
        for axis in suite.axes:
            slope = (
                stats["std"][i]
                if axis.port == "action" and axis.name == str(i)
                else int(axis.port == "origin" and axis.name == str(i) and joint["mode"] == "relative")
            )
            expected.append(Fraction(slope) * Fraction(axis.radius))
            offset += Fraction(slope) * Fraction(axis.center)
        require(
            [Fraction(v) for v in reference.coefficients[i]] == expected
            and Fraction(reference.offset[i]) == offset,
            "scalar contract equals independent affine reference",
        )
        require(
            axes[("origin", str(i))].unit == axes[("current", str(i))].unit == joint["unit"],
            "state-coordinate units",
        )
    files = {"shared/training_stats.json": stats}
    changed = copy.deepcopy(stats)
    changed["mean"][0] += 1
    files["shared/wrong_stats.json"] = changed
    files["shared/checkpoint_config.json"] = {
        "fixture_only": True,
        "state_order": names,
        "action_names": names,
        "relative": True,
        "exclude_joints": ["gripper"],
        "horizon": 2,
        "camera_map": {},
        "use_degrees": True,
    }
    artifacts = [
        {"id": identifier, "path": path, "sha256": hashlib.sha256(encode(files[path])).hexdigest()}
        for identifier, path in (
            ("training_stats", "shared/training_stats.json"),
            ("wrong_stats", "shared/wrong_stats.json"),
            ("checkpoint_config", "shared/checkpoint_config.json"),
        )
    ]
    for name in EXPECTED:
        adapter = copy.deepcopy(base["adapter"])
        adapter.update(camera_map={}, stats_artifact="training_stats", reset=True)
        if name == "moving_anchor":
            adapter["bind_queue"] = False
        if name == "wrong_stats":
            adapter["stats_artifact"] = "wrong_stats"
        if name == "relative_gripper":
            adapter["exclude_joints"] = []
        if name == "absolute_body":
            adapter["relative"] = False
        if name == "missing_controller_units":
            adapter["units"] = None
        if name == "wrong_controller_units":
            adapter["units"][names[0]] = "radian"
        files[f"{name}.bundle.json"] = {
            "schema_version": 1,
            "contract": {
                "embodiment": "Derived SO101 software fixture; not hardware/calibration",
                "authority": None
                if name == "missing_authority"
                else {"source": reference.authority, "physical_labels": BASE.relative_to(ROOT).as_posix()},
                "checks": ["actions"],
                "joints": joints,
                "cameras": [],
                "normalization": {"kind": "mean_std", "artifact": "training_stats"},
                "horizon": 2,
            },
            "adapter": adapter,
            "artifacts": artifacts,
            "scenario": scenario(suite, names),
        }
    files["plan.json"] = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "source_file_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in (CASE, BASE, STATS)},
        "suite_sha256": suite.sha256,
        "reference": reference.to_dict(),
        "expected": EXPECTED,
        "unit": "One reused synthetic development cluster; not eight independent cases",
        "matched_scope": "Second dequeue's independently named scalar commands and declared controller units",
    }
    return files
