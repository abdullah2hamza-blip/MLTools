"""Fixed designed arithmetic inputs; these are not demonstration calibration."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "normalization_certificate.py",
    "certificate_cases.py",
    "normalization_certificate_worker.py",
    "validate_normalization_certificate.py",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = [*sorted((ROOT / "src/mltools").glob("*.py")), ROOT / "uv.lock"]
    paths.extend(ROOT / "scripts" / name for name in SCRIPT_NAMES)
    paths.extend(ROOT / "integrations/openpi" / name for name in ("requirements-cpu.lock", "frameworks.txt"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def fixtures():
    base = {
        "schema_version": 1,
        "profile": "paper_quantile_equations",
        "coordinates": [
            {"name": "axis0", "unit": "synthetic_unit"},
            {"name": "gripper1", "unit": "synthetic_unit"},
        ],
        "intended": {
            "authority": "Independent designed arithmetic metadata",
            "q01": [0, 0],
            "q99": [2, 1],
            "mask": [True, False],
        },
        "candidate": {
            "authority": "Explicit designed decoder intervention",
            "q01": [1, -99],
            "q99": [5, 99],
            "mask": [True, False],
        },
        "calibration": {
            "kind": "designed",
            "space": "normalized",
            "authority": "Fixed hand-calculated intervention inputs, not demonstrations",
            "values": [[0, 0], [0.5, 0.25], [1, 1]],
        },
        "metric": {"indices": [0], "unit": "synthetic_unit"},
        "gripper_indices": [1],
        "thresholds": [0, 2, 3, 4],
        "percentiles": [0, 0.5, 0.9, 1],
    }
    result = {"paper_drift": copy.deepcopy(base)}
    value = copy.deepcopy(base)
    value["candidate"] = copy.deepcopy(value["intended"])
    result["paper_identity"] = value
    value = copy.deepcopy(base)
    value["candidate"]["mask"] = [False, False]
    result["paper_mask_change"] = value
    for framework in ("openpi", "lerobot"):
        value = copy.deepcopy(base)
        value["profile"] = (
            "openpi_quantile_extension" if framework == "openpi" else "lerobot_mean_std_extension"
        )
        value["calibration"]["values"] = [[-1, -1], [0, 0], [1, 1]]
        if framework == "openpi":
            value["intended"] = {"authority": "Designed quantile metadata", "q01": [-2, -1], "q99": [2, 1]}
            value["candidate"] = {
                "authority": "Designed quantile intervention",
                "q01": [-1, -1],
                "q99": [5, 1],
            }
        else:
            value["intended"] = {"authority": "Designed mean/std metadata", "mean": [1, 15], "std": [2, 4]}
            value["candidate"] = {
                "authority": "Designed mean/std intervention",
                "mean": [2, 15],
                "std": [3, 4],
            }
        result[framework + "_drift"] = copy.deepcopy(value)
        if framework == "openpi":
            physical = copy.deepcopy(value)
            physical["calibration"].update(space="physical", values=[[-2, -1], [0, 0], [2, 1]])
            result["openpi_physical_forward"] = physical
        value["candidate"] = copy.deepcopy(value["intended"])
        result[framework + "_identity"] = copy.deepcopy(value)
        if framework == "openpi":
            value["calibration"]["values"] = [[-3, -1], [0, 0], [3, 1]]
            result["openpi_outside_range"] = value
        else:
            value["intended"]["std"][0] = 0
            value["candidate"] = copy.deepcopy(value["intended"])
            value["calibration"].update(space="physical", values=[[0, 1], [1, 2], [2, 3]])
            result["lerobot_lossy_forward"] = value
    return result
