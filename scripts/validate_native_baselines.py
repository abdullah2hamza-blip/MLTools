"""Explicit CPU-only development validation of installed, pinned baseline workflows.

Run in integrations/baselines' isolated environment. Uses complete packages, an
actual preflight CLI subprocess, and Policy Space's dispatcher/MessagePack path.
No sockets, trained models, robot commands or scientific detection scores.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def require(condition, message):
    if not condition:
        raise ValueError(f"Native baseline development validation failed: {message}")


def identities():
    sources = json.loads((ROOT / "integrations/baselines/sources.json").read_bytes())
    for name, source in sources.items():
        distribution = importlib.metadata.distribution(name)
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        require(direct.get("vcs_info", {}).get("commit_id") == source["revision"], f"{name} revision")
        for path, expected in source["blobs"].items():
            content = distribution.locate_file(path).read_bytes().replace(b"\r\n", b"\n")
            blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
            require(blob == expected, f"{name} source {path}")
    return sources


def preflight(output):
    target = output / "preflight"
    target.mkdir()
    base = {
        "fps": 20,
        "observations": {"state": {"shape": [3], "dtype": "float32", "unit": "rad"}},
        "action": {"shape": [3], "dtype": "float32", "unit": "rad", "range": [-2, 2]},
    }
    cases = []
    for name, expected in (("correct", 0), ("wrong_shape", 1), ("wrong_unit", 1), ("missing_unit", 2)):
        folder = target / name
        folder.mkdir()
        dataset, policy, robot = (copy.deepcopy(base) for _ in range(3))
        policy["normalization"] = {"action": {"mean": [0, 0, 0], "std": [1, 1, 1]}}
        if name == "wrong_shape":
            robot["action"]["shape"] = [2]
        if name == "wrong_unit":
            robot["action"]["unit"] = "degree"
        if name == "missing_unit":
            robot["action"]["unit"] = None
        for label, data in (("dataset", dataset), ("policy", policy), ("robot", robot)):
            write(folder / f"{label}.json", data)
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "robot_policy_preflight",
                *(str(folder / f"{label}.json") for label in ("dataset", "policy", "robot")),
                "--output",
                str(folder / "native-report.json"),
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        require(result.returncode == expected, f"preflight {name}: {result.stderr}")
        row = {"case": name, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        if expected != 2:
            row["native_report"] = json.loads((folder / "native-report.json").read_bytes())
        else:
            row["interpretation"] = "Native input rejection; not a detected semantic incompatibility"
        cases.append(row)
    return cases


def policy_space():
    import numpy as np
    from policy_space import PolicyModel
    from policy_space.runtime.server import (
        PolicySpaceServer,
        _pack_array,
        _safe_packb,
        _safe_unpackb,
        _unpack_array,
    )

    class FixtureModel(PolicyModel):
        def __init__(self, actions):
            super().__init__({})
            self.actions, self.initialized, self.finalized = actions, 0, 0

        def initialize_episode(self, episode_info=None):
            self.initialized += 1

        def ingest_observation(self, obs):
            pass

        def infer_actions(self):
            return self.actions.copy()

        def finalize_episode(self, episode_info=None):
            self.finalized += 1

    selected = {
        "observation_schema": "single_arm_rgb_joint@1",
        "action_schema": "single_arm_joint_position@1",
        "embodiment_schema": "franka@1",
        "episode_id": "fixture-1",
    }
    pair = {k: selected[k] for k in ("observation_schema", "action_schema")}
    pair["embodiment_constraints"] = ["franka@1"]
    outcomes = []
    for name in (
        "correct",
        "schema_mismatch",
        "wrong_width",
        "wrong_horizon",
        "missing_camera",
        "nonfinite_action",
        "camera_relabeling",
        "action_permutation",
    ):
        actions = np.array([[0.01 * i for i in range(8)]] * 2, dtype=np.float32)
        observation = {
            "episode_id": "fixture-1",
            "step": 0,
            "images": {
                role: np.full((2, 2, 3), value, dtype=np.uint8)
                for role, value in (("exterior_1", 31), ("exterior_2", 83), ("wrist", 197))
            },
            "state": {"joint_position": np.zeros(7, dtype=np.float32), "gripper_position": np.zeros(1)},
        }
        request = dict(selected)
        if name == "schema_mismatch":
            request["action_schema"] = "bimanual_joint_position@1"
        if name == "wrong_width":
            actions = actions[:, :-1]
        if name == "wrong_horizon":
            actions = actions[:1]
        if name == "missing_camera":
            del observation["images"]["wrist"]
        if name == "nonfinite_action":
            actions[0, 0] = np.nan
        if name == "camera_relabeling":
            images = observation["images"]
            images["exterior_1"], images["wrist"] = images["wrist"], images["exterior_1"]
        if name == "action_permutation":
            actions[:, [0, 1]] = actions[:, [1, 0]]
        model = FixtureModel(actions)
        server = PolicySpaceServer(model, metadata={"supported_schema_pairs": [pair], "action_horizon": 2})

        def dispatch(endpoint, payload, server=server):
            # Use the native codec around the actual dispatcher. No socket is opened.
            decoded = _safe_unpackb(_safe_packb(payload, default=_pack_array), object_hook=_unpack_array)
            response = server._dispatch(endpoint, decoded)
            return _safe_unpackb(_safe_packb(response, default=_pack_array), object_hook=_unpack_array)

        row = {"case": name}
        try:
            dispatch("initialize_episode", request)
            result = dispatch("infer_actions", observation)
        except ValueError as exc:
            row.update(decision="REJECT", error=str(exc))
        else:
            require(np.array_equal(result["actions"], actions), f"{name} action wire preservation")
            dispatch("finalize_episode", {"episode_id": "fixture-1"})
            dispatch("initialize_episode", {**selected, "episode_id": "fixture-2"})
            dispatch("finalize_episode", {"episode_id": "fixture-2"})
            require(model.initialized == model.finalized == 2, f"{name} lifecycle dispatch")
            row.update(decision="ACCEPT", shape=list(result["actions"].shape), lifecycle_calls=2)
        expected = "ACCEPT" if name in ("correct", "camera_relabeling", "action_permutation") else "REJECT"
        require(row["decision"] == expected, f"Policy Space {name}: {row}")
        if name == "schema_mismatch":
            require(model.initialized == 0, "incompatible schema rejected before model initialization")
        outcomes.append(row)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New development evidence directory")
    args = parser.parse_args()
    require(not os.environ.get("POLICY_SPACE_CONTRACTS_DIR"), "custom schema override is not allowed")
    require(importlib.util.find_spec("torch") is None, "use the isolated no-ML baseline environment")
    source_identities = identities()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "classification": "native workflow applicability controls; not a fault-detection benchmark",
        "source_identities": source_identities,
        "preflight": preflight(args.output),
        "policy_space": policy_space(),
        "runtime": {"python": platform.python_version()},
        "scope": [
            "Complete installed packages, with pinned source identity checks before import",
            "Preflight full CLI; Policy Space dispatcher, MessagePack, schema and lifecycle paths",
            "Policy Space synthetic fixture model; no checkpoint adapter, network server or real client",
            "Schema-valid relabelings expose scope; no claim that full products miss a benchmark fault",
            "Missing preflight metadata is an input error; preserve native semantics",
            "No MLTools superiority score, trained inference, simulator, robot or fabricated telemetry",
        ],
    }
    result["runtime"]["wall_seconds"] = time.perf_counter() - start
    result["runtime"]["packages"] = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    result["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(args.output / "summary.json", result)
    print("Verified preflight CLI and Policy Space schema/codec/lifecycle applicability controls")


if __name__ == "__main__":
    main()
