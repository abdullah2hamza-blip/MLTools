"""Isolated genuine Panda/OSC operators and native populated-state reset."""

import base64
import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panda_reference import (  # noqa: E402
    ROOT,
    SEMANTIC_BLOBS,
    distribution,
    fingerprint,
    fixtures,
    require,
    sha,
    sources,
)


def installation():
    expected = dict(
        re.findall(r"(?m)^([\w.-]+)==([^\s\\]+)", (ROOT / "integrations/panda/requirements.lock").read_text())
    )
    versions = {name: importlib.metadata.version(name) for name in expected}
    require(expected == versions, "complete locked runtime versions")
    dist = importlib.metadata.distribution("robosuite")
    root = Path(dist.locate_file("robosuite"))
    record_hashes = {}
    byte_count = 0
    for entry in dist.files:
        if entry.hash is None:
            continue
        require(entry.hash.mode == "sha256", "native wheel RECORD hash algorithm")
        path = Path(dist.locate_file(entry))
        raw = path.read_bytes()
        actual = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
        require(
            actual == entry.hash.value and len(raw) == entry.size,
            "complete installed wheel RECORD: " + str(entry),
        )
        record_hashes[str(entry).replace("\\", "/")] = hashlib.sha256(raw).hexdigest()
        byte_count += len(raw)
    semantic = {}
    for name, expected_blob in SEMANTIC_BLOBS.items():
        raw = (root / name).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        require(actual == expected_blob, "native semantic Git source: " + name)
        semantic[name] = actual
    require(not (root / "macros_private.py").exists(), "no hidden local macro overrides")
    pinned = distribution()
    require(
        fingerprint(record_hashes) == pinned["record_manifest_sha256"]
        and len(record_hashes) == pinned["record_hashed_files"],
        "pinned complete release file manifest",
    )
    require(
        sha(Path(importlib.metadata.distribution("mujoco").locate_file("mujoco/mujoco.dll")))
        == pinned["mujoco_dll_sha256"],
        "pinned genuine dependency DLL",
    )
    return root, {
        "packages": versions,
        "semantic_git_blobs": semantic,
        "record_file_sha256": record_hashes,
        "record_checked_bytes": byte_count,
        "record_sha256": hashlib.sha256(dist.read_text("RECORD").encode()).hexdigest(),
        "bootstrap_source_sha256": {
            name: sha(root / name) for name in ("utils/binding_utils.py", "macros.py")
        },
    }


def import_native(root):
    original = ctypes.WinDLL if os.name == "nt" else None
    redirects = []
    target = Path(importlib.metadata.distribution("mujoco").locate_file("mujoco/mujoco.dll"))
    require(sha(target) == distribution()["mujoco_dll_sha256"], "native DLL load identity")

    def loader(name, *args, **kwargs):
        if Path(name).resolve() == (root / "utils/mujoco.dll").resolve():
            macros = sys.modules["robosuite.macros"]
            redirects.append(
                {
                    "from": "robosuite/utils/mujoco.dll",
                    "to": "mujoco/mujoco.dll",
                    "target_sha256": sha(target),
                    "gpu_macro_before": macros.MUJOCO_GPU_RENDERING,
                    "gpu_macro_after": False,
                }
            )
            # Supported runtime macro, set at the import point before the GL
            # branch. Source bytes and controller/gripper code remain unchanged.
            macros.MUJOCO_GPU_RENDERING = False
            return original(str(target), *args, **kwargs)
        return original(name, *args, **kwargs)

    if original:
        ctypes.WinDLL = loader
    try:
        import robosuite
    finally:
        if original:
            ctypes.WinDLL = original
    require(len(redirects) == (1 if os.name == "nt" else 0), "bounded Windows compatibility bootstrap")
    return robosuite, {
        "windows_dll_redirects": redirects,
        "ctypes_loader_restored": not original or ctypes.WinDLL is original,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "scope": "Import-only Windows DLL path and documented headless macro; no source replacements",
    }


def execute(request):
    require(request["inputs"] == fixtures(), "fixed independent inputs")
    require(request["source_file_sha256"] == sources(), "producer source snapshot")
    suite = request["suite"]
    require(
        suite["axes"] == fixtures()["axes"] and suite["outputs"] == fixtures()["outputs"],
        "bounded translation ports",
    )
    require(len(suite["probes"]) == 19, "canonical translation query budget")
    original_sources = sources()
    events = []

    def audit(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            events.append(event)
            raise RuntimeError("Panda software worker rejects network connections")

    sys.addaudithook(audit)
    start = time.perf_counter()
    root, identity = installation()
    native, bootstrap = import_native(root)
    import numpy as np
    from robosuite.controllers import load_controller_config
    from robosuite.utils import control_utils as goals
    from robosuite.utils import transform_utils as rotations

    np.random.seed(fixtures()["environment"]["seed"])
    config = load_controller_config(default_controller="OSC_POSE")
    env = native.make(
        "Lift",
        robots="Panda",
        controller_configs=config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        initialization_noise=None,
        hard_reset=False,
    )
    env.reset()
    started_time = float(env.sim.data.time)
    captured = {}
    try:
        controller = env.robots[0].controller
        original_min, original_max = controller.output_min.copy(), controller.output_max.copy()
        for condition in fixtures()["affine_conditions"]:
            factor = 2 if condition == "wrong_gain" else 1
            controller.output_min, controller.output_max = original_min * factor, original_max * factor
            controller.action_scale = None
            rows = []
            for query in suite["probes"]:
                ports = {
                    (a["port"], a["name"]): v for a, v in zip(suite["axes"], query["values"], strict=True)
                }
                action, origin, current = (
                    [ports[(p, str(i))] for i in range(3)] for p in ("action", "origin", "current")
                )
                called_action = action[:]
                if condition == "wrong_order":
                    called_action[0], called_action[1] = called_action[1], called_action[0]
                six = np.asarray(called_action + [0, 0, 0], dtype=np.float64)
                scaled = controller.scale_action(six)
                called_current = origin if condition == "wrong_anchor" else current
                goal = goals.set_goal_position(scaled[:3], np.asarray(called_current, dtype=np.float64))
                rows.append(
                    {
                        "query": query["id"],
                        "action": action,
                        "origin": origin,
                        "current": current,
                        "called_action": six.tolist(),
                        "called_current": called_current,
                        "scaled": scaled.tolist(),
                        "goal": goal.tolist(),
                    }
                )
            captured[condition] = {
                "rows": rows,
                "configured_output_min": controller.output_min.tolist(),
                "configured_output_max": controller.output_max.tolist(),
            }
        controller.output_min, controller.output_max = original_min, original_max
        controller.action_scale = None
        full = []
        for action in fixtures()["full_controller_actions"]:
            controller.update(force=True)
            before_pos, before_rot = controller.ee_pos.copy(), controller.ee_ori_mat.copy()
            controller.set_goal(np.asarray(action, dtype=np.float64))
            full.append(
                {
                    "action": action,
                    "current_position": before_pos.tolist(),
                    "current_rotation": before_rot.tolist(),
                    "goal_position": controller.goal_pos.tolist(),
                    "goal_rotation": controller.goal_ori.tolist(),
                }
            )
        orientation = []
        for case in fixtures()["orientation_cases"]:
            current = rotations.quat2mat(rotations.axisangle2quat(np.asarray(case["current_axisangle"])))
            delta = np.asarray(case["delta_axisangle"])
            delta_matrix = rotations.quat2mat(rotations.axisangle2quat(delta))
            for variant in ("native_pre_multiply", "post_multiply_injection"):
                called_delta, called_current = (
                    (delta, current)
                    if variant == "native_pre_multiply"
                    else (np.asarray(case["current_axisangle"]), delta_matrix)
                )
                actual = goals.set_goal_orientation(called_delta, called_current)
                orientation.append(
                    {
                        "case": case["id"],
                        "variant": variant,
                        "current_matrix": current.tolist(),
                        "called_delta": called_delta.tolist(),
                        "called_current_matrix": called_current.tolist(),
                        "goal": actual.tolist(),
                    }
                )
        grippers = []
        for variant in fixtures()["gripper_reset_variants"]:
            env.reset()
            gripper = env.robots[0].gripper
            rows = []
            initial = gripper.current_action.tolist()
            for action in fixtures()["gripper_commands"]:
                before = gripper.current_action.tolist()
                after = gripper.format_action(np.asarray([action], dtype=np.float64)).tolist()
                rows.append({"action": action, "before": before, "after": after})
            populated = gripper.current_action.tolist()
            if variant == "native_reset":
                env.reset()
            cleared = gripper.current_action.tolist()
            same = gripper is env.robots[0].gripper
            first = gripper.format_action(np.asarray([1.0])).tolist()
            grippers.append(
                {
                    "variant": variant,
                    "initial": initial,
                    "rows": rows,
                    "populated": populated,
                    "reset_called": variant == "native_reset",
                    "same_gripper_object": same,
                    "after_reset_boundary": cleared,
                    "first_after_boundary": first,
                }
            )
        ended_time = float(env.sim.data.time)
    finally:
        env.close()
    require(
        installation()[1] == identity and sources() == original_sources, "stable native and producer sources"
    )
    require(events == [], "native network attempts")
    helper_spec = importlib.util.spec_from_file_location("panda_rss", ROOT / "src/mltools/_openpi_worker.py")
    helper = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helper)
    peak = helper._rss()
    require(peak <= 2 * 1024**3, "measured worker peak-RSS budget")
    return {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "source_file_sha256": original_sources,
        "identity": identity,
        "bootstrap": bootstrap,
        "native_config": config,
        "affine": captured,
        "full_controller": full,
        "orientation": orientation,
        "gripper": grippers,
        "simulation": {
            "environment": "Lift",
            "robot": "Panda",
            "physics_time_initial": started_time,
            "physics_time_final": ended_time,
            "env_step_calls": 0,
            "controller_torque_calls": 0,
            "render_calls": 0,
            "checkpoint_mapping": "UNKNOWN",
        },
        "runtime": {
            "python": platform.python_version(),
            "wall_seconds": time.perf_counter() - start,
            "peak_rss_bytes": peak,
            "network_events": events,
        },
    }


def main():
    raw = sys.stdin.buffer.read(2 * 1024**2 + 1)
    require(len(raw) <= 2 * 1024**2, "worker request size")
    result = execute(json.loads(raw))
    result["request_sha256"] = hashlib.sha256(raw).hexdigest()
    sys.stdout.buffer.write(json.dumps(result, allow_nan=False).encode())


if __name__ == "__main__":
    main()
