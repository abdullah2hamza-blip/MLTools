"""Acquire/replay a source-grounded Panda software-controller development stratum."""

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from panda_reference import (
    EXPECTED,
    ROOT,
    SEMANTIC_BLOBS,
    close,
    distribution,
    fingerprint,
    fixtures,
    grip,
    product,
    require,
    rotation,
    sha,
    sources,
    write,
)

from mltools.obligations import Axis, Reference, compile_suite, evaluate
from mltools.research_oracles import pointwise, relational, stateless

FILES = {
    "inputs.json",
    "request.json",
    "native.stdout.txt",
    "native.stderr.txt",
    "capture.json",
    "analysis.json",
}


def suite():
    data = fixtures()
    return compile_suite(tuple(Axis(**a) for a in data["axes"]), tuple(data["outputs"]))


def request():
    return {"inputs": fixtures(), "suite": suite().to_dict(), "source_file_sha256": sources()}


def reference(condition):
    compiled = suite()
    return Reference(
        coefficients=tuple(
            tuple(
                a.radius * (0.05 if a.port == "action" else 1 if a.port == "current" else 0)
                if a.name == str(i)
                else 0
                for a in compiled.axes
            )
            for i in range(3)
        ),
        offset=(0, 0, 0),
        output_units=(None,) * 3 if condition == "missing_units" else ("sim_position_coordinate",) * 3,
        authority=None
        if condition == "missing_authority"
        else "Pinned OSC_POSE scaling and native current-state goal composition; simulator coordinates",
        affine_fragment_verified=True,
    )


def analyze(capture):
    compiled = suite()
    require(set(capture["affine"]) == set(EXPECTED), "complete affine condition inventory")
    affine = {}
    for condition in EXPECTED:
        data = capture["affine"][condition]
        factor = 2 if condition == "wrong_gain" else 1
        require(
            close(data["configured_output_max"], [0.05 * factor] * 3 + [0.5 * factor] * 3, 1e-12)
            and close(data["configured_output_min"], [-0.05 * factor] * 3 + [-0.5 * factor] * 3, 1e-12),
            "actual native configured range",
        )
        require(len(data["rows"]) == len(compiled.probes), "actual native query budget")
        outputs = {}
        for query, row in zip(compiled.probes, data["rows"], strict=True):
            ports = {(a.port, a.name): v for a, v in zip(compiled.axes, query.values, strict=True)}
            values = {p: [ports[(p, str(i))] for i in range(3)] for p in ("action", "origin", "current")}
            require(
                row["query"] == query.id and all(row[p] == values[p] for p in values), "executed query inputs"
            )
            called = values["action"][:]
            if condition == "wrong_order":
                called[0], called[1] = called[1], called[0]
            require(
                row["called_action"] == called + [0, 0, 0]
                and row["called_current"] == values["origin" if condition == "wrong_anchor" else "current"],
                "explicit injected caller wiring",
            )
            require(
                isinstance(row["goal"], list)
                and len(row["goal"]) == 3
                and all(type(x) in (int, float) and math.isfinite(x) for x in row["goal"]),
                "finite native goal",
            )
            actual_scaled = [
                max(-1, min(1, value)) * scale * factor
                for value, scale in zip(row["called_action"], [0.05] * 3 + [0.5] * 3, strict=True)
            ]
            actual_goal = [
                value + delta for value, delta in zip(row["called_current"], actual_scaled[:3], strict=True)
            ]
            require(
                close(row["scaled"], actual_scaled, 1e-12) and close(row["goal"], actual_goal, 1e-12),
                "native outputs match actual injected call and configuration",
            )
            outputs[query.id] = row["goal"]
        evidence = {"suite_sha256": compiled.sha256, "outputs": outputs}
        methods = {
            "compiled": evaluate(compiled, reference(condition), evidence),
            "pointwise": pointwise(compiled, reference(condition), evidence),
            "strong_relational": relational(compiled, reference(condition), evidence),
            "stateless": stateless(compiled, reference(condition), evidence),
        }
        require(
            all(
                methods[m]["status"] == EXPECTED[condition]
                for m in ("compiled", "pointwise", "strong_relational")
            ),
            "independent expected affine outcomes",
        )
        affine[condition] = {
            "queries": len(compiled.probes),
            "methods": methods,
            "scope": "Finite bounded native translation operators with controlled state arguments",
        }
    full = capture["full_controller"]
    require(len(full) == len(fixtures()["full_controller_actions"]), "full native controller inventory")
    for action, row in zip(fixtures()["full_controller_actions"], full, strict=True):
        require(row["action"] == action, "full native controller action inputs")
        expected_pos = [p + 0.05 * a for p, a in zip(row["current_position"], action[:3], strict=True)]
        expected_ori = product(rotation([0.5 * a for a in action[3:]]), row["current_rotation"])
        require(
            close(expected_pos, row["goal_position"], 1e-12)
            and close(expected_ori, row["goal_rotation"], 1e-6),
            "full native OSC set_goal against independent reference",
        )
    orientation = []
    cases = fixtures()["orientation_cases"]
    rows = capture["orientation"]
    require(len(rows) == len(cases) * 2, "orientation case inventory")
    for case, variant, row in zip(
        (c for c in cases for _ in range(2)),
        (v for _ in cases for v in ("native_pre_multiply", "post_multiply_injection")),
        rows,
        strict=True,
    ):
        require(row["case"] == case["id"] and row["variant"] == variant, "orientation case identity")
        correct_current, delta = rotation(case["current_axisangle"]), rotation(case["delta_axisangle"])
        require(
            close(row["current_matrix"], correct_current, 1e-6), "native axis-angle fixture interpretation"
        )
        require(
            close(
                row["called_delta"],
                case["delta_axisangle"] if variant == "native_pre_multiply" else case["current_axisangle"],
                1e-12,
            )
            and close(
                row["called_current_matrix"],
                correct_current if variant == "native_pre_multiply" else delta,
                1e-6,
            ),
            "native orientation call binding",
        )
        expected = product(delta, correct_current)
        require(
            close(row["goal"], product(rotation(row["called_delta"]), row["called_current_matrix"]), 1e-6),
            "native orientation output matches actual call",
        )
        status = "PASS" if close(row["goal"], expected, 1e-6) else "FAIL"
        require(
            status == ("PASS" if variant == "native_pre_multiply" else "FAIL"),
            "noncommuting rotation control outcome",
        )
        orientation.append(
            {
                "case": case["id"],
                "variant": variant,
                "independent_native_reference_status": status,
                "affine_evaluator_status": "UNKNOWN",
                "reason": "Nonlinear orientation composition outside the admitted fragment",
                "expected": expected,
                "actual": row["goal"],
            }
        )
    grippers = []
    require(
        [r["variant"] for r in capture["gripper"]] == fixtures()["gripper_reset_variants"],
        "gripper variant inventory",
    )
    for data in capture["gripper"]:
        require(
            data["initial"] == [0] and len(data["rows"]) == len(fixtures()["gripper_commands"]),
            "native initial state and executed sequence",
        )
        previous, state = [0], [0.0, 0.0]
        for command, row in zip(fixtures()["gripper_commands"], data["rows"], strict=True):
            require(
                row["action"] == command and row["before"] == previous, "native gripper trajectory continuity"
            )
            state = grip(state, command)
            require(close(row["after"], state, 1e-12), "native sign/integration/clipping reference")
            previous = row["after"]
        require(
            data["populated"] == previous and max(abs(v) for v in previous) > 0.5, "populated pre-reset state"
        )
        expected_reset = data["variant"] == "native_reset"
        require(
            data["reset_called"] is expected_reset and data["same_gripper_object"] is True,
            "same-object native reset binding",
        )
        actual = data["after_reset_boundary"]
        require(actual == ([0] if expected_reset else previous), "actual reset/omitted-reset boundary")
        state_after = [0.0, 0.0] if expected_reset else state
        require(
            close(data["first_after_boundary"], grip(state_after, 1), 1e-12),
            "actual post-boundary first action",
        )
        grippers.append(
            {
                "variant": data["variant"],
                "independent_stateful_status": "PASS" if close(actual, [0], 1e-12) else "FAIL",
                "affine_evaluator_status": "UNKNOWN",
                "reason": "Stateful sign/integration/clipping and reset outside the admitted affine fragment",
                "native_format_action_calls": len(data["rows"]) + 1,
            }
        )
    return {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "affine": affine,
        "native_full_controller_checks": len(full),
        "orientation": orientation,
        "gripper": grippers,
        "checkpoint_linkage_status": "UNKNOWN",
        "unit": "One source-grounded Panda software cluster; conditions/queries are repeated measurements",
    }


def verify(folder):
    def read(name):
        return json.loads((folder / name).read_bytes())

    summary = read("summary.json")
    require(
        summary["schema_version"] == 1
        and summary["development_only"] is True
        and summary["final_evaluation"] is False,
        "development scope",
    )
    require(
        summary["source_file_sha256"] == sources() and set(summary["file_sha256"]) == FILES,
        "producer and closed manifest",
    )
    require(
        all(sha(folder / name) == expected for name, expected in summary["file_sha256"].items()),
        "exact saved evidence files",
    )
    require(
        read("inputs.json") == fixtures() and read("request.json") == json.loads(json.dumps(request())),
        "fixed independent inputs and request",
    )
    capture = read("capture.json")
    require(capture == json.loads((folder / "native.stdout.txt").read_bytes()), "raw native stdout binding")
    require(
        capture["source_file_sha256"] == sources()
        and capture["request_sha256"] == sha(folder / "request.json"),
        "native input/source binding",
    )
    require(
        capture["schema_version"] == 1
        and capture["development_only"] is True
        and capture["final_evaluation"] is False,
        "native development scope",
    )
    identity = capture["identity"]
    require(identity["semantic_git_blobs"] == SEMANTIC_BLOBS, "native semantic source identities")
    packages = dict(
        re.findall(r"(?m)^([\w.-]+)==([^\s\\]+)", (ROOT / "integrations/panda/requirements.lock").read_text())
    )
    pinned = distribution()
    require(identity["packages"] == packages, "complete pinned runtime packages")
    require(
        fingerprint(identity["record_file_sha256"]) == pinned["record_manifest_sha256"]
        and identity["record_sha256"] == pinned["installed_record_sha256"]
        and identity["bootstrap_source_sha256"] == pinned["bootstrap_wheel_only_sha256"],
        "complete pinned release source manifest and loader source",
    )
    require(
        identity["packages"]["robosuite"] == "1.4.1"
        and identity["packages"]["mujoco"] == "2.3.7"
        and len(identity["record_file_sha256"]) >= 1000
        and identity["record_checked_bytes"] > 190_000_000,
        "whole native wheel identity scope",
    )
    config = capture["native_config"]
    require(
        config["type"] == "OSC_POSE"
        and config["input_min"] == -1
        and config["input_max"] == 1
        and config["output_max"] == [0.05] * 3 + [0.5] * 3
        and config["output_min"] == [-0.05] * 3 + [-0.5] * 3
        and config["control_delta"] is True
        and config["position_limits"] is config["orientation_limits"] is None,
        "native source-defined controller configuration",
    )
    bootstrap = capture["bootstrap"]
    require(
        bootstrap["ctypes_loader_restored"] is True and bootstrap["mujoco_gl"] == "disable",
        "headless bootstrap and restored loader",
    )
    redirects = bootstrap["windows_dll_redirects"]
    require(
        len(redirects) == 1
        and redirects[0]["from"] == "robosuite/utils/mujoco.dll"
        and redirects[0]["to"] == "mujoco/mujoco.dll"
        and redirects[0]["gpu_macro_before"] is True
        and redirects[0]["gpu_macro_after"] is False,
        "scoped Windows import compatibility",
    )
    require(redirects[0]["target_sha256"] == pinned["mujoco_dll_sha256"], "native redirected DLL identity")
    sim = capture["simulation"]
    require(
        sim["environment"] == "Lift"
        and sim["robot"] == "Panda"
        and sim["physics_time_initial"] == sim["physics_time_final"] == 0
        and sim["env_step_calls"] == sim["controller_torque_calls"] == sim["render_calls"] == 0
        and sim["checkpoint_mapping"] == "UNKNOWN",
        "simulation-only lifecycle scope",
    )
    runtime = capture["runtime"]
    require(
        runtime["network_events"] == []
        and runtime["python"] == "3.11.13"
        and type(runtime["peak_rss_bytes"]) is int
        and 0 < runtime["peak_rss_bytes"] <= 2 * 1024**3
        and type(runtime["wall_seconds"]) in (int, float)
        and math.isfinite(runtime["wall_seconds"])
        and runtime["wall_seconds"] >= 0,
        "measured resource/network controls",
    )
    require(read("analysis.json") == analyze(capture), "deterministic independent analysis")
    require(
        not any(n in sys.modules for n in ("numpy", "robosuite", "mujoco", "torch", "lerobot", "openpi")),
        "data-only replay",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
    else:
        require(args.python is not None, "explicit locked native interpreter")
        folder = args.output.resolve()
        require(folder.is_relative_to(ROOT), "workspace-contained output")
        folder.mkdir(parents=True, exist_ok=False)
        initial = sources()
        write(folder / "inputs.json", fixtures())
        write(folder / "request.json", request())
        env = os.environ.copy()
        env.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            MUJOCO_GL="disable",
            NUMBA_CACHE_DIR=str(ROOT / ".test-tmp/panda-numba-cache"),
            PYTHONUTF8="1",
        )
        native = subprocess.run(
            [str(args.python.resolve()), "-I", str(ROOT / "scripts/panda_worker.py")],
            input=(folder / "request.json").read_bytes(),
            capture_output=True,
            timeout=180,
            env=env,
            check=False,
        )
        (folder / "native.stdout.txt").write_bytes(native.stdout)
        (folder / "native.stderr.txt").write_bytes(native.stderr)
        require(native.returncode == 0, f"actual native worker: {native.stderr[-2000:]!r}")
        capture = json.loads(native.stdout)
        write(folder / "capture.json", capture)
        write(folder / "analysis.json", analyze(capture))
        require(sources() == initial, "stable producer around native acquisition")
        write(
            folder / "summary.json",
            {
                "schema_version": 1,
                "development_only": True,
                "final_evaluation": False,
                "source_file_sha256": initial,
                "file_sha256": {n: sha(folder / n) for n in sorted(FILES)},
            },
        )
        verify(folder)
    print("Verified genuine Panda software operators, full goal setter and populated-state reset")


if __name__ == "__main__":
    main()
