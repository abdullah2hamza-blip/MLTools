"""Compare two pinned baseline modules on one constructed interface mismatch.

This checks module coverage, not the complete products. Source snapshots remain
outside this repository; their Git blob identities are verified before import.
The baseline source has been inspected: these modules use the standard library
and have no import-time network, subprocess, or filesystem side effects.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


SOURCES = {
    "tether": {
        "file": "tether_action_execution_cert.py",
        "blob": "40b136c7ee8703bab133a43fa3180da20be44b2d",
        "url": "https://github.com/FastCrest/tether/blob/dcbcbb5a669fa18ae07b1c22d17bd5bd386a33da/src/tether/action_execution_cert.py",
    },
    "preflight": {
        "file": "robot_policy_preflight.py",
        "blob": "4621c449515b7c9193459db31c29b68c6a225b0c",
        "url": "https://github.com/gbesse/robot-policy-preflight/blob/09137467bbf988bf5f4cc7418f8e300eaa6dd191/robot_policy_preflight.py",
    },
}


def load(name: str, root: Path):
    source = SOURCES[name]
    path = root / source["file"]
    payload = path.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
    if blob != source["blob"]:
        raise ValueError(f"Unrecognized {name} snapshot: {blob}")
    spec = importlib.util.spec_from_file_location(f"research_baseline_{name}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(root: Path) -> dict:
    tether = load("tether", root)
    preflight = load("preflight", root)
    names = ["shoulder", "elbow", "gripper"]
    contract = {
        "fps": 20,
        "observations": {"state": {"shape": [3], "dtype": "float32", "unit": "rad"}},
        "action": {
            "shape": [3], "dtype": "float32", "unit": "rad", "range": [-2, 2],
            "names": names,
        },
        "normalization": {"action": {"mean": [0, 0, 0], "std": [1, 1, 1]}},
    }
    vector = [0.02, 0.04, 0.06]
    receipt = {
        "act_samples": [
            {
                "sample": n,
                "roundtrip_ms": 10,
                "actions": [vector[:] for _ in range(3)],
                "action_execution": {"executed_horizon": 3, "scheduler": "fixture"},
            }
            for n in (1, 2)
        ]
    }
    results = []
    for faulty in (False, True):
        adapter_order = ["elbow", "shoulder", "gripper"] if faulty else names
        expected = dict(zip(names, vector, strict=True))
        actual = dict(zip(adapter_order, vector, strict=True))
        static = preflight.check(contract, contract, contract)
        execution = tether.build_action_execution_certificate(receipt, control_hz=20)
        independent = expected == actual
        assert static["status"] == "pass"
        assert execution["passed"]
        assert independent is not faulty
        results.append({
            "fixture": "permuted_runtime_adapter" if faulty else "correct_runtime_adapter",
            "robot_policy_preflight": static["status"],
            "tether_action_execution_certificate": execution["decision"],
            "tether_check_counts": execution["summary"],
            "independent_named_action_comparison": "PASS" if independent else "FAIL",
            "expected_named_commands": expected,
            "actual_named_commands": actual,
        })
    return {
        "evidence_kind": "executed pinned baseline modules on constructed fixtures",
        "sources": SOURCES,
        "results": results,
        "interpretation": (
            "Static interface checks and numeric execution telemetry can both pass while "
            "a runtime adapter sends the same vector components to different named joints."
        ),
        "limits": [
            "One constructed semantic mismatch and one correct control, not a general benchmark.",
            "The full Tether release/validation workflow and Policy Space were not run.",
            "Neither baseline module promises this independent runtime semantics check.",
            "This shows coverage boundaries, not a defect in either baseline.",
            "No learned policy or robot was executed; the independent contract is supplied.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(compare(args.source_dir), indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
