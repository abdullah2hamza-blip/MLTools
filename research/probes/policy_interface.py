"""Constructed examples for the policy-interface research memo.

These are specification examples, not measurements of an upstream framework.
Each faulty adapter passes a weaker local check. An independently declared
interface contract detects the changed observation or physical action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def camera_roles(swapped: bool) -> dict:
    training_roles = {"camera1": "left", "camera2": "right", "camera3": "wrist"}
    deployment_roles = dict(training_roles)
    if swapped:
        deployment_roles["camera2"], deployment_roles["camera3"] = (
            deployment_roles["camera3"], deployment_roles["camera2"]
        )
    # Distinct observations, with identical shapes and numeric types.
    images = {"left": (1.0,) * 4, "right": (2.0,) * 4, "wrist": (3.0,) * 4}
    expected = {k: images[v] for k, v in training_roles.items()}
    actual = {k: images[v] for k, v in deployment_roles.items()}
    return {
        "case": "camera_role_swap",
        "weak_check": "keys, shapes, and numeric types match",
        "weak_check_passes": expected.keys() == actual.keys()
        and all(len(expected[k]) == len(actual[k]) for k in expected),
        "independent_contract_passes": expected == actual,
        "expected_roles": training_roles,
        "actual_roles": deployment_roles,
    }


def motor_order(swapped: bool) -> dict:
    physical = {"shoulder": 0.1, "elbow": 0.8, "gripper": 0.3}
    checkpoint_order = ["shoulder", "elbow", "gripper"]
    adapter_order = ["elbow", "shoulder", "gripper"] if swapped else checkpoint_order
    encoded = [physical[k] for k in adapter_order]
    decoded = dict(zip(adapter_order, encoded, strict=True))
    return {
        "case": "joint_order",
        "weak_check": "adapter encode/decode round trip",
        "weak_check_passes": decoded == physical,
        "independent_contract_passes": encoded == [physical[k] for k in checkpoint_order],
        "expected_checkpoint_order": checkpoint_order,
        "actual_adapter_order": adapter_order,
    }


def chunk_anchor(moving: bool) -> dict:
    chunk_state = 10.0
    offsets = [1.0, 2.0, 3.0]
    observed_states = [10.0, 11.0, 12.0]
    anchors = observed_states if moving else [chunk_state] * len(offsets)
    targets = [a + d for a, d in zip(anchors, offsets, strict=True)]
    reconstructed = [t - a for t, a in zip(targets, anchors, strict=True)]
    expected = [chunk_state + d for d in offsets]
    return {
        "case": "chunk_anchor",
        "weak_check": "relative/absolute round trip at each control tick",
        "weak_check_passes": reconstructed == offsets,
        "independent_contract_passes": targets == expected,
        "expected_targets": expected,
        "actual_targets": targets,
        "maximum_target_error": max(abs(t - e) for t, e in zip(targets, expected, strict=True)),
    }


def normalization(stale: bool) -> dict:
    training_mean, training_std = 0.0, 2.0
    deployment_mean, deployment_std = (10.0, 2.0) if stale else (0.0, 2.0)
    physical_target = 1.0
    local_encoded = (physical_target - deployment_mean) / deployment_std
    local_decoded = local_encoded * deployment_std + deployment_mean
    model_output = (physical_target - training_mean) / training_std
    actual = model_output * deployment_std + deployment_mean
    return {
        "case": "normalization_provenance",
        "weak_check": "deployment normalize/denormalize round trip",
        "weak_check_passes": local_decoded == physical_target,
        "independent_contract_passes": actual == physical_target,
        "expected_physical_target": physical_target,
        "actual_physical_target": actual,
    }


def run() -> dict:
    cases = (camera_roles, motor_order, chunk_anchor, normalization)
    faulty = [case(True) for case in cases]
    controls = [case(False) for case in cases]
    assert all(x["weak_check_passes"] for x in faulty + controls)
    assert all(not x["independent_contract_passes"] for x in faulty)
    assert all(x["independent_contract_passes"] for x in controls)
    return {
        "evidence_kind": "constructed standard-library specification examples",
        "scope": "four deliberately selected failure modes; no model, GPU, or robot",
        "faulty_cases": faulty,
        "correct_controls": controls,
        "summary": {
            "faulty_cases_passing_weak_checks": len(faulty),
            "faulty_cases_rejected_by_independent_contract": len(faulty),
            "correct_controls_accepted": len(controls),
        },
        "limitations": [
            "Contracts are supplied independently; the probe cannot infer intended semantics.",
            "A shared mistake in the contract and adapter can remain undetected.",
            "The selected cases do not measure general detection accuracy or customer value.",
            "Interface equivalence does not establish learned competence or physical safety.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(run(), indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")
