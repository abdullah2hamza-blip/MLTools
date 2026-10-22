"""Independently authored synthetic controls and elementary CPU score expectations."""

import copy
import math


def require(condition, message):
    if not condition:
        raise ValueError(f"ActionABI development evidence: {message}")


def contract(**changes):
    value = {
        "schema_version": "1.0",
        "target": "delta_position",
        "space": "joint",
        "frame": "unspecified",
        "permutation": [0, 1],
        "sign": [1, 1],
        "scale": [1, 1],
        "lag_steps": 0,
        "gripper_inverted": False,
    }
    return value | changes


def authority():
    return {
        "schema_version": 1,
        "synthetic_only": True,
        "final_evaluation": False,
        "description": "Authored deterministic two-coordinate response fixtures, not robot trajectories",
        "timestamps_ns": [0, 500000000, 1500000000, 2000000000],
        "states": [[[10, -3], [11, -1], [9, 2], [12, 1]], [[2, 5], [4, 4], [3, 7], [7, 5]]],
        "actions": [[[1, 2], [-2, 3], [3, -1], [0, 0]], [[2, -1], [-1, 3], [4, -2], [0, 0]]],
        "intended_contract": contract(),
        "physical_authority": "Synthetic joint/radian coordinate definitions only; no hardware meaning",
        "fixed_fields": ["space", "frame"],
        "calibration_claim": False,
    }


def plans():
    variants = [
        contract(),
        contract(sign=[-1, 1]),
        contract(permutation=[1, 0]),
        contract(target="absolute_position"),
        contract(target="velocity"),
        contract(lag_steps=1),
        contract(scale=[2, 1]),
        contract(gripper_inverted=True),
    ]
    return {
        "identified": {"dataset": "excited", "hypotheses": variants, "unsupported_fields": []},
        "unexcited": {
            "dataset": "constant",
            "hypotheses": [contract(), contract(permutation=[1, 0])],
            "unsupported_fields": [],
        },
        "structural_ambiguity": {
            "dataset": "inverted_last",
            "hypotheses": [contract(sign=[1, -1]), contract(gripper_inverted=True)],
            "unsupported_fields": [],
        },
        "unsupported": {
            "dataset": "excited",
            "hypotheses": variants,
            "unsupported_fields": ["space", "frame"],
        },
        "misspecified_grammar": {
            "dataset": "excited",
            "hypotheses": [contract(sign=[-1, 1])],
            "unsupported_fields": [],
        },
    }


def samples(kind):
    declared = authority()
    states, actions = copy.deepcopy(declared["states"]), copy.deepcopy(declared["actions"])
    if kind == "constant":
        states = [[[10, -3]] * 4, [[2, 5]] * 4]
        actions = [[[0, 0]] * 4, [[0, 0]] * 4]
    elif kind == "inverted_last":
        states = [[[10, -3], [11, -5], [9, -8], [12, -7]], [[2, 5], [4, 6], [3, 3], [7, 5]]]
    else:
        require(kind == "excited", "fixed synthetic dataset kind")
    return [
        {"record_type": "sample", "episode_id": episode, "t_ns": timestamp, "state": state, "action": action}
        for episode in range(2)
        for timestamp, state, action in zip(
            declared["timestamps_ns"], states[episode], actions[episode], strict=True
        )
    ]


def prediction(action, hypothesis):
    """Caller model predictions, not observations or inferred physical authority."""
    decoded = [
        action[p] * sign * scale
        for p, sign, scale in zip(
            hypothesis["permutation"], hypothesis["sign"], hypothesis["scale"], strict=True
        )
    ]
    if hypothesis["gripper_inverted"]:
        decoded[-1] *= -1
    return decoded


def probe_request(name, pulses):
    require(
        name in ("separating", "first_unsafe", "all_unsafe", "inseparable", "greedy"), "probe control name"
    )
    hypotheses = [contract(), contract(permutation=[1, 0])]
    if name == "inseparable":
        hypotheses = [contract(sign=[1, -1]), contract(gripper_inverted=True)]
    elif name == "greedy":
        hypotheses = [contract(scale=[0.5 + i / 10, 1]) for i in range(21)]
    candidates = [
        {
            "action": pulse,
            "predicted_observations": [prediction(pulse, h) for h in hypotheses],
            "safe": name != "all_unsafe" and (name != "first_unsafe" or i != 0),
        }
        for i, pulse in enumerate(pulses)
    ]
    return {
        "operation": "select",
        "observation_tolerance": 0.0,
        "hypotheses": hypotheses,
        "candidates": candidates,
    }


def expected_score(rows, hypothesis):
    """Independent direct Huber calculation for these finite, noise-free controls."""
    losses, dimensions = [], []
    for episode in (0, 1):
        records = [r for r in rows if r["episode_id"] == episode]
        per_axis = [[], []]
        offset = hypothesis["lag_steps"] + 1
        for index in range(len(records) - offset):
            before, after = records[index + offset - 1], records[index + offset]
            dt = (after["t_ns"] - before["t_ns"]) / 1e9
            for j, command in enumerate(prediction(records[index]["action"], hypothesis)):
                observed = after["state"][j]
                if hypothesis["target"] != "absolute_position":
                    observed -= before["state"][j]
                if hypothesis["target"] == "velocity":
                    observed /= dt
                residual = abs(command - observed)
                per_axis[j].append(residual**2 / 2 if residual <= 1 else residual - 0.5)
        flattened = [v for axis in per_axis for v in axis]
        losses.append((sum(flattened), len(flattened)))
        dimensions.append([sum(axis) / len(axis) for axis in per_axis])
    return {
        "train_loss": losses[0][0] / losses[0][1],
        "heldout_loss": losses[1][0] / losses[1][1],
        "residual_count": sum(count for _loss, count in losses),
        "per_dimension_loss": dimensions[1],
    }


def close(actual, expected):
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
