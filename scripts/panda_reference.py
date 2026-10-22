"""Independent source-grounded offline Panda contracts and fixture definitions."""

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_BLOBS = {
    "controllers/config/osc_pose.json": "8dc645e44bb13ba6806e7a74af52d1efaefc79e8",
    "controllers/base_controller.py": "895952a5a4e300ae58ec998cc379853d06fff689",
    "controllers/osc.py": "a45843d087ca2329456a53f5a86e87ded6e9c44a",
    "utils/control_utils.py": "18cc0a6b30cd8011f5267cc39542c86a01867e44",
    "models/grippers/panda_gripper.py": "64650e33d40f6df02d48032c91df3c71213d1751",
    "models/grippers/gripper_model.py": "5ebe18b3ae96aa2a0d04134071388dde1c75de10",
    "robots/single_arm.py": "598a168e0fdf9ba7e774bf75f349d5787448e287",
    "utils/transform_utils.py": "ac371c09da962dcceea8a376876fc1d46c0e52da",
}
EXPECTED = {
    "control": "PASS",
    "wrong_gain": "FAIL",
    "wrong_anchor": "FAIL",
    "wrong_order": "FAIL",
    "missing_authority": "UNKNOWN",
    "missing_units": "UNKNOWN",
}


def require(condition, message):
    if not condition:
        raise ValueError("Panda software contract: " + message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def distribution():
    return json.loads((ROOT / "integrations/panda/distribution.json").read_bytes())


def sources():
    paths = [
        *sorted((ROOT / "src/mltools").glob("*.py")),
        *(ROOT / "scripts" / name for name in ("panda_reference.py", "panda_worker.py", "validate_panda.py")),
        ROOT / "integrations/panda/requirements.in",
        ROOT / "integrations/panda/requirements.lock",
        ROOT / "integrations/panda/distribution.json",
    ]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def fixtures():
    return {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "authority": {
            "repository": "ARISE-Initiative/robosuite",
            "selected_semantic_source_ref": "fbee5844ff5632f5b5698e204ec5357ca50be0df",
            "semantic_git_blobs": SEMANTIC_BLOBS,
            "distribution": "robosuite==1.4.1",
            "wheel_sha256": "62587cd5888d9d3d813f3ffcc7017f6f77efa618952527a0b9bb139ffd0a3741",
            "scope": "Selected semantics match the Git ref; the complete wheel differs from that checkout",
        },
        "environment": {
            "name": "Lift",
            "robot": "Panda",
            "controller": "OSC_POSE",
            "control_freq": 20,
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "use_camera_obs": False,
            "initialization_noise": None,
            "hard_reset": False,
            "seed": 2026,
        },
        "affine_conditions": EXPECTED,
        "axes": [
            {
                "port": p,
                "name": str(i),
                "unit": "normalized_controller_input" if p == "action" else "sim_position_coordinate",
                "lower": -0.8 if p == "action" else -0.25,
                "upper": 0.8 if p == "action" else 0.25,
            }
            for p in ("action", "origin", "current")
            for i in range(3)
        ],
        "outputs": ["goal_x", "goal_y", "goal_z"],
        "translation_scale": [0.05] * 3,
        "rotation_scale": [0.5] * 3,
        "orientation_cases": [
            {
                "id": "noncommuting_x_then_z",
                "current_axisangle": [math.pi / 2, 0, 0],
                "delta_axisangle": [0, 0, math.pi / 2],
            },
            {
                "id": "noncommuting_y_then_x",
                "current_axisangle": [0, 0.7, 0],
                "delta_axisangle": [-0.4, 0, 0],
            },
        ],
        "gripper_commands": [0, 0.2, 0.7, -0.3, 0] + [1] * 110 + [-1] * 3,
        "gripper_speed": 0.01,
        "gripper_reset_variants": ["native_reset", "skip_reset_injection"],
        "full_controller_actions": [[0.3, -0.2, 0.1, 0.2, 0, -0.1], [-0.4, 0.1, 0.2, 0, 0.3, 0.1]],
        "limits": [
            "Source-defined simulator coordinates are not asserted to be calibrated SI metres",
            "No checkpoint-to-controller mapping, policy inference, task-success or hardware claim",
            "Rotation and gripper state are outside the admitted affine evaluator",
            "Translation fixtures supply explicit states to native operators",
            "Full controller sanity calls separately use actual simulator state",
        ],
    }


def rotation(axisangle):
    """Independent Rodrigues formula; no native quaternion helper is reused."""
    angle = math.hypot(*axisangle)
    if angle == 0:
        return [[float(i == j) for j in range(3)] for i in range(3)]
    x, y, z = (v / angle for v in axisangle)
    cross = [[0, -z, y], [z, 0, -x], [-y, x, 0]]
    square = product(cross, cross)
    return [
        [
            float(i == j) + math.sin(angle) * cross[i][j] + (1 - math.cos(angle)) * square[i][j]
            for j in range(3)
        ]
        for i in range(3)
    ]


def product(left, right):
    return [[math.fsum(left[i][k] * right[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def close(left, right, atol=1e-7):
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(close(a, b, atol) for a, b in zip(left, right, strict=True))
    return (
        type(left) in (int, float)
        and type(right) in (int, float)
        and math.isfinite(left)
        and math.isfinite(right)
        and abs(left - right) <= atol
    )


def grip(state, command):
    sign = (command > 0) - (command < 0)
    return [
        max(-1.0, min(1.0, v + direction * 0.01 * sign)) for v, direction in zip(state, (-1, 1), strict=True)
    ]
