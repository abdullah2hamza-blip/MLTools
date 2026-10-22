"""Rebuild public, fabricated calibration fixtures; never downloads weights/data."""

from copy import deepcopy
from pathlib import Path

from mltools.io import digest, write_json

ROOT = Path(__file__).parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REVISION = "5517a326eeb9b457cfe07d59b9bfdf291799c8d9"
CAMERAS = {
    "observation.images.front": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}
PREDICTION = [[0.5, 1, 1.5, 2, 2.5, 0.3], [1, 1.5, 2, 2.5, 3, 0.4], [1.5, 2, 2.5, 3, 3.5, 0.5]]


def tick(values, prediction=None, targets=None):
    event = {
        "kind": "tick",
        "state": dict(zip(JOINTS, values, strict=True)),
        "cameras": {
            "observation.images.front": [[11, 12], [13, 14]],
            "observation.images.wrist": [[21, 22], [23, 24]],
        },
    }
    if prediction is not None:
        event.update(prediction=prediction, training_targets=targets)
    return event


def build():
    (ROOT / "artifacts").mkdir(exist_ok=True)
    training = {
        "kind": "software-calibration-config-not-trained-checkpoint",
        "state_order": JOINTS,
        "action_names": JOINTS,
        "relative": True,
        "exclude_joints": ["gripper"],
        "camera_map": CAMERAS,
        "horizon": 3,
        "use_degrees": True,
        "weights": None,
        "dataset_revision": None,
    }
    stats = {"mean": [0, 0, 0, 0, 0, 50], "std": [2, 2, 2, 2, 2, 20]}
    stale = {"mean": [10, 10, 10, 10, 10, 0], "std": [2, 2, 2, 2, 2, 20]}
    for name, data in (("checkpoint_config", training), ("training_stats", stats), ("stale_stats", stale)):
        write_json(ROOT / "artifacts" / f"{name}.json", data)
    artifacts = [
        {"id": name, "path": f"artifacts/{name}.json", "sha256": digest(ROOT / "artifacts" / f"{name}.json")}
        for name in ("checkpoint_config", "training_stats", "stale_stats")
    ]
    contract = {
        "embodiment": "so101_follower/use_degrees=True/software-fixture",
        "authority": {
            "source": f"https://github.com/huggingface/lerobot/tree/{REVISION}/src/lerobot/robots/so_follower",
            "description": "Upstream joint names/units; fixture author independently specifies camera roles, "
            "relative body/absolute gripper, stats and targets. No trained checkpoint or hardware.",
        },
        "checks": ["artifacts", "cameras", "joint_order", "normalization", "actions", "reset"],
        "joints": [
            {
                "name": n,
                "unit": "range_0_100" if n == "gripper" else "degree",
                "mode": "absolute" if n == "gripper" else "relative",
            }
            for n in JOINTS
        ],
        "cameras": [
            {"source": s, "target": t, "role": "front" if s.endswith("front") else "wrist"}
            for s, t in CAMERAS.items()
        ],
        "normalization": {"kind": "mean_std", "artifact": "training_stats"},
        "horizon": 3,
    }
    adapter = {
        "state_order": JOINTS,
        "action_names": JOINTS,
        "motor_names": JOINTS,
        "camera_map": CAMERAS,
        "relative": True,
        "exclude_joints": ["gripper"],
        "bind_queue": True,
        "reset": True,
        "stats_artifact": "training_stats",
        "units": {n: "range_0_100" if n == "gripper" else "degree" for n in JOINTS},
    }
    scenario = [
        {"kind": "reset"},
        tick(
            [10, 20, 30, 40, 50, 60],
            PREDICTION,
            [[11, 22, 33, 44, 55, 56], [12, 23, 34, 45, 56, 58], [13, 24, 35, 46, 57, 60]],
        ),
        tick([11, 21, 31, 41, 51, 62]),
        tick([12, 22, 32, 42, 52, 64]),
        tick(
            [40, 50, 60, 70, 80, 90],
            PREDICTION,
            [[41, 52, 63, 74, 85, 56], [42, 53, 64, 75, 86, 58], [43, 54, 65, 76, 87, 60]],
        ),
        {"kind": "reset"},
        tick(
            [-10, -20, -30, -40, -50, 20],
            PREDICTION,
            [[-9, -18, -27, -36, -45, 56], [-8, -17, -26, -35, -44, 58], [-7, -16, -25, -34, -43, 60]],
        ),
        tick([-9, -19, -29, -39, -49, 22]),
        tick([-8, -18, -28, -38, -48, 24]),
        {"kind": "reset"},
    ]
    good = {
        "schema_version": 1,
        "contract": contract,
        "adapter": adapter,
        "artifacts": artifacts,
        "scenario": scenario,
    }
    cases = {"good": deepcopy(good)}
    for name in (
        "camera_swap",
        "joint_swap",
        "state_swap",
        "stale_stats",
        "moving_anchor",
        "relative_gripper",
        "no_reset",
        "absolute_body",
        "missing_identity",
        "unsupported_frames",
    ):
        cases[name] = deepcopy(good)
    cases["camera_swap"]["adapter"]["camera_map"] = dict(
        zip(CAMERAS, reversed(list(CAMERAS.values())), strict=True)
    )
    cases["joint_swap"]["adapter"]["motor_names"][0:2] = JOINTS[1::-1]
    cases["state_swap"]["adapter"]["state_order"][0:2] = JOINTS[1::-1]
    cases["stale_stats"]["adapter"]["stats_artifact"] = "stale_stats"
    cases["moving_anchor"]["adapter"]["bind_queue"] = False
    cases["relative_gripper"]["adapter"]["exclude_joints"] = []
    cases["no_reset"]["adapter"]["reset"] = False
    cases["absolute_body"]["adapter"]["relative"] = False
    cases["missing_identity"]["artifacts"][1]["sha256"] = None
    cases["unsupported_frames"]["contract"]["checks"].append("frames")
    for name, bundle in cases.items():
        write_json(ROOT / f"{name}.json", bundle)


if __name__ == "__main__":
    build()
