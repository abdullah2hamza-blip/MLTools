import importlib.util

import pytest
from conftest import EXAMPLES

from mltools.checker import check
from mltools.io import digest, read_json, write_json
from mltools.lerobot import REVISION, capture
from mltools.reproduce import export_counterexample

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("lerobot") is None,
    reason="Install optional pinned integration: uv sync --extra lerobot",
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("good", "PASS"),
        ("camera_swap", "FAIL"),
        ("joint_swap", "FAIL"),
        ("state_swap", "FAIL"),
        ("stale_stats", "FAIL"),
        ("moving_anchor", "FAIL"),
        ("relative_gripper", "FAIL"),
        ("no_reset", "FAIL"),
        ("absolute_body", "FAIL"),
        ("missing_identity", "UNKNOWN"),
        ("unsupported_frames", "UNKNOWN"),
    ],
)
def test_actual_upstream_composed_training_and_deployment(name, expected):
    path = EXAMPLES / f"{name}.json"
    trace = capture(path)
    assert trace["execution"]["revision"] == REVISION
    assert len(trace["execution"]["git_blobs"]) == 8
    assert check(path, trace)["status"] == expected
    if name == "moving_anchor":
        assert trace["events"][2]["action"]["commands"]["shoulder_pan.pos"] == 13.0
    if name == "relative_gripper":
        assert trace["events"][1]["action"]["commands"]["gripper.pos"] == 116.0


@pytest.mark.parametrize("case", ["moving_anchor", "no_reset", "camera_swap"])
def test_exported_regression_reexecutes_real_fault(tmp_path, case):
    path = EXAMPLES / f"{case}.json"
    trace = capture(path)
    report = check(path, trace)
    destination = tmp_path / "repro"
    export_counterexample(path, trace, report, destination)
    reduced = destination / "bundle.json"
    assert check(reduced, capture(reduced))["status"] == "FAIL"
    if case == "no_reset":
        assert len(read_json(reduced)["scenario"]) == 6


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("use_degrees", False, "joint_order"),
        ("horizon", 64, "actions"),
        (
            "camera_map",
            {
                "observation.images.front": "observation.images.camera2",
                "observation.images.wrist": "observation.images.camera1",
            },
            "cameras",
        ),
    ],
)
def test_training_metadata_contradiction_cannot_pass(sample, field, value, failed_check):
    path, bundle, _ = sample
    config_path = path.parent / "artifacts" / "checkpoint_config.json"
    config = read_json(config_path)
    config[field] = value
    write_json(config_path, config)
    bundle["artifacts"][0]["sha256"] = digest(config_path)
    write_json(path, bundle)
    assert check(path, capture(path))["checks"][failed_check] == "FAIL"


def test_training_state_permutation_is_detected_even_without_relative_actions(sample):
    path, bundle, _ = sample
    for joint in bundle["contract"]["joints"]:
        joint["mode"] = "absolute"
    bundle["adapter"]["relative"] = False
    config_path = path.parent / "artifacts" / "checkpoint_config.json"
    config = read_json(config_path)
    config["relative"] = False
    config["state_order"][0:2] = config["state_order"][1::-1]
    write_json(config_path, config)
    bundle["artifacts"][0]["sha256"] = digest(config_path)
    write_json(path, bundle)
    report = check(path, capture(path))
    assert report["checks"]["joint_order"] == "FAIL"
    assert report["first_divergence"]["boundary"] == "training.state_names"
