from copy import deepcopy

import pytest
from conftest import EXAMPLES

from mltools.checker import check
from mltools.io import InputError, digest, read_json, write_json
from mltools.reproduce import export_counterexample


def rebind(path, bundle, trace):
    write_json(path, bundle)
    trace["bundle_sha256"] = digest(path)


def test_real_recorded_control_and_known_physical_targets(sample):
    path, _, trace = sample
    report = check(path, trace)
    assert report["status"] == "PASS"
    assert trace["events"][1]["action"]["commands"]["shoulder_pan.pos"] == 11.0
    assert trace["events"][2]["action"]["commands"]["shoulder_pan.pos"] == 12.0
    assert trace["events"][1]["action"]["commands"]["gripper.pos"] == 56.0
    assert trace["events"][6]["action"]["commands"]["shoulder_pan.pos"] == -9.0


def test_actual_moving_anchor_counterexample():
    report = check(EXAMPLES / "moving_anchor.json", EXAMPLES / "moving_anchor.trace.json")
    first = report["first_divergence"]
    assert report["status"] == "FAIL"
    assert first["event"] == 2
    assert first["expected"]["shoulder_pan.pos"] == 12.0
    assert first["actual"]["shoulder_pan.pos"] == 13.0
    assert len(report["counterexample"]["events"]) == 3


@pytest.mark.parametrize("missing", ["authority", "unit", "mode", "camera_role", "stats_identity", "units"])
def test_missing_semantic_evidence_cannot_pass(sample, missing):
    path, bundle, trace = sample
    if missing == "authority":
        bundle["contract"].pop("authority")
    elif missing in ("unit", "mode"):
        bundle["contract"]["joints"][0].pop(missing)
    elif missing == "camera_role":
        bundle["contract"]["cameras"][0].pop("role")
    elif missing == "stats_identity":
        bundle["artifacts"][1]["sha256"] = None
    else:
        trace["events"][1]["action"].pop("units")
    rebind(path, bundle, trace)
    assert check(path, trace)["status"] == "UNKNOWN"


def test_no_trace_is_unknown(sample):
    assert check(sample[0])["status"] == "UNKNOWN"


def test_unsupported_semantics_are_unknown(sample):
    path, bundle, trace = sample
    bundle["contract"]["checks"].append("rotation_frames")
    rebind(path, bundle, trace)
    report = check(path, trace)
    assert report["checks"]["rotation_frames"] == "UNKNOWN"
    assert report["status"] == "UNKNOWN"


def test_artifact_change_is_failure_even_if_trace_looks_good(sample):
    path, _, trace = sample
    stats = path.parent / "artifacts" / "training_stats.json"
    data = read_json(stats)
    data["mean"][0] = 999
    write_json(stats, data)
    report = check(path, trace)
    assert report["status"] == "FAIL"
    assert report["checks"]["normalization"] == "UNKNOWN"


def test_missing_artifact_is_unknown(sample):
    path, _, trace = sample
    (path.parent / "artifacts" / "training_stats.json").unlink()
    assert check(path, trace)["status"] == "UNKNOWN"


def test_changed_bundle_rejects_stale_trace(sample):
    path, bundle, trace = sample
    bundle["adapter"]["bind_queue"] = False
    write_json(path, bundle)
    with pytest.raises(InputError, match="digest mismatch"):
        check(path, trace)


def test_report_is_roundtrip_json(sample, tmp_path):
    report = check(sample[0], sample[2])
    write_json(tmp_path / "report.json", report)
    assert read_json(tmp_path / "report.json") == report


@pytest.mark.parametrize("boundary", ["cameras", "state_values", "action"])
def test_absent_trace_boundary_is_unknown(sample, boundary):
    path, _, trace = sample
    trace["events"][1].pop(boundary)
    assert check(path, trace)["status"] == "UNKNOWN"


def test_exported_moving_anchor_replays_failure(tmp_path):
    path = EXAMPLES / "moving_anchor.json"
    trace = read_json(EXAMPLES / "moving_anchor.trace.json")
    report = check(path, trace)
    output = tmp_path / "repro"
    export_counterexample(path, trace, report, output)
    assert check(output / "bundle.json", output / "trace.json")["status"] == "FAIL"
    assert len(read_json(output / "bundle.json")["scenario"]) == 3
    with pytest.raises(FileExistsError):
        export_counterexample(path, trace, report, output)


def test_reference_is_not_derived_from_deployment_roundtrip(sample):
    path, _, trace = sample
    swapped = deepcopy(trace)
    for event in swapped["events"]:
        if event.get("action"):
            commands = event["action"]["commands"]
            commands["shoulder_pan.pos"], commands["elbow_flex.pos"] = (
                commands["elbow_flex.pos"],
                commands["shoulder_pan.pos"],
            )
    assert check(path, swapped)["checks"]["actions"] == "FAIL"


@pytest.mark.parametrize("malformed", [[], [1], "command"])
def test_malformed_recorded_action_is_operational_error(sample, malformed):
    path, _, trace = sample
    trace["events"][1]["action"] = malformed
    with pytest.raises(InputError, match="action must be an object"):
        check(path, trace)


def test_execution_scope_is_required(sample):
    path, _, trace = sample
    trace.pop("execution")
    assert check(path, trace)["status"] == "UNKNOWN"


def test_indistinguishable_camera_content_cannot_prove_role_mapping(sample):
    path, bundle, trace = sample
    for event in bundle["scenario"]:
        if event["kind"] == "tick":
            event["cameras"]["observation.images.front"] = [[1, 2], [3, 4]]
            event["cameras"]["observation.images.wrist"] = [[1.0, 2.0], [3.0, 4.0]]
    rebind(path, bundle, trace)
    assert check(path, trace)["checks"]["cameras"] == "UNKNOWN"


def test_missing_output_units_cannot_be_interpreted_as_wrong_degrees(sample):
    path, _, trace = sample
    for event in trace["events"]:
        if event.get("action"):
            event["action"].pop("units")
            event["action"]["commands"]["shoulder_pan.pos"] = 0.19
    assert check(path, trace)["checks"]["actions"] == "UNKNOWN"


def test_unsupported_normalization_does_not_parse_other_statistics_as_mean_std(sample):
    path, bundle, trace = sample
    stats_path = path.parent / "artifacts" / "training_stats.json"
    write_json(stats_path, {"min": [0] * 6, "max": [100] * 6})
    bundle["contract"]["normalization"]["kind"] = "min_max"
    bundle["artifacts"][1]["sha256"] = digest(stats_path)
    rebind(path, bundle, trace)
    assert check(path, trace)["checks"]["normalization"] == "UNKNOWN"


def test_partial_output_units_are_unknown_not_a_shape_failure(sample):
    path, _, trace = sample
    trace["events"][1]["action"]["units"].pop("shoulder_pan")
    trace["events"][1]["action"]["commands"]["shoulder_pan.pos"] = 0.19
    assert check(path, trace)["checks"]["actions"] == "UNKNOWN"


def test_missing_camera_frame_cannot_prove_routing(sample):
    path, bundle, trace = sample
    bundle["scenario"][1]["cameras"]["observation.images.front"] = None
    rebind(path, bundle, trace)
    assert check(path, trace)["checks"]["cameras"] == "UNKNOWN"
