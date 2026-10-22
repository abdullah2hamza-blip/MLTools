"""Same-observation replay, partial-unit guards and independently bound witnesses."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import compare_stateful_development as comparison  # noqa: E402

from mltools.checker import check  # noqa: E402
from mltools.io import digest, read_json  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8", newline="\n")


@pytest.mark.parametrize(
    "tamper",
    [None, "state", "unexpected_control", "request", "manifest", "old_relation", "compiled_relation"],
)
def test_optimized_stateful_evidence(tmp_path, tamper):
    source = ROOT / "validation/stateful-comparison-001"
    target = tmp_path / "evidence"
    shutil.copytree(source, target)
    summary_path = target / "summary.json"
    summary = read_json(summary_path)
    if tamper == "manifest":
        del summary["file_sha256"]["full.stderr.txt"]
    elif tamper == "request":
        request_path = target / "full.request.json"
        data = read_json(request_path)
        data["bundles"]["control"] = data["bundles"]["moving_anchor"]
        write(request_path, data)
    elif tamper in ("state", "unexpected_control"):
        data = read_json(target / "full.capture.json")
        trace = data["captures"]["control"]
        if tamper == "state":
            trace["events"][1]["state_values"][0] += 1
        else:
            # Recompute every dependent verdict: hash consistency alone must
            # not turn an independently specified correct control into FAIL.
            trace["events"][2]["action"]["commands"]["shoulder_pan.pos"] += 10
            old, compiled, matched, _ = comparison.analyze(
                target / "inputs/control.bundle.json", trace, "control"
            )
            assert old["status"] == compiled["status"] == "FAIL"
            write(target / "control.old-report.json", old)
            write(target / "control.compiled-report.json", compiled)
            write(target / "control.comparison.json", matched)
            summary["cases"]["control"] = matched
        write(target / "full.capture.json", data)
        write(target / "full.stdout.txt", data)
    elif tamper in ("old_relation", "compiled_relation"):
        data = read_json(target / "witness.capture.json")
        if tamper == "old_relation":
            trace = data["captures"]["moving_anchor:old"]
            # Still FAIL, but at an earlier event for a different coordinate.
            trace["events"][1]["action"]["commands"]["shoulder_lift.pos"] += 10
            report = check(target / "inputs/moving_anchor.old-witness.bundle.json", trace)
            assert report["status"] == "FAIL"
            write(target / "moving_anchor.old-witness-report.json", report)
        else:
            trace = data["captures"]["wrong_stats:compiled"]
            commands = trace["events"][2]["action"]["commands"]
            commands["shoulder_pan.pos"] = 1.0
            commands["gripper.pos"] += 10
            suite = comparison.Suite.from_dict(read_json(target / "wrong_stats.witness-suite.json"))
            _, report, matched, _ = comparison.analyze(
                target / "inputs/wrong_stats.compiled-witness.bundle.json", trace, "wrong_stats", suite
            )
            assert matched["compiled_matched_status"] == "FAIL"
            assert report["first_divergence"]["output"] == "gripper.pos"
            write(target / "wrong_stats.compiled-witness-report.json", report)
        write(target / "witness.capture.json", data)
        write(target / "witness.stdout.txt", data)
    for name in summary["file_sha256"]:
        summary["file_sha256"][name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    write(summary_path, summary)
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            str(ROOT / "scripts/compare_stateful_development.py"),
            "--verify",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) is (tamper is None)
    if tamper:
        assert "Stateful comparison:" in result.stderr
        assert "Verified same-observation" not in result.stdout
    else:
        assert "Verified same-observation" in result.stdout


def test_partial_units_retain_grounded_other_failures(tmp_path):
    source = ROOT / "validation/stateful-comparison-001"
    target = tmp_path / "evidence"
    shutil.copytree(source, target)
    data = read_json(target / "full.capture.json")
    trace = copy.deepcopy(data["captures"]["wrong_stats"])
    bundle_path = target / "inputs/wrong_stats.bundle.json"
    bundle = read_json(bundle_path)
    # Withhold the gripper unit, while the shoulder mismatch stays grounded.
    del bundle["adapter"]["units"]["gripper"]
    write(bundle_path, bundle)
    trace["bundle_sha256"] = digest(bundle_path)
    for event in trace["events"]:
        if event["kind"] == "tick":
            del event["action"]["units"]["gripper"]
    _, report, matched, _ = comparison.analyze(bundle_path, trace, "wrong_stats")
    assert matched["common_authority_and_unit_status"] == "UNKNOWN"
    assert matched["original_matched_status"] == matched["compiled_matched_status"] == "FAIL"
    assert report["first_divergence"]["output"] == "shoulder_pan.pos"


def test_supported_unit_mismatch_has_precedence():
    bundle = comparison.inputs()["control.bundle.json"]
    units = {j["name"]: j["unit"] for j in bundle["contract"]["joints"]}
    units["gripper"] = "unestablished"
    assert comparison.unit_status(bundle, {"action": {"units": units}}) == "UNKNOWN"
    units["shoulder_pan"] = "radian"
    assert comparison.unit_status(bundle, {"action": {"units": units}}) == "FAIL"
