"""Versioned workflow tests; manufactured traces are labeled unit fixtures."""

import copy
import importlib.util
import json
import os
from pathlib import Path

import pytest

from mltools import probe_cases
from mltools.cli import main
from mltools.io import InputError, digest, read_json, write_json
from mltools.lerobot import BLOBS, REVISION
from mltools.obligations import Axis, Reference, compile_suite


@pytest.fixture
def probe_case(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    stats = root / "stats.json"
    write_json(stats, {"mean": [3.0, 5.0], "std": [2.0, 4.0]})
    axes = tuple(
        Axis(
            port,
            str(i),
            "normalized" if port == "action" else "degree" if i == 0 else "range_0_100",
            0 if port != "action" and i == 1 else -1 if port == "action" else -10,
            1 if port == "action" else 100 if i == 1 else 10,
        )
        for port in ("action", "origin", "current")
        for i in range(2)
    )
    suite = compile_suite(axes, ("joint.pos", "gripper.pos"))
    ref = Reference(
        ((2, 0, 10, 0, 0, 0), (0, 4, 0, 0, 0, 0)),
        (3, 5),
        ("degree", "range_0_100"),
        "Independent synthetic unit fixture",
        True,
    )
    case = {
        "schema_version": 1,
        "format": "mltools-affine-case",
        "development_only": True,
        "suite": suite.to_dict(),
        "reference": ref.to_dict(),
        "artifacts": [{"id": "stats", "path": "stats.json", "sha256": digest(stats)}],
        "adapter": {
            "framework": "lerobot",
            "stats_artifact": "stats",
            "configuration": {
                "relative": True,
                "bind_queue": True,
                "exclude_joints": ["gripper"],
                "action_names": ["joint", "gripper"],
                "motor_names": ["joint", "gripper"],
            },
        },
    }
    path = root / "case.json"
    write_json(path, case)
    return path, case, suite


def manufactured_trace(path, case, suite, *, moving=False):
    return {
        "schema_version": 1,
        "case_sha256": digest(path),
        "suite_sha256": suite.sha256,
        "framework": "lerobot",
        "configuration": copy.deepcopy(case["adapter"]["configuration"]),
        "identity": {"revision": REVISION, "git_blobs": dict(BLOBS)},
        "stats_sha256": case["artifacts"][0]["sha256"],
        "unit_test_only": True,
        "outputs": {
            p.id: [2 * p.values[0] + p.values[4 if moving else 2] + 3, 4 * p.values[1] + 5]
            for p in suite.probes
        },
    }


def test_data_only_replay_requires_all_provenance_and_independent_obligations(probe_case):
    path, case, suite = probe_case
    trace = manufactured_trace(path, case, suite)
    assert probe_cases.check(path, trace)["status"] == "PASS"
    assert probe_cases.check(path)["status"] == "UNKNOWN"
    del trace["identity"]
    assert probe_cases.check(path, trace)["status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "field", ["case_sha256", "suite_sha256", "framework", "configuration", "stats_sha256"]
)
def test_changed_binding_cannot_hide_behind_matching_numbers(probe_case, field):
    path, case, suite = probe_case
    trace = manufactured_trace(path, case, suite)
    trace[field] = "wrong"
    assert probe_cases.check(path, trace)["status"] == "FAIL"


def test_unsupported_requested_checks_remain_unknown(probe_case):
    path, case, suite = probe_case
    case["checks"] = ["camera_extrinsics"]
    write_json(path, case)
    report = probe_cases.check(path, manufactured_trace(path, case, suite))
    assert report["status"] == "UNKNOWN"
    assert report["checks"]["camera_extrinsics"] == "UNKNOWN"


def test_missing_hash_cannot_be_inferred_from_observed_file(probe_case):
    path, case, suite = probe_case
    case["artifacts"][0]["sha256"] = None
    write_json(path, case)
    trace = manufactured_trace(path, case, suite)
    trace["stats_sha256"] = digest(path.parent / "stats.json")
    assert probe_cases.check(path, trace)["status"] == "UNKNOWN"


def test_reference_erasure_retains_other_known_failure(probe_case):
    path, case, suite = probe_case
    case["reference"]["offset"] = [None, None]
    write_json(path, case)
    report = probe_cases.check(path, manufactured_trace(path, case, suite, moving=True))
    assert report["status"] == "FAIL"
    assert report["numeric_report"]["first_divergence"]["obligation"] == "effect:origin:0"


def test_export_preserves_original_binding_and_does_not_fabricate_witness_trace(probe_case, tmp_path):
    path, case, suite = probe_case
    trace = manufactured_trace(path, case, suite, moving=True)
    report = probe_cases.check(path, trace)
    destination = tmp_path / "regression"
    probe_cases.export(path, trace, report, destination)
    info = read_json(destination / "export.json")
    assert (
        probe_cases.check(destination / info["offline_case"], destination / info["offline_trace"]) == report
    )
    child = destination / info["witness_case"]
    assert len(probe_cases.load(child)[2].probes) == 2
    assert probe_cases.check(child)["status"] == "UNKNOWN"
    assert not (destination / "witness.trace.json").exists()
    assert digest(destination / info["offline_case"]) == digest(path)


def test_export_handles_artifact_names_that_match_control_files(probe_case, tmp_path):
    path, case, suite = probe_case
    (path.parent / "stats.json").rename(path.parent / "report.json")
    case["artifacts"][0]["path"] = "report.json"
    write_json(path, case)
    trace = manufactured_trace(path, case, suite, moving=True)
    report = probe_cases.check(path, trace)
    destination = tmp_path / "regression"
    probe_cases.export(path, trace, report, destination)
    assert digest(destination / "original/report.json") == digest(path.parent / "report.json")
    assert read_json(destination / "report.json")["status"] == "FAIL"


def test_export_finds_free_witness_name_with_maximum_reserved_artifacts(probe_case, tmp_path):
    path, case, suite = probe_case
    for i in range(31):
        file = path.parent / f"witness-{i}.case.json"
        write_json(file, {"metadata": i})
        case["artifacts"].append({"id": f"artifact_{i}", "path": file.name, "sha256": digest(file)})
    (path.parent / "stats.json").rename(path.parent / "witness-31.case.json")
    case["artifacts"][0]["path"] = "witness-31.case.json"
    renamed = path.parent / "witness-32.case.json"
    path.rename(renamed)
    write_json(renamed, case)
    trace = manufactured_trace(renamed, case, suite, moving=True)
    report = probe_cases.check(renamed, trace)
    destination = tmp_path / "regression"
    probe_cases.export(renamed, trace, report, destination)
    child = destination / read_json(destination / "export.json")["witness_case"]
    assert child.is_file()
    assert len(probe_cases.load(child)[2].probes) == 2


def test_export_and_cli_do_not_overwrite_inputs(probe_case, tmp_path, capsys):
    path, case, suite = probe_case
    trace = manufactured_trace(path, case, suite, moving=True)
    original = path.read_bytes()
    trace_path = tmp_path / "trace.json"
    write_json(trace_path, trace)
    assert main(["probe", str(path), "--trace", str(trace_path), "--json", str(path)]) == 3
    assert path.read_bytes() == original
    assert main(["probe", str(path), "--trace", str(trace_path), "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "FAIL"
    assert main(["probe", str(path)]) == 2


@pytest.mark.parametrize("output", ["trace.json", "original/stats.json", "original/case.json"])
def test_cli_output_cannot_corrupt_regression_export(probe_case, tmp_path, output):
    path, case, suite = probe_case
    trace_path = tmp_path / "input-trace.json"
    write_json(trace_path, manufactured_trace(path, case, suite, moving=True))
    destination = tmp_path / "regression"
    assert (
        main(
            [
                "probe",
                str(path),
                "--trace",
                str(trace_path),
                "--counterexample",
                str(destination),
                "--json",
                str(destination / output),
            ]
        )
        == 3
    )
    assert not destination.exists()


@pytest.mark.parametrize(
    "mutate", ["escape", "duplicate", "wrong_ports", "bool_version", "malformed_exclusions"]
)
def test_case_input_validation(probe_case, mutate):
    path, case, _suite = probe_case
    if mutate == "escape":
        case["artifacts"][0]["path"] = "../outside.json"
    elif mutate == "duplicate":
        case["artifacts"].append(case["artifacts"][0])
    elif mutate == "wrong_ports":
        case["adapter"]["framework"] = "openpi"
        case["adapter"]["configuration"] = {
            "model_type": "pi0",
            "mask": [True] * 7,
            "use_quantiles": False,
            "anchor": "origin",
        }
    elif mutate == "bool_version":
        case["schema_version"] = True
    else:
        case["adapter"]["configuration"]["exclude_joints"] = ""
    write_json(path, case)
    with pytest.raises(InputError):
        probe_cases.load(path)


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="Optional LeRobot integration")
@pytest.mark.parametrize("kind", ["json", "safetensors"])
def test_capture_rejects_statistics_replaced_during_execution(probe_case, monkeypatch, kind):
    import torch
    from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
    from safetensors.torch import save

    from mltools.probe_capture import capture

    path, case, suite = probe_case
    stats = path.parent / f"stats.{kind}"
    if kind == "safetensors":
        stats.write_bytes(
            save({"action.mean": torch.tensor([3.0, 5.0]), "action.std": torch.tensor([2.0, 4.0])})
        )
    call = UnnormalizerProcessorStep.__call__

    def replace_statistics(step, *args, **kwargs):
        stats.write_bytes(b"replacement bytes, not the executed statistics")
        return call(step, *args, **kwargs)

    monkeypatch.setattr(UnnormalizerProcessorStep, "__call__", replace_statistics)
    with pytest.raises(InputError, match="Statistics changed during execution"):
        capture(suite, stats, case["adapter"]["configuration"])


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="Optional LeRobot integration")
@pytest.mark.parametrize("bind_queue", [True, False])
def test_genuine_lerobot_generated_probes_and_real_witness_reexecution(probe_case, tmp_path, bind_queue):
    path, case, _suite = probe_case
    case["adapter"]["configuration"]["bind_queue"] = bind_queue
    write_json(path, case)
    trace = probe_cases.capture(path)
    report = probe_cases.check(path, trace)
    assert report["status"] == ("PASS" if bind_queue else "FAIL")
    assert all(t["reset_cleared_before_query"] for t in trace["traces"])
    if not bind_queue:
        destination = tmp_path / "regression"
        probe_cases.export(path, trace, report, destination)
        child = destination / read_json(destination / "export.json")["witness_case"]
        rerun = probe_cases.capture(child)
        assert len(rerun["outputs"]) == 2
        assert probe_cases.check(child, rerun)["status"] == "FAIL"


@pytest.mark.skipif(not os.environ.get("MLTOOLS_OPENPI_PYTHON"), reason="Opt-in isolated OpenPI environment")
def test_genuine_openpi_cli_case_and_fresh_witness(probe_case, tmp_path):
    path, case, _suite = probe_case
    prior = read_json(
        Path(__file__).resolve().parents[1]
        / "research/publication/development/openpi-001/moving_anchor.case.json"
    )
    (path.parent / "stats.json").write_text(prior["stats_json"], encoding="utf-8", newline="")
    case["suite"], case["reference"] = prior["suite"], prior["reference"]
    case["adapter"] = {
        "framework": "openpi",
        "stats_artifact": "stats",
        "configuration": {
            "model_type": "pi0",
            "use_quantiles": False,
            "mask": [True] * 6 + [False],
            "anchor": "current",
        },
    }
    case["artifacts"][0]["sha256"] = digest(path.parent / "stats.json")
    write_json(path, case)
    trace = probe_cases.capture(path, python=os.environ["MLTOOLS_OPENPI_PYTHON"])
    report = probe_cases.check(path, trace)
    assert report["status"] == "FAIL"
    destination = tmp_path / "regression"
    probe_cases.export(path, trace, report, destination)
    child = destination / read_json(destination / "export.json")["witness_case"]
    replay = probe_cases.capture(child, python=os.environ["MLTOOLS_OPENPI_PYTHON"])
    assert probe_cases.check(child, replay)["status"] == "FAIL"
    assert len(replay["outputs"]) == 2
