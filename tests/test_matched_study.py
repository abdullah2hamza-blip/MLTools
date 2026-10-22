"""Shared-byte fidelity, source-role gates and native-report replay integrity."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_matched_study as matched  # noqa: E402
import study_assets as assets  # noqa: E402
from action_study_common import write  # noqa: E402
from prepare_act_candidates import tensors  # noqa: E402


def materialize(folder, role="development"):
    plan, files = assets.blueprint(role)
    for name, raw in files.items():
        p = folder / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    write(folder / "plan.json", plan)
    return plan


def test_f64_encoding_retains_values_and_other_tensors(tmp_path):
    plan, files = assets.blueprint("development")
    cell = plan["cells"]["act_move_cube_lerobot"]
    control, same = (cell["conditions"][k] for k in ("numeric_control", "equivalent_f64_serialization"))
    assert control["specification"] == same["specification"]
    name = cell["post_state_file"]
    raw_control = files[control["directory"] + "/" + name]
    raw_same = files[same["directory"] + "/" + name]
    assert raw_control != raw_same
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    a.write_bytes(raw_control)
    b.write_bytes(raw_same)
    first, second = tensors(a), tensors(b)
    assert set(first) == set(second)
    for key, value in first.items():
        assert value["values"] == second[key]["values"]
        assert value["shape"] == second[key]["shape"]
        if key in ("action.mean", "action.std"):
            assert second[key]["dtype"] == "F64"
        else:
            assert value["dtype"] == second[key]["dtype"]
    assert matched.combined_statistics_hash(cell, control) != matched.combined_statistics_hash(cell, same)


def test_development_assignment_rejects_replaced_manifest(tmp_path, monkeypatch):
    original = assets.read(ROOT / assets.DEVELOPMENT_MANIFEST)
    altered = copy.deepcopy(original)
    altered["model"] = "candidate/not-development"
    altered["revision"] = "0" * 40
    write(tmp_path / "manifest.json", altered)
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "DEVELOPMENT_MANIFEST", "manifest.json")
    with pytest.raises(ValueError, match="pinned development assignment"):
        assets.selections("development")


@pytest.mark.parametrize("change", ["extra", "missing", "bytes", "version_type"])
def test_closed_assets_regenerate_from_pinned_sources(tmp_path, change):
    plan = materialize(tmp_path)
    assert assets.verify(tmp_path) == plan
    if change == "extra":
        (tmp_path / "extra.json").write_text("{}")
    elif change == "missing":
        (tmp_path / next(iter(plan["file_sha256"]))).unlink()
    elif change == "bytes":
        name = next(iter(plan["file_sha256"]))
        (tmp_path / name).write_bytes(b"rewritten")
    else:
        plan["schema_version"] = True
        write(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError):
        assets.verify(tmp_path)


def test_prospective_compilation_is_not_native_authorization():
    scratch = ROOT / ".test-tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prospective-gate-", dir=scratch) as value:
        folder = Path(value).resolve()
        assert folder.is_relative_to(ROOT)  # Check before automatic recursive cleanup.
        plan = materialize(folder, "prospective")
        assert len(plan["cells"]) == 4 and plan["compilation_only"] is True
        req = matched.request(folder, "robotruth")
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts/matched_study_worker.py")],
            input=json.dumps(req).encode(),
            capture_output=True,
            timeout=30,
        )
        assert result.returncode != 0
        assert b"no prospective engine execution before committed protocol freeze" in result.stderr


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "stats_finding",
        "drop_gripper",
        "duplicate_finding",
        "manifest_value",
        "overlays",
        "weights",
        "aggregate",
        "loaded_values",
        "response",
        "roundtrip",
        "calls",
        "numeric_identity",
        "double_count",
        "inventory",
        "source",
    ],
)
def test_optimized_matched_replay(tmp_path, tamper):
    folder = tmp_path / "native"
    shutil.copytree(ROOT / "validation/matched-workflows-development-001", folder)
    summary = assets.read(folder / "summary.json")
    engine = "robotruth"
    data = assets.read(folder / (engine + ".native.json"))
    cell = "act_move_cube_lerobot"
    row = data["cells"][cell]["offset_intervention"]
    if tamper == "stats_finding":
        next(f for f in row["findings"] if f["field"] == "action.normalization.stats_sha256")["deployed"] = (
            "0" * 64
        )
    elif tamper == "drop_gripper":
        row["findings"] = [f for f in row["findings"] if f["field"] != "action.gripper_convention"]
        fatal = sum(f["severity"] == "fatal" for f in row["findings"])
        warn = sum(f["severity"] == "warn" for f in row["findings"])
        row["native_summary"] = f"FAIL: {fatal} fatal, {warn} warn, {len(row['findings'])} checks"
    elif tamper == "duplicate_finding":
        row["findings"].append(copy.deepcopy(row["findings"][0]))
    elif tamper == "manifest_value":
        next(f for f in row["findings"] if f["field"] == "runtime.device")["evaluated"] = (
            "__tampered_device__"
        )
    elif tamper == "overlays":
        row["overlays"]["deployed"]["action.units"] = "rad"
    elif tamper == "weights":
        row["deployed"]["policy"]["weights_sha256"] = "0" * 64
    elif tamper == "aggregate":
        row["native_passed"] = True
    elif tamper in ("loaded_values", "response", "roundtrip", "calls"):
        engine = "lerobot"
        data = assets.read(folder / (engine + ".native.json"))
        row = data["cells"][cell]["offset_intervention"]
        if tamper == "loaded_values":
            row["loaded_artifact"]["statistics"]["mean"][0] += 1
        elif tamper == "response":
            for output in row["designs"]["canonical"]:
                output["output"] = [0.0] * 6
        elif tamper == "roundtrip":
            row["roundtrips"][0]["cross_decoded"] = [0.0] * 6
        else:
            data["runtime"]["primary_decoder_calls"] += 1
    elif tamper in ("numeric_identity", "double_count"):
        analysis = assets.read(folder / "analysis.json")
        row = analysis["cells"][cell]["equivalent_f64_serialization"]
        if tamper == "numeric_identity":
            row["numerical_methods"]["compiled"]["status"] = "FAIL"
        else:
            row["robotruth"]["underlying_artifact_differences_counted"] = 2
        write(folder / "analysis.json", analysis)
    elif tamper == "inventory":
        del summary["file_sha256"]["robotruth.stderr.txt"]
    elif tamper == "source":
        summary["source_file_sha256"]["scripts/matched_study_worker.py"] = "0" * 64
    if tamper in (
        "stats_finding",
        "drop_gripper",
        "duplicate_finding",
        "manifest_value",
        "overlays",
        "weights",
        "aggregate",
        "loaded_values",
        "response",
        "roundtrip",
        "calls",
    ):
        write(folder / (engine + ".native.json"), data)
        write(folder / (engine + ".stdout.txt"), data)
    summary["file_sha256"] = {
        n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in summary["file_sha256"]
    }
    write(folder / "summary.json", summary)
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/run_matched_study.py"), "--verify", str(folder)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert (result.returncode == 0) is (tamper is None), result.stdout + result.stderr
