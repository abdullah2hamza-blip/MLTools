"""Evidence-matched transport scope, lifecycle, conversion and replay integrity."""

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
import policy_space_numeric as profile  # noqa: E402
from action_study_common import write  # noqa: E402
from study_assets import read  # noqa: E402

CAPTURE = ROOT / "validation/policy-space-numeric-development-001"


def test_profile_preserves_unknown_embodiment_and_equal_numerical_evidence():
    plan = read(ROOT / "validation/matched-assets-development-001/plan.json")
    registry = profile.registry_documents(plan)
    assert registry["embodiments"] == {}
    cell = plan["cells"]["act_move_cube_lerobot"]
    meta = profile.metadata(cell, "offset_intervention")
    assert "embodiment_constraints" not in meta["supported_schema_pairs"][0]
    assert meta["study_evidence"]["intended"] != meta["study_evidence"]["deployment"]
    assert meta["study_evidence"]["checkpoint_physical_authority_status"] == "UNKNOWN"
    selected = profile.selection("cell", "condition", meta)
    assert set(selected) == {"episode_id", "study_evidence"}
    assert all("unit" not in v for v in registry["actions"].values())


def test_wire_conversion_reports_float32_roundoff():
    original = [1.0, 0.1, -17.123456789]
    wire = profile.float32(original)
    assert wire[0] == original[0]
    assert wire[1] != original[1] and abs(wire[1] - original[1]) < 1e-8
    assert wire[2] != original[2]
    with pytest.raises(ValueError):
        profile.float32([float("nan")])
    with pytest.raises(ValueError):
        profile.float32([True])


def test_prospective_native_request_rejected_before_transport_import():
    req = {
        "schema_version": 1,
        "matched_path": "research/publication/prepared/action-study-001",
        "contracts_path": "research/publication/prepared/action-study-001",
    }
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/policy_space_numeric_worker.py")],
        input=json.dumps(req).encode(),
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert b"no prospective native execution" in result.stderr


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "input",
        "ingested",
        "replayed",
        "wire",
        "dtype",
        "shape",
        "query",
        "drop_design",
        "selected_schema",
        "invented_embodiment",
        "metadata",
        "lifecycle",
        "negative_error",
        "negative_calls",
        "negative_array",
        "negative_nan_tag",
        "drop_negative",
        "total",
        "analysis",
        "source",
        "manifest",
        "extra",
        "contract",
        "version_type",
    ],
)
def test_optimized_native_transport_replay(tmp_path, tamper):
    folder = tmp_path / "capture"
    shutil.copytree(CAPTURE, folder)
    data = read(folder / "native.json")
    summary = read(folder / "summary.json")
    cell = "act_move_cube_lerobot"
    condition = data["cells"][cell]["offset_intervention"]
    query = condition["designs"]["canonical"][0]
    if tamper in ("input", "ingested", "replayed", "wire"):
        field = {
            "input": "input",
            "ingested": "ingested_input",
            "replayed": "replayed_output",
            "wire": "wire_output",
        }[tamper]
        query[field][0] += 1
    elif tamper == "dtype":
        query["wire_dtype"] = "float64"
    elif tamper == "shape":
        query["wire_shape"] = [6]
    elif tamper == "query":
        query["query"] = "not-the-captured-query"
    elif tamper == "drop_design":
        del condition["designs"]["canonical"]
    elif tamper == "selected_schema":
        condition["selected_schema_pair"]["action_schema"] = "invented@1"
    elif tamper == "invented_embodiment":
        data["registry"]["embodiments"]["robot@1"] = {"id": "robot@1"}
    elif tamper == "metadata":
        condition["metadata"]["study_evidence"]["deployment"] = copy.deepcopy(
            condition["metadata"]["study_evidence"]["intended"]
        )
    elif tamper == "lifecycle":
        condition["callback_counts"]["finalize"] = 0
    elif tamper == "negative_error":
        data["negative_controls"][cell]["wrong_output_width"]["native_error"] = "unrelated shape error"
    elif tamper == "negative_calls":
        data["negative_controls"][cell]["wrong_input_width"]["callback_counts"]["infer_replay"] = 1
    elif tamper == "negative_array":
        data["negative_controls"][cell]["wrong_output_width"]["actual_arrays"]["model_output"]["values"][
            0
        ] += 1
    elif tamper == "negative_nan_tag":
        data["negative_controls"][cell]["nonfinite_input"]["actual_arrays"]["request_input"]["values"][0] = {
            "nonfinite": "+infinity"
        }
    elif tamper == "drop_negative":
        del data["negative_controls"][cell]["nonfinite_output"]
    elif tamper == "total":
        data["runtime"]["accepted_replay_queries"] += 1
    elif tamper == "analysis":
        analysis = read(folder / "analysis.json")
        analysis["cells"][cell]["offset_intervention"]["policy_space_transport"] = "FAIL"
        write(folder / "analysis.json", analysis)
    elif tamper == "source":
        summary["source_file_sha256"]["scripts/policy_space_numeric.py"] = "0" * 64
    elif tamper == "manifest":
        del summary["file_sha256"]["stderr.txt"]
    elif tamper == "extra":
        write(folder / "extra.json", {})
    elif tamper == "contract":
        path = folder / "contracts/actions/mltools_raw_vector_6@1.yaml"
        contract = read(path)
        contract["action_dim"] = 7
        write(path, contract)
    elif tamper == "version_type":
        data["schema_version"] = True
    if tamper not in (None, "analysis", "source", "manifest", "extra", "contract"):
        write(folder / "native.json", data)
        write(folder / "stdout.txt", data)
    summary["file_sha256"] = {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in summary["file_sha256"]
    }
    write(folder / "summary.json", summary)
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/run_policy_space_numeric.py"), "--verify", str(folder)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert (result.returncode == 0) is (tamper is None), result.stdout + result.stderr
