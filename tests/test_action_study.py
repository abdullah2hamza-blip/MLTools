"""Development recipe math, scope separation and optimized evidence tampering."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_action_study as study  # noqa: E402
from action_study_common import development_inputs, interventions  # noqa: E402


def test_interventions_are_visible_artifacts_chosen_without_outputs():
    value = {
        "schema_version": 1,
        "profile": "lerobot_mean_std_f64_v1",
        "statistics": {"mean": [8.0, 2.0], "std": [0.0, 4.0]},
    }
    cases = interventions(value)
    assert all(v["selected_coordinate"] == 1 for v in cases.values())
    assert cases["offset_intervention"]["specification"]["statistics"] == {"mean": [8, 4], "std": [0, 4]}
    assert cases["slope_intervention"]["specification"]["statistics"] == {"mean": [8, 2], "std": [0, 8]}
    assert cases["fixed_offset_slope_composition"]["specification"]["statistics"] == {
        "mean": [8, 4],
        "std": [0, 8],
    }
    assert value["statistics"] == {"mean": [8, 2], "std": [0, 4]}


def test_fixed_development_gate_and_authority_erasure():
    cells = development_inputs()
    assert len(cells) == 4
    assert all("boccia" not in k and "orange" not in k and "libero" not in k for k in cells)
    for cell in cells.values():
        erased = study.reference(cell, erased="authority")
        assert erased.authority is None
        unitless = study.reference(cell, erased="units")
        assert all(u is None for u in unitless.output_units)
        assert erased.coefficients == unitless.coefficients == study.reference(cell).coefficients
        assert cell["checkpoint_physical_status"] == "UNKNOWN"


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "fault_output",
        "roundtrip",
        "counts",
        "schedule",
        "ablation",
        "witness_relation",
        "witness_response",
        "scope",
        "manifest",
        "identity",
        "source",
        "auxiliary",
        "statistics",
        "openpi_client",
        "openpi_commit",
    ],
)
def test_optimized_action_study_replay(tmp_path, tamper):
    destination = tmp_path / "evidence"
    shutil.copytree(ROOT / "validation/action-study-development-001", destination)

    def read(name):
        return json.loads((destination / name).read_bytes())

    def write(name, value):
        (destination / name).write_bytes((json.dumps(value, indent=2) + "\n").encode())

    summary = read("summary.json")
    prefix = "lerobot.main"
    data = read(prefix + ".capture.json")
    cell = "designed_lerobot_mean_std_f64_v1"
    if tamper == "fault_output":
        for row in data["measurements"][cell]["offset_intervention"]["designs"]["canonical"]["rows"]:
            row["output"] = [0.0, 0.0]
    elif tamper == "roundtrip":
        data["roundtrips"][cell]["numeric_control"][0]["cross_encoded"] = [99.0, 99.0]
    elif tamper == "counts":
        data["runtime"]["main_decoder_calls"] -= 1
    elif tamper == "schedule":
        value = read("designs.json")
        value[cell]["canonical"]["schedule"]["queue_operations"] = True
        write("designs.json", value)
    elif tamper == "ablation":
        value = read("analysis.json")
        value["cells"][cell]["reference_authority_erased"]["methods"]["compiled"]["status"] = "PASS"
        write("analysis.json", value)
    elif tamper == "witness_relation":
        value = read("witnesses.json")
        value["lerobot"][cell]["offset_intervention"]["methods"]["compiled"]["relation"]["queries"] = []
        write("witnesses.json", value)
    elif tamper == "witness_response":
        prefix = "lerobot.witness"
        data = read(prefix + ".capture.json")
        data["measurements"][cell]["offset_intervention"]["designs"]["canonical"]["rows"][0]["output"] = [
            0.0,
            0.0,
        ]
    elif tamper == "scope":
        summary["final_evaluation"] = True
    elif tamper == "manifest":
        del summary["file_sha256"]["lerobot.main.stderr.txt"]
    elif tamper == "identity":
        data["upstream_identity"]["git_blobs"]["processor/normalize_processor.py"] = "0" * 40
    elif tamper == "source":
        summary["source_file_sha256"]["scripts/action_study_worker.py"] = "0" * 64
    elif tamper == "auxiliary":
        data["roundtrips"] = {}
        data["runtime"]["auxiliary_native_calls"] = 0
    elif tamper == "statistics":
        value = read("lerobot.main.request.json")
        value["cells"][cell]["conditions"]["offset_intervention"]["specification"]["statistics"]["mean"][
            0
        ] = 0
        write("lerobot.main.request.json", value)
    elif tamper in ("openpi_client", "openpi_commit"):
        prefix = "openpi.main"
        data = read(prefix + ".capture.json")
        if tamper == "openpi_client":
            data["upstream_identity"]["client_image_tools_blob"] = "0" * 40
        else:
            data["upstream_identity"]["direct_urls"]["openpi-client"]["vcs_info"]["commit_id"] = "0" * 40
    write(prefix + ".capture.json", data)
    write(prefix + ".stdout.txt", data)
    summary["file_sha256"] = {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in summary["file_sha256"]
    }
    write("summary.json", summary)
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/run_action_study.py"), "--verify", str(destination)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert (result.returncode == 0) is (tamper is None), result.stdout + result.stderr
