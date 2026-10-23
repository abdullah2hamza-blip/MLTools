"""Committed source gates, exact subset acquisition and full-study replay integrity."""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import publication_protocol as protocol  # noqa: E402
import run_publication_study as runner  # noqa: E402
from action_study_common import sha, write  # noqa: E402
from study_assets import read  # noqa: E402

CAPTURE = ROOT / "validation/publication-wrapper-development-001"


def test_committed_inventory_reads_one_immutable_commit():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result = protocol.committed_bytes(ROOT, commit, ["pyproject.toml", "scripts/policy_space_numeric.py"])
    assert result["pyproject.toml"].startswith(b"[project]\n")
    assert b'"mltools"' in result["pyproject.toml"]
    assert (
        result["scripts/policy_space_numeric.py"] == (ROOT / "scripts/policy_space_numeric.py").read_bytes()
    )
    with pytest.raises(ValueError, match="same commit"):
        protocol.committed_bytes(ROOT, commit, ["__missing_publication_protocol_test__"])


@pytest.mark.parametrize("commit", ["HEAD", "main", "a" * 39, "a" * 40 + ":file", "--help"])
def test_gate_requires_full_commit_identity(commit):
    with pytest.raises(ValueError, match="immutable Git commit"):
        protocol.committed_bytes(ROOT, commit, ["pyproject.toml"])


def test_prepared_inputs_are_not_a_committed_protocol():
    req = {
        "schema_version": 1,
        "protocol_path": "research/publication/prepared/action-study-001/plan.json",
        "freeze_commit": "a" * 40,
    }
    native = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/publication_worker.py")],
        input=json.dumps(req).encode(),
        capture_output=True,
        timeout=30,
    )
    assert native.returncode != 0
    assert b"complete publication protocol format" in native.stderr


def test_development_and_selected_denominators_remain_distinct():
    dev = protocol.build("development")
    assert dev["planned_calls"]["primary_decoder"] == 680
    assert dev["planned_calls"]["roundtrip_auxiliary"] == 340
    assert dev["role"] == "development"
    selected = read(ROOT / "research/publication/prepared/action-study-001/plan.json")
    assert len(selected["cells"]) == 4
    assert len({c["cluster"] for c in selected["cells"].values()}) == 3
    assert selected["compilation_only"] is True


@pytest.mark.parametrize("guard", ["phase_order", "existing_claim"])
def test_native_entry_requires_order_and_exclusive_claim(guard):
    scratch = ROOT / ".test-tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="publication-entry-", dir=scratch) as value:
        folder = Path(value).resolve()
        assert folder.is_relative_to(ROOT)
        spec = protocol.build("development")
        path = folder / "protocol.json"
        write(path, spec)
        write(
            folder / "attempt-start.json",
            {
                "schema_version": 1,
                "role": "development",
                "freeze_commit": None,
                "protocol_path": path.relative_to(ROOT).as_posix(),
                "protocol_sha256": sha(path),
                "attempt_path": folder.relative_to(ROOT).as_posix(),
                "reproduction_of": None,
                "environments": {"robotruth": runner.environment(Path(sys.executable))},
            },
        )
        write(folder / "attempt-status.json", {"status": "STARTED", "completed_phases": []})
        phase = "lerobot" if guard == "phase_order" else "robotruth"
        if guard == "existing_claim":
            write(folder / "robotruth.claim.json", {"prior_execution_claim": True})
        req = runner.request(folder, path, spec, None, phase)
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts/publication_worker.py")],
            input=(json.dumps(req, indent=2) + "\n").encode(),
            capture_output=True,
            timeout=30,
        )
        assert result.returncode != 0
        expected = b"exact preceding phase prefix" if guard == "phase_order" else b"FileExistsError"
        assert expected in result.stderr
        assert not (folder / (phase + ".native.json")).exists()


def test_completed_native_request_cannot_be_executed_again():
    req = (CAPTURE / "lerobot.request.json").read_bytes()
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "scripts/publication_worker.py")],
        input=req,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert b"native execution requires STARTED" in result.stderr


@pytest.mark.parametrize("change", ["same_environment", "semantic_difference"])
def test_clean_reproduction_requires_semantics_and_separate_environments(tmp_path, change):
    folder = tmp_path / "reproduction"
    shutil.copytree(CAPTURE, folder)
    start = read(folder / "attempt-start.json")
    start["reproduction_of"] = {
        "path": CAPTURE.relative_to(ROOT).as_posix(),
        "summary_sha256": sha(CAPTURE / "summary.json"),
    }
    write(folder / "attempt-start.json", start)
    if change == "semantic_difference":
        (folder / "conditions.csv").write_bytes(
            (folder / "conditions.csv").read_bytes().replace(b"PASS", b"FAIL", 1)
        )
    code = (
        "import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
        "from run_publication_study import reproduction_comparison; "
        "reproduction_comparison(Path(sys.argv[2]))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(ROOT / "scripts"), str(folder)],
        capture_output=True,
        timeout=90,
    )
    assert result.returncode != 0
    expected = (
        b"separate native environment prefixes"
        if change == "same_environment"
        else b"clean semantic artifacts reproduce"
    )
    assert expected in result.stderr


def test_timeout_retains_partial_native_streams_and_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "request", lambda *_: {"test": "timeout"})

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1, output=b"partial stdout", stderr=b"partial stderr")

    monkeypatch.setattr(runner.subprocess, "run", timed_out)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.acquire(
            tmp_path,
            tmp_path / "protocol.json",
            {"specification": {"resources": {"worker_timeout_seconds": 1}}},
            None,
            "robotruth",
            Path(sys.executable),
        )
    assert (tmp_path / "robotruth.stdout.txt").read_bytes() == b"partial stdout"
    assert (tmp_path / "robotruth.stderr.txt").read_bytes() == b"partial stderr"
    assert read(tmp_path / "robotruth.receipt.json")["timed_out"] is True


def repair_request_bindings(folder):
    start = read(folder / "attempt-start.json")
    path = ROOT / start["protocol_path"]
    spec = read(path)
    for phase in runner.PHASES:
        write(
            folder / f"{phase}.request.json",
            runner.request(folder, path, spec, start["freeze_commit"], phase),
        )
        data = read(folder / f"{phase}.native.json")
        data["request_sha256"] = sha(folder / f"{phase}.request.json")
        for suffix in ("claim", "receipt"):
            bound = read(folder / f"{phase}.{suffix}.json")
            bound["request_sha256"] = data["request_sha256"]
            write(folder / f"{phase}.{suffix}.json", bound)
        write(folder / f"{phase}.native.json", data)
        write(folder / f"{phase}.stdout.txt", data)


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "main_input",
        "main_output",
        "roundtrip",
        "fresh_output",
        "openpi_identity",
        "openpi_client",
        "openpi_dtype",
        "fresh_count",
        "robo_inventory",
        "transport_value",
        "transport_negative_arrays",
        "derived_status",
        "witness_relation",
        "metrics_denominator",
        "metrics_loo",
        "csv",
        "missing_manifest",
        "extra",
        "phase_header",
        "version_type",
        "request_source",
        "protocol_bytes",
        "denied_probe",
        "receipt_wall",
        "claim_type",
    ],
)
def test_optimized_full_study_replay_rejects_rehashed_substitutions(tmp_path, tamper):
    folder = tmp_path / "study"
    shutil.copytree(CAPTURE, folder)
    summary = read(folder / "summary.json")
    phase = "lerobot"
    data = read(folder / f"{phase}.native.json")
    cell_id, condition = "act_move_cube_lerobot", "offset_intervention"
    actual = data["cells"][cell_id][condition]
    if tamper == "main_input":
        actual["designs"]["canonical"][0]["input"][0] += 0.1
    elif tamper == "main_output":
        actual["designs"]["canonical"][0]["output"][0] += 1
    elif tamper == "roundtrip":
        actual["roundtrips"][0]["cross_decoded"][0] += 1
    elif tamper in ("fresh_output", "fresh_count"):
        phase = "lerobot_witness"
        data = read(folder / f"{phase}.native.json")
        if tamper == "fresh_output":
            data["cells"][cell_id][condition]["designs"]["canonical"][0]["output"][0] += 1
        else:
            data["runtime"]["primary_decoder_calls"] += 1
    elif tamper in ("openpi_identity", "openpi_client", "openpi_dtype"):
        phase = "openpi"
        data = read(folder / f"{phase}.native.json")
        if tamper == "openpi_identity":
            data["upstream_identity"]["revision"] = "0" * 40
        elif tamper == "openpi_client":
            data["upstream_identity"]["direct_urls"]["openpi-client"]["vcs_info"]["commit_id"] = "0" * 40
        else:
            data["cells"]["authored_quantile_openpi"][condition]["loaded_artifact"]["tensor_dtypes"][
                "q01"
            ] = "torch.float32"
    elif tamper == "robo_inventory":
        phase = "robotruth"
        data = read(folder / f"{phase}.native.json")
        row = data["cells"][cell_id][condition]
        row["findings"] = [f for f in row["findings"] if f["field"] != "action.gripper_convention"]
    elif tamper in ("transport_value", "transport_negative_arrays"):
        phase = "policy_space"
        data = read(folder / f"{phase}.native.json")
        if tamper == "transport_value":
            data["cells"][cell_id][condition]["designs"]["canonical"][0]["wire_output"][0] += 1
        else:
            data["negative_controls"][cell_id]["wrong_input_width"]["actual_arrays"]["request_input"][
                "values"
            ][0] += 0.1
    elif tamper == "derived_status":
        derived = read(folder / "analysis.json")
        derived["cells"][cell_id]["reference_authority_erased"]["numerical_methods"]["compiled"]["status"] = (
            "PASS"
        )
        write(folder / "analysis.json", derived)
    elif tamper == "witness_relation":
        witness = read(folder / "witnesses.json")
        witness["lerobot"][cell_id][condition]["methods"]["compiled"]["relation"]["queries"] = [
            "not-the-original-query"
        ]
        write(folder / "witnesses.json", witness)
    elif tamper in ("metrics_denominator", "metrics_loo"):
        m = read(folder / "metrics.json")
        if tamper == "metrics_denominator":
            m["counts"]["artifact_groups"] += 1
        else:
            next(iter(m["leave_one_artifact_out_macro_detection_fraction"].values()))["compiled"] = 0.0
        write(folder / "metrics.json", m)
    elif tamper == "csv":
        (folder / "conditions.csv").write_bytes(
            (folder / "conditions.csv").read_bytes().replace(b"PASS", b"FAIL", 1)
        )
    elif tamper == "missing_manifest":
        del summary["file_sha256"]["robotruth.stderr.txt"]
    elif tamper == "extra":
        write(folder / "extra.json", {})
    elif tamper == "phase_header":
        data["engine"] = "decoder_witness"
    elif tamper == "version_type":
        data["schema_version"] = True
    elif tamper == "request_source":
        req = read(folder / "lerobot.request.json")
        req["source_file_sha256"]["scripts/publication_worker.py"] = "0" * 64
        write(folder / "lerobot.request.json", req)
    elif tamper == "protocol_bytes":
        spec = read(folder / "protocol.json")
        spec["role"] = "prospective"
        write(folder / "protocol.json", spec)
    elif tamper == "denied_probe":
        phase = "openpi"
        data = read(folder / f"{phase}.native.json")
        data["runtime"]["denied_dependency_capability_probes"][0]["operation_executed"] = True
    elif tamper == "receipt_wall":
        receipt = read(folder / "lerobot.receipt.json")
        receipt["subprocess_wall_seconds"] = 0.0
        write(folder / "lerobot.receipt.json", receipt)
        repair_request_bindings(folder)
    elif tamper == "claim_type":
        claim = read(folder / "lerobot.claim.json")
        claim["schema_version"] = True
        write(folder / "lerobot.claim.json", claim)
        repair_request_bindings(folder)
    native_mutations = {
        "main_input",
        "main_output",
        "roundtrip",
        "fresh_output",
        "openpi_identity",
        "openpi_client",
        "openpi_dtype",
        "fresh_count",
        "robo_inventory",
        "transport_value",
        "transport_negative_arrays",
        "phase_header",
        "version_type",
        "denied_probe",
    }
    if tamper in native_mutations:
        write(folder / f"{phase}.native.json", data)
        repair_request_bindings(folder)
    summary["file_sha256"] = {name: sha(folder / name) for name in summary["file_sha256"]}
    write(folder / "summary.json", summary)
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/run_publication_study.py"), "--verify", str(folder)],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert (result.returncode == 0) is (tamper is None), result.stdout + result.stderr
