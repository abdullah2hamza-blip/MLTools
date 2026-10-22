"""Rehashed changes cannot rewrite designed calibration or native parity evidence."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8", newline="\n")


@pytest.mark.parametrize("tamper", [None, "native", "request", "report", "upstream", "manifest"])
def test_optimized_certificate_replay(tmp_path, tamper):
    target = tmp_path / "evidence"
    shutil.copytree(ROOT / "validation/normalization-certificate-001", target)
    summary = json.loads((target / "summary.json").read_bytes())
    if tamper == "native":
        data = json.loads((target / "openpi.capture.json").read_bytes())
        data["captures"]["openpi_drift"]["candidate_decoded"][0][0] += 0.125
        write(target / "openpi.capture.json", data)
        write(target / "openpi.stdout.txt", data)
    elif tamper == "request":
        data = json.loads((target / "lerobot.request.json").read_bytes())
        # Identical values must not be relabeled demonstration-derived.
        data["cases"]["lerobot_drift"]["calibration"]["kind"] = "demonstration"
        write(target / "lerobot.request.json", data)
        capture = json.loads((target / "lerobot.capture.json").read_bytes())
        capture["request_sha256"] = hashlib.sha256((target / "lerobot.request.json").read_bytes()).hexdigest()
        write(target / "lerobot.capture.json", capture)
        write(target / "lerobot.stdout.txt", capture)
    elif tamper == "report":
        data = json.loads((target / "paper_identity.report.json").read_bytes())
        data["metric"]["tails"][0]["fraction_greater_or_equal"] = 0
        write(target / "paper_identity.report.json", data)
    elif tamper == "upstream":
        data = json.loads((target / "openpi.capture.json").read_bytes())
        data["upstream_identity"]["direct_urls"]["openpi-client"]["vcs_info"]["commit_id"] = "0" * 40
        write(target / "openpi.capture.json", data)
        write(target / "openpi.stdout.txt", data)
    elif tamper == "manifest":
        del summary["file_sha256"]["lerobot.stderr.txt"]
    for name in summary["file_sha256"]:
        summary["file_sha256"][name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    write(target / "summary.json", summary)
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            str(ROOT / "scripts/validate_normalization_certificate.py"),
            "--verify",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) is (tamper is None)
    if tamper:
        assert "Normalization certificate:" in result.stderr
        assert "Verified designed normalization" not in result.stdout
    else:
        assert "Verified designed normalization" in result.stdout
