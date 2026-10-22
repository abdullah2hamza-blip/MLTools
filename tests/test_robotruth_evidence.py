"""Stock external-baseline evidence is replayable without its dependencies or ML."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tamper", [None, "finding", "manifest", "scope", "revision"])
def test_optimized_robotruth_record_verification(tmp_path, tamper):
    source = ROOT / "validation/robotruth-native-001"
    evidence = tmp_path / "copy"
    shutil.copytree(source, evidence)
    summary_path = evidence / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    if tamper in ("finding", "manifest"):
        name = (
            "cases/synthetic_statistics.result.json"
            if tamper == "finding"
            else "cases/synthetic_statistics.deployed.json"
        )
        path = evidence / name
        data = json.loads(path.read_bytes())
        if tamper == "finding":
            row = next(f for f in data["findings"] if f["field"] == "action.normalization.stats_sha256")
            row["deployed"] = row["evaluated"]
        else:
            data["action"]["normalization"]["stats_sha256"] = "f" * 64
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
        # Exercise substantive checks rather than the outer file digest alone.
        summary["file_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif tamper == "scope":
        summary["development_only"] = False
    elif tamper == "revision":
        summary["upstream_identity"]["direct_url"]["vcs_info"]["commit_id"] = "0" * 40
    summary_path.write_text(json.dumps(summary), encoding="utf-8", newline="\n")
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/validate_robotruth.py"), "--verify", str(evidence)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) is (tamper is None)
    if tamper:
        assert "development evidence:" in result.stderr
        assert "Verified 11 robotruth" not in result.stdout
    else:
        assert "Verified 11 robotruth development" in result.stdout
