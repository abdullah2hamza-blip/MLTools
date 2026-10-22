"""Saved genuine captures remain data-only and reject substantive tampering under -O."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (None, None),
        ("training", "after observed training targets"),
        ("identity", "before package inventory"),
        ("scope", "before research scope"),
        ("measurement", "after deterministic independent analysis"),
    ],
)
def test_optimized_history_replay(tmp_path, tamper, message):
    source = ROOT / "validation/openpi-history-001"
    for path in source.iterdir():
        if path.is_file():
            shutil.copy2(path, tmp_path / path.name)
    if tamper:
        phase = "after" if tamper in ("training", "measurement") else "before"
        path = tmp_path / f"{phase}.trace.json"
        trace = json.loads(path.read_bytes())
        if tamper == "training":
            trace["boundary"]["training_targets"][1][4] = 9.0
        elif tamper == "identity":
            trace["identity"]["direct_urls"] = {}
        elif tamper == "scope":
            trace["final_evaluation"] = True
        else:
            trace["outputs"]["center"][0] += 0.5
        path.write_text(json.dumps(trace), encoding="utf-8", newline="\n")
        # Rehash the modified file: rejection must exercise semantic/identity checks,
        # not merely notice that a raw file no longer matches its manifest digest.
        summary_path = tmp_path / "summary.json"
        summary = json.loads(summary_path.read_bytes())
        summary["file_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary_path.write_text(json.dumps(summary), encoding="utf-8", newline="\n")
    process = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/reproduce_openpi_history.py"), "--verify", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (process.returncode == 0) is (tamper is None)
    if message:
        assert message in process.stderr
        assert "Verified genuine historical" not in process.stdout
    else:
        assert "Verified genuine historical" in process.stdout
