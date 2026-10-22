"""Data-only subset fixtures must reject rehashed measured-input/output tampering."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tamper", [None, "input", "output", "request"])
def test_optimized_design_replay(tmp_path, tamper):
    source = ROOT / "research/publication/development/query-designs-001"
    summary = json.loads((source / "summary.json").read_bytes())
    # A test-only subset of the immutable actual capture, not new research data.
    summary["cases"] = summary["cases"][:1]
    summary["file_sha256"] = {
        name: sha for name, sha in summary["file_sha256"].items() if name.startswith("lerobot_good.")
    }
    for name in summary["file_sha256"]:
        shutil.copy2(source / name, tmp_path / name)
    if tamper:
        name = "lerobot_good.request.json" if tamper == "request" else "lerobot_good.capture.json"
        path = tmp_path / name
        data = json.loads(path.read_bytes())
        if tamper == "request":
            data["designs"]["random_dense-7"]["seed"] = 999
        elif tamper == "input":
            data["measurements"]["manual_axis"]["traces"][0]["origin"][0] += 1
        else:
            data["measurements"]["manual_axis"]["outputs"]["q000"][0] += 1
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
        summary["file_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8", newline="\n")
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/compare_query_designs.py"), "--verify", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (result.returncode == 0) is (tamper is None)
    if tamper:
        assert "Research design:" in result.stderr
        assert "Verified 1 development query-design" not in result.stdout
    else:
        assert "Verified 1 development query-design" in result.stdout
