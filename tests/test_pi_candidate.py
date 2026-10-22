"""Metadata-only eligibility must preserve original bytes and unexecuted labels."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("pi_selection_tests", ROOT / "scripts/prepare_pi_candidate.py")
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def test_optimized_offline_pi_selection():
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/prepare_pi_candidate.py")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Verified 7 pinned PI05 metadata artifacts; no case outcomes observed" in result.stdout


@pytest.mark.parametrize("tamper", ["raw", "decoded", "executed"])
def test_pi_selection_cannot_be_rehashed_or_relabelled(tmp_path, monkeypatch, tamper):
    target = tmp_path / "metadata"
    shutil.copytree(selection.BASE, target)
    monkeypatch.setattr(selection, "BASE", target)
    if tamper == "raw":
        path = target / "config.json"
        path.write_bytes(path.read_bytes().replace(b'"pi05"', b'"pi00"'))
    else:
        path = target / ("decoded_statistics.json" if tamper == "decoded" else "manifest.json")
        data = json.loads(path.read_bytes())
        if tamper == "decoded":
            data["tensors"]["postprocessor"]["action.q01"]["values"][0] += 1
        else:
            data["execution_outcomes_observed"] = True
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="PI metadata selection:"):
        selection.snapshot()
