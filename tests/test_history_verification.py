"""Data-only verification must reject altered evidence even under Python -O."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("tamper", [False, True])
def test_historical_verifier_optimized_mode(tmp_path, tamper):
    for phase in ("before", "after"):
        data = json.loads((ROOT / "validation/lerobot-history-001" / f"{phase}.json").read_bytes())
        if tamper and phase == "before":
            data["events"][1]["actual"] = 999
        (tmp_path / f"{phase}.json").write_text(json.dumps(data), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/reproduce_lerobot_history.py"), "--verify", str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert (result.returncode == 0) is (not tamper)
    if tamper:
        assert "Historical capture verification failed: before observed values" in result.stderr
        assert "Verified native parent" not in result.stdout
