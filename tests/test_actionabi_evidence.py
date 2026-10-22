"""Independent hand arithmetic and substantive native-baseline replay tampering."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "actionabi_fixture_tests", ROOT / "scripts/actionabi_fixtures.py"
)
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)


def test_hand_calculated_huber_control():
    rows = fixture.samples("excited")
    correct = fixture.expected_score(rows, fixture.contract())
    wrong = fixture.expected_score(rows, fixture.contract(sign=[-1, 1]))
    assert correct["train_loss"] == correct["heldout_loss"] == 0
    assert correct["residual_count"] == 12
    assert wrong["train_loss"] == 1.75
    assert wrong["heldout_loss"] == pytest.approx(25 / 12)
    assert fixture.expected_score(rows, fixture.contract(lag_steps=1))["residual_count"] == 8


def test_structural_sign_gripper_ambiguity():
    for action in ([0.5, -0.3], [-0.25, 1.0], [0, 0]):
        assert fixture.prediction(action, fixture.contract(sign=[1, -1])) == fixture.prediction(
            action, fixture.contract(gripper_inverted=True)
        )


@pytest.mark.parametrize(
    "tamper", [None, "selector", "converter_request", "converter_binding", "score", "coverage", "upstream"]
)
def test_optimized_actionabi_evidence(tmp_path, tamper):
    target = tmp_path / "evidence"
    shutil.copytree(ROOT / "validation/actionabi-native-001", target)
    summary_path = target / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    name = None
    if tamper == "selector":
        name = "native/all_unsafe-select.request.json"
        data = json.loads((target / name).read_bytes())
        # This altered control has the same expected selection/inseparability,
        # but would never exercise the original unsafe-filtering path.
        data["hypotheses"] = [fixture.contract(), fixture.contract()]
        for candidate in data["candidates"]:
            candidate["safe"] = True
            candidate["predicted_observations"] = [candidate["action"], candidate["action"]]
    elif tamper == "converter_request":
        name = "native/unsupported-convert.request.json"
        data = json.loads((target / name).read_bytes())
        data["unsupported_fields"] = []
    elif tamper == "converter_binding":
        name = "native/identified-convert.stdout.txt"
        data = json.loads((target / name).read_bytes())
        data["converter"]["evidence_report_sha256"] = "0" * 64
    elif tamper == "score":
        name = "native/identified-score.stdout.txt"
        data = json.loads((target / name).read_bytes())
        data["native_evidence"]["ranked_hypotheses"][0]["score"]["train_loss"] = 0.125
    elif tamper == "coverage":
        summary["coverage_claim"] = True
    elif tamper == "upstream":
        del summary["upstream_identities"]["cli11"]
    if name is not None:
        (target / name).write_text(json.dumps(data), encoding="utf-8", newline="\n")
        summary["file_sha256"][name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8", newline="\n")
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/validate_actionabi.py"), "--verify", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) is (tamper is None)
    if tamper:
        assert "ActionABI development evidence:" in result.stderr
        assert "Verified stock ActionABI" not in result.stdout
    else:
        assert "Verified stock ActionABI CPU scoring" in result.stdout
