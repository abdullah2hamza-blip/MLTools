"""Independent rotation/state arithmetic and optimized native-evidence replay."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("panda_reference_tests", ROOT / "scripts/panda_reference.py")
reference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reference)


def test_noncommuting_quarter_turn_hand_calculation():
    half_pi = reference.math.pi / 2
    x = reference.rotation([half_pi, 0, 0])
    z = reference.rotation([0, 0, half_pi])
    expected = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    assert reference.close(reference.product(z, x), expected, 1e-12)
    assert not reference.close(reference.product(x, z), expected, 1e-6)
    assert not reference.close(reference.rotation([half_pi, 0, half_pi]), expected, 1e-6)


def test_gripper_sign_integration_clipping_and_zero():
    assert reference.grip([0, 0], 0.25) == [-0.01, 0.01]
    assert reference.grip([-0.01, 0.01], -0.8) == [0, 0]
    assert reference.grip([-0.2, 0.2], 0) == [-0.2, 0.2]
    state = [0, 0]
    for _ in range(110):
        state = reference.grip(state, 1)
    assert state == [-1, 1]
    for _ in range(3):
        state = reference.grip(state, -1)
    assert state == pytest.approx([-0.97, 0.97])


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "input",
        "control",
        "wrong_gain",
        "orientation_output",
        "nonlinear",
        "reset",
        "dll",
        "source",
        "manifest",
        "simulation",
        "request",
    ],
)
def test_optimized_panda_replay(tmp_path, tamper):
    target = tmp_path / "evidence"
    shutil.copytree(ROOT / "validation/panda-software-001", target)
    summary = json.loads((target / "summary.json").read_bytes())
    capture = json.loads((target / "capture.json").read_bytes())
    if tamper == "input":
        capture["affine"]["control"]["rows"][0]["action"][0] += 0.1
    elif tamper == "control":
        capture["affine"]["control"]["rows"][0]["goal"][0] += 1
    elif tamper == "wrong_gain":
        for row in capture["affine"]["wrong_gain"]["rows"]:
            row["goal"] = [0, 0, 0]
    elif tamper == "orientation_output":
        capture["orientation"][1]["goal"] = [[0, 0, 0]] * 3
    elif tamper == "reset":
        capture["gripper"][0]["after_reset_boundary"] = capture["gripper"][0]["populated"]
    elif tamper == "dll":
        capture["bootstrap"]["windows_dll_redirects"][0]["target_sha256"] = "0" * 64
    elif tamper == "source":
        capture["identity"]["record_file_sha256"]["robosuite/controllers/osc.py"] = "0" * 64
    elif tamper == "simulation":
        capture["simulation"]["env_step_calls"] = 1
    elif tamper == "nonlinear":
        path = target / "analysis.json"
        data = json.loads(path.read_bytes())
        data["orientation"][0]["affine_evaluator_status"] = "PASS"
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
    elif tamper == "manifest":
        del summary["file_sha256"]["native.stderr.txt"]
    elif tamper == "request":
        path = target / "request.json"
        data = json.loads(path.read_bytes())
        data["inputs"]["environment"]["hard_reset"] = True
        path.write_text(json.dumps(data), encoding="utf-8", newline="\n")
        capture["request_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("capture.json", "native.stdout.txt"):
        (target / name).write_text(json.dumps(capture), encoding="utf-8", newline="\n")
    for name in summary["file_sha256"]:
        summary["file_sha256"][name] = hashlib.sha256((target / name).read_bytes()).hexdigest()
    (target / "summary.json").write_text(json.dumps(summary), encoding="utf-8", newline="\n")
    result = subprocess.run(
        [sys.executable, "-O", str(ROOT / "scripts/validate_panda.py"), "--verify", str(target)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert (result.returncode == 0) is (tamper is None)
    if tamper:
        assert "Panda software contract:" in result.stderr
        assert "Verified genuine Panda" not in result.stdout
        if tamper in ("control", "wrong_gain"):
            assert "native outputs match actual injected call and configuration" in result.stderr
        if tamper == "orientation_output":
            assert "native orientation output matches actual call" in result.stderr
    else:
        assert "Verified genuine Panda software operators" in result.stdout
