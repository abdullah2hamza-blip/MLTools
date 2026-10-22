import json
import subprocess
import sys

import pytest
from conftest import EXAMPLES

from mltools.cli import main
from mltools.io import MAX_BYTES, InputError, load_bundle, read_json, write_json


@pytest.mark.parametrize(
    "path", ["../outside.json", "C:/private.json", "artifacts/../../outside.json", "https://x"]
)
def test_artifact_path_cannot_escape(sample, path):
    location, bundle, _ = sample
    bundle["artifacts"][0]["path"] = path
    write_json(location, bundle)
    with pytest.raises(InputError):
        load_bundle(location)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', "[]"])
def test_unsafe_or_ambiguous_json_rejected(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(InputError):
        read_json(path)


def test_size_limit(tmp_path):
    path = tmp_path / "large.json"
    path.write_bytes(b" " * (MAX_BYTES + 1))
    with pytest.raises(InputError, match="exceeds"):
        read_json(path)


@pytest.mark.parametrize("text", ['{"a":1e1000}', '{"a":' + "9" * 310 + "}"])
def test_float_overflow_and_huge_integers_are_input_errors(tmp_path, text):
    path = tmp_path / "overflow.json"
    path.write_text(text)
    with pytest.raises(InputError):
        read_json(path)


@pytest.mark.parametrize("bad", [[True, 2, 3, 4, 5, 6], [1, 2], [1, 2, 3, 4, 5, "6"]])
def test_prediction_requires_finite_numeric_dimensions(sample, bad):
    path, bundle, _ = sample
    bundle["scenario"][1]["prediction"][0] = bad
    write_json(path, bundle)
    with pytest.raises(InputError):
        load_bundle(path)


def test_cli_exit_statuses_and_machine_json(capsys):
    assert (
        main(
            [
                "check",
                str(EXAMPLES / "good.json"),
                "--trace",
                str(EXAMPLES / "good.trace.json"),
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"
    assert (
        main(
            [
                "check",
                str(EXAMPLES / "moving_anchor.json"),
                "--trace",
                str(EXAMPLES / "moving_anchor.trace.json"),
            ]
        )
        == 1
    )
    capsys.readouterr()
    assert main(["check", str(EXAMPLES / "good.json")]) == 2
    capsys.readouterr()
    assert main(["check", "missing.json"]) == 3
    assert "ERROR:" in capsys.readouterr().err


def test_core_replay_does_not_import_framework():
    script = (
        "import sys; from mltools.checker import check; "
        f"result=check({str(EXAMPLES / 'good.json')!r}, {str(EXAMPLES / 'good.trace.json')!r}); "
        "assert result['status']=='PASS'; "
        "assert 'torch' not in sys.modules; assert 'lerobot' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


def test_output_cannot_overwrite_reference_bundle(sample, capsys):
    path, _, _ = sample
    original = path.read_bytes()
    assert main(["check", str(path), "--json", str(path)]) == 3
    assert "cannot overwrite" in capsys.readouterr().err
    assert path.read_bytes() == original


def test_invalid_invocation_has_operational_exit_code():
    with pytest.raises(SystemExit) as result:
        main(["check", "--unrecognized"])
    assert result.value.code == 3
