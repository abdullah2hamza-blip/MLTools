import importlib.util
from pathlib import Path

import pytest

from mltools.checker import check
from mltools.io import digest, read_json
from mltools.lerobot import capture

ROOT = Path(__file__).resolve().parents[1] / "examples" / "published_so101"


@pytest.mark.parametrize(
    ("name", "status", "normalization"),
    [
        ("metadata", "UNKNOWN", "PASS"),
        ("wrong_stats", "FAIL", "FAIL"),
    ],
)
def test_published_metadata_replay_has_honest_partial_scope(name, status, normalization):
    report = check(ROOT / f"{name}.json", ROOT / f"{name}.trace.json")
    assert report["status"] == status
    assert report["checks"]["normalization"] == normalization
    assert report["checks"]["joint_order"] == "UNKNOWN"
    assert report["checks"]["training_dataset_revision"] == "UNKNOWN"
    if name == "wrong_stats":
        failure = report["first_divergence"]
        assert failure["boundary"] == "action.unnormalize"
        assert failure["expected"][0] == 3.9553866386413574
        assert failure["actual"][0] == 13.955386638641357


def test_published_artifact_identities_and_no_weights():
    manifest = read_json(ROOT / "source_manifest.json")
    assert manifest["revision"] == "0220b37dbcf1e0b2f6f5e21c64c75bf16fac6b69"
    assert manifest["weights_downloaded"] is False
    for artifact in manifest["files"]:
        assert digest(ROOT / artifact["path"]) == artifact["sha256"]
    assert not (ROOT / "source" / "model.safetensors").exists()


@pytest.mark.skipif(importlib.util.find_spec("lerobot") is None, reason="Optional real integration required")
def test_published_real_processor_path():
    path = ROOT / "metadata.json"
    trace = capture(path)
    report = check(path, trace)
    assert report["checks"]["normalization"] == "PASS"
    assert report["status"] == "UNKNOWN"
    assert trace["events"][1]["action"]["commands"] is None
