"""Release-specific integrity, privacy and notebook regressions (no credentials)."""

import ast
import io
import json
import stat
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build
import integrity
from bootstrap import prepare_payload
from notebook import notebook


@pytest.mark.parametrize("name", ["../x", "/x", "C:/x", "a\\b", "a/../b", "a//b", "./x", "a/./b"])
def test_unsafe_name(name):
    with pytest.raises(ValueError):
        integrity.safe_name(name)


def archive(entries):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as z:
        for name, raw, mode in entries:
            item = zipfile.ZipInfo(name)
            item.external_attr = mode << 16
            z.writestr(item, raw)
    data.seek(0)
    return data


@pytest.mark.parametrize(
    "entries",
    [
        [("../escape", b"x", 0o100644)],
        [("link", b"target", stat.S_IFLNK | 0o777)],
        [("x", b"1", 0o100644), ("X", b"2", 0o100644)],
    ],
)
def test_archive_rejected_before_writes(tmp_path, entries):
    destination = tmp_path / "payload"
    with pytest.raises(ValueError):
        integrity.extract(archive(entries), destination)
    assert not destination.exists()


def test_closed_inventory_and_tampering(tmp_path):
    raw = b"a\n"
    manifest = build.encoded({"schema_version": 1, "files": {"data.csv": integrity.sha(raw)}})
    target = tmp_path / "payload"
    integrity.extract(archive([("manifest.json", manifest, 0o100644), ("data.csv", raw, 0o100644)]), target)
    (target / "extra").write_bytes(b"x")
    with pytest.raises(ValueError, match="closed"):
        integrity.verify(target)
    (target / "extra").unlink()
    (target / "data.csv").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        integrity.verify(target)


def test_exact_redactions_preserve_numbers():
    original = {
        "runtime": {"environment_prefix": "C:\\Users\\Example\\env", "calls": 7},
        "observations": [0.125, 1.0, -3.5],
        "upstream": "https://github.com/Physical-Intelligence/openpi.git",
    }
    projected, pointers = build.project(original)
    assert pointers == ["/runtime/environment_prefix"]
    assert projected["observations"] == original["observations"]
    assert projected["upstream"] == original["upstream"]
    assert original["runtime"]["environment_prefix"] != projected["runtime"]["environment_prefix"]
    with pytest.raises(ValueError, match="unexpected sensitive"):
        build.project({"new_unreviewed_field": "C:\\Users\\Example\\sensitive"})


@pytest.mark.parametrize(
    "text",
    [
        "KGAT_" + "a" * 30,
        "hf_" + "a" * 34,
        "github_pat_" + "a" * 70,
        "-----BEGIN PRIVATE KEY-----",
        "C:\\Users\\Example\\secret",
    ],
)
def test_privacy_scan(text):
    with pytest.raises(ValueError):
        build.privacy(text.encode(), "fixture")


def test_public_repository_link_does_not_allow_embedded_credentials():
    public = b"https://github.com/abdullah2hamza-blip/MLTools"
    build.privacy(public, "public citation")
    with pytest.raises(ValueError):
        build.privacy(public + b"?token=KGAT_" + b"a" * 32, "credential-bearing link")


def test_zip_is_deterministic():
    assert build.make_zip({"b": b"2", "a": b"1"}) == build.make_zip({"a": b"1", "b": b"2"})


def test_notebook_code_and_no_hidden_state():
    doc = notebook("a" * 64, "b" * 64, "boundary-conformance-replay", "c" * 64)
    assert doc["nbformat"] == 4
    for cell in doc["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "<cell>", "exec")
            for node in ast.walk(ast.parse(cell["source"])):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "runner" for t in node.targets
                ):
                    compile(ast.literal_eval(node.value), "<replay-subprocess>", "exec")
            assert cell["execution_count"] is None and cell["outputs"] == []
            assert "import kaggle" not in cell["source"]


def test_unapproved_license_fails_before_build(tmp_path):
    config = json.loads((Path(__file__).parent / "config.json").read_bytes())
    config["license_approved"] = False
    with pytest.raises(ValueError, match="approved original license"):
        build.build(tmp_path / "out", config)
    assert not (tmp_path / "out").exists()


def small_payload(root):
    raw = b"saved observation\n"
    manifest = build.encoded(
        {"schema_version": 1, "release_id": "fixture", "files": {"data.txt": integrity.sha(raw)}}
    )
    files = {"manifest.json": manifest, "data.txt": raw}
    data = build.make_zip(files)
    root.mkdir()
    (root / "payload.zip").write_bytes(data)
    return data, manifest


@pytest.mark.parametrize("expanded", [False, True])
def test_kaggle_input_representations(tmp_path, expanded):
    source = tmp_path / "input"
    data, manifest = small_payload(source)
    if expanded:
        integrity.extract(source / "payload.zip", source / "payload")
        (source / "payload.zip").unlink()
    mode = prepare_payload(source, tmp_path / "output", integrity.sha(data), integrity.sha(manifest))
    assert mode == ("expanded-directory" if expanded else "archive")
    integrity.verify(tmp_path / "output")


@pytest.mark.parametrize("mutation", ["changed-observation", "extra-file", "wrong-manifest"])
def test_expanded_input_rejects_tampering(tmp_path, mutation):
    source = tmp_path / "input"
    data, manifest = small_payload(source)
    integrity.extract(source / "payload.zip", source / "payload")
    (source / "payload.zip").unlink()
    if mutation == "changed-observation":
        (source / "payload/data.txt").write_bytes(b"substitute")
    elif mutation == "extra-file":
        (source / "payload/extra.py").write_bytes(b"print('unexpected')")
    else:
        (source / "payload/manifest.json").write_bytes(b"{}")
    with pytest.raises(AssertionError):
        prepare_payload(source, tmp_path / "output", integrity.sha(data), integrity.sha(manifest))
