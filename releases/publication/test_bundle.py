"""Consequential supplement regressions; no Kaggle imports or remote calls."""

import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "kaggle")]
import bundle  # noqa: E402
import publication_guard as guard  # noqa: E402
import test_publish as fixtures  # noqa: E402


@pytest.fixture(scope="session")
def all_source_files():
    source = bundle.revision_files(HERE.parents[1], "HEAD")
    source["releases/publication/LICENSE.LPPL-1.3c.txt"] = (HERE / "LICENSE.LPPL-1.3c.txt").read_bytes()
    return source


@pytest.fixture(scope="session")
def source_files(all_source_files):
    source = all_source_files
    selected, omitted = bundle.source_snapshot(source)
    assert len(guard.parse(source[guard.PROTOCOL])["file_sha256"]) == 406
    assert omitted
    return selected


@pytest.fixture
def release(source_files):
    fixture = fixtures.PublisherTests("test_default_plan_never_discovers_cli_or_writes")
    fixture.setUp()
    main = fixture.root / "dataset/release-manifest.json"
    replay = guard.parse(main.read_bytes())
    manuscript = {
        name.removeprefix(bundle.PAPER): data
        for name, data in source_files.items()
        if name.startswith(bundle.PAPER) and not name.startswith(bundle.PAPER + "output/")
    }
    extras = {
        "manuscript.pdf": source_files[bundle.PAPER + "output/main.pdf"],
        "manuscript-source.zip": bundle.archive(manuscript),
        "research-source.zip": bundle.archive(source_files),
    }
    public = {
        "schema_version": 1,
        "profile": guard.PROFILE,
        "source_tag": "unit-fixture-do-not-publish",
        "source_commit": "a" * 40,
        "freeze_commit": guard.FREEZE_COMMIT,
        "files": {name: guard.sha(raw) for name, raw in extras.items()},
        "zip_members": {
            "manuscript-source.zip": {name: guard.sha(raw) for name, raw in manuscript.items()},
            "research-source.zip": {name: guard.sha(raw) for name, raw in source_files.items()},
        },
        "replay": {
            name: replay[name] for name in ("release_id", "payload_sha256", "expected_replay_outputs")
        },
    }
    extras["public-release-manifest.json"] = guard.encoded(public)
    for name, raw in extras.items():
        (fixture.dataset / name).write_bytes(raw)
    replay["distribution_profile"] = guard.PROFILE
    replay["files"].update({name: guard.sha(raw) for name, raw in extras.items()})
    main.write_bytes(guard.encoded(replay))
    try:
        yield fixture, public
    finally:
        fixture.doCleanups()


def rebind(fixture, public):
    path = fixture.dataset / "public-release-manifest.json"
    path.write_bytes(guard.encoded(public))
    main = fixture.dataset / "release-manifest.json"
    value = guard.parse(main.read_bytes())
    for name in guard.EXTRA_FILES:
        value["files"][name] = guard.sha((fixture.dataset / name).read_bytes())
    main.write_bytes(guard.encoded(value))


def test_explicit_profile_retains_all_original_replay_guards(release):
    fixture, _ = release
    result = fixtures.publish.validate(fixture.root, fixture.config)
    assert result["public_supplement"]["protected_file_count"] == 406
    assert not fixtures.publish.plan(fixture.root, fixture.config)["credentials_accessed"]


def test_default_profile_cannot_admit_extra_files(release):
    fixture, _ = release
    fixture.manifest()
    with pytest.raises(ValueError, match="closed inventory"):
        fixtures.publish.validate(fixture.root, fixture.config)


def test_optional_profile_requires_exact_inventory(release):
    fixture, _ = release
    (fixture.dataset / "extra-unreviewed.txt").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="closed inventory"):
        fixtures.publish.validate(fixture.root, fixture.config)


def test_rehashed_source_change_cannot_evade_original_frozen_closure(release, source_files):
    fixture, public = release
    changed = dict(source_files)
    name = next(iter(guard.parse(changed[guard.PROTOCOL])["file_sha256"]))
    changed[name] += b"\nchanged frozen input"
    raw = bundle.archive(changed)
    (fixture.dataset / "research-source.zip").write_bytes(raw)
    public["files"]["research-source.zip"] = guard.sha(raw)
    public["zip_members"]["research-source.zip"][name] = guard.sha(changed[name])
    rebind(fixture, public)
    with pytest.raises(ValueError, match="frozen source/input bytes"):
        fixtures.publish.validate(fixture.root, fixture.config)


def test_expanded_member_inventory_is_independently_bound(release):
    fixture, public = release
    raw = bundle.archive({"different.tex": b"new source"})
    (fixture.dataset / "manuscript-source.zip").write_bytes(raw)
    public["files"]["manuscript-source.zip"] = guard.sha(raw)
    rebind(fixture, public)
    with pytest.raises(ValueError, match="expanded ZIP member hashes"):
        fixtures.publish.validate(fixture.root, fixture.config)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a\\b", "a//b", ".git/config", "x/.env", "runtime.dll"]
)
def test_source_archive_names_are_bounded(name):
    with pytest.raises(ValueError):
        bundle.archive({name: b"unsafe"})


@pytest.mark.parametrize("value", ["KGAT_" + "a" * 32, "hf_" + "a" * 34, "github_pat_" + "a" * 70])
def test_source_guard_checks_values_and_permits_variable_names(value):
    guard.check_values(b'os.environ.get("KAGGLE_API_TOKEN"); HF_TOKEN; C:\\Users\\Example', "source.py")
    with pytest.raises(ValueError, match="credential value"):
        bundle.archive({"source.py": value.encode()})


def test_zip_links_rejected_before_acceptance(tmp_path):
    path = tmp_path / "link.zip"
    with zipfile.ZipFile(path, "w") as archive:
        item = zipfile.ZipInfo("link")
        item.external_attr = 0o120777 << 16
        archive.writestr(item, b"outside")
    with pytest.raises(ValueError, match="regular unencrypted"):
        guard.archive_members(path.read_bytes())


def test_archive_generation_is_byte_deterministic():
    assert bundle.archive({"b": b"2", "a": b"1"}) == bundle.archive({"a": b"1", "b": b"2"})


def test_snapshot_records_omissions_without_losing_frozen_files(source_files):
    assert guard.verify_frozen_closure(source_files) == 406
    assert not any(name in source_files for name in bundle.STYLES)
    assert not any(name.startswith("releases/kaggle/artifacts/") for name in source_files)


def test_built_supplements_bind_compiled_pdf_and_editable_sources(all_source_files):
    replay = {"release_id": "unit-test", "payload_sha256": "a" * 64, "expected_replay_outputs": {"a": "b"}}
    extras = bundle.supplements(all_source_files, "unit-test-only", "b" * 40, replay)
    release = {**replay, "distribution_profile": guard.PROFILE}
    result = guard.validate({"dataset/" + name: raw for name, raw in extras.items()}, release)
    assert result["protected_file_count"] == 406


def test_editing_manuscript_requires_an_updated_pdf_build(all_source_files):
    changed = dict(all_source_files)
    changed[bundle.PAPER + "article.tex"] += b"\nChanged after compilation"
    replay = {"release_id": "unit-test", "payload_sha256": "a" * 64, "expected_replay_outputs": {"a": "b"}}
    with pytest.raises(ValueError, match="manuscript changed since compile"):
        bundle.supplements(changed, "unit-test-only", "b" * 40, replay)


def test_tag_build_refuses_a_dirty_tracked_checkout(monkeypatch, tmp_path):
    def fake_git(root, *args, **kwargs):
        if args[0] == "rev-parse":
            return ("a" * 40 + "\n").encode()
        assert args[0] == "status"
        return b" M research/publication/paper/article.tex\n"

    monkeypatch.setattr(bundle, "git", fake_git)
    with pytest.raises(ValueError, match="clean tracked checkout"):
        bundle.clean_tag(tmp_path, "unit-test-only")
