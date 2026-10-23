"""Copy either Kaggle-expanded or archived payloads with the same pinned closure."""


def prepare_payload(input_root, destination, archive_sha256, manifest_sha256):
    # This complete function is embedded verbatim into the generated notebook.
    import hashlib
    import json
    import stat
    import zipfile
    from pathlib import Path, PurePosixPath

    def digest(raw):
        return hashlib.sha256(raw).hexdigest()

    def safe(name):
        p = PurePosixPath(name)
        assert name and "\\" not in name and not p.is_absolute(), "Unsafe payload path"
        assert p.as_posix() == name and all(x not in ("", ".", "..") and ":" not in x for x in p.parts)
        return p

    input_root, destination = Path(input_root), Path(destination)
    assert not destination.exists(), "Use a new payload destination"
    archives = [
        p for p in sorted(input_root.rglob("payload.zip")) if digest(p.read_bytes()) == archive_sha256
    ]
    if archives:
        assert len(archives) == 1, "Attach only one matching dataset version"
        with zipfile.ZipFile(archives[0]) as archive:
            entries = archive.infolist()
            assert 0 < len(entries) <= 2000
            assert sum(i.file_size for i in entries) <= 64 * 1024 * 1024
            assert len({i.filename.casefold() for i in entries}) == len(entries)
            for item in entries:
                safe(item.filename)
                assert not item.is_dir() and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
                assert not item.flag_bits & 1
            for item in entries:
                path = destination.joinpath(*safe(item.filename).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(item))
        representation = "archive"
    else:
        candidates = [
            p for p in sorted(input_root.rglob("manifest.json")) if digest(p.read_bytes()) == manifest_sha256
        ]
        assert len(candidates) == 1, "Attach the dataset version with the pinned payload manifest"
        source = candidates[0].parent
        manifest = json.loads(candidates[0].read_bytes())
        expected = manifest["files"]
        assert 0 < len(expected) < 2000 and manifest["schema_version"] == 1
        paths = list(source.rglob("*"))
        assert not any(p.is_symlink() for p in paths), "No payload symlinks"
        assert {p.relative_to(source).as_posix() for p in paths if p.is_file()} == set(expected) | {
            "manifest.json"
        }
        assert sum(p.stat().st_size for p in paths if p.is_file()) <= 64 * 1024 * 1024
        # Check bytes before copying or importing any payload source.
        for name, expected_sha in expected.items():
            path = source.joinpath(*safe(name).parts)
            assert digest(path.read_bytes()) == expected_sha, "Expanded payload hash mismatch: " + name
        for name in [*expected, "manifest.json"]:
            raw = source.joinpath(*safe(name).parts).read_bytes()
            path = destination.joinpath(*safe(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        representation = "expanded-directory"
    assert digest((destination / "manifest.json").read_bytes()) == manifest_sha256
    manifest = json.loads((destination / "manifest.json").read_bytes())
    expected = manifest["files"]
    assert {p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()} == (
        set(expected) | {"manifest.json"}
    )
    for name, expected_sha in expected.items():
        assert digest(destination.joinpath(*safe(name).parts).read_bytes()) == expected_sha
    return representation
