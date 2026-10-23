"""Closed, bounded export integrity. Hashes are integrity checks, not attestation."""

import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path, PurePosixPath

MAX_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 2000


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_name(name):
    p = PurePosixPath(name)
    require(
        name
        and "\\" not in name
        and not p.is_absolute()
        and p.as_posix() == name
        and all(x not in ("", ".", "..") for x in p.parts)
        and not any(":" in x for x in p.parts),
        "unsafe archive/member path",
    )
    return p


def extract(archive, destination):
    """Validate the entire archive before writing; never use extractall."""
    destination = Path(destination)
    require(not destination.exists(), "extraction destination must be new")
    with zipfile.ZipFile(archive) as z:
        entries = z.infolist()
        require(0 < len(entries) <= MAX_MEMBERS, "bounded member count")
        names = [i.filename for i in entries]
        require(len(set(n.casefold() for n in names)) == len(names), "duplicate member")
        require(sum(i.file_size for i in entries) <= MAX_BYTES, "bounded uncompressed size")
        for i in entries:
            safe_name(i.filename)
            mode = i.external_attr >> 16
            require(not i.is_dir() and stat.S_IFMT(mode) in (0, stat.S_IFREG), "regular files only")
            require(not i.flag_bits & 1, "encrypted archive forbidden")
        destination.mkdir(parents=True)
        for i in entries:
            path = destination.joinpath(*PurePosixPath(i.filename).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(i))
    verify(destination)


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    require(manifest.get("schema_version") == 1, "manifest version")
    expected = manifest["files"]
    require(isinstance(expected, dict) and expected, "closed file map")
    for name, digest in expected.items():
        safe_name(name)
        require(re.fullmatch(r"[a-f0-9]{64}", digest) is not None, "SHA256 required")
    paths = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "symlinks forbidden")
    actual = {p.relative_to(root).as_posix() for p in paths if p.is_file()}
    require(actual == set(expected) | {"manifest.json"}, "closed payload inventory")
    require(sum(p.stat().st_size for p in paths if p.is_file()) <= MAX_BYTES, "bounded payload")
    for name, digest in expected.items():
        require(sha((root / name).read_bytes()) == digest, "payload hash mismatch: " + name)
    return manifest
