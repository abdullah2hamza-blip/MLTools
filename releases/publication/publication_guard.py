"""Closed research supplement checks; credential variable names are ordinary source."""

import hashlib
import io
import json
import re
import stat
import zipfile
from pathlib import PurePosixPath

PROFILE = "public_research_supplement_v1"
EXTRA_FILES = {
    "manuscript.pdf",
    "manuscript-source.zip",
    "research-source.zip",
    "public-release-manifest.json",
}
ARCHIVES = {"manuscript-source.zip", "research-source.zip"}
FREEZE_COMMIT = "7513755e9627098c3529c46b3ae9a056d76434a5"
PROTOCOL = "research/publication/protocol-freeze-001.json"
PROTOCOL_SHA256 = "76a825f41b72e5cbe945a4f7e6930ef5e00df0a43bc461ccfb293be6773481fc"
MAX_FILE = 64 * 1024 * 1024
MAX_ARCHIVE = 128 * 1024 * 1024
FORBIDDEN_PARTS = {
    ".git",
    ".aws",
    ".ssh",
    ".kaggle",
    ".codex",
    ".agents",
    ".test-tmp",
    "private-state",
    "__pycache__",
    "node_modules",
}
FORBIDDEN_SUFFIXES = {
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".whl",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
    ".bundle",
    ".bin",
}
VALUES = re.compile(
    rb"(?:KGAT_[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    rb"github_pat_[A-Za-z0-9_]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{35}|"
    rb"xox[baprs]-[A-Za-z0-9-]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{40,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=]{32,})"
)


def require(condition, message):
    if not condition:
        raise ValueError("Publication supplement: " + message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def bad(value):
        raise ValueError("Publication supplement: nonfinite JSON " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name and "\x00" not in name, "relative file name")
    parts = name.split("/")
    require(
        not name.startswith("/") and ":" not in name and all(x not in ("", ".", "..") for x in parts),
        "safe relative file name",
    )
    require(
        not any(x.lower() in FORBIDDEN_PARTS or x.lower().startswith((".venv", ".env")) for x in parts)
        and PurePosixPath(name).suffix.lower() not in FORBIDDEN_SUFFIXES,
        "excluded private/runtime file",
    )
    return name


def check_values(raw, label):
    require(len(raw) <= MAX_FILE, "bounded file: " + label)
    require(VALUES.search(raw) is None, "credential value/private key in " + label)
    require(not raw.startswith((b"MZ", b"\x7fELF")), "native runtime binary in " + label)
    if label.endswith(".safetensors"):
        require(10 <= len(raw) <= 128 * 1024, "small numeric statistics/fixture only")
        size = int.from_bytes(raw[:8], "little")
        require(2 <= size <= len(raw) - 8, "bounded Safetensors header")
        tensors = {k: v for k, v in parse(raw[8 : 8 + size]).items() if k != "__metadata__"}
        fields = {"count", "max", "mean", "min", "q01", "q10", "q50", "q90", "q99", "std"}
        require(
            tensors and all(v.get("dtype") in ("F32", "F64") for v in tensors.values()), "numeric tensors"
        )
        require(
            all(k.split(".")[-1] in fields for k in tensors)
            or tensors == {"fixture_weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
            "statistics or the authored two-float identity fixture; no learned weights",
        )


def archive_members(raw):
    check_values(raw, "source archive")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= 6000, "bounded source archive members")
        require(sum(x.file_size for x in entries) <= MAX_ARCHIVE, "bounded expanded source archive")
        require(len({x.filename.casefold() for x in entries}) == len(entries), "unique archive names")
        output = {}
        for item in entries:
            name = safe_name(item.filename)
            require(
                not item.is_dir()
                and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
                and not item.flag_bits & 1,
                "regular unencrypted archive files",
            )
            data = archive.read(item)
            check_values(data, name)
            require(not data.startswith(b"PK\x03\x04"), "nested historical archives are excluded")
            output[name] = data
        return output


def verify_frozen_closure(files):
    require(PROTOCOL in files and sha(files[PROTOCOL]) == PROTOCOL_SHA256, "exact frozen protocol bytes")
    protected = parse(files[PROTOCOL])["file_sha256"]
    require(len(protected) == 406, "complete 406-file frozen closure")
    for path, digest in protected.items():
        safe_name(path)
        require(path in files and sha(files[path]) == digest, "frozen source/input bytes: " + path)
    return len(protected)


def validate(snapshots, release):
    """Inspect the optional supplement without weakening focused replay checks."""
    require(release.get("distribution_profile") == PROFILE, "explicit research supplement profile")
    raw = {name: snapshots["dataset/" + name] for name in EXTRA_FILES}
    for name, data in raw.items():
        check_values(data, name)
    require(raw["manuscript.pdf"].startswith(b"%PDF-"), "manuscript PDF signature")
    manifest = parse(raw["public-release-manifest.json"])
    require(
        type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1,
        "public manifest version",
    )
    require(manifest.get("profile") == PROFILE, "public manifest profile")
    require(manifest.get("freeze_commit") == FREEZE_COMMIT, "original freeze identity")
    require(re.fullmatch(r"[a-f0-9]{40}", manifest.get("source_commit", "")) is not None, "source commit")
    require(isinstance(manifest.get("source_tag"), str) and manifest["source_tag"], "source tag")
    expected = manifest.get("files")
    require(
        isinstance(expected, dict) and set(expected) == EXTRA_FILES - {"public-release-manifest.json"},
        "extra file closure",
    )
    require(all(sha(raw[name]) == digest for name, digest in expected.items()), "extra file byte hashes")
    inventory = manifest.get("zip_members")
    require(isinstance(inventory, dict) and set(inventory) == ARCHIVES, "both source ZIP member inventories")
    contents = {name: archive_members(raw[name]) for name in ARCHIVES}
    for name, files in contents.items():
        require(
            {path: sha(data) for path, data in files.items()} == inventory[name],
            "exact expanded ZIP member hashes: " + name,
        )
    protected = verify_frozen_closure(contents["research-source.zip"])
    paper_root = "research/publication/paper/"
    research = contents["research-source.zip"]
    manuscript = contents["manuscript-source.zip"]
    require(
        research.get(paper_root + "output/main.pdf") == raw["manuscript.pdf"], "PDF matches source snapshot"
    )
    for path, data in research.items():
        if path.startswith(paper_root) and not path.startswith(paper_root + "output/"):
            require(
                manuscript.get(path.removeprefix(paper_root)) == data, "editable manuscript matches snapshot"
            )
    replay = manifest.get("replay", {})
    require(
        replay.get("release_id") == release.get("release_id")
        and replay.get("payload_sha256") == release.get("payload_sha256")
        and replay.get("expected_replay_outputs") == release.get("expected_replay_outputs"),
        "supplement binds reviewed numerical replay",
    )
    return {
        "profile": PROFILE,
        "source_commit": manifest["source_commit"],
        "source_tag": manifest["source_tag"],
        "protected_file_count": protected,
        "zip_member_counts": {name: len(files) for name, files in contents.items()},
        "scope": (
            "Byte-bound source snapshot; full original Git verification requires the documented public clone"
        ),
    }
