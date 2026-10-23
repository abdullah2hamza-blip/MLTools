"""Byte-exact, committed protocol gate for the selected metadata evaluation."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

from action_study_common import ROOT, fingerprint, sha, write
from normalization_profiles import require
from publication_assets import DEVELOPMENT
from publication_assets import verify as verify_assets
from run_policy_space_numeric import sources as baseline_sources
from study_assets import read

SPEC = "research/publication/protocol-specification.json"
FREEZE = "research/publication/protocol-freeze-001.json"
PLANS = {
    "development": DEVELOPMENT,
    "prospective": "research/publication/prepared/action-study-001",
}


def relative(path):
    resolved = (ROOT / path).resolve()
    require(resolved.is_relative_to(ROOT), "workspace-contained protocol path")
    return resolved.relative_to(ROOT).as_posix()


def source_inventory():
    paths = [*sorted((ROOT / "scripts").glob("*.py")), *sorted((ROOT / "src/mltools").glob("*.py"))]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def build(role):
    require(role in PLANS, "explicit protocol role")
    folder = ROOT / PLANS[role]
    plan = verify_assets(folder, role)
    require(plan["role"] == role, "exact role-specific assets")
    spec = read(ROOT / SPEC)
    require(type(spec["schema_version"]) is int and spec["schema_version"] == 1, "protocol version")
    files = {**baseline_sources(), **source_inventory()}
    for path in (
        SPEC,
        "pyproject.toml",
        ".python-version",
        "research/publication/study-design.json",
        "research/publication/method-applicability.json",
    ):
        files[path] = sha(ROOT / path)
    files[PLANS[role] + "/plan.json"] = sha(folder / "plan.json")
    files.update({PLANS[role] + "/" + k: v for k, v in plan["file_sha256"].items()})
    files.update(plan["original_input_sha256"])
    cells = plan["cells"]
    environments = None
    primary = sum(
        len(d["probes"]) * len(cell["conditions"])
        for cell in cells.values()
        for d in cell["designs"].values()
    )
    auxiliary = sum(
        4 * len(cell["designs"]["canonical"]["probes"]) * len(cell["conditions"]) for cell in cells.values()
    )
    if role == "prospective":
        record = "validation/publication-wrapper-development-verification.json"
        validation = read(ROOT / record)
        require(
            validation["status"] == "VALIDATED"
            and validation["runtime_source_sha256"] == source_inventory()
            and validation["prospective_native_outcomes_observed"] is False,
            "complete current wrapper development validation before freeze",
        )
        environments = validation["native_environments"]
        require(
            set(environments) == {"lerobot", "openpi", "robotruth", "policy_space"},
            "all native profiles frozen",
        )
        files[record] = sha(ROOT / record)
        require(
            validation["configuration_sha256"]
            == {
                p: sha(ROOT / p)
                for p in (
                    SPEC,
                    "research/publication/study-design.json",
                    "research/publication/method-applicability.json",
                )
            },
            "validated complete current configuration",
        )
        for path, digest in validation["test_file_sha256"].items():
            require(sha(ROOT / path) == digest, "validated exact test sources")
            files[path] = digest
        for path, digest in validation["development_evidence_sha256"].items():
            require(sha(ROOT / path) == digest, "validated development evidence bytes")
            files[path] = digest
        require(primary == 1792 and auxiliary == 896 and len(cells) == 4, "selected design denominators")
    return {
        "schema_version": 1,
        "format": "mltools-committed-publication-protocol",
        "role": role,
        "status": "FROZEN_WHEN_PRESENT_IN_VERIFIED_COMMIT"
        if role == "prospective"
        else "DEVELOPMENT_VALIDATION",
        "assets_path": PLANS[role],
        "specification": spec,
        "source_file_sha256": source_inventory(),
        "native_environments": environments,
        "file_sha256": dict(sorted(files.items())),
        "planned_calls": {
            "primary_decoder": primary,
            "roundtrip_auxiliary": auxiliary,
            "fresh_witness_decoder_maximum": sum(
                len(cell["designs"]["canonical"]["probes"]) * (len(cell["conditions"]) - 1)
                for cell in cells.values()
            ),
            "stock_extract": sum(len(cell["conditions"]) + 1 for cell in cells.values()),
            "stock_contract_check": sum(len(cell["conditions"]) for cell in cells.values()),
            "policy_space_primary_replays": primary,
            "policy_space_structural_controls": 4 * len(cells),
        },
        "no_prospective_execution_performed_by_compilation": True,
    }


def committed_bytes(root, commit, paths):
    """Read all protected files from one full immutable commit, not the index."""
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "full immutable Git commit")
    require(
        all(p and not p.startswith(("/", "-")) and ":" not in p and "\n" not in p for p in paths), "Git paths"
    )
    raw = subprocess.check_output(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input="".join(commit + ":" + p + "\n" for p in paths).encode(),
    )
    result, offset = {}, 0
    for path in paths:
        end = raw.index(b"\n", offset)
        header = raw[offset:end].split()
        require(len(header) == 3 and header[1] == b"blob", "every frozen file exists in the same commit")
        size = int(header[2])
        offset = end + 1
        result[path] = raw[offset : offset + size]
        offset += size + 1
    require(offset == len(raw), "complete Git byte inventory")
    return result


def verify(path, commit=None):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), "contained protocol")
    raw = path.read_bytes()
    protocol = json.loads(raw)
    require(
        type(protocol["schema_version"]) is int and protocol["schema_version"] == 1, "typed protocol version"
    )
    require(
        protocol.get("format") == "mltools-committed-publication-protocol",
        "complete publication protocol format",
    )
    require(fingerprint(protocol) == fingerprint(build(protocol["role"])), "exact complete frozen protocol")
    require(
        all(sha(ROOT / relative(name)) == digest for name, digest in protocol["file_sha256"].items()),
        "all executed/configuration/input bytes match the protocol",
    )
    if protocol["role"] == "prospective":
        require(path == ROOT / FREEZE, "canonical prospective freeze location")
        stored = committed_bytes(ROOT, commit, [FREEZE, *protocol["file_sha256"]])
        require(stored[FREEZE] == raw, "protocol itself committed before native execution")
        require(
            all(
                hashlib.sha256(stored[name]).hexdigest() == digest
                for name, digest in protocol["file_sha256"].items()
            ),
            "all protected bytes in the same immutable freeze commit",
        )
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT, check=False
        )
        require(ancestor.returncode == 0, "freeze commit belongs to current history")
    else:
        require(commit is None, "development protocol is not a committed prospective freeze")
    return protocol, verify_assets(ROOT / protocol["assets_path"], protocol["role"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--compile", action="store_true")
    group.add_argument("--verify", type=Path)
    parser.add_argument("--freeze-commit")
    args = parser.parse_args()
    if args.compile:
        path = ROOT / FREEZE
        require(not path.exists(), "freeze compilation cannot overwrite an earlier protocol")
        write(path, build("prospective"))
        print("Compiled full prospective protocol; commit it before every selected native execution")
    else:
        verify(args.verify, args.freeze_commit)
        print("Verified complete protocol and exact committed bytes where required")


if __name__ == "__main__":
    main()
