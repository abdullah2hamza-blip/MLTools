"""Execute stock robotruth extraction/checking controls or verify saved development evidence.

No policy inference, external services, final benchmark candidates or physical robots.
The --verify path is stdlib-only and validates records, not a new native execution.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import struct
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "integrations/robotruth"
SYNTHETIC = (
    "synthetic_complete",
    "synthetic_weights",
    "synthetic_statistics",
    "synthetic_chunk",
    "synthetic_camera",
    "synthetic_units",
    "synthetic_missing_gripper",
    "synthetic_missing_weights",
)
PUBLISHED = {
    "published_smolvla": "examples/published_so101/source_manifest.json",
    "published_act": "examples/probes/act/source_manifest.json",
    "published_act_statistics": "examples/probes/act/source_manifest.json",
}
CASES = (*SYNTHETIC, *PUBLISHED)


def require(condition, message):
    if not condition:
        raise ValueError(f"Robottruth development evidence: {message}")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def project_sources():
    paths = [Path(__file__), *sorted(PROFILE.glob("*"))]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths if p.is_file() and p.suffix != ".md"}


def identities():
    ledger = read(PROFILE / "sources.json")
    distribution = importlib.metadata.distribution("robotruth")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(direct.get("vcs_info", {}).get("commit_id") == ledger["revision"], "installed revision")
    require(direct.get("url") == "https://github.com/mulkakhileshmj/robotruth.git", "installed origin")
    result = {}
    for name, expected in ledger["package_python_git_blobs"].items():
        raw = distribution.locate_file(name).read_bytes()

        def blob(data):
            return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()

        exact = blob(raw) == expected
        require(exact or blob(raw.replace(b"\r\n", b"\n")) == expected, f"installed source: {name}")
        result[name] = {
            "git_blob": expected,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "verification": "exact_git_blob" if exact else "CRLF_to_LF_matches_pinned_git_blob",
        }
    actual = {
        Path(str(f)).as_posix()
        for f in distribution.files
        if str(f).replace("\\", "/").startswith("robotruth/") and str(f).endswith(".py")
    }
    require(actual == set(result), "complete installed Python inventory")
    return {"direct_url": direct, "python_sources": result}


def synthetic_inputs(folder, condition, role):
    """Independent synthetic declarations; tiny valid tensor bytes are not trained weights."""
    deployed = role == "deployed"
    cfg = {
        "type": "act",
        "chunk_size": 3 if deployed and condition == "synthetic_chunk" else 2,
        "n_action_steps": 2,
        "relative_actions": False,
        "normalization_mapping": {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [2]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 8, 8]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [2]}},
    }
    if deployed and condition == "synthetic_camera":
        cfg["input_features"]["observation.images.wrist"] = cfg["input_features"].pop(
            "observation.images.top"
        )
    write(folder / "config.json", cfg)
    write(folder / "train_config.json", {"dataset": {"fps": 20, "repo_id": "synthetic/two-axis"}})
    stats = {"action": {"mean": [0.2, -0.1], "std": [1.5, 2.0]}}
    if deployed and condition == "synthetic_statistics":
        stats["action"]["mean"][0] += 1
    write(folder / "stats.json", stats)
    if not (deployed and condition == "synthetic_missing_weights"):
        header = json.dumps(
            {"fixture_weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
        ).encode()
        header += b" " * (-len(header) % 8)
        first = 0.75 if deployed and condition == "synthetic_weights" else 0.25
        (folder / "model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + struct.pack("<2f", first, -0.5)
        )
    overlay = {
        "action.gripper_convention": "0_open_1_closed",
        "action.units": "degree" if deployed and condition == "synthetic_units" else "rad",
        "embodiment.robot": "synthetic_two_axis_fixture",
    }
    if deployed and condition == "synthetic_missing_gripper":
        del overlay["action.gripper_convention"]
    return overlay


def published_inputs(folder, condition, role):
    source_path = ROOT / PUBLISHED[condition]
    manifest = read(source_path)
    origins = {source_path.relative_to(ROOT).as_posix(): digest(source_path)}
    count = 0
    for item in manifest["files"]:
        # Only already inspected development checkpoint originals; derived controls
        # and other candidate lineages are excluded from extraction.
        if not item.get("url") or item["path"].startswith("derived/"):
            continue
        name = Path(item["path"]).name
        if name not in (
            "config.json",
            "train_config.json",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        ) and not name.endswith(".safetensors"):
            continue
        source = source_path.parent / item["path"]
        require(source.resolve().is_relative_to(source_path.parent.resolve()), "contained publisher file")
        require(digest(source) == item["sha256"], f"published source identity: {source}")
        (folder / name).write_bytes(source.read_bytes())
        origins[source.relative_to(ROOT).as_posix()] = item["sha256"]
        count += 1
    require(count == 6, "complete bounded published metadata input")
    if condition == "published_act_statistics" and role == "deployed":
        path = folder / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        raw = bytearray(path.read_bytes())
        size = struct.unpack("<Q", raw[:8])[0]
        require(size <= 65536, "bounded statistics header")
        header = json.loads(raw[8 : 8 + size])
        tensor = header["action.mean"]
        require(tensor["dtype"] == "F32" and tensor["shape"] == [6], "explicit statistics intervention")
        offset = 8 + size + tensor["data_offsets"][0]
        original = struct.unpack_from("<f", raw, offset)[0]
        struct.pack_into("<f", raw, offset, original + 1)
        path.write_bytes(raw)
    return origins


def expectations(name):
    missing = [
        ("policy.weights_sha256", "fatal", "unknown"),
        ("action.control_hz", "fatal", "unknown"),
        ("action.gripper_convention", "fatal", "unknown"),
    ]
    fields = {
        "synthetic_complete": [],
        "synthetic_weights": [("policy.weights_sha256", "fatal", "mismatch")],
        "synthetic_statistics": [("action.normalization.stats_sha256", "fatal", "mismatch")],
        "synthetic_chunk": [("action.chunk_size", "fatal", "mismatch")],
        "synthetic_camera": [("observation.cameras", "fatal", "mismatch")],
        "synthetic_units": [("action.units", "warn", "mismatch")],
        "synthetic_missing_gripper": [("action.gripper_convention", "fatal", "unknown")],
        "synthetic_missing_weights": [("policy.weights_sha256", "fatal", "unknown")],
        "published_smolvla": missing,
        "published_act": missing,
        "published_act_statistics": [*missing, ("action.normalization.stats_sha256", "fatal", "mismatch")],
    }
    return {"passed": name in ("synthetic_complete", "synthetic_units"), "required_findings": fields[name]}


def validate_result(name, result, specs=None):
    expected = expectations(name)
    findings = result["findings"]
    require(result["native_passed"] == expected["passed"], f"native decision: {name}")
    require(
        result["native_passed"] == (not any(f["severity"] == "fatal" for f in findings)), "native aggregation"
    )
    require(result["allow_unknown"] is False and result["ignore"] == [], "stock fail-closed scope")
    for field, severity, relation in expected["required_findings"]:
        matches = [f for f in findings if f["field"] == field and f["severity"] == severity]
        require(len(matches) == 1, f"required native finding: {name}:{field}")
        finding = matches[0]
        if specs is not None:
            for role in ("evaluated", "deployed"):
                if field == "observation.cameras":
                    value = sorted(c["name"] for c in specs[role]["observation"]["cameras"])
                else:
                    value = specs[role]
                    for part in field.split("."):
                        value = value[part]
                require(finding[role] == value, f"native finding/manifest binding: {name}:{field}")
        if relation == "mismatch":
            require(finding["evaluated"] != finding["deployed"], f"grounded difference: {field}")
            require(finding["evaluated"] is not None and finding["deployed"] is not None, "known mismatch")
        else:
            require(finding["evaluated"] is None or finding["deployed"] is None, "missing evidence finding")


def execute(folder):
    require(
        all(importlib.util.find_spec(m) is None for m in ("torch", "lerobot", "openpi")),
        "isolated no-ML environment",
    )
    initial = project_sources()
    source_identity = identities()
    network_events = []

    def audit(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            network_events.append(event)
            raise RuntimeError("Robottruth applicability controls forbid network connections")

    sys.addaudithook(audit)
    started = time.perf_counter()
    from robotruth.contract import check_contract, extract_spec

    rows, origins = [], {}
    for name in CASES:
        overlays, specs = {}, {}
        for role in ("evaluated", "deployed"):
            inputs = folder / "inputs" / name / role
            inputs.mkdir(parents=True)
            overlay = synthetic_inputs(inputs, name, role) if name in SYNTHETIC else {}
            if name in PUBLISHED:
                origins.update(published_inputs(inputs, name, role))
            require(not (inputs / "robotruth.spec.json").exists(), "no hidden native overlay")
            overlays[role] = overlay
            specs[role] = extract_spec(inputs, family="lerobot", overlay=overlay, role=role)
        write(folder / "cases" / f"{name}.overlays.json", overlays)
        for role, spec in specs.items():
            write(folder / "cases" / f"{name}.{role}.json", spec.model_dump(mode="json"))
        result = check_contract(specs["evaluated"], specs["deployed"], allow_unknown=False)
        recorded = {
            "native_passed": result.passed,
            "native_summary": result.summary(),
            "findings": dataclasses.asdict(result)["findings"],
            "allow_unknown": False,
            "ignore": [],
            "classification": "published metadata, missing certification evidence"
            if name in PUBLISHED
            else "explicit synthetic workflow control",
        }
        validate_result(name, recorded)
        write(folder / "cases" / f"{name}.result.json", recorded)
        rows.append({"case": name, "native_passed": result.passed, "summary": result.summary()})
    require(source_identity == identities() and initial == project_sources(), "source stability")
    require(all(digest(ROOT / name) == sha for name, sha in origins.items()), "published input stability")
    require(not network_events, "network attempts")
    require(not any(m in sys.modules for m in ("torch", "lerobot", "openpi")), "no ML imports")
    summary = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "source_file_sha256": initial,
        "upstream_identity": source_identity,
        "original_input_sha256": origins,
        "file_sha256": {
            p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()
        },
        "cases": rows,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "wall_seconds": time.perf_counter() - started,
            "network_events": network_events,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
        "scope": (
            "Complete installed base package, stock extraction and checking only; "
            "no trained inference, temporal execution or superiority result"
        ),
        "interpretation": (
            "Native FAIL on unknown fatal fields withholds certification; "
            "native PASS with warnings is not full physical conformance"
        ),
    }
    write(folder / "summary.json", summary)
    return verify(folder)


def verify(folder):
    summary = read(folder / "summary.json")
    require(summary["development_only"] is True and summary["final_evaluation"] is False, "development scope")
    require(summary["source_file_sha256"] == project_sources(), "project source identity")
    ledger = read(PROFILE / "sources.json")
    identity = summary["upstream_identity"]
    require(
        identity["direct_url"]["vcs_info"]["commit_id"] == ledger["revision"], "recorded upstream revision"
    )
    require(
        {k: v["git_blob"] for k, v in identity["python_sources"].items()}
        == ledger["package_python_git_blobs"],
        "complete recorded upstream source set",
    )
    for name, sha in summary["file_sha256"].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()), "contained evidence path")
        require(digest(path) == sha, f"saved file identity: {name}")
    for name, sha in summary["original_input_sha256"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "contained original development input")
        require(digest(path) == sha, f"original development input: {name}")
    require([row["case"] for row in summary["cases"]] == list(CASES), "case inventory")
    for row in summary["cases"]:
        name = row["case"]
        result = read(folder / "cases" / f"{name}.result.json")
        specs = {role: read(folder / "cases" / f"{name}.{role}.json") for role in ("evaluated", "deployed")}
        validate_result(name, result, specs)
        require(
            row["native_passed"] == result["native_passed"] and row["summary"] == result["native_summary"],
            "summary outcome binding",
        )
        if name in PUBLISHED:
            for role in ("evaluated", "deployed"):
                spec = read(folder / "cases" / f"{name}.{role}.json")
                require(spec["policy"]["weights_sha256"] is None, "no invented published weights")
                require(
                    spec["action"]["gripper_convention"] is None, "no invented published gripper authority"
                )
                require(
                    read(folder / "cases" / f"{name}.overlays.json")[role] == {},
                    "no invented published overlays",
                )
    require(summary["runtime"]["network_events"] == [], "recorded network attempts")
    require(not any(m in sys.modules for m in ("torch", "lerobot", "openpi")), "no ML imports")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--output", type=Path)
    source.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
        require("robotruth" not in sys.modules, "verification must not import native baseline")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        result = execute(args.output.resolve())
    print(f"Verified {len(result['cases'])} robotruth development extraction/checking controls")


if __name__ == "__main__":
    main()
