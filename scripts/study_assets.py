"""Compile exact metadata artifacts for matched workflows; never execute a policy.

Prospective compilation is metadata inspection only. Its separate native gate
must verify a committed complete protocol before any selected-cell execution.
"""

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

from action_designs import generate_all
from action_study_common import ROOT, fingerprint, interventions, sha, write
from normalization_profiles import require, validate

from mltools.obligations import Axis, compile_suite

DEVELOPMENT_MANIFEST = "examples/probes/act/source_manifest.json"
DEVELOPMENT_MANIFEST_SHA256 = "fff39c104a97239efb7e9529faada09d59d7f1977d379ffbf0433956650f5256"
BASE_FILES = {"config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
MAX_BYTES = 128 * 1024


def read(path):
    return json.loads(Path(path).read_bytes())


def action_statistics(raw, n, fields):
    require(10 <= len(raw) <= MAX_BYTES, "bounded statistics snapshot")
    size = struct.unpack("<Q", raw[:8])[0]
    require(2 <= size <= len(raw) - 8, "bounded snapshot header")
    header, payload = json.loads(raw[8 : 8 + size]), raw[8 + size :]
    stats = {}
    for field in fields:
        item = header["action." + field]
        require(item["dtype"] in ("F32", "F64") and item["shape"] == [n], "declared numeric action vector")
        code, width = ("f", 4) if item["dtype"] == "F32" else ("d", 8)
        start, end = item["data_offsets"]
        require(
            type(start) is type(end) is int
            and 0 <= start <= end <= len(payload)
            and end - start == n * width,
            "snapshot action tensor span",
        )
        stats[field] = list(struct.unpack("<" + code * n, payload[start:end]))
    return stats


def edited_safetensors(raw, spec):
    """Replace selected action arrays by exact F64 data; copy all other values.

    This is a declared binary artifact intervention. Headers/offsets are rebuilt
    and may differ even for unchanged arrays; controls retain original bytes.
    """
    n = validate(spec)
    require(10 <= len(raw) <= MAX_BYTES, "bounded original statistics bytes")
    size = struct.unpack("<Q", raw[:8])[0]
    require(2 <= size <= len(raw) - 8, "statistics header span")
    header = json.loads(raw[8 : 8 + size])
    payload = raw[8 + size :]
    result, data, changed = {}, bytearray(), set()
    for key, value in header.items():
        if key == "__metadata__":
            result[key] = value
            continue
        require(isinstance(value, dict) and value["dtype"] in ("F32", "F64"), "reviewed numeric statistics")
        shape = value["shape"]
        require(
            isinstance(shape, list) and shape and all(type(x) is int and 1 <= x <= 4096 for x in shape),
            "bounded tensor shape",
        )
        count = math.prod(shape)
        start, end = value["data_offsets"]
        width = 4 if value["dtype"] == "F32" else 8
        require(
            count <= 4096
            and type(start) is type(end) is int
            and 0 <= start <= end <= len(payload)
            and end - start == count * width,
            "tensor data span",
        )
        field = key.removeprefix("action.") if key.startswith("action.") else None
        if field in spec["statistics"]:
            require(shape == [n], "declared action dimensions")
            chunk = struct.pack("<" + "d" * n, *spec["statistics"][field])
            dtype = "F64"
            changed.add(field)
        else:
            chunk, dtype = payload[start:end], value["dtype"]
        result[key] = {"dtype": dtype, "shape": shape, "data_offsets": [len(data), len(data) + len(chunk)]}
        data.extend(chunk)
    require(changed == set(spec["statistics"]), "all intended action statistics exist in native artifact")
    encoded = json.dumps(result, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    output = struct.pack("<Q", len(encoded)) + encoded + data
    require(len(output) <= MAX_BYTES, "bounded derivative statistics")
    return output


def metadata(manifest_path, expected_sha):
    path = (ROOT / manifest_path).resolve()
    require(path.is_relative_to(ROOT), "contained source manifest")
    raw_manifest = path.read_bytes()
    require(hashlib.sha256(raw_manifest).hexdigest() == expected_sha, "pinned role-specific source manifest")
    manifest = json.loads(raw_manifest)
    files, origins = {}, {manifest_path: expected_sha}
    for item in manifest["files"]:
        name = item["path"]
        if name not in BASE_FILES and not (name.startswith("policy_") and name.endswith(".safetensors")):
            continue
        require(Path(name).name == name and item.get("url"), "fixed original metadata allowlist")
        source = path.parent / name
        raw = source.read_bytes()
        require(
            len(raw) <= MAX_BYTES and hashlib.sha256(raw).hexdigest() == item["sha256"],
            "pinned publisher bytes",
        )
        files[name] = raw
        origins[source.relative_to(ROOT).as_posix()] = item["sha256"]
    require(BASE_FILES <= set(files) and len(files) == 6, "complete saved metadata pipeline")
    license_path = path.parent / "LICENSE-2.0.txt"
    require(
        license_path.read_bytes() == (ROOT / "examples/published_so101/LICENSE-2.0.txt").read_bytes(),
        "included publisher-declared license text",
    )
    files["LICENSE-2.0.txt"] = license_path.read_bytes()
    origins[license_path.relative_to(ROOT).as_posix()] = sha(license_path)
    config, post, pre = (
        json.loads(files[n]) for n in ("config.json", "policy_postprocessor.json", "policy_preprocessor.json")
    )
    steps = [s for s in post["steps"] if s["registry_name"] == "unnormalizer_processor"]
    psteps = [s for s in pre["steps"] if s["registry_name"] == "normalizer_processor"]
    require(len(steps) == len(psteps) == 1, "single explicit normalization step on each side")
    step, pstep = steps[0], psteps[0]
    mode = step["config"]["norm_map"]["ACTION"]
    require(
        mode in ("MEAN_STD", "QUANTILES")
        and pstep["config"]["norm_map"]["ACTION"] == mode
        and step["config"]["eps"] == pstep["config"]["eps"] == 1e-8,
        "saved source normalization mode and epsilon",
    )
    n = config["output_features"]["action"]["shape"][0]
    require(
        step["config"]["features"]["action"]["shape"] == [n] and 1 <= n <= 32,
        "saved declared action dimensions",
    )
    field_names = ("mean", "std") if mode == "MEAN_STD" else ("q01", "q99")
    stats = action_statistics(files[step["state_file"]], n, field_names)
    spec = {
        "schema_version": 1,
        "profile": "lerobot_mean_std_f64_v1" if mode == "MEAN_STD" else "lerobot_quantiles_f64_v1",
        "statistics": stats,
    }
    validate(spec)
    return {
        "manifest": manifest,
        "files": files,
        "origins": origins,
        "intended": spec,
        "post_state_file": step["state_file"],
        "pre_state_file": pstep["state_file"],
        "dimensions": n,
        "policy_type": config["type"],
    }


def selections(role):
    if role == "development":
        require(
            sha(ROOT / DEVELOPMENT_MANIFEST) == DEVELOPMENT_MANIFEST_SHA256, "pinned development assignment"
        )
        return [
            {
                "id": "act_move_cube_lerobot",
                "artifact_manifest": DEVELOPMENT_MANIFEST,
                "artifact_manifest_sha256": DEVELOPMENT_MANIFEST_SHA256,
                "family": "ACT",
                "framework": "lerobot",
                "cluster": "yutofujita-so101-move-cube",
            }
        ]
    require(role == "prospective", "metadata selection role")
    design = read(ROOT / "research/publication/study-design.json")
    stratum = next(s for s in design["strata"] if s["id"] == "prospective_metadata")
    artifacts = {a["id"]: a for a in stratum["artifacts"]}
    result = []
    for entry in stratum["execution_cells"]:
        artifact = artifacts[entry["artifact"]]
        require(
            sha(ROOT / artifact["artifact_manifest"]) == artifact["artifact_manifest_sha256"],
            "selected prospective artifact identity, without native execution",
        )
        result.append(
            {
                "id": entry["id"],
                "artifact_manifest": artifact["artifact_manifest"],
                "artifact_manifest_sha256": artifact["artifact_manifest_sha256"],
                "family": artifact["family"],
                "framework": entry["framework"],
                "cluster": artifact["declared_dataset_cluster"],
            }
        )
    return result


def blueprint(role):
    files, cells, origins = {}, {}, {}
    for selected in selections(role):
        source = metadata(selected["artifact_manifest"], selected["artifact_manifest_sha256"])
        if role == "development":
            require(
                source["manifest"]["model"] == "yutofujita/act-so101-move-cube"
                and source["manifest"]["revision"] == "81ecbfed016e9ec41a2c17d78cfc20f7e08f47d5",
                "fixed already-executed development checkpoint revision",
            )
        require(
            source["policy_type"] == {"ACT": "act", "SmolVLA": "smolvla", "PI": "pi05"}[selected["family"]],
            "selected family matches pinned publisher configuration",
        )
        origins.update(source["origins"])
        intended = source["intended"]
        if selected["framework"] == "openpi":
            intended = {**intended, "profile": "openpi_quantiles_f64_v1"}
        conditions = interventions(intended)
        if role == "development":
            # Identity-sensitivity control: same numeric values with a different
            # valid serialization. It is not a selected main-study condition.
            conditions["equivalent_f64_serialization"] = {
                **conditions["numeric_control"],
                "annotation": "same numerical statistics; explicit F64 reserialization",
            }
        for condition, entry in conditions.items():
            prefix = f"cells/{selected['id']}/{condition}"
            packet = dict(source["files"])
            if condition != "numeric_control":
                name = source["post_state_file"]
                packet[name] = edited_safetensors(packet[name], entry["specification"])
            runtime = {
                "schema_version": 1,
                "target_framework": selected["framework"],
                "source_pipeline_family": "lerobot",
                "dtype": "float64",
                "device": "cpu",
                "operator_profile": intended["profile"],
                "scope": "Current-runtime numeric experiment; no publisher training/inference replication",
            }
            packet["runtime_profile.json"] = (json.dumps(runtime, indent=2) + "\n").encode()
            for name, raw in packet.items():
                files[prefix + "/" + name] = raw
            entry["directory"] = prefix
            entry["post_state_file"] = source["post_state_file"]
            entry["file_sha256"] = {name: hashlib.sha256(raw).hexdigest() for name, raw in packet.items()}
            entry["serialization"] = (
                "pristine source bytes"
                if condition == "numeric_control"
                else ("postprocessor action statistics rewritten as F64; all other tensor values retained")
            )
        n = source["dimensions"]
        s = compile_suite(
            tuple(Axis("action", str(i), "normalized_output_index_" + str(i), -0.8, 0.8) for i in range(n)),
            tuple("raw_action_" + str(i) for i in range(n)),
        )
        cells[selected["id"]] = {
            **selected,
            "model": source["manifest"]["model"],
            "revision": source["manifest"]["revision"],
            "intended": intended,
            "conditions": conditions,
            "designs": {k: d.to_dict() for k, d in generate_all(s).items()},
            "authority": "Pinned independently decoded publisher statistics; current-runtime profile",
            "checkpoint_physical_authority_status": "UNKNOWN",
            "source_pipeline_family": "lerobot",
            "post_state_file": source["post_state_file"],
            "pre_state_file": source["pre_state_file"],
        }
    plan = {
        "schema_version": 1,
        "format": "mltools-matched-statistics-artifacts",
        "role": role,
        "native_execution_performed_by_compilation": False,
        "compilation_only": True,
        "cells": cells,
        "original_input_sha256": origins,
        "builder_source_sha256": sha(Path(__file__)),
        "file_sha256": {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())},
        "limits": [
            "Metadata compilation is not execution evidence or a frozen final protocol",
            "No weights, episodes, physical calibration, controller mapping or task-success result",
            "OpenPI cells transport LeRobot metadata under a separate decoder; no OpenPI checkpoint claim",
        ],
    }
    return plan, files


def verify(folder):
    plan = read(folder / "plan.json")
    expected, files = blueprint(plan["role"])
    require(fingerprint(plan) == fingerprint(expected), "exact independently regenerated asset plan")
    require(
        {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} == {"plan.json", *files},
        "closed asset inventory",
    )
    for name, raw in files.items():
        require((folder / name).read_bytes() == raw, "exact matched artifact bytes: " + name)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--role", choices=("development", "prospective"), default="development")
    args = parser.parse_args()
    if args.verify:
        plan = verify(args.verify.resolve())
    else:
        folder = args.output.resolve()
        require(folder.is_relative_to(ROOT), "workspace-contained output")
        folder.mkdir(parents=True, exist_ok=False)
        plan, files = blueprint(args.role)
        for name, raw in files.items():
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        write(folder / "plan.json", plan)
        verify(folder)
    require(not any(m in sys.modules for m in ("torch", "lerobot", "openpi")), "metadata-only builder CLI")
    print(f"Verified {len(plan['cells'])} exact matched metadata cells; compilation only, no native outcomes")


if __name__ == "__main__":
    main()
