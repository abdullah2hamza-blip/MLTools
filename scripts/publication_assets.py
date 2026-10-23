"""Exact development wrapper inputs, including an already known OpenPI fixture.

The selected prospective input compiler is unchanged. The authored two-axis
fixture exists only to test the full new multi-framework acquisition wrapper.
"""

import argparse
import hashlib
import json
import struct

from action_designs import generate_all
from action_study_common import ROOT, fingerprint, interventions, sha, write
from normalization_profiles import require
from run_action_study import suite
from study_assets import blueprint as original_blueprint
from study_assets import edited_safetensors, read
from study_assets import verify as original_verify

DEVELOPMENT = "validation/publication-assets-development-001"
AUTHORED = "research/publication/development/openpi-wrapper-fixture.json"


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def blueprint():
    plan, files = original_blueprint("development")
    source = read(ROOT / AUTHORED)
    spec = source["intended"]
    header, values = {}, bytearray()
    for field, data in spec["statistics"].items():
        start = len(values)
        values.extend(struct.pack("<" + "d" * len(data), *data))
        header["action." + field] = {
            "dtype": "F64",
            "shape": [len(data)],
            "data_offsets": [start, len(values)],
        }
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    raw = struct.pack("<Q", len(raw_header)) + raw_header + values
    pre_file = "policy_preprocessor_step_0_normalizer_processor.safetensors"
    post_file = "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    shared = {
        "config.json": encoded(
            {
                "type": "authored_openpi_transport_fixture",
                "output_features": {"action": {"type": "ACTION", "shape": [2]}},
                "normalization_mapping": {"ACTION": "QUANTILES"},
            }
        ),
        "train_config.json": encoded(
            {"scope": "Authored numeric fixture; no training/checkpoint/robot claim"}
        ),
        "policy_preprocessor.json": encoded(
            {"steps": [{"registry_name": "normalizer_processor", "state_file": pre_file}]}
        ),
        "policy_postprocessor.json": encoded(
            {"steps": [{"registry_name": "unnormalizer_processor", "state_file": post_file}]}
        ),
        pre_file: bytes(raw),
        "AUTHORITY.json": encoded(source),
    }
    cell_id, conditions = "authored_quantile_openpi", {}
    for condition, entry in interventions(spec).items():
        directory = f"cells/{cell_id}/{condition}"
        leaf = {
            **shared,
            post_file: bytes(raw)
            if condition == "numeric_control"
            else edited_safetensors(raw, entry["specification"]),
            "runtime_profile.json": encoded(
                {
                    "framework": "openpi",
                    "dtype": "float64",
                    "device": "cpu",
                    "specification": entry["specification"],
                }
            ),
        }
        for name, data in leaf.items():
            files[directory + "/" + name] = data
        conditions[condition] = {
            **entry,
            "directory": directory,
            "post_state_file": post_file,
            "file_sha256": {n: hashlib.sha256(data).hexdigest() for n, data in leaf.items()},
        }
    cell = {
        "id": cell_id,
        "artifact_manifest": AUTHORED,
        "artifact_manifest_sha256": sha(ROOT / AUTHORED),
        "family": "AUTHORED_NUMERIC_FIXTURE",
        "framework": "openpi",
        "cluster": "authored-known-two-axis-openpi-development",
        "model": None,
        "revision": None,
        "intended": spec,
        "conditions": conditions,
        "authority": source["authority"],
        "checkpoint_physical_authority_status": "UNKNOWN",
        "source_pipeline_family": "authored_lerobot_style_container",
        "post_state_file": post_file,
        "pre_state_file": pre_file,
    }
    cell["designs"] = {name: d.to_dict() for name, d in generate_all(suite(cell)).items()}
    plan["cells"][cell_id] = cell
    plan["original_input_sha256"][AUTHORED] = sha(ROOT / AUTHORED)
    plan["base_builder_source_sha256"] = plan["builder_source_sha256"]
    plan["builder_source_sha256"] = sha(ROOT / "scripts/publication_assets.py")
    plan["file_sha256"] = {name: hashlib.sha256(raw).hexdigest() for name, raw in sorted(files.items())}
    plan["limits"].append(
        "Additional authored known OpenPI fixture is development-only, not a policy-family/artifact sample"
    )
    return plan, files


def verify(folder, role):
    if role == "prospective":
        return original_verify(folder)
    require(role == "development", "explicit wrapper input role")
    plan, files = blueprint()
    require(
        fingerprint(read(folder / "plan.json")) == fingerprint(plan), "exact authored/development input plan"
    )
    require(
        {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} == {"plan.json", *files},
        "closed wrapper development inventory",
    )
    require(
        all((folder / name).read_bytes() == raw for name, raw in files.items()), "exact wrapper input bytes"
    )
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile-development", action="store_true", required=True)
    parser.parse_args()
    folder = ROOT / DEVELOPMENT
    folder.mkdir(parents=True, exist_ok=False)
    plan, files = blueprint()
    for name, raw in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    write(folder / "plan.json", plan)
    verify(folder, "development")
    print("Compiled exact reused ACT and authored known OpenPI development inputs; no native execution")


if __name__ == "__main__":
    main()
