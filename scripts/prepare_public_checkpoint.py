"""Prepare metadata-only real-checkpoint evidence; downloads require --download.

Only fixed, immutable, size-limited source artifacts are accepted. Never downloads weights,
loads a saved arbitrary processor registry, or initializes a tokenizer/model.
"""

import argparse
import hashlib
from copy import deepcopy
from pathlib import Path
from urllib.request import urlopen

from mltools.io import MAX_BYTES, InputError, digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1] / "examples" / "published_so101"
MODEL = "yutofujita/smolvla-so101-move-cube"
REVISION = "0220b37dbcf1e0b2f6f5e21c64c75bf16fac6b69"
STATS_SHA256 = "77b14eb2066885f13079760eae89edf7a18c3a4b1a3ad2abc3e0e0f2dd579232"
SOURCE_HASHES = {
    "config.json": "11b22f5b1e48951b0c7d20000ab5364916e72322a0c216c57c5a1956607e1a77",
    "train_config.json": "f7815d405e674b4b1ba8d94fecbcd0f222b385817d5ecb34be451b4c56409a1a",
    "policy_preprocessor.json": "1cac4c93d9d9fc2bbd8f367d3205ffd20e79fc21fa8176331453b966b2da897b",
    "policy_postprocessor.json": "2b78bb742065288df2ec63b0ee35f97a1f6950171cf2272f7e34b1fe3873b17b",
    "policy_preprocessor_step_5_normalizer_processor.safetensors": STATS_SHA256,
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": STATS_SHA256,
}


def prepare(download=False):
    from safetensors.numpy import load_file

    (ROOT / "source").mkdir(parents=True, exist_ok=True)
    manifest = []
    for filename, expected in SOURCE_HASHES.items():
        path = ROOT / "source" / filename
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{filename}"
        if not path.exists():
            if not download:
                raise InputError("Pinned sources missing; explicitly run with --download for initial setup")
            with urlopen(url, timeout=30) as response:
                raw = response.read(MAX_BYTES + 1)
            if len(raw) > MAX_BYTES or hashlib.sha256(raw).hexdigest() != expected:
                raise InputError(f"Source size/digest mismatch: {filename}")
            path.write_bytes(raw)
        if digest(path) != expected:
            raise InputError(f"Local source differs from pinned public artifact: {filename}")
        manifest.append({"id": filename, "path": f"source/{filename}", "sha256": expected, "url": url})
    config = read_json(ROOT / "source" / "config.json")
    train = read_json(ROOT / "source" / "train_config.json")
    pre = read_json(ROOT / "source" / "policy_preprocessor.json")["steps"]
    post = read_json(ROOT / "source" / "policy_postprocessor.json")["steps"]
    if [s["registry_name"] for s in post] != ["unnormalizer_processor", "device_processor"]:
        raise InputError("Unsupported published postprocessor; review before extending supported extraction")
    rename = next(s for s in pre if s["registry_name"] == "rename_observations_processor")["config"][
        "rename_map"
    ]
    normal = next(s for s in pre if s["registry_name"] == "normalizer_processor")
    inverse = post[0]
    if (
        normal["config"]["norm_map"]["ACTION"] != "MEAN_STD"
        or inverse["config"]["norm_map"]["ACTION"] != "MEAN_STD"
    ):
        raise InputError("Only MEAN_STD is supported")
    if train["rename_map"] != rename or normal["config"]["eps"] != 1e-8 or inverse["config"]["eps"] != 1e-8:
        raise InputError("Training mapping or epsilon requires explicit compatibility review")
    if any("relative" in s["registry_name"] for s in pre + post):
        raise InputError("This metadata-only extraction expects no relative-action processor")
    (ROOT / "derived").mkdir(exist_ok=True)
    for name, step in (("training_stats", normal), ("deployment_stats", inverse)):
        filename = step["state_file"]
        if filename not in SOURCE_HASHES:
            raise InputError("Unexpected safetensors state file")
        tensors = load_file(ROOT / "source" / filename)
        stats = {key: tensors[f"action.{key}"].tolist() for key in ("mean", "std")}
        write_json(ROOT / "derived" / f"{name}.json", stats)
    stats = read_json(ROOT / "derived" / "training_stats.json")
    size = config["output_features"]["action"]["shape"][0]
    horizon = config["chunk_size"]
    dimensions = [f"dimension_{i}" for i in range(size)]
    derived_config = {
        "state_order": dimensions,
        "action_names": dimensions,
        "relative": False,
        "exclude_joints": [],
        "camera_map": rename,
        "horizon": horizon,
        "use_degrees": None,
        "source_model": MODEL,
        "source_revision": REVISION,
        "scope": "Numeric saved-processor path; physical modes/units/names are not established",
    }
    write_json(ROOT / "derived" / "checkpoint_config.json", derived_config)
    stale = deepcopy(stats)
    stale["mean"][0] += 10
    write_json(ROOT / "derived" / "wrong_deployment_stats.json", stale)
    for name in ("checkpoint_config", "training_stats", "deployment_stats", "wrong_deployment_stats"):
        manifest.append(
            {"id": name, "path": f"derived/{name}.json", "sha256": digest(ROOT / "derived" / f"{name}.json")}
        )
    source = f"https://huggingface.co/{MODEL}/tree/{REVISION}"
    normalized = [[(i % 3) * 0.25] * size for i in range(horizon)]
    raw_targets = [[stats["mean"][j] + stats["std"][j] * x for j, x in enumerate(row)] for row in normalized]
    scenario = [{"kind": "reset"}]
    for i in range(horizon):
        event = {
            "kind": "tick",
            "state": dict.fromkeys(dimensions, 0),
            "cameras": {
                "observation.images.front": [[11, 12], [13, 14]],
                "observation.images.wrist": [[21, 22], [23, 24]],
            },
        }
        if i == 0:
            event.update(prediction=normalized, training_targets=raw_targets)
        scenario.append(event)
    scenario.append({"kind": "reset"})
    bundle = {
        "schema_version": 1,
        "contract": {
            "embodiment": "published-so101/physical-interface-unverified",
            "authority": {
                "source": source,
                "description": "Immutable published processor metadata; positional dimensions only. "
                "Camera labels are publisher labels; physical identity remains unverified.",
            },
            "checks": [
                "artifacts",
                "cameras",
                "normalization",
                "actions",
                "joint_order",
                "camera3_role",
                "model_preprocessing",
                "training_dataset_revision",
                "checkpoint_weights",
                "state_normalization",
            ],
            "joints": [{"name": n, "unit": None, "mode": None} for n in dimensions],
            "cameras": [
                {"source": k, "target": v, "role": k.removeprefix("observation.images.")}
                for k, v in rename.items()
            ],
            "normalization": {"kind": "mean_std", "artifact": "training_stats"},
            "training_action_transform": "identity",
            "horizon": horizon,
        },
        "adapter": {
            "state_order": dimensions,
            "action_names": dimensions,
            "motor_names": dimensions,
            "camera_map": rename,
            "relative": False,
            "exclude_joints": [],
            "bind_queue": True,
            "reset": True,
            "stats_artifact": "deployment_stats",
            "units": None,
            "emit_commands": False,
        },
        "artifacts": [{k: a[k] for k in ("id", "path", "sha256")} for a in manifest],
        "scenario": scenario,
    }
    write_json(ROOT / "metadata.json", bundle)
    wrong = deepcopy(bundle)
    wrong["adapter"]["stats_artifact"] = "wrong_deployment_stats"
    write_json(ROOT / "wrong_stats.json", wrong)
    write_json(
        ROOT / "source_manifest.json",
        {
            "schema_version": 1,
            "model": MODEL,
            "revision": REVISION,
            "files": manifest,
            "license": "Apache-2.0 per pinned model card",
            "model_card": f"https://huggingface.co/{MODEL}/blob/{REVISION}/README.md",
            "weights_downloaded": False,
            "limits": [
                "No authoritative physical joint names/units/action modes in downloaded checkpoint metadata",
                "Training dataset revision is null; current dataset metadata is not its proven snapshot",
                "camera3 physical role, 32-dimensional model padding and image preprocessing are unverified",
            ],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true", help="Allow initial pinned metadata downloads only"
    )
    prepare(parser.parse_args().download)
