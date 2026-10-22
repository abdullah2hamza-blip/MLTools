"""Fetch a fixed small metadata allowlist; no weights or dataset episodes."""

import argparse
import hashlib
import json
import re
import struct
import urllib.error
import urllib.request
from pathlib import Path

from mltools.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_3_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)
CANDIDATES = (
    (
        "act-so101-move-cube",
        "81ecbfed016e9ec41a2c17d78cfc20f7e08f47d5",
        "9155f3cb55dc35771a721dfd14d63e0ed7e785519e36b2c4794a72f556ac02dc",
        "so101-move-cube",
        "1b983c5a3e27385c595ebd298ab1b90c43f2c1b2",
    ),
    (
        "act-so101-boccia-center-only",
        "16e80dfa6c37f53f319a3d2c98eca12bb5006fbb",
        "fd85df21d39dd71bf79103d118cd668b98077167d586630d7d8e55c4df1aecae",
        "so101-boccia-center-only",
        "9a6bfb650042e2c486ea3335d50cebd02c071e78",
    ),
)
LIMIT = 128 * 1024


def fetch(url, path, download):
    if not path.exists():
        if not download:
            raise ValueError(f"Missing {path.name}; use --download for the fixed public metadata allowlist")
        with urllib.request.urlopen(url, timeout=45) as response:
            raw = response.read(LIMIT + 1)
        if len(raw) > LIMIT:
            raise ValueError("Metadata download exceeded the fixed size bound")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = path.read_bytes()
    if len(raw) > LIMIT:
        raise ValueError("Metadata file exceeds size bound")
    return {"path": path.name, "url": url, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def tensors(path):
    """Decode only bounded F32/F64 arrays from the fixed tiny statistics files."""
    raw = path.read_bytes()
    if not 8 <= len(raw) <= LIMIT:
        raise ValueError("Unsupported statistics file size")
    header_size = struct.unpack("<Q", raw[:8])[0]
    if not 2 <= header_size <= len(raw) - 8:
        raise ValueError("Invalid safetensors header size")
    header = json.loads(raw[8 : 8 + header_size])
    data = raw[8 + header_size :]
    result = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if tensor["dtype"] not in ("F32", "F64"):
            raise ValueError("Only reviewed floating point statistics are supported")
        shape = tensor["shape"]
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(n) is not int or not 1 <= n <= 4096 for n in shape)
        ):
            raise ValueError("Invalid tensor shape")
        count = 1
        for n in shape:
            count *= n
        if count > 4096:
            raise ValueError("Statistics tensor exceeds element bound")
        code, width = ("f", 4) if tensor["dtype"] == "F32" else ("d", 8)
        start, end = tensor["data_offsets"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(data):
            raise ValueError("Invalid tensor offsets")
        if end - start != count * width:
            raise ValueError("Tensor shape/byte span mismatch")
        result[name] = {
            "shape": shape,
            "dtype": tensor["dtype"],
            "values": list(struct.unpack("<" + code * count, data[start:end])),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    base = ROOT / "research/publication/selection/act"
    rows = []
    for model, revision, expected_stats, dataset, dataset_revision in CANDIDATES:
        folder = base / model
        records = []
        for file in FILES:
            url = f"https://huggingface.co/yutofujita/{model}/resolve/{revision}/{file}"
            item = fetch(url, folder / file, args.download)
            if file.endswith(".safetensors") and item["sha256"] != expected_stats:
                raise ValueError("Statistics disagree with the independently inspected pinned artifact")
            records.append(item)
        card = (folder / "README.md").read_text(encoding="utf-8")
        frontmatter = card.split("---", 2)[1] if card.startswith("---") else ""
        if not re.search(r"(?m)^license:\s*apache-2\.0\s*$", frontmatter):
            raise ValueError("Pinned publisher card does not declare the reviewed license")
        config, training = read_json(folder / "config.json"), read_json(folder / "train_config.json")
        if config["type"] != "act":
            raise ValueError("Candidate is not the independently selected policy family")
        dataset_url = (
            f"https://huggingface.co/datasets/yutofujita/{dataset}/resolve/{dataset_revision}/meta/info.json"
        )
        records.append(fetch(dataset_url, folder / "current_dataset_info.json", args.download))
        info = read_json(folder / "current_dataset_info.json")
        card_url = (
            f"https://huggingface.co/datasets/yutofujita/{dataset}/resolve/{dataset_revision}/README.md"
        )
        dataset_card_path = folder / "current_dataset_README.md"
        status_path = folder / "current_dataset_card_status.json"
        try:
            if not args.download and status_path.exists() and not dataset_card_path.exists():
                dataset_card = ""
            else:
                records.append(fetch(card_url, dataset_card_path, args.download))
                dataset_card = dataset_card_path.read_text(encoding="utf-8")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
            dataset_card = ""
            write_json(
                status_path,
                {
                    "url": card_url,
                    "http_status": 404,
                    "scope": "No dataset card at this pinned revision; model license not inherited",
                },
            )
        if not dataset_card:
            records.append(
                {
                    "path": status_path.name,
                    "kind": "retrieval_observation",
                    "sha256": hashlib.sha256(status_path.read_bytes()).hexdigest(),
                }
            )
        dataset_header = dataset_card.split("---", 2)[1] if dataset_card.startswith("---") else ""
        dataset_license = re.search(r"(?m)^license:\s*(\S+)\s*$", dataset_header)
        decoded = tensors(folder / "policy_preprocessor_step_3_normalizer_processor.safetensors")
        write_json(
            folder / "decoded_statistics.json",
            {
                "schema_version": 1,
                "source_sha256": expected_stats,
                "decoding": "bounded little-endian F32/F64; no framework deserialization",
                "tensors": decoded,
            },
        )
        records.append(
            {
                "path": "decoded_statistics.json",
                "derivation": "numeric decoding of pinned statistics",
                "sha256": hashlib.sha256((folder / "decoded_statistics.json").read_bytes()).hexdigest(),
            }
        )
        (folder / "LICENSE-2.0.txt").write_bytes(
            (ROOT / "examples/published_so101/LICENSE-2.0.txt").read_bytes()
        )
        row = {
            "schema_version": 1,
            "candidate_only": True,
            "final_evaluation": False,
            "model": f"yutofujita/{model}",
            "revision": revision,
            "policy_family": "ACT",
            "license": {"declared": "Apache-2.0", "authority": "pinned publisher model-card frontmatter"},
            "files": records,
            "declared_dataset": training["dataset"],
            "current_dataset_metadata": {
                "revision": dataset_revision,
                "robot_type": info.get("robot_type"),
                "training_snapshot_attested": False,
                "license": dataset_license.group(1) if dataset_license else None,
                "license_authority": "pinned current dataset card; separate from model-card licensing",
            },
            "selection_notes": [
                "Metadata inspected before any case outcome evaluation",
                "Dataset revision is null; current schema is corroboration, not training snapshot authority",
                "Physical units, action modes, calibration and training source revision remain unverified",
                "Shared move-cube lineage with existing development SmolVLA must remain in one split cluster",
            ],
            "weights_downloaded": False,
        }
        write_json(folder / "manifest.json", row)
        rows.append(
            {
                "model": row["model"],
                "revision": revision,
                "policy_family": "ACT",
                "manifest": f"{model}/manifest.json",
                "data_lineage": f"yutofujita/{dataset}",
            }
        )
        print(f"Verified metadata only: {row['model']}@{revision}", flush=True)
    write_json(
        base / "index.json",
        {
            "schema_version": 1,
            "status": "SELECTION_ONLY_NOT_FROZEN",
            "candidates": rows,
            "selection_event": "Metadata retrieval only; experiment outcomes recorded separately",
        },
    )


if __name__ == "__main__":
    main()
