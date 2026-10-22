"""Retrieve/verify a fixed, bounded SmolVLA metadata catalog; never execute a policy."""

import argparse
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

from prepare_act_candidates import tensors

from mltools.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "research/publication/selection/smolvla"
LIMIT = 128 * 1024
FILES = {
    "README.md",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
}
MODELS = {
    "edge-inference/smolvla-so101-pick-orange": "71cf4a9d35ce317f6706efe1a9f9d4cbb2b8fb4d",
    "shenjianliang/smolvla_policy": "5753a3bff43a5014b9320476458ae9e4c5aae02d",
}


def require(condition, message):
    if not condition:
        raise ValueError(f"SmolVLA selection verification: {message}")


def artifact(folder, model, revision, entry, download):
    name = entry["path"]
    require(name in FILES and 0 < entry["size"] <= LIMIT, "artifact allowlist/size")
    path = folder / name
    url = f"https://huggingface.co/{model}/resolve/{revision}/{name}"
    if path.exists():
        raw = path.read_bytes()
    else:
        require(download, f"{model}/{name} missing; explicit --download required")
        with urllib.request.urlopen(url, timeout=45) as response:
            raw = response.read(LIMIT + 1)
    require(len(raw) == entry["size"] <= LIMIT, f"{name} exact pinned size")
    digest = hashlib.sha256(raw).hexdigest()
    if "lfs" in entry:
        require(digest == entry["lfs"]["oid"], f"{name} upstream SHA-256")
    else:
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        require(blob == entry["oid"], f"{name} upstream Git blob")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return {"path": name, "url": url, "bytes": len(raw), "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    catalog = read_json(BASE / "catalog.json")
    require({c["model"]: c["revision"] for c in catalog["candidates"]} == MODELS, "model allowlist")
    rows = []
    for candidate in catalog["candidates"]:
        model, revision = candidate["model"], candidate["revision"]
        require({e["path"] for e in candidate["files"]} == FILES, "complete fixed metadata inventory")
        folder = BASE / model.replace("/", "--")
        records = [artifact(folder, model, revision, e, args.download) for e in candidate["files"]]
        card = (folder / "README.md").read_text(encoding="utf-8")
        header = card.split("---", 2)[1] if card.startswith("---") else ""
        require(re.search(r"(?m)^license:\s*apache-2\.0\s*$", header), "publisher license declaration")
        config, train = read_json(folder / "config.json"), read_json(folder / "train_config.json")
        require(config["type"] == "smolvla", "policy family")
        require(
            train["dataset"]["repo_id"] == candidate["declared_lineage_cluster"], "declared dataset lineage"
        )
        base_match = re.search(r"(?m)^base_model:\s*(\S+)\s*$", header)
        pre, post = (
            read_json(folder / f"policy_{p}.json")["steps"] for p in ("preprocessor", "postprocessor")
        )
        pre_normal = [s for s in pre if s["registry_name"] == "normalizer_processor"]
        post_normal = [s for s in post if s["registry_name"] == "unnormalizer_processor"]
        require(len(pre_normal) == len(post_normal) == 1, "normalization metadata inventory")
        decoded = {}
        for side, step in (("pre", pre_normal[0]), ("post", post_normal[0])):
            require(step["state_file"] in FILES, "statistics allowlist")
            decoded[side] = tensors(folder / step["state_file"])
        write_json(
            folder / "decoded_statistics.json", {"decoding": "bounded stdlib F32/F64", "tensors": decoded}
        )
        (folder / "LICENSE-2.0.txt").write_bytes(
            (ROOT / "examples/published_so101/LICENSE-2.0.txt").read_bytes()
        )
        manifest = {
            "schema_version": 1,
            "candidate_only": True,
            "final_evaluation": False,
            "execution_outcomes_observed": False,
            "model": model,
            "revision": revision,
            "policy_family": "SmolVLA",
            "license": {"declared": "Apache-2.0", "authority": "pinned publisher model-card frontmatter"},
            "files": records,
            "derived_statistics_sha256": hashlib.sha256(
                (folder / "decoded_statistics.json").read_bytes()
            ).hexdigest(),
            "declared_training_dataset": train.get("dataset"),
            "declared_lineage_cluster": candidate["declared_lineage_cluster"],
            "base_model": {
                "publisher_card": base_match.group(1) if base_match else None,
                "configured_pretrained_path": config.get("pretrained_path"),
                "training_base_revision_attested": False,
            },
            "saved_preprocessor_steps": [s["registry_name"] for s in pre],
            "saved_postprocessor_steps": [s["registry_name"] for s in post],
            "action_dimensions": config["output_features"]["action"]["shape"],
            "state_dimensions": config["input_features"]["observation.state"]["shape"],
            "normalization": {"pre": pre_normal[0]["config"], "post": post_normal[0]["config"]},
            "scope": [
                "Metadata inspected/decoded for eligibility only; not blinded and no execution outcomes",
                "Cards declare SmolVLA base; exact training base/dataset snapshots are unattested",
                "Physical units/modes, calibration, camera identity and training source remain unverified",
                "No dataset episodes, weights, model/tokenizer or native processor execution",
                "Dataset licensing is not inherited from the model card",
                "Freeze lineage assignment, intervention recipes and analysis before any evaluation",
            ],
            "weights_downloaded": False,
        }
        write_json(folder / "manifest.json", manifest)
        rows.append({"model": model, "revision": revision, "manifest": folder.name + "/manifest.json"})
        print(f"Verified metadata only: {model}@{revision}", flush=True)
    require(
        not any(name in sys.modules for name in ("torch", "lerobot", "safetensors")), "unexpected ML import"
    )
    write_json(BASE / "index.json", {"schema_version": 1, "status": catalog["status"], "candidates": rows})


if __name__ == "__main__":
    main()
