"""Pin a small PI05/LIBERO metadata reserve; never load or execute the policy."""

import argparse
import hashlib
import math
import re
import sys
import urllib.request
from pathlib import Path

from prepare_act_candidates import tensors

from mltools.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
MODEL = "lianqing11/pi05_libero"
REVISION = "dacdd57a16713e7273b97943ce95ee3539763b1a"
BASE = ROOT / "research/publication/selection/pi/lianqing11--pi05_libero"
LIMIT = 128 * 1024
# Git/LFS identities inspected from the immutable primary Hub tree endpoint.
FILES = {
    "README.md": (2134, "f3c170e83885f39330f6b922854812f06d65fbae"),
    "config.json": (1982, "53803452599f6d0b134e2e5047f6a3cb15f2da9b"),
    "train_config.json": (5375, "f9fd053742e874df4e12ea3484c7c34a59b775b2"),
    "policy_preprocessor.json": (1722, "4c438f02dacfcaa14670ff0a63007bc930ef7feb"),
    "policy_postprocessor.json": (662, "5a9832a7594764709ad630c45b0e2f4682fa89a1"),
    "policy_preprocessor_step_2_normalizer_processor.safetensors": (
        7668,
        "a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9",
    ),
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors": (
        7668,
        "a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9",
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError("PI metadata selection: " + message)


def snapshot(download=False):
    records = []
    for name, (size, expected) in FILES.items():
        path = BASE / name
        url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
        if path.exists():
            raw = path.read_bytes()
        else:
            require(download, "missing fixed artifact; explicit --download required: " + name)
            with urllib.request.urlopen(url, timeout=45) as response:
                raw = response.read(LIMIT + 1)
        require(len(raw) == size <= LIMIT, "fixed artifact size: " + name)
        identity = (
            hashlib.sha256(raw).hexdigest()
            if len(expected) == 64
            else hashlib.sha1(b"blob " + str(size).encode() + b"\0" + raw).hexdigest()
        )
        require(identity == expected, "immutable upstream identity: " + name)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        records.append(
            {
                "path": name,
                "url": url,
                "bytes": size,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "upstream_identity": expected,
            }
        )
    card = (BASE / "README.md").read_text(encoding="utf-8")
    require(
        card.startswith("---") and re.search(r"(?m)^license:\s*apache-2\.0\s*$", card.split("---", 2)[1]),
        "pinned publisher license declaration",
    )
    config, train = read_json(BASE / "config.json"), read_json(BASE / "train_config.json")
    require(
        config["type"] == "pi05"
        and config["output_features"]["action"]["shape"] == [7]
        and config["input_features"]["observation.state"]["shape"] == [8],
        "declared family/dimensions",
    )
    require(
        train["dataset"]["repo_id"] == "HuggingFaceVLA/libero" and train["dataset"]["revision"] is None,
        "declared dataset and missing revision",
    )
    decoded = {}
    for side, registry in (
        ("preprocessor", "normalizer_processor"),
        ("postprocessor", "unnormalizer_processor"),
    ):
        steps = read_json(BASE / f"policy_{side}.json")["steps"]
        selected = [s for s in steps if s["registry_name"] == registry]
        require(len(selected) == 1, "single declared normalization step")
        step = selected[0]
        require(
            step["config"]["norm_map"]["ACTION"] == "QUANTILES"
            and step["config"]["eps"] == 1e-8
            and step["state_file"] in FILES
            and step["state_file"].endswith(".safetensors"),
            "saved normalization declaration",
        )
        values = tensors(BASE / step["state_file"])
        require(
            all(math.isfinite(v) for value in values.values() for v in value["values"]),
            "finite numeric statistics",
        )
        require(
            values["action.q01"]["shape"] == values["action.q99"]["shape"] == [7],
            "declared action quantile statistics",
        )
        decoded[side] = values
    derived = {
        "decoding": "bounded stdlib F32/F64; metadata only, no processor execution",
        "tensors": decoded,
    }
    license_text = (ROOT / "examples/published_so101/LICENSE-2.0.txt").read_bytes()
    license_path = BASE / "LICENSE-2.0.txt"
    if download:
        license_path.write_bytes(license_text)
    require(license_path.read_bytes() == license_text, "accompanying publisher-declared license text")
    manifest = {
        "schema_version": 1,
        "candidate_only": True,
        "final_evaluation": False,
        "execution_outcomes_observed": False,
        "model": MODEL,
        "revision": REVISION,
        "policy_family_label": "PI05 metadata, not model execution",
        "pipeline": "LeRobot saved normalization processors; not an original OpenPI checkpoint",
        "identity_authority": f"https://huggingface.co/api/models/{MODEL}/tree/{REVISION}?recursive=true&expand=false",
        "files": records,
        "publisher_license_declaration": {
            "license": "Apache-2.0",
            "authority": "pinned publisher card; declaration, not an independent ownership audit",
            "weights_retrieved": False,
            "upstream_weight_terms_overridden": False,
        },
        "license_text_sha256": hashlib.sha256(license_text).hexdigest(),
        "declared_dataset": train["dataset"],
        "declared_lineage_cluster": "HuggingFaceVLA/libero; shared task corpus with other LIBERO metadata",
        "configured_base": {
            "path": config["pretrained_path"],
            "revision_attested": False,
            "terms": "No upstream model weights retrieved or relicensed",
        },
        "normalization": {"mode": "QUANTILES", "eps": 1e-8, "action_dimensions": 7, "state_dimensions": 8},
        "split_status": "Metadata-inspected reserve; freeze role/ancestry exclusions before case outcomes",
        "missing_authority": [
            "exact training dataset/base snapshot",
            "training processor source revision",
            "physical units/frame/gripper conventions",
            "attested mapping to independently audited native LIBERO controller",
        ],
        "limits": [
            "No weights, tokenizer, model, dataset episodes, processor, simulator or hardware execution",
            "Card contains generic ACT/SO100 example commands; these are not PI05 embodiment authority",
            "OpenPI use is a cross-framework metadata experiment, not native checkpoint replication",
            "Does not resolve original GCS asset licensing or establish second-embodiment execution coverage",
        ],
    }
    for name, data in (("decoded_statistics.json", derived), ("manifest.json", manifest)):
        path = BASE / name
        if download:
            write_json(path, data)
        else:
            require(read_json(path) == data, "derived offline metadata: " + name)
    require(
        not any(name in sys.modules for name in ("torch", "lerobot", "openpi")), "metadata-only inspection"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = snapshot(args.download)
    print(f"Verified {len(result['files'])} pinned PI05 metadata artifacts; no case outcomes observed")


if __name__ == "__main__":
    main()
