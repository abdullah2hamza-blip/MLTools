"""Create a deterministic allowlisted Kaggle release. No Kaggle client/credentials."""

import argparse
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from integrity import require, safe_name, sha
from notebook import notebook

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FREEZE_COMMIT = "7513755e9627098c3529c46b3ae9a056d76434a5"
RESEARCH_COMMIT = "86e35c0fb4cc7c7f0e363d1e250a281f673d72ce"
ASSETS = "research/publication/prepared/action-study-001"
RUNS = ("prospective-001", "clean-reproduction-001")
PHASES = ("robotruth", "lerobot", "openpi", "lerobot_witness", "openpi_witness", "policy_space")
PRODUCTS = ("analysis.json", "witnesses.json", "workflow-analysis.json", "metrics.json", "conditions.csv")
SELECTED = (
    "research/publication/selection/act/act-so101-boccia-center-only",
    "research/publication/selection/smolvla/edge-inference--smolvla-so101-pick-orange",
    "research/publication/selection/pi/lianqing11--pi05_libero",
)
PRIVATE = re.compile(
    r"(?i)(?:[A-Z]:[\\/]Users[\\/]|/Users/|/home/(?:abdul|abdullah)|"
    r"github\.com/abdullah2hamza-blip/MLTools|KGAT_[A-Za-z0-9]|gh[pousr]_[A-Za-z0-9]{16}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def privacy(raw, name):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return
    require(PRIVATE.search(text) is None, "private path/secret marker in export: " + name)


def project(data):
    """Only redact known administrative fields; reject unexpected local paths."""
    pointers = []

    def walk(value, pointer=""):
        if isinstance(value, dict):
            return {
                k: walk(v, pointer + "/" + k.replace("~", "~0").replace("/", "~1")) for k, v in value.items()
            }
        if isinstance(value, list):
            return [walk(v, pointer + "/" + str(i)) for i, v in enumerate(value)]
        if isinstance(value, str) and PRIVATE.search(value):
            require(
                pointer.endswith("/environment_prefix") or pointer.endswith("/policy/source_path"),
                "unexpected sensitive JSON field: " + pointer,
            )
            pointers.append(pointer)
            return "[REDACTED_LOCAL_PATH]"
        return value

    return walk(data), pointers


def validate_original():
    proofs = {}
    for run in RUNS:
        folder = "research/publication/runs/" + run
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/run_publication_study.py"), "--verify", folder],
            cwd=ROOT,
            check=True,
            capture_output=True,
            timeout=180,
        )
        require(b"Verified frozen study" in result.stdout, "original verifier did not confirm success")
        proofs[run] = {
            "status": "ORIGINAL_COMMITTED_GATE_VERIFIED_BEFORE_EXPORT",
            "summary_sha256": sha((ROOT / folder / "summary.json").read_bytes()),
        }
    return proofs


def collect(proofs):
    files, ledger = {}, {}

    def add(path, exported=None, projection=False):
        source = ROOT / path
        require(not source.is_symlink() and source.is_file(), "allowlist source must be regular: " + path)
        raw = source.read_bytes()
        output = exported or path
        safe_name(output)
        require(output not in files, "duplicate export destination")
        changes = []
        if projection:
            value, changes = project(json.loads(raw))
            result = encoded(value) if changes else raw
        else:
            result = raw
        privacy(result, output)
        files[output] = result
        ledger[output] = {
            "source_path": path,
            "original_sha256": sha(raw),
            "exported_sha256": sha(result),
            "transform": "json-path-redaction-and-LF-serialization" if changes else "byte-exact-copy",
            "redacted_json_pointers": changes,
        }

    protocol = read(ROOT / "research/publication/protocol-freeze-001.json")
    # Closed frozen source inventory, not an arbitrary repository walk.
    for path, digest in protocol["source_file_sha256"].items():
        require(
            path.startswith(("scripts/", "src/mltools/")) and path.endswith(".py"),
            "frozen Python source only",
        )
        require(sha((ROOT / path).read_bytes()) == digest, "frozen source hash")
        add(path)
    plan = read(ROOT / ASSETS / "plan.json")
    add(ASSETS + "/plan.json")
    for path, digest in plan["file_sha256"].items():
        safe_name(path)
        require(sha((ROOT / ASSETS / path).read_bytes()) == digest, "prepared file hash")
        add(ASSETS + "/" + path)
    for leaf in SELECTED:
        manifest = read(ROOT / leaf / "manifest.json")
        declared = manifest.get("license", {}).get("declared") or manifest.get(
            "publisher_license_declaration", {}
        ).get("license")
        require(declared == "Apache-2.0", "publisher-declared license")
        names = {x["path"] for x in manifest["files"]} | {"manifest.json", "LICENSE-2.0.txt"}
        for name in sorted(names):
            safe_name(name)
            p = ROOT / leaf / name
            require(p.stat().st_size <= 128 * 1024, "metadata-only publisher leaf")
            add(leaf + "/" + name)
    for path in (
        "integrations/robotruth/sources.json",
        "integrations/baselines/sources.json",
        "LICENSE",
        "NOTICE",
    ):
        add(path)
    expected_outputs = {}
    for run in RUNS:
        folder = "research/publication/runs/" + run
        for phase in PHASES:
            add(folder + "/" + phase + ".native.json", "evidence/" + run + "/" + phase + ".native.json", True)
        for path in sorted((ROOT / folder / "contracts").rglob("*")):
            if path.is_file():
                rel = path.relative_to(ROOT).as_posix()
                add(
                    rel,
                    "evidence/"
                    + run
                    + "/contracts/"
                    + path.relative_to(ROOT / folder / "contracts").as_posix(),
                )
        for name in PRODUCTS:
            target = "evidence/" + run + "/" + name
            add(folder + "/" + name, target, name.endswith(".json"))
            if run == RUNS[0]:
                raw = files[target]
                expected_outputs[name] = sha(raw if name.endswith(".csv") else encoded(json.loads(raw)))
    for name in ("integrity.py", "replay.py"):
        add("releases/kaggle/" + name, name)
    add("releases/kaggle/DATASET_CARD.md", "README.md")
    add("releases/kaggle/THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")
    provenance = {
        "schema_version": 1,
        "research_commit": RESEARCH_COMMIT,
        "freeze_commit": FREEZE_COMMIT,
        "original_verification": proofs,
        "files": ledger,
        "scope": "Public export projection; not original byte-exact capture when a transformation is listed",
        "omitted": [
            "private Git history and full 406-file freeze closure",
            "attempt/request/stdout/environment/setup logs",
            "native dependency packages, model weights, episodes, credentials",
            "unresolved OpenPI GCS assets and Tether source",
            "TeX/style/compiler assets and manuscript",
            "retrospective and Panda/SO101 development strata",
        ],
    }
    files["release-provenance.json"] = encoded(provenance)
    return files, expected_outputs


def make_zip(files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for name, raw in sorted(files.items()):
            safe_name(name)
            i = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            i.create_system = 3
            i.external_attr = 0o100644 << 16
            i.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(i, raw, compresslevel=9)
    return out.getvalue()


def build(output, config):
    require(
        config.get("schema_version") == 1
        and config.get("license") == "apache-2.0"
        and config.get("license_approved") is True,
        "approved original license required",
    )
    require(config["owner"] == "abdullah2hamza", "confirmed release owner")
    require(
        config["dataset_slug"] == config["notebook_slug"] == "boundary-conformance-replay", "reviewed slugs"
    )
    output = Path(output).resolve()
    require(not output.exists(), "build destination must be new")
    proofs = validate_original()
    files, expected_outputs = collect(proofs)
    release_id = sha(encoded({name: sha(raw) for name, raw in sorted(files.items())}))
    files["manifest.json"] = encoded(
        {
            "schema_version": 1,
            "release_id": release_id,
            "files": {name: sha(raw) for name, raw in sorted(files.items())},
        }
    )
    payload = make_zip(files)
    payload_sha = sha(payload)
    title, subtitle = (
        config["dataset_title"],
        "Offline CPU replay of saved normalization-boundary research traces",
    )
    require(6 <= len(title) <= 50 and 20 <= len(subtitle) <= 80, "Kaggle title/subtitle limits")
    descriptions = {
        "cell": "Execution cell; OpenPI reuses the PI artifact",
        "cluster": "Declared dependency-linked ancestry group",
        "condition": "Actual condition or derived authority-erased view",
        "derived": "Reuses observations, no new acquisition",
        "compiled": "Compiled numerical PASS/FAIL/UNKNOWN",
        "pointwise": "Independent pointwise numerical status",
        "strong": "Strong relational numerical status",
        "physical": "UNKNOWN: no calibrated physical authority",
        "robotruth_identity_difference": "Union statistics identity differs from control",
        "robotruth_certification": "WITHHELD for missing evidence",
        "policy_space_transport": "Structural acceptance only",
    }
    fields = [
        {
            "name": name,
            "description": desc,
            "type": "boolean" if name in ("derived", "robotruth_identity_difference") else "string",
        }
        for name, desc in descriptions.items()
    ]
    metadata = {
        "title": title,
        "subtitle": subtitle,
        "id": config["owner"] + "/" + config["dataset_slug"],
        "licenses": [{"name": config["license"]}],
        "description": (HERE / "DATASET_CARD.md").read_text(encoding="utf-8"),
        "keywords": ["machine learning", "reproducible research"],
        "resources": [
            {
                "path": "conditions.csv",
                "description": "16 actual conditions and 8 reused derived views",
                "schema": {"fields": fields},
            }
        ],
    }
    dataset = {
        "dataset-metadata.json": encoded(metadata),
        "payload.zip": payload,
        "conditions.csv": files["evidence/prospective-001/conditions.csv"],
        "README.md": files["README.md"],
        "LICENSE": files["LICENSE"],
        "THIRD_PARTY_NOTICES.md": files["THIRD_PARTY_NOTICES.md"],
    }
    dataset["release-manifest.json"] = encoded(
        {
            "schema_version": 1,
            "release_id": release_id,
            "payload_sha256": payload_sha,
            "files": {name: sha(raw) for name, raw in sorted(dataset.items())},
            "expected_replay_outputs": expected_outputs,
        }
    )
    nb = notebook(payload_sha, release_id, config["dataset_slug"])
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], "<notebook-cell>", "exec")
    kernel = {
        "id": config["owner"] + "/" + config["notebook_slug"],
        "title": config["notebook_title"],
        "code_file": "replay.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "dataset_sources": [metadata["id"]],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    for folder, entries in (
        ("dataset", dataset),
        ("notebook", {"replay.ipynb": encoded(nb), "kernel-metadata.json": encoded(kernel)}),
    ):
        for name, raw in entries.items():
            if name != "payload.zip":
                privacy(raw, name)
            p = output / folder / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
    return {
        "schema_version": 1,
        "release_id": release_id,
        "payload_sha256": payload_sha,
        "payload_files": len(files),
        "payload_zip_bytes": len(payload),
        "files": {
            p.relative_to(output).as_posix(): sha(p.read_bytes())
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "original_verification": proofs,
        "credentials_used": False,
        "hosted_validation": "PENDING_TOKEN_AND_AUTHORIZATION",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=HERE / "config.json")
    p.add_argument("--record", type=Path)
    args = p.parse_args()
    record = build(args.output, read(args.config))
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_bytes(encoded(record))
    print(json.dumps(record, indent=2))
