"""Development-only native custom transport over actual matched decoder captures."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import run_matched_study as matched
from action_study_common import ROOT, fingerprint, sha, write
from normalization_profiles import finite, require
from policy_space_numeric import contracts, validate_result
from study_assets import read
from study_assets import verify as verify_assets

FILES = {"request.json", "stdout.txt", "stderr.txt", "native.json", "analysis.json"}


def sources():
    result = matched.sources()
    paths = [
        ROOT / "scripts" / name
        for name in (
            "policy_space_numeric.py",
            "policy_space_numeric_worker.py",
            "run_policy_space_numeric.py",
            "validate_native_baselines.py",
        )
    ]
    paths += [p for p in (ROOT / "integrations/baselines").iterdir() if p.is_file() and p.suffix != ".md"]
    return {**result, **{p.relative_to(ROOT).as_posix(): sha(p) for p in paths}}


def inputs(folder):
    # Reject a prepared prospective plan before any native import or extraction.
    if (folder / "plan.json").is_file():
        require(read(folder / "plan.json").get("role") == "development", "no prospective native execution")
    summary = read(folder / "summary.json")
    assets = (ROOT / summary["assets_path"]).resolve()
    require(assets.is_relative_to(ROOT), "contained matched asset input")
    plan = verify_assets(assets)
    require(plan["role"] == "development", "no prospective native execution")
    matched.verify(folder)
    return plan, read(folder / "lerobot.native.json")


def contract_bytes(plan):
    return {
        name: (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode()
        for name, doc in contracts(plan).items()
    }


def verify_contracts(folder, plan):
    expected = contract_bytes(plan)
    require(
        {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()} == set(expected),
        "closed authored contract inventory",
    )
    for name, raw in expected.items():
        require((folder / name).read_bytes() == raw, "exact authored numeric schema: " + name)


def request(input_folder, contract_dir, plan):
    return {
        "schema_version": 1,
        "engine": "policy_space_numeric",
        "source_file_sha256": sources(),
        "matched_path": input_folder.relative_to(ROOT).as_posix(),
        "matched_summary_sha256": sha(input_folder / "summary.json"),
        "contracts_path": contract_dir.relative_to(ROOT).as_posix(),
        "contracts_file_sha256": {
            name: hashlib.sha256(raw).hexdigest() for name, raw in contract_bytes(plan).items()
        },
    }


def analyze(data, plan, decoder, matched_analysis):
    count, negatives = validate_result(data, plan, decoder)
    results = {}
    for cell_id, cell in plan["cells"].items():
        results[cell_id] = {}
        for condition in cell["conditions"]:
            native = data["cells"][cell_id][condition]
            source = matched_analysis["cells"][cell_id][condition]
            errors = [
                abs(a - b)
                for observations in native["designs"].values()
                for row in observations
                for a, b in zip(row["wire_output"], row["replayed_output"], strict=True)
            ]
            results[cell_id][condition] = {
                "policy_space_transport": "ACCEPT",
                "replayed_queries": len(errors) // len(next(iter(cell["intended"]["statistics"].values()))),
                "maximum_float32_conversion_error_raw_coordinate": max(errors),
                "numerical_methods": {k: v["status"] for k, v in source["numerical_methods"].items()},
                "robotruth": source["robotruth"],
                "checkpoint_physical_authority_status": "UNKNOWN",
                "interpretation": (
                    "Valid raw-vector transport acceptance; no numerical equality or safety verdict"
                ),
            }
    return {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "accepted_replay_queries": count,
        "rejected_structural_controls": negatives,
        "cells": results,
        "limits": [
            "One reused development checkpoint lineage; conditions and queries are correlated",
            "Actual captured decoder outputs replayed by an adapter; no checkpoint policy inference",
            "Stock dispatcher/MessagePack/selected custom schemas; no WebSocket client/server run",
            "Authored raw-index schemas do not supply physical units, gripper conventions or robot state",
            "All available numerical statistics supplied as metadata; "
            "native structural validation has a different scope",
            "Float64 decoder values become float32 on the native wire; "
            "roundoff is reported without a safety threshold",
            "Transport acceptance is not a semantic miss, certification, "
            "task success or evidence of superiority",
        ],
    }


def verify(folder):
    summary = read(folder / "summary.json")
    require(
        type(summary["schema_version"]) is int
        and summary["schema_version"] == 1
        and summary["development_only"] is True
        and summary["final_evaluation"] is False
        and summary["source_file_sha256"] == sources(),
        "development/source binding",
    )
    input_folder = (ROOT / summary["matched_path"]).resolve()
    capture = (ROOT / summary["capture_path"]).resolve()
    require(input_folder.is_relative_to(ROOT) and capture.is_relative_to(ROOT), "contained saved paths")
    plan, decoder = inputs(input_folder)
    verify_contracts(folder / "contracts", plan)
    wanted = FILES | {"contracts/" + n for n in contract_bytes(plan)}
    require(set(summary["file_sha256"]) == wanted, "closed evidence manifest")
    require(
        {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
        - {"README.md", "summary.json"}
        == wanted,
        "closed evidence directory",
    )
    require(all(sha(folder / n) == h for n, h in summary["file_sha256"].items()), "exact evidence bytes")
    req, data = (read(folder / name) for name in ("request.json", "native.json"))
    require(
        fingerprint(req) == fingerprint(request(input_folder, capture / "contracts", plan))
        and fingerprint(data) == fingerprint(read(folder / "stdout.txt"))
        and type(data["schema_version"]) is int
        and data["schema_version"] == 1
        and data["development_only"] is True
        and data["final_evaluation"] is False
        and data["engine"] == "policy_space_numeric"
        and data["source_file_sha256"] == sources()
        and data["request_sha256"] == sha(folder / "request.json")
        and data["upstream_identity"] == read(ROOT / "integrations/baselines/sources.json"),
        "native request/stdout/source/installed-identity binding",
    )
    runtime = data["runtime"]
    require(
        runtime["network_events"] == []
        and type(runtime["peak_rss_bytes"]) is int
        and 0 < runtime["peak_rss_bytes"] <= 2 * 1024**3
        and all(
            finite(runtime[k]) and runtime[k] >= 0
            for k in ("setup_seconds", "dispatch_seconds", "total_wall_seconds")
        ),
        "observed native controls and cost",
    )
    expected = analyze(data, plan, decoder, read(input_folder / "analysis.json"))
    require(
        fingerprint(read(folder / "analysis.json")) == fingerprint(expected), "independent transport analysis"
    )
    require(
        not any(
            m in sys.modules for m in ("torch", "lerobot", "openpi", "robotruth", "policy_space", "numpy")
        ),
        "data-only replay process",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", type=Path)
    group.add_argument("--output", type=Path)
    parser.add_argument("--matched", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
    else:
        require(args.matched is not None and args.python is not None, "explicit matched capture/interpreter")
        input_folder, folder = args.matched.resolve(), args.output.resolve()
        require(
            input_folder.is_relative_to(ROOT) and folder.is_relative_to(ROOT),
            "workspace-contained inputs/output",
        )
        plan, decoder = inputs(input_folder)
        before = sources()
        folder.mkdir(parents=True, exist_ok=False)
        for name, raw in contract_bytes(plan).items():
            path = folder / "contracts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        req = request(input_folder, folder / "contracts", plan)
        write(folder / "request.json", req)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUTF8="1")
        native = subprocess.run(
            [str(args.python.resolve()), "-I", str(ROOT / "scripts/policy_space_numeric_worker.py")],
            input=(folder / "request.json").read_bytes(),
            capture_output=True,
            timeout=180,
            env=env,
            check=False,
        )
        (folder / "stdout.txt").write_bytes(native.stdout)
        (folder / "stderr.txt").write_bytes(native.stderr)
        require(native.returncode == 0, f"actual native transport worker: {native.stderr[-2500:]!r}")
        data = json.loads(native.stdout)
        write(folder / "native.json", data)
        write(folder / "analysis.json", analyze(data, plan, decoder, read(input_folder / "analysis.json")))
        require(sources() == before, "stable transport producer sources")
        files = FILES | {"contracts/" + n for n in contract_bytes(plan)}
        write(
            folder / "summary.json",
            {
                "schema_version": 1,
                "development_only": True,
                "final_evaluation": False,
                "source_file_sha256": before,
                "matched_path": input_folder.relative_to(ROOT).as_posix(),
                "capture_path": folder.relative_to(ROOT).as_posix(),
                "file_sha256": {n: sha(folder / n) for n in sorted(files)},
            },
        )
        verify(folder)
    print("Verified native Policy Space custom numeric transport over actual decoder observations")


if __name__ == "__main__":
    main()
