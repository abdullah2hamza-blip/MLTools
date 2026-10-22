"""Matched stock robotruth and genuine decoder workflows on exact shared files.

This command executes development assets only. Prospective metadata compilation
does not authorize selected-case execution without a committed frozen protocol.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import query_designs as qd
import run_action_study as numerical
from action_designs import ActionDesign
from action_study_common import ROOT, fingerprint, sha, write
from normalization_profiles import finite, require, transform
from prepare_act_candidates import tensors
from study_assets import read
from study_assets import verify as verify_assets

FILES = {
    "analysis.json",
    *(
        f"{engine}.{suffix}"
        for engine in ("robotruth", "lerobot")
        for suffix in ("request.json", "stdout.txt", "stderr.txt", "native.json")
    ),
}

# Pinned native scalar comparison inventory. This replay check preserves the
# upstream report contract; it is not a replacement robotruth execution.
SCALAR_FIELDS = {
    "fatal": (
        "policy.weights_sha256",
        "action.normalization.stats_sha256",
        "action.normalization.type",
        "action.space",
        "action.mode",
        "action.dim",
        "action.chunk_size",
        "action.n_action_steps",
        "action.control_hz",
        "action.gripper_convention",
        "observation.normalization.stats_sha256",
        "observation.preprocessing_sha256",
        "observation.state_dim",
        "embodiment.robot",
    ),
    "warn": (
        "runtime.dtype",
        "runtime.export_format",
        "runtime.device",
        "runtime.inference_hz",
        "embodiment.unit_id",
        "embodiment.controller",
        "embodiment.controller_hz",
        "observation.image_resize",
        "action.units",
    ),
    "info": ("policy.name", "policy.family", "runtime.framework_version"),
}


def expected_findings(evaluated, deployed):
    def get(value, field):
        for part in field.split("."):
            value = value.get(part) if value is not None else None
        return value

    def unknown(value):
        return value is None or value == "unknown" or value == [] or value == ""

    expected = {}
    for severity, fields in SCALAR_FIELDS.items():
        for field in fields:
            left, right = get(evaluated, field), get(deployed, field)
            if unknown(left) or unknown(right):
                actual = "fatal" if severity == "fatal" else "info"
            else:
                actual = severity if left != right else "ok"
            expected[field] = (actual, left, right)
    cameras = {
        role: {c["name"]: c for c in spec["observation"]["cameras"]}
        for role, spec in (("evaluated", evaluated), ("deployed", deployed))
    }
    left, right = cameras["evaluated"], cameras["deployed"]
    if set(left) != set(right):
        expected["observation.cameras"] = ("fatal", sorted(left), sorted(right))
    else:
        count = 0
        for name, camera in left.items():
            other = right[name]
            a, b = [camera["width"], camera["height"]], [other["width"], other["height"]]
            if a != b:
                expected[f"observation.cameras[{name}].resolution"] = ("warn", a, b)
                count += 1
            for attr in ("intrinsics_sha256", "extrinsics_sha256"):
                a, b = camera[attr], other[attr]
                if a is None or b is None or a != b:
                    expected[f"observation.cameras[{name}].{attr}"] = (
                        "warn" if a is None or b is None else "fatal",
                        a,
                        b,
                    )
                    count += 1
        if count == 0:
            expected["observation.cameras"] = ("ok", sorted(left), sorted(right))
    return expected


def sources():
    names = (
        "study_assets.py",
        "matched_study_worker.py",
        "run_matched_study.py",
        "validate_robotruth.py",
        "action_study_worker.py",
        "action_study_common.py",
        "run_action_study.py",
        "action_designs.py",
        "normalization_profiles.py",
        "normalization_certificate.py",
        "prepare_act_candidates.py",
        "query_designs.py",
    )
    paths = [ROOT / "scripts" / n for n in names]
    paths += list((ROOT / "src/mltools").glob("*.py")) + [ROOT / "uv.lock"]
    paths += [p for p in (ROOT / "integrations/robotruth").iterdir() if p.is_file() and p.suffix != ".md"]
    paths += [ROOT / "integrations/openpi" / n for n in ("frameworks.txt", "requirements-cpu.lock")]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def combined_statistics_hash(cell, entry):
    digest = hashlib.sha256()
    for name in sorted({cell["pre_state_file"], cell["post_state_file"]}):
        digest.update(name.encode())
        digest.update(bytes.fromhex(entry["file_sha256"][name]))
    return digest.hexdigest()


def validate_robotruth(data, plan):
    ledger = read(ROOT / "integrations/robotruth/sources.json")
    identity = data["upstream_identity"]
    require(
        identity["direct_url"]["vcs_info"]["commit_id"] == ledger["revision"]
        and identity["direct_url"]["url"] == "https://github.com/mulkakhileshmj/robotruth.git",
        "actual complete stock robotruth identity",
    )
    require(
        {k: v["git_blob"] for k, v in identity["python_sources"].items()}
        == ledger["package_python_git_blobs"],
        "all pinned installed package source files",
    )
    require(set(data["cells"]) == set(plan["cells"]), "stock baseline cell inventory")
    checks = 0
    for cell_id, cell in plan["cells"].items():
        require(set(data["cells"][cell_id]) == set(cell["conditions"]), "all native baseline conditions")
        base_hash = combined_statistics_hash(cell, cell["conditions"]["numeric_control"])
        for condition, entry in cell["conditions"].items():
            result = data["cells"][cell_id][condition]
            require(
                result["overlays"] == {"evaluated": {}, "deployed": {}}
                and result["allow_unknown"] is False
                and result["ignore"] == []
                and result["extractor_family"] == "lerobot"
                and result["target_decoder_framework"] == cell["framework"],
                "actual source extractor/no invented evidence",
            )
            expected = {"evaluated": base_hash, "deployed": combined_statistics_hash(cell, entry)}
            for role, digest in expected.items():
                spec = result[role]
                require(
                    spec["policy"]["weights_sha256"] is None and spec["action"]["gripper_convention"] is None,
                    "missing published weights/gripper remain missing",
                )
                require(
                    spec["action"]["normalization"]["stats_sha256"] == digest
                    and spec["observation"]["normalization"]["stats_sha256"] == digest,
                    "native union statistics hash, not independent action/observation changes",
                )
            findings = result["findings"]
            wanted = expected_findings(result["evaluated"], result["deployed"])
            require(
                len({f["field"] for f in findings}) == len(findings)
                and {f["field"] for f in findings} == set(wanted),
                "complete unique pinned scalar/camera finding inventory",
            )
            require(
                all((f["severity"], f["evaluated"], f["deployed"]) == wanted[f["field"]] for f in findings),
                "every native finding bound to the saved manifests and severity policy",
            )
            for field in ("action.normalization.stats_sha256", "observation.normalization.stats_sha256"):
                row = [f for f in findings if f["field"] == field]
                require(
                    len(row) == 1 and all(row[0][role] == value for role, value in expected.items()),
                    "known statistics finding bound to actual input files",
                )
                require(
                    row[0]["severity"] == ("ok" if len(set(expected.values())) == 1 else "fatal"),
                    "native known equality/mismatch severity",
                )
            require(
                any(
                    f["field"] == "policy.weights_sha256"
                    and f["severity"] == "fatal"
                    and f["evaluated"] is None
                    and f["deployed"] is None
                    for f in findings
                ),
                "native certification withheld for missing evidence",
            )
            n_fatal = sum(f["severity"] == "fatal" for f in findings)
            n_warn = sum(f["severity"] == "warn" for f in findings)
            require(
                result["native_passed"] is False
                and n_fatal > 0
                and result["native_summary"]
                == f"FAIL: {n_fatal} fatal, {n_warn} warn, {len(findings)} checks",
                "native aggregation/vocabulary",
            )
            checks += 1
    require(
        data["runtime"]["check_calls"] == checks
        and data["runtime"]["extract_calls"] == checks + len(plan["cells"]),
        "actual native workflow calls",
    )


def validate_decoder(data, plan, folder):
    identity = data["upstream_identity"]
    require(
        data["framework"] == "lerobot"
        and identity["revision"] == numerical.LEROBOT_REVISION
        and identity["git_blobs"] == numerical.LEROBOT_BLOBS
        and identity["source_newlines"] == "canonical LF",
        "genuine decoder source identity",
    )
    require(set(data["cells"]) == set(plan["cells"]), "decoder cell inventory")
    primary, auxiliary = 0, 0
    for cell_id, cell in plan["cells"].items():
        require(set(data["cells"][cell_id]) == set(cell["conditions"]), "decoder condition inventory")
        n = len(numerical.suite(cell).outputs)
        mode = "MEAN_STD" if "mean_std" in cell["intended"]["profile"] else "QUANTILES"
        config = {
            "eps": 1e-8,
            "features": {"action": {"type": "ACTION", "shape": [n]}},
            "norm_map": {"ACTION": mode},
        }
        for condition, entry in cell["conditions"].items():
            result = data["cells"][cell_id][condition]
            require(
                result["configuration"]
                == {"encoder": config, "decoder": config, "device": "cpu", "dtype": "float64"},
                "actual normalization operator configuration",
            )
            native = result["loaded_artifact"]
            decoded = tensors(folder / entry["directory"] / entry["post_state_file"])
            require(
                native["artifact_sha256"] == entry["file_sha256"][entry["post_state_file"]]
                and native["statistics"] == entry["specification"]["statistics"]
                and native["tensor_dtypes"]
                == {
                    k: "torch.float" + ("32" if decoded["action." + k]["dtype"] == "F32" else "64")
                    for k in native["statistics"]
                },
                "actual native loaded bytes/dtypes/statistics",
            )
            require(set(result["designs"]) == set(cell["designs"]), "all equal-budget designs")
            for name, rows in result["designs"].items():
                design = ActionDesign.from_dict(cell["designs"][name])
                require(len(rows) == len(design.probes), "one decoder call per design query")
                for row, query in zip(rows, design.probes, strict=True):
                    require(
                        row["query"] == query.id
                        and row["input"] == list(query.values)
                        and numerical.near(
                            row["output"], transform(entry["specification"], row["input"], inverse=True)
                        ),
                        "actual file-backed native response, including injections",
                    )
                    primary += 1
            main = result["designs"]["canonical"]
            require(len(result["roundtrips"]) == len(main), "native roundtrip count")
            for row, p in zip(result["roundtrips"], main, strict=True):
                require(row["query"] == p["query"], "roundtrip query binding")
                expected = {
                    "self_encoded": transform(entry["specification"], p["output"], inverse=False),
                    "intended_decoded": transform(cell["intended"], p["input"], inverse=True),
                    "cross_encoded": transform(cell["intended"], row["intended_decoded"], inverse=False),
                    "cross_decoded": transform(entry["specification"], row["cross_encoded"], inverse=True),
                }
                require(
                    all(numerical.near(row[k], v) for k, v in expected.items()),
                    "actual native roundtrip boundaries",
                )
                auxiliary += 4
    require(
        data["runtime"]["primary_decoder_calls"] == primary
        and data["runtime"]["auxiliary_native_calls"] == auxiliary,
        "primary and auxiliary native calls charged separately",
    )


def analyze(robo, decoder, plan):
    results = {}
    for cell_id, cell in plan["cells"].items():
        results[cell_id] = {}
        for condition, entry in cell["conditions"].items():
            row = decoder["cells"][cell_id][condition]
            s, ref = numerical.suite(cell), numerical.reference(cell)
            rows = row["designs"]["canonical"]
            methods = numerical.comparisons(s, ref, numerical.evidence(s, rows))
            native = robo["cells"][cell_id][condition]
            base = combined_statistics_hash(cell, cell["conditions"]["numeric_control"])
            results[cell_id][condition] = {
                "numerical_methods": methods,
                "geometry_shared_oracle": {
                    name: qd.analyze(
                        ActionDesign.from_dict(cell["designs"][name]),
                        ref.to_dict(),
                        {
                            "design_sha256": ActionDesign.from_dict(cell["designs"][name]).sha256,
                            "outputs": {r["query"]: r["output"] for r in values},
                        },
                        **numerical.TOLERANCES,
                        measurement_allowance=1e-12,
                    )
                    for name, values in row["designs"].items()
                },
                "certificate": numerical.certificate_adapter(cell, condition, rows, ref),
                "robotruth": {
                    "native_passed": native["native_passed"],
                    "full_certification": "WITHHELD",
                    "applicability": "STATIC_SOURCE_ARTIFACT_COMPARISON",
                    "known_statistics_identity_difference": combined_statistics_hash(cell, entry) != base,
                    "underlying_artifact_differences_counted": int(
                        combined_statistics_hash(cell, entry) != base
                    ),
                    "limits": (
                        "One union hash covers both normalizers; no separate observation-numeric finding"
                    ),
                },
                "same_numeric_statistics_as_control": entry["specification"] == cell["intended"],
                "physical_checkpoint_authority_status": "UNKNOWN",
            }
    return {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "cells": results,
        "scope": "Shared artifacts; numerical equality and strict byte identity have distinct requirements",
    }


def request(assets, engine):
    return {
        "schema_version": 1,
        "engine": engine,
        "assets_path": assets.relative_to(ROOT).as_posix(),
        "plan_sha256": sha(assets / "plan.json"),
        "source_file_sha256": sources(),
    }


def acquire(folder, assets, engine, python):
    path = folder / (engine + ".request.json")
    write(path, request(assets, engine))
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUTF8="1",
    )
    native = subprocess.run(
        [str(python.resolve()), "-I", str(ROOT / "scripts/matched_study_worker.py")],
        input=path.read_bytes(),
        capture_output=True,
        timeout=180,
        env=env,
        check=False,
    )
    (folder / (engine + ".stdout.txt")).write_bytes(native.stdout)
    (folder / (engine + ".stderr.txt")).write_bytes(native.stderr)
    require(native.returncode == 0, f"actual matched engine: {native.stderr[-2500:]!r}")
    write(folder / (engine + ".native.json"), json.loads(native.stdout))


def verify(folder):
    summary = read(folder / "summary.json")
    require(
        summary["development_only"] is True
        and summary["final_evaluation"] is False
        and summary["source_file_sha256"] == sources()
        and set(summary["file_sha256"]) == FILES,
        "development scope and closed native evidence",
    )
    assets = (ROOT / summary["assets_path"]).resolve()
    require(assets.is_relative_to(ROOT), "contained exact assets")
    plan = verify_assets(assets)
    require(
        plan["role"] == "development" and sha(assets / "plan.json") == summary["plan_sha256"],
        "same development assets for every workflow",
    )
    require(
        all(sha(folder / n) == h for n, h in summary["file_sha256"].items()), "exact native evidence files"
    )
    data = {}
    for engine in ("robotruth", "lerobot"):
        req = read(folder / (engine + ".request.json"))
        row = read(folder / (engine + ".native.json"))
        require(
            fingerprint(req) == fingerprint(request(assets, engine))
            and row == read(folder / (engine + ".stdout.txt"))
            and row["request_sha256"] == sha(folder / (engine + ".request.json"))
            and row["source_file_sha256"] == sources()
            and row["plan_sha256"] == sha(assets / "plan.json")
            and row["schema_version"] == 1
            and type(row["schema_version"]) is int
            and row["engine"] == ("robotruth" if engine == "robotruth" else "decoder")
            and row["development_only"] is True
            and row["final_evaluation"] is False,
            "source/input/native stdout binding",
        )
        runtime = row["runtime"]
        require(
            runtime["network_events"] == []
            and type(runtime["peak_rss_bytes"]) is int
            and 0 < runtime["peak_rss_bytes"] <= 2 * 1024**3
            and finite(runtime["total_wall_seconds"])
            and runtime["total_wall_seconds"] >= 0,
            "actual resource/network controls",
        )
        (validate_robotruth(row, plan) if engine == "robotruth" else validate_decoder(row, plan, assets))
        data[engine] = row
    require(
        read(folder / "analysis.json") == analyze(data["robotruth"], data["lerobot"], plan),
        "independent matched analysis",
    )
    require(
        not any(m in sys.modules for m in ("torch", "lerobot", "openpi", "robotruth")), "data-only parent"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--robotruth-python", type=Path)
    parser.add_argument("--lerobot-python", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
    else:
        require(
            args.assets is not None and args.robotruth_python is not None and args.lerobot_python is not None,
            "explicit assets and isolated native environments",
        )
        assets = args.assets.resolve()
        require(assets.is_relative_to(ROOT), "contained assets")
        plan = verify_assets(assets)
        require(
            plan["role"] == "development", "prospective execution still prohibited before protocol freeze"
        )
        folder = args.output.resolve()
        require(folder.is_relative_to(ROOT), "contained evidence")
        folder.mkdir(parents=True, exist_ok=False)
        initial = sources()
        acquire(folder, assets, "robotruth", args.robotruth_python)
        acquire(folder, assets, "lerobot", args.lerobot_python)
        write(
            folder / "analysis.json",
            analyze(read(folder / "robotruth.native.json"), read(folder / "lerobot.native.json"), plan),
        )
        require(sources() == initial, "stable matched producer")
        write(
            folder / "summary.json",
            {
                "schema_version": 1,
                "development_only": True,
                "final_evaluation": False,
                "source_file_sha256": initial,
                "assets_path": assets.relative_to(ROOT).as_posix(),
                "plan_sha256": sha(assets / "plan.json"),
                "file_sha256": {n: sha(folder / n) for n in sorted(FILES)},
            },
        )
        verify(folder)
    print("Verified exact shared-artifact native decoder and stock robotruth workflows")


if __name__ == "__main__":
    main()
