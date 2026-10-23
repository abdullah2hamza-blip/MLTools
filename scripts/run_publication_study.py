"""Run/replay the frozen multi-framework study; prospective attempts are single-use."""

import argparse
import csv
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import publication_analysis as analysis
import run_matched_study as matched
import run_policy_space_numeric as transport
from action_study_common import ROOT, fingerprint, sha, write
from normalization_profiles import finite, require
from publication_network import validate as validate_network
from publication_protocol import build, relative
from publication_protocol import verify as verify_protocol
from study_assets import read

PHASES = ("robotruth", "lerobot", "openpi", "lerobot_witness", "openpi_witness", "policy_space")
PRODUCTS = {"analysis.json", "witnesses.json", "workflow-analysis.json", "metrics.json", "conditions.csv"}


def verify_attempt(folder, protocol, commit, phase=None):
    start = read(folder / "attempt-start.json")
    require(relative(start["protocol_path"]) == start["protocol_path"], "contained saved protocol path")
    require(
        type(start["schema_version"]) is int
        and start["schema_version"] == 1
        and start["role"] == protocol["role"]
        and start["freeze_commit"] == commit
        and start["protocol_sha256"] == sha(ROOT / start["protocol_path"]),
        "immutable attempt-start binding",
    )
    require(
        (folder / "protocol.json").read_bytes() == (ROOT / start["protocol_path"]).read_bytes(),
        "saved protocol bytes",
    )
    require(relative(start["attempt_path"]) == start["attempt_path"], "contained original attempt path")
    if protocol["role"] == "prospective":
        require(
            {
                k: {field: v[field] for field in ("python", "packages")}
                for k, v in start["environments"].items()
            }
            == protocol["native_environments"],
            "exact frozen native distribution profiles",
        )
        policy = protocol["specification"]["attempt_policy"]
        key = "primary_output" if start["reproduction_of"] is None else "clean_reproduction_output"
        require(start["attempt_path"] == policy[key], "single declared prospective attempt directory")
        if start["reproduction_of"] is not None:
            original = start["reproduction_of"]
            require(original["path"] == policy["primary_output"], "fixed original reproduction source")
            require(
                sha(ROOT / original["path"] / "summary.json") == original["summary_sha256"],
                "bound successful original",
            )
            require(
                read(ROOT / original["path"] / "attempt-status.json")["status"] == "SUCCEEDED",
                "successful original required",
            )
    else:
        require(start["reproduction_of"] is None, "development validation is not clean main reproduction")
    if phase is not None:
        require(phase in PHASES, "known native phase before claiming execution")
        require(
            read(folder / "attempt-status.json")
            == {"status": "STARTED", "completed_phases": list(PHASES[: PHASES.index(phase)])},
            "native execution requires STARTED and exact preceding phase prefix",
        )
    return start


def request(folder, protocol_path, protocol, commit, phase):
    require(phase in PHASES, "fixed native phase")
    dependent = (
        {}
        if phase in PHASES[:3]
        else {
            f"{engine}.{suffix}": sha(folder / f"{engine}.{suffix}")
            for engine in ("lerobot", "openpi")
            for suffix in (
                "request.json",
                "native.json",
                "stdout.txt",
                "stderr.txt",
                "claim.json",
                "receipt.json",
            )
        }
    )
    start = read(folder / "attempt-start.json")
    return {
        "schema_version": 1,
        "protocol_path": relative(protocol_path),
        "protocol_sha256": sha(protocol_path),
        "freeze_commit": commit,
        "role": protocol["role"],
        "phase": phase,
        "attempt_path": start["attempt_path"],
        "attempt_start_sha256": sha(folder / "attempt-start.json"),
        "source_file_sha256": protocol["source_file_sha256"],
        "phase_input_sha256": dependent,
    }


def load_phase(folder, protocol_path, protocol, commit, phase):
    req, data = (read(folder / f"{phase}.{name}.json") for name in ("request", "native"))
    require(
        fingerprint(req) == fingerprint(request(folder, protocol_path, protocol, commit, phase))
        and fingerprint(data) == fingerprint(read(folder / f"{phase}.stdout.txt"))
        and type(data["schema_version"]) is int
        and data["schema_version"] == 1
        and data["format"] == "mltools-publication-native-phase"
        and data["role"] == protocol["role"]
        and data["phase"] == phase
        and data["engine"]
        == (
            "robotruth"
            if phase == "robotruth"
            else "policy_space_numeric"
            if phase == "policy_space"
            else "decoder_witness"
            if phase.endswith("_witness")
            else "decoder"
        )
        and data["freeze_commit"] == commit
        and data["protocol_sha256"] == sha(protocol_path)
        and data["request_sha256"] == sha(folder / f"{phase}.request.json")
        and data["source_file_sha256"] == protocol["source_file_sha256"],
        "source/protocol/native stdout/request binding",
    )
    runtime = data["runtime"]
    validate_network(runtime)
    digest = sha(folder / f"{phase}.request.json")
    require(
        fingerprint(read(folder / f"{phase}.claim.json"))
        == fingerprint(
            {
                "schema_version": 1,
                "phase": phase,
                "status": "STARTED",
                "request_sha256": digest,
                "freeze_commit": commit,
                "protocol_sha256": sha(protocol_path),
            }
        ),
        "exclusive exact-request native execution claim",
    )
    receipt = read(folder / f"{phase}.receipt.json")
    require(
        receipt["request_sha256"] == digest
        and type(receipt["return_code"]) is int
        and receipt["return_code"] == 0
        and receipt["timed_out"] is False
        and finite(receipt["subprocess_wall_seconds"])
        and receipt["subprocess_wall_seconds"] >= runtime["total_wall_seconds"],
        "parent-observed complete subprocess wall-time receipt",
    )
    start = read(folder / "attempt-start.json")
    environment = start["environments"][phase.removesuffix("_witness")]
    require(
        runtime["packages"] == environment["packages"]
        and runtime["python"] == environment["python"]
        and runtime["environment_prefix"] == environment["prefix"],
        "actual preflighted native environment",
    )
    require(
        runtime["network_events"] == []
        and type(runtime["peak_rss_bytes"]) is int
        and 0
        < runtime["peak_rss_bytes"]
        <= protocol["specification"]["resources"]["observed_peak_rss_limit_bytes"]
        and all(
            finite(runtime[k]) and runtime[k] >= 0 for k in ("total_wall_seconds", "process_cpu_seconds")
        ),
        "actual finite native cost/control fields",
    )
    return data


def load_main(folder, protocol_path, protocol, plan, commit):
    captures = {
        framework: load_phase(folder, protocol_path, protocol, commit, framework)
        for framework in ("lerobot", "openpi")
    }
    for framework, data in captures.items():
        analysis.validate_decoder(data, plan, ROOT / protocol["assets_path"], framework)
    return analysis.merge_decoders(plan, captures)


def tables(plan, numerical, workflows):
    target = io.StringIO(newline="")
    writer = csv.writer(target, lineterminator="\n")
    writer.writerow(
        (
            "cell",
            "cluster",
            "condition",
            "derived",
            "compiled",
            "pointwise",
            "strong",
            "physical",
            "robotruth_identity_difference",
            "robotruth_certification",
            "policy_space_transport",
        )
    )
    for cell_id, cell in plan["cells"].items():
        for condition, row in numerical["cells"][cell_id].items():
            derived = condition not in cell["conditions"]
            source = "numeric_control" if derived else condition
            external = workflows["cells"][cell_id][source]["robotruth"]
            writer.writerow(
                (
                    cell_id,
                    cell["cluster"],
                    condition,
                    str(derived).lower(),
                    *(
                        row["numerical_methods"][k]["status"]
                        for k in ("compiled", "independent_pointwise", "strong_relational")
                    ),
                    row["checkpoint_physical_status"],
                    str(external["known_statistics_identity_difference"]).lower(),
                    external["full_certification"],
                    workflows["transport"][cell_id][source]["policy_space_transport"],
                )
            )
    return target.getvalue().encode()


def products(folder, protocol_path, protocol, plan, commit):
    phases = {p: load_phase(folder, protocol_path, protocol, commit, p) for p in PHASES}
    decoder = load_main(folder, protocol_path, protocol, plan, commit)
    numerical, selections = analysis.analyze(plan, decoder)
    fresh = {framework: phases[framework + "_witness"] for framework in ("lerobot", "openpi")}
    for framework, data in fresh.items():
        analysis.validate_decoder(
            data, plan, ROOT / protocol["assets_path"], framework, selections[framework]
        )
    witnesses = analysis.verify_witnesses(plan, decoder, fresh, numerical, selections)
    matched.validate_robotruth(phases["robotruth"], plan)
    shared = {"cells": {}}
    for cell_id, cell in plan["cells"].items():
        shared["cells"][cell_id] = {}
        base = matched.combined_statistics_hash(cell, cell["conditions"]["numeric_control"])
        for condition, entry in cell["conditions"].items():
            different = matched.combined_statistics_hash(cell, entry) != base
            shared["cells"][cell_id][condition] = {
                "numerical_methods": numerical["cells"][cell_id][condition]["numerical_methods"],
                "robotruth": {
                    "native_passed": phases["robotruth"]["cells"][cell_id][condition]["native_passed"],
                    "full_certification": "WITHHELD",
                    "applicability": "STATIC_SOURCE_ARTIFACT_COMPARISON",
                    "known_statistics_identity_difference": different,
                    "underlying_artifact_differences_counted": int(different),
                    "limits": (
                        "One union hash covers both normalizers; no separate observation-numeric finding"
                    ),
                },
                "same_numeric_statistics_as_control": entry["specification"] == cell["intended"],
                "physical_checkpoint_authority_status": "UNKNOWN",
            }
    transport.verify_contracts(folder / "contracts", plan)
    native_transport = phases["policy_space"]
    require(
        native_transport["upstream_identity"] == read(ROOT / "integrations/baselines/sources.json"),
        "pinned native transport package assertions",
    )
    transported = transport.analyze(native_transport, plan, decoder, shared)
    workflows = {
        "schema_version": 1,
        "role": protocol["role"],
        "cells": {
            cell_id: {
                condition: {
                    key: row[key]
                    for key in (
                        "numerical_methods",
                        "robotruth",
                        "same_numeric_statistics_as_control",
                        "physical_checkpoint_authority_status",
                    )
                }
                for condition, row in conditions.items()
            }
            for cell_id, conditions in shared["cells"].items()
        },
        "transport": transported["cells"],
    }
    counts = analysis.metrics(plan, numerical, witnesses, phases)
    actual = counts["counts"]
    planned = protocol["planned_calls"]
    require(
        actual["primary_decoder_calls"] == planned["primary_decoder"]
        and actual["auxiliary_native_calls"] == planned["roundtrip_auxiliary"]
        and actual["actual_shared_fresh_decoder_calls"] <= planned["fresh_witness_decoder_maximum"]
        and actual["policy_space_replay_queries"] == planned["policy_space_primary_replays"]
        and actual["policy_space_structural_controls"] == planned["policy_space_structural_controls"],
        "complete predeclared actual call denominators",
    )
    cpu = sum(v["runtime"]["process_cpu_seconds"] for v in phases.values())
    require(
        cpu <= protocol["specification"]["resources"]["main_native_process_cpu_seconds_maximum"],
        "main native CPU budget",
    )
    return {
        "analysis.json": numerical,
        "witnesses.json": witnesses,
        "workflow-analysis.json": workflows,
        "metrics.json": counts,
        "conditions.csv": tables(plan, numerical, workflows),
    }


def expected_files(plan):
    return {
        "attempt-start.json",
        "attempt-status.json",
        "protocol.json",
        "evaluation-cost.json",
        "reproduction-comparison.json",
        *PRODUCTS,
        *(
            f"{p}.{suffix}"
            for p in PHASES
            for suffix in (
                "request.json",
                "native.json",
                "stdout.txt",
                "stderr.txt",
                "claim.json",
                "receipt.json",
            )
        ),
        *("contracts/" + name for name in transport.contract_bytes(plan)),
    }


def verify(folder):
    summary = read(folder / "summary.json")
    start = read(folder / "attempt-start.json")
    protocol_path = ROOT / start["protocol_path"]
    protocol, plan = verify_protocol(protocol_path, start["freeze_commit"])
    verify_attempt(folder, protocol, start["freeze_commit"])
    require(
        type(summary["schema_version"]) is int
        and summary["schema_version"] == 1
        and summary["role"] == protocol["role"]
        and summary["freeze_commit"] == start["freeze_commit"]
        and summary["protocol_sha256"] == sha(protocol_path)
        and summary["source_file_sha256"] == protocol["source_file_sha256"]
        and set(summary["file_sha256"]) == expected_files(plan)
        and all(sha(folder / n) == digest for n, digest in summary["file_sha256"].items()),
        "closed exact study evidence",
    )
    require(
        {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
        - {"README.md", "summary.json"}
        == expected_files(plan),
        "closed study directory",
    )
    require(
        read(folder / "attempt-status.json") == {"status": "SUCCEEDED", "completed_phases": list(PHASES)},
        "all phases actually completed",
    )
    for name, value in products(folder, protocol_path, protocol, plan, start["freeze_commit"]).items():
        require(
            (folder / name).read_bytes() == value
            if isinstance(value, bytes)
            else fingerprint(read(folder / name)) == fingerprint(value),
            "regenerated complete study product: " + name,
        )
    cost = read(folder / "evaluation-cost.json")
    require(
        fingerprint(read(folder / "reproduction-comparison.json"))
        == fingerprint(reproduction_comparison(folder)),
        "actual semantic clean reproduction comparison",
    )
    require(
        all(finite(cost[k]) and cost[k] >= 0 for k in ("wall_seconds", "process_cpu_seconds")),
        "actual evaluator cost",
    )
    require(
        not any(
            m in sys.modules for m in ("torch", "lerobot", "openpi", "robotruth", "numpy", "policy_space")
        ),
        "data-only final replay",
    )
    after, _ = verify_protocol(protocol_path, start["freeze_commit"])
    require(
        fingerprint(after) == fingerprint(protocol), "source/configuration/input stability after final replay"
    )
    return summary


def environment(python):
    code = (
        "import importlib.metadata,json,platform,sys; from pathlib import Path; "
        "print(json.dumps({'python':platform.python_version(),'prefix':str(Path(sys.prefix).resolve()),"
        "'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}))"
    )
    result = subprocess.run(
        [str(python.resolve()), "-I", "-c", code], capture_output=True, timeout=30, check=True
    )
    return json.loads(result.stdout)


def reproduction_comparison(folder):
    start = read(folder / "attempt-start.json")
    if start["reproduction_of"] is None:
        return {"schema_version": 1, "status": "NOT_A_REPRODUCTION"}
    original = ROOT / start["reproduction_of"]["path"]
    verify(original)
    names = ("analysis.json", "witnesses.json", "workflow-analysis.json", "conditions.csv")
    require(
        all((folder / n).read_bytes() == (original / n).read_bytes() for n in names),
        "clean semantic artifacts reproduce byte for byte",
    )
    left, right = (read(p / "metrics.json") for p in (folder, original))
    left.pop("phase_costs")
    right.pop("phase_costs")
    require(fingerprint(left) == fingerprint(right), "same finite metrics, denominators and sensitivity")
    different = {
        phase: Path(read(folder / f"{phase}.native.json")["runtime"]["environment_prefix"]).resolve()
        != Path(read(original / f"{phase}.native.json")["runtime"]["environment_prefix"]).resolve()
        for phase in PHASES
    }
    require(all(different.values()), "separate native environment prefixes for clean reproduction")
    return {
        "schema_version": 1,
        "status": "SEMANTIC_ARTIFACTS_REPRODUCED",
        "original_summary_sha256": start["reproduction_of"]["summary_sha256"],
        "semantic_file_sha256": {n: sha(folder / n) for n in names},
        "metrics_equal_excluding_measured_phase_costs": True,
        "distinct_native_environment_prefixes": different,
        "scope": (
            "New separately prepared pinned environments; "
            "not an independent research sample or execution attestation"
        ),
    }


def acquire(folder, protocol_path, protocol, commit, phase, python):
    write(folder / f"{phase}.request.json", request(folder, protocol_path, protocol, commit, phase))
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONUTF8="1",
    )
    started = time.perf_counter()
    receipt = {"request_sha256": sha(folder / f"{phase}.request.json")}
    try:
        result = subprocess.run(
            [str(python.resolve()), "-I", str(ROOT / "scripts/publication_worker.py")],
            input=(folder / f"{phase}.request.json").read_bytes(),
            capture_output=True,
            timeout=protocol["specification"]["resources"]["worker_timeout_seconds"],
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        (folder / f"{phase}.stdout.txt").write_bytes(exc.stdout or b"")
        (folder / f"{phase}.stderr.txt").write_bytes(exc.stderr or b"")
        write(
            folder / f"{phase}.receipt.json",
            {
                **receipt,
                "return_code": None,
                "timed_out": True,
                "subprocess_wall_seconds": time.perf_counter() - started,
            },
        )
        raise
    write(
        folder / f"{phase}.receipt.json",
        {
            **receipt,
            "return_code": result.returncode,
            "timed_out": False,
            "subprocess_wall_seconds": time.perf_counter() - started,
        },
    )
    (folder / f"{phase}.stdout.txt").write_bytes(result.stdout)
    (folder / f"{phase}.stderr.txt").write_bytes(result.stderr)
    require(result.returncode == 0, f"native phase {phase}: {result.stderr[-2500:]!r}")
    write(folder / f"{phase}.native.json", json.loads(result.stdout))
    load_phase(folder, protocol_path, protocol, commit, phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--verify", type=Path)
    group.add_argument("--development", action="store_true")
    group.add_argument("--protocol", type=Path)
    parser.add_argument("--freeze-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reproduce", type=Path)
    for name in ("lerobot", "openpi", "robotruth", "policy-space"):
        parser.add_argument("--" + name + "-python", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
        print(
            "Verified frozen study, exact native evidence, all workflows "
            "and fresh original-relation witnesses"
        )
        return
    require(args.output is not None, "explicit new attempt output")
    folder = args.output.resolve()
    require(folder.is_relative_to(ROOT), "workspace-contained attempt output")
    if args.development:
        require(args.freeze_commit is None and args.reproduce is None, "development-only options")
        protocol = build("development")
        protocol_path = folder / "protocol.json"
    else:
        protocol_path = args.protocol.resolve()
        protocol, _ = verify_protocol(protocol_path, args.freeze_commit)
        require(protocol["role"] == "prospective", "explicit prospective protocol")
    interpreters = {
        name: getattr(args, name + "_python") for name in ("lerobot", "openpi", "robotruth", "policy_space")
    }
    require(
        all(p is not None and p.is_file() for p in interpreters.values()),
        "all four explicit isolated interpreters",
    )
    environments = {name: environment(python) for name, python in interpreters.items()}
    if protocol["native_environments"] is not None:
        require(
            {k: {field: v[field] for field in ("python", "packages")} for k, v in environments.items()}
            == protocol["native_environments"],
            "frozen native environments before creating an attempt",
        )
    reproduction = None
    if args.reproduce is not None:
        original = args.reproduce.resolve()
        summary = verify(original)
        require(
            summary["freeze_commit"] == args.freeze_commit, "same frozen original before clean reproduction"
        )
        reproduction = {"path": relative(original), "summary_sha256": sha(original / "summary.json")}
        for engine, actual in environments.items():
            original_prefix = read(original / f"{engine}.native.json")["runtime"]["environment_prefix"]
            require(
                Path(actual["prefix"]).resolve() != Path(original_prefix).resolve(),
                "fresh isolated reproduction environments before attempt creation",
            )
    if protocol["role"] == "prospective":
        key = "primary_output" if reproduction is None else "clean_reproduction_output"
        require(
            relative(folder) == protocol["specification"]["attempt_policy"][key],
            "predeclared single-use prospective output",
        )
    folder.mkdir(parents=True, exist_ok=False)
    if args.development:
        write(protocol_path, protocol)
    else:
        (folder / "protocol.json").write_bytes(protocol_path.read_bytes())
    write(
        folder / "attempt-start.json",
        {
            "schema_version": 1,
            "role": protocol["role"],
            "freeze_commit": args.freeze_commit,
            "protocol_path": relative(protocol_path),
            "protocol_sha256": sha(protocol_path),
            "attempt_path": relative(folder),
            "reproduction_of": reproduction,
            "environments": environments,
        },
    )
    complete = []
    write(folder / "attempt-status.json", {"status": "STARTED", "completed_phases": complete})
    try:
        protocol, plan = verify_protocol(protocol_path, args.freeze_commit)
        for name, raw in transport.contract_bytes(plan).items():
            path = folder / "contracts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        for phase in PHASES:
            python = interpreters[phase.removesuffix("_witness")]
            acquire(folder, protocol_path, protocol, args.freeze_commit, phase, python)
            complete.append(phase)
            require(
                sum(read(folder / f"{p}.native.json")["runtime"]["process_cpu_seconds"] for p in complete)
                <= protocol["specification"]["resources"]["main_native_process_cpu_seconds_maximum"],
                "stop on cumulative native process CPU budget",
            )
            write(folder / "attempt-status.json", {"status": "STARTED", "completed_phases": complete})
        start, cpu_start = time.perf_counter(), time.process_time()
        for name, value in products(folder, protocol_path, protocol, plan, args.freeze_commit).items():
            if isinstance(value, bytes):
                (folder / name).write_bytes(value)
            else:
                write(folder / name, value)
        write(
            folder / "evaluation-cost.json",
            {
                "wall_seconds": time.perf_counter() - start,
                "process_cpu_seconds": time.process_time() - cpu_start,
            },
        )
        write(folder / "reproduction-comparison.json", reproduction_comparison(folder))
        write(folder / "attempt-status.json", {"status": "SUCCEEDED", "completed_phases": complete})
        write(
            folder / "summary.json",
            {
                "schema_version": 1,
                "role": protocol["role"],
                "freeze_commit": args.freeze_commit,
                "protocol_sha256": sha(protocol_path),
                "source_file_sha256": protocol["source_file_sha256"],
                "file_sha256": {name: sha(folder / name) for name in sorted(expected_files(plan))},
            },
        )
        verify(folder)
    except BaseException as exc:
        write(
            folder / "attempt-status.json",
            {
                "status": "FAILED",
                "completed_phases": complete,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print("Completed and verified every declared native phase and regenerated study product")


if __name__ == "__main__":
    main()
