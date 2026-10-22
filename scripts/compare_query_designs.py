"""Acquire/replay equal-budget development designs; never runs final benchmark cases."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from query_designs import analyze, fingerprint, generate, require

import mltools
from mltools.io import digest, local_path, read_json, write_json
from mltools.probe_cases import _sources, load
from mltools.report import aggregate

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/query_design_worker.py"
CASES = (
    "lerobot/good",
    "lerobot/moving_anchor",
    "openpi/good",
    "openpi/moving_anchor",
    "act/metadata",
    "act/wrong_stats.case",
)
SEEDS = (7, 23, 101)


def source_hashes():
    modules = list((ROOT / "src/mltools").glob("*.py"))
    package = Path(mltools.__file__).parent
    require(
        all((package / p.name).read_bytes() == p.read_bytes() for p in modules), "installed MLTools differs"
    )
    paths = modules + [
        WORKER,
        Path(__file__),
        ROOT / "scripts/query_designs.py",
        ROOT / "uv.lock",
        ROOT / "integrations/openpi/requirements-cpu.lock",
        ROOT / "integrations/openpi/frameworks.txt",
        ROOT / "integrations/openpi/build.lock",
    ]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def designs(suite):
    result = {method: generate(suite.to_dict(), method) for method in ("compiled", "manual_axis")}
    result.update(
        {
            f"{method}-{seed}": generate(suite.to_dict(), method, seed)
            for seed in SEEDS
            for method in ("random_axis", "random_dense")
        }
    )
    return result


def guarded_analysis(case_path, plan, trace):
    case, root, _suite, reference = load(case_path)
    adapter = case["adapter"]
    guards = []

    def compare(label, expected, actual):
        status = "UNKNOWN" if expected is None or actual is None else "PASS" if expected == actual else "FAIL"
        guards.append({"field": label, "status": status, "expected": expected, "actual": actual})

    compare("case_sha256", digest(case_path), trace.get("case_sha256"))
    compare("design_sha256", plan.sha256, trace.get("design_sha256"))
    compare("schedule_sha256", fingerprint(plan.schedule), trace.get("schedule_sha256"))
    compare("framework", adapter["framework"], trace.get("framework"))
    compare("configuration", adapter["configuration"], trace.get("configuration"))
    revision, blobs, client = _sources(adapter["framework"])
    compare("revision", revision, trace.get("identity", {}).get("revision"))
    for name, blob in blobs.items():
        compare(f"source:{name}", blob, trace.get("identity", {}).get("git_blobs", {}).get(name))
    if client:
        compare("client_image_tools_blob", client, trace.get("identity", {}).get("client_image_tools_blob"))
    for artifact in case["artifacts"]:
        path = local_path(root, artifact["path"])
        actual = digest(path) if path.is_file() else None
        compare(f"artifact:{artifact['id']}", artifact["sha256"], actual)
        if artifact["id"] == adapter["stats_artifact"]:
            compare("executed_stats_sha256", actual, trace.get("stats_sha256"))
    rows = trace.get("traces") or []
    by_id = {row["query"]: row for row in rows}
    schedule_checks, schedule_failure = [], None

    def schedule_compare(field, expected, actual):
        nonlocal schedule_failure
        status = "UNKNOWN" if actual is None else "PASS" if expected == actual else "FAIL"
        schedule_checks.append(status)
        if status != "PASS" and schedule_failure is None:
            schedule_failure = {"field": field, "status": status, "expected": expected, "actual": actual}

    schedule_compare("query_inventory", sorted(p.id for p in plan.probes), sorted(by_id))
    schedule_compare("query_row_count", len(plan.probes), len(rows))
    for point in plan.probes:
        row = by_id.get(point.id, {})
        coordinates = {(a.port, a.name): value for a, value in zip(plan.axes, point.values, strict=True)}
        for port, field in (
            ("origin", "origin"),
            ("current", "current"),
            ("action", "normalized_prediction"),
        ):
            expected = [
                coordinates[(a.port, a.name)]
                for a in sorted(plan.axes, key=lambda a: int(a.name))
                if a.port == port
            ]
            schedule_compare(f"{point.id}:{field}", expected, row.get(field))
        schedule_compare(
            f"{point.id}:output", trace.get("outputs", {}).get(point.id), row.get("second_action")
        )
        schedule_compare(f"{point.id}:queue_remaining", 0, row.get("queue_remaining"))
        if adapter["framework"] == "lerobot":
            schedule_compare(f"{point.id}:reset_cleared", True, row.get("reset_cleared_before_query"))
    guards.append(
        {
            "field": "executed_queries_and_schedule",
            "status": aggregate(schedule_checks),
            "comparisons": len(schedule_checks),
            "first_divergence": schedule_failure,
        }
    )
    guard_status = aggregate(row["status"] for row in guards)
    evidence = {"design_sha256": plan.sha256, "outputs": trace["outputs"] if guard_status == "PASS" else {}}
    numeric = analyze(plan, reference.to_dict(), evidence)
    unsupported = [
        c for c in case.get("checks", []) if c not in ("binding", "producer", "artifacts", "obligations")
    ]
    return {
        "guards": guards,
        "guard_status": guard_status,
        "numeric": numeric,
        "unsupported_requested_checks": unsupported,
        "full_requested_status": aggregate(
            [guard_status, numeric["full_affine_request_status"], *(["UNKNOWN"] if unsupported else [])]
        ),
    }


def verify_case(folder, name):
    require(name in CASES, "development case allowlist")
    case_path = ROOT / "examples/probes" / f"{name}.json"
    case, root, suite, _ref = load(case_path)
    stem = name.replace("/", "_")
    request_path = folder / f"{stem}.request.json"
    request = read_json(request_path)
    plans = designs(suite)
    require(
        request["designs"] == json.loads(json.dumps({k: p.to_dict() for k, p in plans.items()})),
        "generated design identities",
    )
    require(request["case_path"] == case_path.relative_to(ROOT).as_posix(), "requested source case")
    require(request["case_sha256"] == digest(case_path), "case identity")
    artifact = next(a for a in case["artifacts"] if a["id"] == case["adapter"]["stats_artifact"])
    require(
        request["stats_sha256"] == digest(local_path(root, artifact["path"])), "requested statistics identity"
    )
    capture = read_json(folder / f"{stem}.capture.json")
    require(
        capture["development_only"] is True and capture["final_evaluation"] is False,
        "development capture scope",
    )
    require(capture["request_sha256"] == digest(request_path), "capture request binding")
    expected_sources = {
        k: v
        for k, v in source_hashes().items()
        if k.startswith("src/") or k in ("scripts/query_design_worker.py", "scripts/query_designs.py")
    }
    require(capture["source_file_sha256"] == expected_sources, "capture source identity")
    require(set(capture["measurements"]) == set(plans), "complete design batch")
    require(capture["runtime"]["network_events"] == [], "network attempts")
    require(capture["runtime"]["peak_working_set_bytes"] <= 2 * 1024**3, "observed resource budget")
    reports = {
        key: guarded_analysis(case_path, plan, capture["measurements"][key]) for key, plan in plans.items()
    }
    require(reports == read_json(folder / f"{stem}.analysis.json"), "deterministic guarded analysis")
    for key, plan in plans.items():
        require(
            set(capture["measurements"][key]["outputs"]) == {p.id for p in plan.probes},
            "executed query budget",
        )
        require(reports[key]["guard_status"] == "PASS", "bound native evidence")
    return {
        "case": name,
        "queries_per_design": len(next(iter(plans.values())).probes),
        "full_requested": {k: r["full_requested_status"] for k, r in reports.items()},
        "numeric": {k: r["numeric"]["full_affine_request_status"] for k, r in reports.items()},
    }


def verify(folder):
    summary = read_json(folder / "summary.json")
    require(summary["development_only"] is True and summary["final_evaluation"] is False, "summary scope")
    require(summary["source_file_sha256"] == source_hashes(), "source identities")
    for name, sha in summary["file_sha256"].items():
        require(digest(folder / name) == sha, f"file identity: {name}")
    require(
        [verify_case(folder, row["case"]) for row in summary["cases"]] == summary["cases"], "summary outcomes"
    )
    require(not any(m in sys.modules for m in ("torch", "openpi", "lerobot")), "data-only replay")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--output", type=Path)
    source.add_argument("--verify", type=Path)
    parser.add_argument("--lerobot-python", type=Path)
    parser.add_argument("--openpi-python", type=Path)
    parser.add_argument("--case", choices=CASES, action="append")
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
        print(f"Verified {len(result['cases'])} development query-design batches without ML imports")
        return
    require(args.lerobot_python and args.openpi_python, "explicit pinned interpreters required")
    args.output.mkdir(parents=True, exist_ok=False)
    initial_sources = source_hashes()
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUTF8="1",
        PYTHONNOUSERSITE="1",
    )
    rows = []
    for name in args.case or CASES:
        require(source_hashes() == initial_sources, "source changed before dispatch")
        path = ROOT / "examples/probes" / f"{name}.json"
        case, root, suite, _ref = load(path)
        plans = designs(suite)
        artifact = next(a for a in case["artifacts"] if a["id"] == case["adapter"]["stats_artifact"])
        request = {
            "case_path": path.relative_to(ROOT).as_posix(),
            "case_sha256": digest(path),
            "stats_sha256": digest(local_path(root, artifact["path"])),
            "designs": {key: plan.to_dict() for key, plan in plans.items()},
        }
        stem = name.replace("/", "_")
        request_path = args.output / f"{stem}.request.json"
        write_json(request_path, request)
        interpreter = args.lerobot_python if case["adapter"]["framework"] == "lerobot" else args.openpi_python
        process = subprocess.run(
            [str(interpreter), "-I", str(WORKER)],
            input=request_path.read_bytes(),
            capture_output=True,
            env=env,
            timeout=180,
            check=False,
        )
        (args.output / f"{stem}.stdout.txt").write_bytes(process.stdout)
        (args.output / f"{stem}.stderr.txt").write_bytes(process.stderr)
        require(
            process.returncode == 0,
            f"native worker {name}: {process.stdout[:1000]!r} {process.stderr[:1000]!r}",
        )
        require(source_hashes() == initial_sources, "sources changed during capture")
        captured = json.loads(process.stdout)
        write_json(args.output / f"{stem}.capture.json", captured)
        started = time.perf_counter()
        reports, timings = {}, {}
        for key, plan in plans.items():
            analysis_started = time.perf_counter()
            reports[key] = guarded_analysis(path, plan, captured["measurements"][key])
            timings[key] = time.perf_counter() - analysis_started
        write_json(args.output / f"{stem}.analysis.json", reports)
        write_json(
            args.output / f"{stem}.analysis-timing.json",
            {
                "seconds": timings,
                "scope": "One analysis per design before verification; not a repeated timing estimate",
            },
        )
        row = verify_case(args.output, name)
        rows.append(row)
        print(f"{name}: {row['numeric']}; analysis {time.perf_counter() - started:.2f}s", flush=True)
    result = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "source_file_sha256": initial_sources,
        "file_sha256": {p.name: digest(p) for p in sorted(args.output.iterdir()) if p.is_file()},
        "cases": rows,
        "seeds": list(SEEDS),
        "unit": "Integration/checkpoint/root-cause cluster; seeds and probes are repeated measurements",
        "scope": "New genuine processor queries; unfrozen development only, no superiority/speed claim",
        "comparison_scope": "Query designs share the same research oracle; compiled denotes query geometry",
    }
    write_json(args.output / "summary.json", result)
    verify(args.output)


if __name__ == "__main__":
    main()
