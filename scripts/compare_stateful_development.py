"""Genuine same-observation old-checker/compiled comparison and fresh witness replay."""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from stateful_fixtures import (
    EXPECTED,
    NUMERIC_FAULTS,
    ROOT,
    inputs,
    intended_reference,
    require,
    scenario,
    sources,
    spec,
    write,
)

from mltools.checker import check
from mltools.io import digest, read_json
from mltools.lerobot import BLOBS, REVISION
from mltools.obligations import Suite, evaluate
from mltools.report import aggregate


def expected_files():
    """The closed evidence inventory, independently reconstructed from the plan."""
    names = {"inputs/" + name for name in inputs()}
    names.update(
        f"inputs/{name}.{method}-witness.bundle.json"
        for name in NUMERIC_FAULTS
        for method in ("old", "compiled")
    )
    names.update(
        f"{stem}.{suffix}"
        for stem in ("full", "witness")
        for suffix in ("request.json", "stdout.txt", "stderr.txt", "capture.json")
    )
    names.update(
        f"{name}.{suffix}.json"
        for name in EXPECTED
        for suffix in ("old-report", "compiled-report", "comparison")
    )
    names.update(
        f"{name}.{suffix}.json"
        for name in NUMERIC_FAULTS
        for suffix in ("witness-suite", "old-witness-report", "compiled-witness-report")
    )
    return names | {"timings.json"}


def require_expected(comparison, name):
    require(
        comparison["original_matched_status"] == comparison["compiled_matched_status"] == EXPECTED[name],
        f"independent expected outcome and parity: {name}",
    )


def require_witness_relation(original, fresh, method):
    expected = copy.deepcopy(original["first_divergence"])
    if method == "old":
        expected["event"] -= original["counterexample"]["start_event"]
    require(
        fresh["first_divergence"] == expected,
        f"fresh {method} witness reproduces original failed relation and values",
    )


def unit_status(bundle, event):
    if bundle["contract"]["authority"] is None:
        return "UNKNOWN"
    actual = event.get("action", {}).get("units")
    expected = {j["name"]: j["unit"] for j in bundle["contract"]["joints"]}
    if actual is None:
        return "UNKNOWN"
    supported = {"degree", "radian", "meter", "range_0_100"}
    known = {n: v for n, v in actual.items() if n in expected and v in supported}
    if any(expected[n] != v for n, v in known.items()):
        return "FAIL"
    return "PASS" if known == expected else "UNKNOWN"


def bound_capture(path, trace):
    bundle = read_json(path)
    require(trace["bundle_sha256"] == digest(path), "actual native bundle binding")
    require(
        trace["execution"]["revision"] == REVISION and trace["execution"]["git_blobs"] == BLOBS,
        "actual native source identity",
    )
    require(
        trace["execution"]["framework"] == "lerobot" and trace["execution"]["dtype"] == "float64",
        "native framework/dtype",
    )
    require(len(trace["events"]) == len(bundle["scenario"]), "actual event count")
    for expected, actual in zip(bundle["scenario"], trace["events"], strict=True):
        require(expected["kind"] == actual["kind"], "actual event kind")
        if expected["kind"] == "reset":
            require(
                actual["anchor_cleared"] is True and actual["queue_remaining"] == 0, "executed reset schedule"
            )
        else:
            names = bundle["adapter"]["state_order"]
            require(
                actual["state_names"] == names
                and actual["state_values"] == [expected["state"][n] for n in names],
                "executed state inputs",
            )
    return bundle


def analyze(path, trace, name, suite=None):
    suite = suite or spec()[0]
    bundle = bound_capture(path, trace)
    require(
        bundle["scenario"] == scenario(suite, [j["name"] for j in bundle["contract"]["joints"]]),
        "actual canonical query schedule",
    )
    units, outputs, selected_events = [], {}, []
    for i, point in enumerate(suite.probes):
        selected = 3 * i + 2
        first, second = trace["events"][selected - 1], trace["events"][selected]
        require(
            first["queue_remaining"] == 1 and second["queue_remaining"] == 0,
            "actual two-dequeue query budget",
        )
        require(first["prediction_horizon"] == first["training_horizon"] == 2, "actual fixture horizon")
        units.append(unit_status(bundle, second))
        values = second["action"]["commands"]
        require(set(values) == set(suite.outputs), "actual named command ports")
        declared = second["action"].get("units") or {}
        expected_units = {j["name"] + ".pos": j["unit"] for j in bundle["contract"]["joints"]}
        outputs[point.id] = [
            values[n] if declared.get(n.removesuffix(".pos")) == expected_units[n] else None
            for n in suite.outputs
        ]
        selected_events.append(selected)
    common = aggregate(units)
    started = time.perf_counter()
    original = check(path, trace)
    old_seconds = time.perf_counter() - started
    projected = [
        f
        for f in original["findings"]
        if f["event"] in selected_events
        and f["boundary"] in ("controller.units", "controller.commands", "controller.commands.known_units")
    ]
    require(len(projected) >= len(suite.probes), "old checker has actual matched-boundary findings")
    started = time.perf_counter()
    compiled = evaluate(
        suite,
        intended_reference(name),
        {"suite_sha256": suite.sha256, "outputs": outputs},
    )
    new_seconds = time.perf_counter() - started
    comparison = {
        "matched_scope": "Same second-dequeue named scalar commands and declared units",
        "queries": len(suite.probes),
        "native_actions": len(suite.probes) * 2,
        "common_authority_and_unit_status": common,
        "original_native_full_status": original["status"],
        "original_matched_status": aggregate([common, aggregate(f["status"] for f in projected)]),
        "compiled_matched_status": aggregate([common, compiled["status"]]),
        "old_projection_findings": projected,
        "scope_limits": (
            "Original full report also checks first actions/queues/horizon; "
            "matched projection is explicit, not a renamed native verdict"
        ),
    }
    return (
        original,
        compiled,
        comparison,
        {"original_check_seconds": old_seconds, "compiled_evaluate_seconds": new_seconds},
    )


def acquire(folder, python, bundles, stem):
    original_sources = sources()
    files = {
        p.relative_to(ROOT).as_posix(): digest(p)
        for p in sorted((folder / "inputs").rglob("*"))
        if p.is_file()
    }
    request = {
        "file_sha256": files,
        "bundles": {k: p.relative_to(ROOT).as_posix() for k, p in bundles.items()},
    }
    write(folder / f"{stem}.request.json", request)
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUTF8="1",
    )
    process = subprocess.run(
        [str(python), "-I", str(ROOT / "scripts/stateful_comparison_worker.py")],
        input=json.dumps(request).encode(),
        capture_output=True,
        env=env,
        timeout=180,
        check=False,
    )
    (folder / f"{stem}.stdout.txt").write_bytes(process.stdout)
    (folder / f"{stem}.stderr.txt").write_bytes(process.stderr)
    require(process.returncode == 0, f"native acquisition: {process.stderr[-1500:]!r}")
    require(sources() == original_sources, "source changed around native dispatch")
    result = json.loads(process.stdout)
    require(
        result["source_file_sha256"] == original_sources and result["input_file_sha256"] == files,
        "native producer/input snapshot",
    )
    require(set(result["captures"]) == set(bundles), "native batch inventory")
    write(folder / f"{stem}.capture.json", result)
    return result


def run(folder, python):
    initial = sources()
    for name, value in inputs().items():
        write(folder / "inputs" / name, value)
    suite, _ref = spec()
    full = acquire(
        folder, python, {name: folder / "inputs" / f"{name}.bundle.json" for name in EXPECTED}, "full"
    )
    rows, timings, witness_inputs = {}, {}, {}
    for name in EXPECTED:
        path = folder / "inputs" / f"{name}.bundle.json"
        old, compiled, comparison, timing = analyze(path, full["captures"][name], name)
        require_expected(comparison, name)
        write(folder / f"{name}.old-report.json", old)
        write(folder / f"{name}.compiled-report.json", compiled)
        write(folder / f"{name}.comparison.json", comparison)
        rows[name], timings[name] = comparison, timing
        if name not in NUMERIC_FAULTS:
            continue
        prefix = copy.deepcopy(read_json(path))
        prefix["scenario"] = old["counterexample"]["events"]
        old_path = folder / "inputs" / f"{name}.old-witness.bundle.json"
        write(old_path, prefix)
        witness_inputs[name + ":old"] = old_path
        query_ids = compiled["first_divergence"]["queries"]
        subset = Suite(suite.axes, suite.outputs, tuple(p for p in suite.probes if p.id in query_ids))
        reduced = copy.deepcopy(read_json(path))
        reduced["scenario"] = scenario(subset, [j["name"] for j in reduced["contract"]["joints"]])
        new_path = folder / "inputs" / f"{name}.compiled-witness.bundle.json"
        write(new_path, reduced)
        witness_inputs[name + ":compiled"] = new_path
        write(folder / f"{name}.witness-suite.json", subset.to_dict())
    fresh = acquire(folder, python, witness_inputs, "witness")
    witness_rows = {}
    for name in NUMERIC_FAULTS:
        old_trace = fresh["captures"][name + ":old"]
        old_path = witness_inputs[name + ":old"]
        bound_capture(old_path, old_trace)
        old_report = check(old_path, old_trace)
        require(old_report["status"] == "FAIL", "fresh original witness failure")
        require_witness_relation(read_json(folder / f"{name}.old-report.json"), old_report, "old")
        subset = Suite.from_dict(read_json(folder / f"{name}.witness-suite.json"))
        new_trace = fresh["captures"][name + ":compiled"]
        _old, new_report, matched, _timing = analyze(
            witness_inputs[name + ":compiled"], new_trace, name, subset
        )
        require(matched["compiled_matched_status"] == "FAIL", "fresh compiled witness failure")
        require_witness_relation(read_json(folder / f"{name}.compiled-report.json"), new_report, "compiled")
        write(folder / f"{name}.old-witness-report.json", old_report)
        write(folder / f"{name}.compiled-witness-report.json", new_report)
        witness_rows[name] = {
            "original_native_actions": sum(e["kind"] == "tick" for e in old_trace["events"]),
            "compiled_native_actions": sum(e["kind"] == "tick" for e in new_trace["events"]),
            "original_fresh_status": old_report["status"],
            "compiled_fresh_status": matched["compiled_matched_status"],
            "original_scope": "Existing reset-prefix exporter",
            "compiled_scope": "Existing local numerical witness queries",
        }
    require(sources() == initial, "stable comparison implementation")
    write(folder / "timings.json", timings)
    write(
        folder / "summary.json",
        {
            "schema_version": 1,
            "development_only": True,
            "final_evaluation": False,
            "capture_root_relative": folder.relative_to(ROOT).as_posix(),
            "source_file_sha256": initial,
            "cases": rows,
            "fresh_witnesses": witness_rows,
            "framework_scope": (
                "LeRobot named scalar SO101 software fixture; "
                "OpenPI end-effector/raw metadata domains not force-mapped"
            ),
            "unit": "One reused integration cluster; faults/probes/witnesses are repeated measurements",
            "file_sha256": {
                p.relative_to(folder).as_posix(): digest(p) for p in sorted(folder.rglob("*")) if p.is_file()
            },
        },
    )
    verify(folder)


def verify(folder):
    summary = read_json(folder / "summary.json")
    require(
        summary["schema_version"] == 1
        and summary["development_only"] is True
        and summary["final_evaluation"] is False,
        "development scope",
    )
    require(summary["source_file_sha256"] == sources(), "project source identity")
    require(set(summary["file_sha256"]) == expected_files(), "complete closed evidence manifest")
    for name, sha in summary["file_sha256"].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()) and digest(path) == sha, f"saved file identity: {name}")
    fixed = inputs()
    capture_root = Path(summary["capture_root_relative"])
    require(
        not capture_root.is_absolute() and ".." not in capture_root.parts, "relative recorded capture root"
    )
    for name, value in fixed.items():
        require(read_json(folder / "inputs" / name) == value, "fixed independent fixture inputs")
    require(set(summary["cases"]) == set(EXPECTED), "case inventory")
    require(set(summary["fresh_witnesses"]) == set(NUMERIC_FAULTS), "fresh witness inventory")
    full, fresh = read_json(folder / "full.capture.json"), read_json(folder / "witness.capture.json")
    for stem, batch in (("full", full), ("witness", fresh)):
        require(
            batch["schema_version"] == 1
            and batch["development_only"] is True
            and batch["final_evaluation"] is False,
            "native batch development scope",
        )
        require(
            json.loads((folder / f"{stem}.stdout.txt").read_bytes()) == batch,
            "exact native stdout/parsed capture",
        )
        request = read_json(folder / f"{stem}.request.json")
        names = list(fixed)
        if stem == "witness":
            names.extend(
                f"{name}.{method}-witness.bundle.json"
                for name in NUMERIC_FAULTS
                for method in ("old", "compiled")
            )
        expected_inputs = {
            (capture_root / "inputs" / name).as_posix(): digest(folder / "inputs" / name) for name in names
        }
        expected_bundles = (
            {name: (capture_root / "inputs" / f"{name}.bundle.json").as_posix() for name in EXPECTED}
            if stem == "full"
            else {
                f"{name}:{method}": (
                    capture_root / "inputs" / f"{name}.{method}-witness.bundle.json"
                ).as_posix()
                for name in NUMERIC_FAULTS
                for method in ("old", "compiled")
            }
        )
        require(
            request == {"file_sha256": expected_inputs, "bundles": expected_bundles},
            "complete fixed native batch request",
        )
        require(set(batch["captures"]) == set(expected_bundles), "complete native capture batch")
        require(
            batch["source_file_sha256"] == sources() and batch["input_file_sha256"] == request["file_sha256"],
            "producer snapshot identity",
        )
        for name, sha in request["file_sha256"].items():
            require(
                digest(folder / Path(name).relative_to(capture_root)) == sha, "native request input identity"
            )
        require(batch["runtime"]["network_events"] == [], "native network attempts")
        require(set(batch["acquisition_seconds"]) == set(expected_bundles), "native timing inventory")
        require(
            all(
                isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
                for value in (
                    *batch["acquisition_seconds"].values(),
                    batch["runtime"]["imports_seconds"],
                    batch["runtime"]["wall_seconds"],
                )
            ),
            "finite measured durations",
        )
    timings = read_json(folder / "timings.json")
    require(set(timings) == set(EXPECTED), "analysis timing inventory")
    for row in timings.values():
        require(set(row) == {"original_check_seconds", "compiled_evaluate_seconds"}, "analysis timing scope")
        require(
            all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in row.values()),
            "finite analysis timings",
        )
    for name in EXPECTED:
        old, compiled, comparison, _timing = analyze(
            folder / "inputs" / f"{name}.bundle.json", full["captures"][name], name
        )
        require_expected(comparison, name)
        require(
            old == read_json(folder / f"{name}.old-report.json")
            and compiled == read_json(folder / f"{name}.compiled-report.json"),
            "deterministic actual method reports",
        )
        require(
            comparison == summary["cases"][name] == read_json(folder / f"{name}.comparison.json"),
            "matched comparison binding",
        )
    for name in NUMERIC_FAULTS:
        old_path = folder / "inputs" / f"{name}.old-witness.bundle.json"
        original = read_json(folder / f"{name}.old-report.json")
        expected_old = copy.deepcopy(fixed[f"{name}.bundle.json"])
        expected_old["scenario"] = original["counterexample"]["events"]
        require(read_json(old_path) == expected_old, "complete original exported witness bundle")
        old_trace = fresh["captures"][name + ":old"]
        bound_capture(old_path, old_trace)
        old_report = check(old_path, old_trace)
        require_witness_relation(original, old_report, "old")
        require(
            old_report == read_json(folder / f"{name}.old-witness-report.json")
            and old_report["status"] == "FAIL",
            "fresh original witness replay",
        )
        selected = read_json(folder / f"{name}.compiled-report.json")["first_divergence"]["queries"]
        suite = Suite.from_dict(read_json(folder / f"{name}.witness-suite.json"))
        original_suite = spec()[0]
        expected_suite = Suite(
            original_suite.axes,
            original_suite.outputs,
            tuple(p for p in original_suite.probes if p.id in selected),
        )
        require(suite == expected_suite, "complete compiler witness suite")
        expected_new = copy.deepcopy(fixed[f"{name}.bundle.json"])
        expected_new["scenario"] = scenario(suite, [j["name"] for j in expected_new["contract"]["joints"]])
        require(
            read_json(folder / "inputs" / f"{name}.compiled-witness.bundle.json") == expected_new,
            "complete compiled witness bundle",
        )
        _old, report, matched, _timing = analyze(
            folder / "inputs" / f"{name}.compiled-witness.bundle.json",
            fresh["captures"][name + ":compiled"],
            name,
            suite,
        )
        require_witness_relation(read_json(folder / f"{name}.compiled-report.json"), report, "compiled")
        require(
            report == read_json(folder / f"{name}.compiled-witness-report.json")
            and matched["compiled_matched_status"] == "FAIL",
            "fresh compiler witness replay",
        )
        require(
            summary["fresh_witnesses"][name]
            == {
                "original_native_actions": sum(e["kind"] == "tick" for e in old_trace["events"]),
                "compiled_native_actions": len(suite.probes) * 2,
                "original_fresh_status": old_report["status"],
                "compiled_fresh_status": matched["compiled_matched_status"],
                "original_scope": "Existing reset-prefix exporter",
                "compiled_scope": "Existing local numerical witness queries",
            },
            "fresh witness outcome and cost binding",
        )
    require(not any(m in sys.modules for m in ("torch", "lerobot", "openpi")), "data-only analysis")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
    else:
        require(args.python is not None, "explicit pinned native interpreter")
        folder = args.output.resolve()
        require(folder.is_relative_to(ROOT), "workspace-contained capture destination")
        folder.mkdir(parents=True, exist_ok=False)
        run(folder, args.python.resolve())
    print("Verified same-observation original-stateful/compiled development comparison and fresh witnesses")


if __name__ == "__main__":
    main()
