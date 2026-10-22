"""Persist real OpenPI development captures; never writes final benchmark results."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from mltools.io import write_json
from mltools.obligations import Axis, Reference, Suite, compile_suite, evaluate
from mltools.openpi import capture


def synthetic_fixture():
    axes = tuple(
        Axis(port, str(i), "synthetic_normalized" if port == "action" else "synthetic_state", -r, r)
        for port, n, r in (("action", 7, 1), ("origin", 8, 0.25), ("current", 8, 0.25))
        for i in range(n)
    )
    suite = compile_suite(axes, tuple(f"command.{i}" for i in range(7)))
    stats = {
        "norm_stats": {
            name: {
                "mean": [0.3 * (i + 1) for i in range(n)],
                "std": [0.5 + i * 0.1 for i in range(n)],
                "q01": [-2 - i * 0.2 for i in range(n)],
                "q99": [3 + i * 0.3 for i in range(n)],
            }
            for name, n in (("actions", 7), ("state", 8))
        }
    }
    return suite, json.dumps(stats)


def synthetic_reference(suite, quantiles):
    scales = [(5 + i * 0.5 + 1e-6) / 2 if quantiles else 0.5 + i * 0.1 + 1e-6 for i in range(7)]
    offsets = [0.5 + i * 0.05 + 0.5e-6 if quantiles else 0.3 * (i + 1) for i in range(7)]
    coefficients = tuple(
        tuple(
            scales[i]
            if a.port == "action" and a.name == str(i)
            else 0.25
            if a.port == "origin" and a.name == str(i) and i < 6
            else 0.0
            for a in suite.axes
        )
        for i in range(7)
    )
    return Reference(
        coefficients,
        tuple(offsets),
        ("synthetic_command",) * 7,
        "Hand-authored development fixture; not checkpoint ground truth",
        True,
    )


def legacy_fixture():
    # Independently chosen native controller vectors, not learned predictions.
    origin = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.02, -0.02]
    desired = [0.25, -0.5, 0.75, 0.1, -0.2, 0.3, -0.75]
    residual = [0.15, -0.3, 0.45, -0.3, 0.3, -0.3, -0.75]
    axes = tuple(
        Axis(
            port,
            str(i),
            "legacy_model_coordinate"
            if port == "action"
            else "sim_position_coordinate"
            if i < 3
            else "radian"
            if i < 6
            else "gripper_joint_coordinate",
            center - 0.125,
            center + 0.125,
        )
        for port, values in (("action", residual), ("origin", origin), ("current", origin))
        for i, center in enumerate(values)
    )
    suite = compile_suite(axes, tuple(f"controller.{i}" for i in range(7)))
    coefficients = tuple(
        tuple(
            0.125 if a.name == str(i) and (a.port == "action" or (a.port == "origin" and i < 6)) else 0.0
            for a in axes
        )
        for i in range(7)
    )
    ref = Reference(
        coefficients,
        tuple(desired),
        ("normalized_controller_command",) * 7,
        "Merged OpenPI PR351 legacy composition plus independent numeric fixture",
        True,
    )
    # Identity inverse isolates the restored action conversion. These are
    # synthetic statistics, not the unknown historical checkpoint statistics.
    stats = json.dumps(
        {
            "norm_stats": {
                name: {"mean": [0.0] * n, "std": [1 - 1e-6] * n} for name, n in (("actions", 7), ("state", 8))
            }
        }
    )
    return suite, ref, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="Pinned OpenPI worker interpreter")
    parser.add_argument("--output", type=Path, required=True, help="New development evidence directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    rows = []

    def run(name, suite, reference, stats, expected, label, **configuration):
        trace = capture(suite, stats, python=args.python, **configuration)
        report = evaluate(suite, reference, trace)
        record = {
            "schema_version": 1,
            "development_only": True,
            "label": label,
            "suite": suite.to_dict(),
            "reference": asdict(reference),
            "stats_json": stats,
            "configuration": configuration,
            "expected_status": expected,
        }
        write_json(args.output / f"{name}.case.json", record)
        write_json(args.output / f"{name}.trace.json", trace)
        write_json(args.output / f"{name}.report.json", report)
        if report["status"] != expected:
            raise AssertionError(f"{name}: expected {expected}, got {report['status']}")
        rows.append(
            {
                "case": name,
                "label": label,
                "status": report["status"],
                "queries": len(suite.probes),
                "runtime": trace["runtime"],
                "first_divergence": report["first_divergence"],
            }
        )
        print(f"{name}: {report['status']} ({len(suite.probes)} queries)", flush=True)
        return report

    suite, stats = synthetic_fixture()
    for model, quantiles in (("pi0", False), ("pi05", True), ("pi0_fast", True)):
        run(
            model,
            suite,
            synthetic_reference(suite, quantiles),
            stats,
            "PASS",
            "synthetic_correct_control",
            model_type=model,
            use_quantiles=quantiles,
        )
    partial = replace(synthetic_reference(suite, False), offset=(None,) * 7)
    failure = run(
        "moving_anchor", suite, partial, stats, "FAIL", "injected_composition_fault", anchor="current"
    )
    identifiers = {row["query"] for row in failure["witness"]}
    reduced = Suite(suite.axes, suite.outputs, tuple(p for p in suite.probes if p.id in identifiers))
    replay = run(
        "moving_anchor_witness",
        reduced,
        partial,
        stats,
        "FAIL",
        "development_witness_replay",
        anchor="current",
    )
    if replay["first_divergence"]["obligation"] != failure["first_divergence"]["obligation"]:
        raise AssertionError("Fresh-worker witness failed to retain the same local violation")
    suite, reference, stats = legacy_fixture()
    run(
        "pr351_omitted",
        suite,
        reference,
        stats,
        "FAIL",
        "retrospective_processor_reconstruction",
        mask=(False,) * 7,
    )
    run("pr351_restored", suite, reference, stats, "PASS", "retrospective_legacy_correct_control")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.glob("*.json"))}
    write_json(
        args.output / "summary.json",
        {
            "schema_version": 1,
            "development_only": True,
            "final_evaluation": False,
            "cases": rows,
            "file_sha256": files,
            "wall_seconds": time.perf_counter() - started,
            "scope": [
                "Genuine pinned processor execution, no trained weights or policy inference",
                "Synthetic fixtures and explicitly labeled historical composition reconstruction",
                "No new benchmark detection rate, robot competence or hardware safety claim",
                "Each capture is a fresh process; filesystem caches may already be warm",
                "Worker RSS is observed and checked after execution, not an enforced OS memory limit",
            ],
        },
    )


if __name__ == "__main__":
    main()
