"""Execute the labeled development matrix through the public CLI and replay real witnesses."""

import argparse
import contextlib
import io
import json
from pathlib import Path

from mltools.cli import main as cli
from mltools.io import digest, read_json, write_json
from mltools.probe_cases import capture, check, export

ROOT = Path(__file__).resolve().parents[1]


def compact(path, value):
    path.write_text(
        json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New development evidence directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ("lerobot/good", "PASS", "synthetic_correct_control"),
        ("lerobot/moving_anchor", "FAIL", "injected_composition_fault"),
        ("act/metadata", "UNKNOWN", "published_artifact_partial_authority"),
        ("act/wrong_stats.case", "FAIL", "published_artifact_injected_normalization_fault"),
        ("openpi/good", "PASS", "synthetic_correct_control"),
        ("openpi/moving_anchor", "FAIL", "injected_composition_fault"),
    ]
    records = []
    for name, expected, label in cases:
        case_path = ROOT / "examples/probes" / f"{name}.json"
        key = name.replace("/", "_")
        trace_path, report_path = args.output / f"{key}.trace.json", args.output / f"{key}.report.json"
        framework = "openpi" if name.startswith("openpi/") else "lerobot"
        command = [
            "probe",
            str(case_path),
            "--execute",
            framework,
            "--save-trace",
            str(trace_path),
            "--json",
            str(report_path),
        ]
        if framework == "openpi":
            command.extend(["--python", str(args.openpi_python)])
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            exit_code = cli(command)
        report, trace = read_json(report_path), read_json(trace_path)
        if report["status"] != expected or exit_code != {"PASS": 0, "FAIL": 1, "UNKNOWN": 2}[expected]:
            raise AssertionError(f"Unexpected development outcome: {name}: {report['status']}")
        compact(trace_path, trace)
        compact(report_path, report)
        # Also exercise the data-only CLI path with exactly the saved evidence.
        with contextlib.redirect_stdout(io.StringIO()):
            replay_exit = cli(["probe", str(case_path), "--trace", str(trace_path)])
        if replay_exit != exit_code:
            raise AssertionError("Recorded replay disagrees with explicit execution")
        record = {
            "case": name,
            "case_sha256": digest(case_path),
            "label": label,
            "status": report["status"],
            "checks": report["checks"],
            "trace": trace_path.name,
            "trace_sha256": digest(trace_path),
            "report": report_path.name,
            "report_sha256": digest(report_path),
        }
        if expected == "FAIL":
            destination = args.output / f"{key}.regression"
            export(case_path, trace, report, destination)
            manifest = read_json(destination / "export.json")
            witness_case = destination / manifest["witness_case"]
            witness = capture(witness_case, python=args.openpi_python if framework == "openpi" else None)
            witness_report = check(witness_case, witness)
            if witness_report["status"] != "FAIL":
                raise AssertionError("Fresh reduced case did not replay the failure")
            if (
                witness_report["numeric_report"]["first_divergence"]["obligation"]
                != report["numeric_report"]["first_divergence"]["obligation"]
            ):
                raise AssertionError("Witness changed the violated local obligation")
            compact(destination / "witness.trace.json", witness)
            compact(destination / "witness.report.json", witness_report)
            record["fresh_witness"] = {
                "queries": len(witness["outputs"]),
                "status": "FAIL",
                "source_case": manifest["witness_case"],
            }
        records.append(record)
        print(f"{name}: {expected}; CLI, recorded replay and applicable witness verified", flush=True)
    write_json(
        args.output / "summary.json",
        {
            "schema_version": 1,
            "development_only": True,
            "final_evaluation": False,
            "cases": records,
            "scope": [
                "Published ACT statistics are genuinely loaded via native safetensors, no ACT inference",
                "Synthetic temporal profiles and separately labeled injected faults are development cases",
                "Published ACT physical semantics remain UNKNOWN despite matching numeric normalization",
                "ACT boccia selection candidate was not executed or used for tuning",
            ],
        },
    )


if __name__ == "__main__":
    main()
