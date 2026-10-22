"""Replay recorded development captures through the current affine evaluator, without ML imports."""

import argparse
import hashlib
from pathlib import Path

from mltools import obligations
from mltools.io import read_json, write_json
from mltools.obligations import Reference, Suite, evaluate


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = read_json(args.input / "summary.json")
    for name, expected in original["file_sha256"].items():
        if Path(name).name != name or sha(args.input / name) != expected:
            raise AssertionError(f"Recorded development artifact changed: {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(args.input.glob("*.case.json")):
        name = path.name.removesuffix(".case.json")
        case = read_json(path)
        trace = read_json(args.input / f"{name}.trace.json")
        report = evaluate(Suite.from_dict(case["suite"]), Reference(**case["reference"]), trace)
        if report["status"] != case["expected_status"]:
            raise AssertionError(f"Reanalysis changed intended development outcome: {name}")
        write_json(args.output / f"{name}.report.json", report)
        rows.append(
            {
                "case": name,
                "status": report["status"],
                "label": case["label"],
                "trace_file_sha256": sha(args.input / f"{name}.trace.json"),
                "report_file_sha256": sha(args.output / f"{name}.report.json"),
            }
        )
    write_json(
        args.output / "summary.json",
        {
            "schema_version": 1,
            "development_only": True,
            "final_evaluation": False,
            "input_summary_sha256": sha(args.input / "summary.json"),
            "evaluator_sha256": sha(Path(obligations.__file__)),
            "cases": rows,
            "scope": "Offline reanalysis of existing captures; original bytes and measurements preserved",
            "capture_limit": "Evolving development worktree; exact initial worker hashes unrecorded",
        },
    )
    print(f"Reanalyzed {len(rows)} development captures without ML imports")


if __name__ == "__main__":
    main()
