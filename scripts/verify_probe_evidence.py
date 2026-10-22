"""Verify saved development CLI captures and fresh witnesses without importing ML."""

import argparse
from pathlib import Path

from mltools.io import digest, local_path, read_json
from mltools.probe_cases import check

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(f"Development evidence verification failed: {message}")


def verify(folder):
    folder = folder.resolve()
    summary = read_json(folder / "summary.json")
    require(summary["development_only"] is True and summary["final_evaluation"] is False, "scope")
    records = []
    for row in summary["cases"]:
        case = local_path(ROOT / "examples/probes", f"{row['case']}.json")
        trace = local_path(folder, row["trace"])
        report = local_path(folder, row["report"])
        for file, expected in (
            (case, row["case_sha256"]),
            (trace, row["trace_sha256"]),
            (report, row["report_sha256"]),
        ):
            require(digest(file) == expected, f"{row['case']} identity of {file.name}")
        saved = read_json(report)
        require(check(case, trace) == saved, f"{row['case']} current offline replay")
        require(saved["status"] == row["status"] and saved["checks"] == row["checks"], "summary binding")
        if "fresh_witness" in row:
            export_root = folder / f"{row['case'].replace('/', '_')}.regression"
            manifest = read_json(export_root / "export.json")
            original = local_path(export_root, manifest["offline_case"])
            original_trace = local_path(export_root, manifest["offline_trace"])
            require(check(original, original_trace) == saved, "original regression replay")
            witness = local_path(export_root, manifest["witness_case"])
            witness_trace = read_json(export_root / "witness.trace.json")
            witness_report = read_json(export_root / "witness.report.json")
            require(check(witness, witness_trace) == witness_report, "fresh witness replay")
            require(witness_report["status"] == row["fresh_witness"]["status"] == "FAIL", "witness status")
            require(len(witness_trace["outputs"]) == row["fresh_witness"]["queries"], "witness query count")
            require(
                witness_report["numeric_report"]["first_divergence"]["obligation"]
                == saved["numeric_report"]["first_divergence"]["obligation"],
                "witness relation identity",
            )
        records.append({"case": row["case"], "status": row["status"]})
    require(
        len(records) == 6 and len({r["case"] for r in records}) == 6, "complete six-case development matrix"
    )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    for row in verify(args.folder):
        print(f"{row['case']}: {row['status']}; identities, replay and applicable witnesses verified")


if __name__ == "__main__":
    main()
