"""Data-only verification of saved matched-evidence development oracle analyses."""

import argparse
import sys
from pathlib import Path

from verify_probe_evidence import verify as verify_captures

import mltools
from mltools.io import digest, local_path, read_json
from mltools.research_compare import compare

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(f"Oracle evidence verification failed: {message}")


def verify(folder):
    folder = folder.resolve()
    summary = read_json(folder / "summary.json")
    require(summary["development_only"] is True and summary["final_evaluation"] is False, "scope")
    source = local_path(ROOT, summary["source_evidence"])
    require(digest(source / "summary.json") == summary["source_summary_sha256"], "capture summary identity")
    verify_captures(source)
    package = Path(mltools.__file__).resolve().parent
    for name, expected in summary["source_file_sha256"].items():
        require(digest(local_path(ROOT, name)) == expected, f"workspace implementation {name}")
        if name.startswith("src/mltools/"):
            require(digest(package / Path(name).name) == expected, f"installed implementation {name}")
    source_rows = {r["case"]: r for r in read_json(source / "summary.json")["cases"]}
    require(
        len(summary["cases"]) == len(source_rows)
        and {r["case"] for r in summary["cases"]} == source_rows.keys(),
        "analysis inventory differs from capture inventory",
    )
    for row in summary["cases"]:
        case = local_path(ROOT / "examples/probes", f"{row['case']}.json")
        trace = local_path(source, source_rows[row["case"]]["trace"])
        analysis = local_path(folder, row["analysis"])
        require(digest(case) == row["source_case_sha256"], "case identity")
        require(digest(trace) == row["source_trace_sha256"], "trace identity")
        require(digest(analysis) == row["analysis_sha256"], "analysis identity")
        report = read_json(analysis)
        require(compare(case, trace) == report, "deterministic current reanalysis")
        require(
            row["original_numeric"] == {k: v["status"] for k, v in report["methods"].items()},
            "numeric summary",
        )
        require(
            row["original_full_request"]
            == {k: v["full_requested_status"] for k, v in report["methods"].items()},
            "full-scope summary",
        )
        partial = report["ablations"]["offset_authority_erased"]["methods"]
        require(
            row["offset_erased_numeric"] == {k: v["status"] for k, v in partial.items()}, "ablation summary"
        )
        require(row["saved_fresh_witness"] == source_rows[row["case"]].get("fresh_witness"), "witness source")
    require(
        not any(module.split(".")[0] in {"torch", "openpi", "lerobot"} for module in sys.modules),
        "data-only verification imported an ML framework",
    )
    return len(summary["cases"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(f"Verified {verify(args.folder)} development analyses, source identities and original captures")


if __name__ == "__main__":
    main()
