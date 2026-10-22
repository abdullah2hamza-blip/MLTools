"""Replay genuine development captures with matched-evidence numerical oracles."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from verify_probe_evidence import verify

import mltools
from mltools.io import digest, local_path, read_json, write_json
from mltools.research_compare import compare

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(f"Development oracle analysis failed: {message}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "validation/probe-cases-002")
    parser.add_argument("--output", type=Path, required=True, help="New analysis directory")
    args = parser.parse_args()
    source = args.evidence.resolve()
    verify(source)  # Identities, original replay and saved freshly executed witnesses.
    # Verify installed-wheel and workspace source equality before claiming clean replay.
    package = Path(mltools.__file__).resolve().parent
    source_hashes = {}
    for path in sorted((ROOT / "src/mltools").glob("*.py")):
        installed = package / path.name
        require(installed.read_bytes() == path.read_bytes(), f"installed source differs: {path.name}")
        source_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    for path in (
        Path(__file__),
        ROOT / "scripts/verify_probe_evidence.py",
        ROOT / "scripts/verify_oracle_development.py",
    ):
        source_hashes[path.relative_to(ROOT).as_posix()] = digest(path)
    original = read_json(source / "summary.json")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in original["cases"]:
        case = local_path(ROOT / "examples/probes", f"{row['case']}.json")
        trace = local_path(source, row["trace"])
        start = time.perf_counter()
        analysis = compare(case, trace)
        elapsed = time.perf_counter() - start
        path = args.output / f"{row['case'].replace('/', '_')}.json"
        write_json(path, analysis)
        partial = analysis["ablations"]["offset_authority_erased"]["methods"]
        record = {
            "case": row["case"],
            "label": row["label"],
            "source_case_sha256": digest(case),
            "source_trace_sha256": digest(trace),
            "analysis": path.name,
            "analysis_sha256": digest(path),
            "original_numeric": {name: result["status"] for name, result in analysis["methods"].items()},
            "original_full_request": {
                name: result["full_requested_status"] for name, result in analysis["methods"].items()
            },
            "offset_erased_numeric": {name: result["status"] for name, result in partial.items()},
            "stateless_eligible_queries": analysis["methods"]["stateless_diagonal"]["coverage"][
                "eligible_queries"
            ],
            "analysis_wall_seconds": elapsed,
            "saved_fresh_witness": row.get("fresh_witness"),
        }
        records.append(record)
        print(f"{row['case']}: {json.dumps(record['original_numeric'])}", flush=True)
    require(
        not any(module.split(".")[0] in {"torch", "openpi", "lerobot"} for module in sys.modules),
        "data-only analysis imported an ML framework",
    )
    write_json(
        args.output / "summary.json",
        {
            "schema_version": 1,
            "status": "DEVELOPMENT_REANALYSIS_NOT_FINAL_EVALUATION",
            "development_only": True,
            "final_evaluation": False,
            "source_evidence": source.relative_to(ROOT).as_posix(),
            "source_summary_sha256": digest(source / "summary.json"),
            "source_file_sha256": source_hashes,
            "package_source": str(package),
            "runtime": {"python": platform.python_version(), "platform": platform.platform()},
            "tolerances": {"atol": 1e-7, "rtol": 1e-7, "status": "development defaults, not frozen"},
            "numeric_policy": "Propagated basis/closure envelopes with binary64 resolution floors",
            "cases": records,
            "scope": [
                "Same saved processor observations for all numerical oracles; no new adapter execution",
                "Reference erasure changes analysis authority, not original case/capture identity",
                "Strong all-pairs oracle is allowed to match the compiler; no superiority claim",
                "Stateless subdomain PASS leaves excluded temporal obligations UNKNOWN",
                "Recorded fresh witness executions predate this analysis; no new replay timing is claimed",
                "Analysis timing includes all oracles/ablations; not per-method speed or acquisition cost",
                "Manual/random query generation and external baseline comparisons remain required",
            ],
        },
    )
    print("Saved development oracle analysis; no ML imports or final benchmark evaluation")


if __name__ == "__main__":
    main()
