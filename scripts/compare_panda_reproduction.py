"""Verify two complete Panda captures and compare their semantic output fields."""

import argparse
import json
from pathlib import Path

from panda_reference import require, sha
from validate_panda import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("reproduction", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left, right = args.original.resolve(), args.reproduction.resolve()
    verify(left)
    verify(right)
    fields = (
        "source_file_sha256",
        "identity",
        "bootstrap",
        "native_config",
        "affine",
        "full_controller",
        "orientation",
        "gripper",
        "simulation",
    )
    a, b = (json.loads((p / "capture.json").read_bytes()) for p in (left, right))
    require(all(a[k] == b[k] for k in fields), "clean reproduction semantic outputs")
    exact = ["inputs.json", "request.json", "analysis.json"]
    require(
        all((left / name).read_bytes() == (right / name).read_bytes() for name in exact),
        "clean reproduction exact data-only inputs/reports",
    )
    result = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "comparison_source_sha256": sha(Path(__file__)),
        "original_summary_sha256": sha(left / "summary.json"),
        "reproduction_summary_sha256": sha(right / "summary.json"),
        "identical_semantic_fields": list(fields),
        "byte_identical_files": exact,
        "scope": "New environment/cached pins; no main-study reproduction or independent sample",
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Verified clean-environment Panda semantic reproduction and three byte-identical data files")


if __name__ == "__main__":
    main()
