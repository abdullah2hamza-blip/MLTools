"""Reproduce/verify the pinned processor incident; no weights, robot or simulator."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from lerobot_history_worker import SOURCES

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/lerobot_history_worker.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(f"Historical capture verification failed: {message}")


def verify(folder):
    predicted = {"before": [1.1, 5.2, 9.3, 20.4, 30.5], "after": [1.1, 1.2, 1.3, 20.4, 30.5]}
    records = []
    for phase, values in predicted.items():
        path = folder / f"{phase}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        revision, blobs = SOURCES[phase]
        require(
            data["identity"]["revision"] == revision and data["identity"]["git_blobs"] == blobs,
            f"{phase} source identity",
        )
        require(data["worker_sha256"] == sha(WORKER), f"{phase} worker identity")
        require(data["status"] == ("FAIL" if phase == "before" else "PASS"), f"{phase} status")
        require(len(data["events"]) == len(values), f"{phase} event count")
        require(
            all(
                abs(event["actual"] - expected) < 1e-12
                for event, expected in zip(data["events"], values, strict=True)
            ),
            f"{phase} observed values",
        )
        require(
            data["reset"]["anchor_cleared"] and data["reset"]["pending_before_reset"] == 1,
            f"{phase} interrupted reset",
        )
        require(abs(data["reset"]["new_chunk_actual"] - 40.7) < 1e-12, f"{phase} refill")
        records.append(
            {
                "phase": phase,
                "revision": revision,
                "file": path.name,
                "sha256": sha(path),
                "status": data["status"],
                "observed": [e["actual"] for e in data["events"]],
            }
        )
    return {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "classification": "genuine historical processors with explicit synthetic queue boundary inputs",
        "worker_sha256": sha(WORKER),
        "phases": records,
        "reference_authority": "research/publication/lerobot-history.md",
        "scope": "Processor before/fix reproduction with reset/refill; not full policy evaluation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--verify", type=Path, help="Data-only verification of saved captures")
    source.add_argument("--output", type=Path, help="New capture directory")
    parser.add_argument("--before-python", type=Path)
    parser.add_argument("--after-python", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        print("Verified native parent drift, fixed conformance, reset and refill without ML imports")
        return
    if not args.before_python or not args.after_python:
        parser.error("Capture requires both pinned historical interpreters")
    args.output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUTF8="1",
    )
    for phase, python in (("before", args.before_python), ("after", args.after_python)):
        subprocess.run(
            [
                str(python),
                "-I",
                str(WORKER),
                "--phase",
                phase,
                "--output",
                str(args.output / f"{phase}.json"),
            ],
            env=env,
            timeout=180,
            check=True,
        )
    result = verify(args.output)
    result["dependency_lock_sha256"] = sha(ROOT / "uv.lock")
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
