"""Execute and replay designed certificate equations/native parity controls."""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from certificate_cases import ROOT, fixtures, sha, sources, write
from normalization_certificate import certificate, require

from mltools._openpi_worker import BLOBS as OPENPI_BLOBS
from mltools._openpi_worker import REVISION as OPENPI_REVISION
from mltools.lerobot import BLOBS as LEROBOT_BLOBS
from mltools.lerobot import REVISION as LEROBOT_REVISION


def request(framework):
    return {
        "framework": framework,
        "source_file_sha256": sources(),
        "cases": {n: c for n, c in fixtures().items() if n.startswith(framework + "_")},
    }


def parity(captures, cases):
    require(set(captures) == set(cases), "complete native case inventory")
    for name, case in cases.items():
        calculated = certificate(case)
        native = captures[name]
        require(
            set(native) == {"normalized_actions", "intended_decoded", "candidate_decoded"},
            "native boundary inventory",
        )
        for key in native:
            require(len(native[key]) == len(calculated[key]), "native sample inventory")
            for left, right in zip(native[key], calculated[key], strict=True):
                require(len(left) == len(right), "native coordinate inventory")
                require(
                    all(
                        type(a) in (int, float)
                        and math.isfinite(a)
                        and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                        for a, b in zip(left, right, strict=True)
                    ),
                    f"native decoder parity: {name} {key}",
                )


def acquire(folder, framework, python):
    path = folder / f"{framework}.request.json"
    write(path, request(framework))
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUTF8="1",
        PYTHONNOUSERSITE="1",
    )
    result = subprocess.run(
        [str(python), "-I", str(ROOT / "scripts/normalization_certificate_worker.py")],
        input=path.read_bytes(),
        capture_output=True,
        timeout=180,
        env=env,
        check=False,
    )
    (folder / f"{framework}.stdout.txt").write_bytes(result.stdout)
    (folder / f"{framework}.stderr.txt").write_bytes(result.stderr)
    require(result.returncode == 0, f"native {framework} capture: {result.stderr[-1500:]!r}")
    data = json.loads(result.stdout)
    write(folder / f"{framework}.capture.json", data)
    parity(data["captures"], request(framework)["cases"])


def expected_files():
    return {
        "inputs.json",
        *(f"{name}.report.json" for name in fixtures()),
        *(
            f"{framework}.{suffix}"
            for framework in ("openpi", "lerobot")
            for suffix in ("request.json", "stdout.txt", "stderr.txt", "capture.json")
        ),
    }


def run(folder, lerobot_python, openpi_python):
    initial = sources()
    write(folder / "inputs.json", fixtures())
    # Sequential native workers: never import both ML frameworks in the parent.
    acquire(folder, "lerobot", lerobot_python)
    acquire(folder, "openpi", openpi_python)
    require(sources() == initial, "producer stability around both workers")
    for name, case in fixtures().items():
        write(folder / f"{name}.report.json", certificate(case))
    write(
        folder / "summary.json",
        {
            "schema_version": 1,
            "development_only": True,
            "final_evaluation": False,
            "source_file_sha256": initial,
            "cases": list(fixtures()),
            "paper_source": "https://arxiv.org/html/2606.03724v1",
            "calibration_kind": "designed; no demonstration distribution or task-success experiment",
            "native_parity": {
                "absolute_tolerance": 1e-12,
                "relative_tolerance": 1e-12,
                "scope": "Measured float64 designed controls, not a universal upstream error bound",
            },
            "decision_rule": None,
            "file_sha256": {name: sha(folder / name) for name in sorted(expected_files())},
        },
    )
    verify(folder)


def verify(folder):
    summary = json.loads((folder / "summary.json").read_bytes())
    require(
        summary["schema_version"] == 1
        and summary["development_only"] is True
        and summary["final_evaluation"] is False,
        "development scope",
    )
    require(summary["source_file_sha256"] == sources(), "producer source identity")
    require(summary["cases"] == list(fixtures()), "complete fixed case inventory")
    require(
        summary["paper_source"] == "https://arxiv.org/html/2606.03724v1"
        and summary["calibration_kind"]
        == "designed; no demonstration distribution or task-success experiment",
        "paper/calibration scope",
    )
    require(
        summary["native_parity"]
        == {
            "absolute_tolerance": 1e-12,
            "relative_tolerance": 1e-12,
            "scope": "Measured float64 designed controls, not a universal upstream error bound",
        }
        and summary["decision_rule"] is None,
        "numerical profile and absence of safety/detection rule",
    )
    require(set(summary["file_sha256"]) == expected_files(), "closed evidence manifest")
    for name, expected in summary["file_sha256"].items():
        require(sha(folder / name) == expected, "saved evidence identity: " + name)
    cases = fixtures()
    require(json.loads((folder / "inputs.json").read_bytes()) == cases, "independent designed inputs")
    for framework, revision, blobs in (
        ("openpi", OPENPI_REVISION, OPENPI_BLOBS),
        ("lerobot", LEROBOT_REVISION, LEROBOT_BLOBS),
    ):
        path = folder / f"{framework}.request.json"
        require(json.loads(path.read_bytes()) == request(framework), "complete native request")
        data = json.loads((folder / f"{framework}.capture.json").read_bytes())
        require(
            json.loads((folder / f"{framework}.stdout.txt").read_bytes()) == data,
            "exact native stdout/parsed capture",
        )
        require(
            data["schema_version"] == 1
            and data["development_only"] is True
            and data["final_evaluation"] is False,
            "native development scope",
        )
        require(
            data["source_file_sha256"] == sources()
            and data["request_sha256"] == sha(path)
            and data["framework"] == framework,
            "native producer/input/framework binding",
        )
        identity = data["upstream_identity"]
        require(
            identity["revision"] == revision and identity["git_blobs"] == blobs, "native upstream identity"
        )
        if framework == "openpi":
            require(
                identity["client_image_tools_blob"] == "7a971b9d5f6b1495fd6cdea202ffa607d8b34bf0"
                and set(identity["direct_urls"]) == {"openpi", "openpi-client"},
                "native OpenPI client identity",
            )
            require(
                all(d["vcs_info"]["commit_id"] == revision for d in identity["direct_urls"].values()),
                "native OpenPI package revisions",
            )
        runtime = data["runtime"]
        require(
            runtime["network_events"] == []
            and type(runtime["peak_rss_bytes"]) is int
            and 0 < runtime["peak_rss_bytes"] <= 2 * 1024**3,
            "network and measured peak-RSS controls",
        )
        require(
            type(runtime["wall_seconds"]) in (int, float)
            and math.isfinite(runtime["wall_seconds"])
            and runtime["wall_seconds"] >= 0,
            "native measured duration",
        )
        parity(data["captures"], request(framework)["cases"])
    for name, case in cases.items():
        require(
            json.loads((folder / f"{name}.report.json").read_bytes()) == certificate(case),
            "independent deterministic certificate replay: " + name,
        )
    require(
        not any(name in sys.modules for name in ("torch", "lerobot", "openpi")), "data-only parent/replay"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--verify", type=Path)
    parser.add_argument("--lerobot-python", type=Path)
    parser.add_argument("--openpi-python", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify.resolve())
    else:
        require(
            args.lerobot_python is not None and args.openpi_python is not None, "explicit native interpreters"
        )
        folder = args.output.resolve()
        require(folder.is_relative_to(ROOT), "workspace-contained capture output")
        folder.mkdir(parents=True, exist_ok=False)
        run(folder, args.lerobot_python.resolve(), args.openpi_python.resolve())
    print("Verified designed normalization-certificate controls and genuine native decoder parity")


if __name__ == "__main__":
    main()
