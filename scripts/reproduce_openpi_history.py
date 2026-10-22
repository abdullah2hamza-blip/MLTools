"""Capture or data-only verify the genuine historical OpenPI PR351 data boundary."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import mltools
from mltools.io import write_json
from mltools.obligations import Axis, Reference, Suite, compile_suite, evaluate
from mltools.research_oracles import pointwise, relational

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "integrations/openpi-history"
WORKER = ROOT / "scripts/openpi_history_worker.py"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(f"Historical OpenPI verification failed: {message}")


def design():
    fixture = json.loads((PROFILE / "fixture.json").read_bytes())
    radius = fixture["probe_radius"]
    axes = tuple(
        Axis(port, str(i), "legacy_configuration_coordinate", value - radius, value + radius)
        for port, values in (("action", fixture["legacy_targets"][0]), ("origin", fixture["origin"]))
        for i, value in enumerate(values)
    )
    suite = compile_suite(axes, tuple(f"dataset_action.{i}" for i in range(7)))
    reference = Reference(
        tuple(
            tuple(radius if a.name == str(i) and (a.port == "action" or i < 6) else 0.0 for a in axes)
            for i in range(7)
        ),
        tuple(fixture["dataset_actions"][0]),
        ("dataset_action_coordinate",) * 7,
        "Pre-removal bf25a4d LeRobotLiberoDataConfig active wiring plus independent numeric fixture; "
        "configured extra legacy conversion only, not physical controller/calibration authority",
        True,
    )
    return fixture, suite, reference


def close(actual, expected):
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(close(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return type(actual) in (int, float) and math.isfinite(actual) and abs(actual - expected) <= 1e-12


def analysis(suite, reference, trace):
    return {
        "compiled": evaluate(suite, reference, trace),
        "pointwise": pointwise(suite, reference, trace),
        "strong_relational": relational(suite, reference, trace),
    }


def source_hashes():
    modules = list((ROOT / "src/mltools").glob("*.py"))
    installed = Path(mltools.__file__).parent
    require(
        {p.name for p in installed.glob("*.py")} == {p.name for p in modules}
        and all((installed / p.name).read_bytes() == p.read_bytes() for p in modules),
        "installed MLTools package differs from recorded workspace source",
    )
    files = modules + [
        WORKER,
        Path(__file__),
        *PROFILE.glob("*.*"),
        ROOT / "integrations/openpi/build.lock",
        ROOT / "integrations/openpi/requirements-cpu.lock",
    ]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files) if p.suffix != ".md"}


def verify_capture(folder, name, phase, suite, reference):
    trace = json.loads((folder / f"{name}.trace.json").read_bytes())
    request = json.loads((folder / f"{name}.request.json").read_bytes())
    ledger = json.loads((PROFILE / "sources.json").read_bytes())["phases"][phase]
    require(
        trace["schema_version"] == 1
        and trace["development_only"] is True
        and trace["final_evaluation"] is False
        and trace["classification"]
        == "genuine_historical_data_factory_with_isolated_unused_model_preprocessing",
        f"{name} research scope",
    )
    require(trace["phase"] == phase, f"{name} phase")
    for key, value in ledger.items():
        require(trace["identity"][key] == value, f"{name} source inventory: {key}")
    require(trace["worker_sha256"] == sha(WORKER), f"{name} worker identity")
    require(trace["source_ledger_sha256"] == sha(PROFILE / "sources.json"), f"{name} source ledger")
    require(trace["fixture_sha256"] == sha(PROFILE / "fixture.json"), f"{name} fixture")
    require(trace["request_sha256"] == sha(folder / f"{name}.request.json"), f"{name} request identity")
    require(
        request
        == {
            "suite": suite.to_dict(),
            "suite_sha256": suite.sha256,
            "fixture_sha256": sha(PROFILE / "fixture.json"),
        },
        f"{name} requested experiment",
    )
    require(trace["suite_sha256"] == suite.sha256, f"{name} suite identity")
    require(set(trace["outputs"]) == {p.id for p in suite.probes}, f"{name} query inventory")
    require(set(trace["identity"]["direct_urls"]) == {"openpi", "openpi-client"}, f"{name} package inventory")
    for direct in trace["identity"]["direct_urls"].values():
        require(direct["vcs_info"]["commit_id"] == ledger["revision"], f"{name} installed revision")
    factory = trace["factory"]
    require(
        factory["name"] == "pi0_libero"
        and factory["model_action_dim"] == 32
        and factory["normalization_statistics"] is None
        and factory["original_method_restored"] is True
        and factory["substitution_calls"] == ["Pi0Config"],
        f"{name} disclosed factory scope",
    )
    require(
        factory["repo_id"] == "physical-intelligence/libero"
        and factory["model_inputs"] == []
        and factory["model_outputs"] == [],
        f"{name} named configuration and model-transform exclusion",
    )
    require(
        factory["data_inputs"]
        == (["LiberoInputs"] if phase == "before" else ["LiberoInputs", "DeltaActions"])
        and factory["data_outputs"]
        == (["LiberoOutputs"] if phase == "before" else ["AbsoluteActions", "LiberoOutputs"]),
        f"{name} actual transform order",
    )
    fixture, _, _ = design()
    boundary = trace["boundary"]
    target = fixture["dataset_actions"] if phase == "before" else fixture["legacy_targets"]
    output = fixture["legacy_targets"] if phase == "before" else fixture["dataset_actions"]
    require(close(boundary["training_targets"], target), f"{name} observed training targets")
    require(close(boundary["legacy_output_actions"], output), f"{name} observed legacy outputs")
    require(close(boundary["native_roundtrip"], fixture["dataset_actions"]), f"{name} round trip")
    require(
        boundary["training_shape"] == [2, 32]
        and boundary["training_padding_zero"] is True
        and boundary["state_padding_zero"] is True
        and boundary["image_pixels"] == {"base_0_rgb": 17, "left_wrist_0_rgb": 91, "right_wrist_0_rgb": 0}
        and boundary["image_masks"]
        == {"base_0_rgb": True, "left_wrist_0_rgb": True, "right_wrist_0_rgb": False},
        f"{name} padding and labeled images",
    )
    require(trace["runtime"]["blocked_network_events"] == [], f"{name} network attempts")
    result = analysis(suite, reference, trace)
    require(
        result == json.loads((folder / f"{name}.analysis.json").read_bytes()),
        f"{name} deterministic independent analysis",
    )
    expected_status = "FAIL" if phase == "before" else "PASS"
    require(all(r["status"] == expected_status for r in result.values()), f"{name} expected verdicts")
    return trace, result


def verify(folder):
    summary = json.loads((folder / "summary.json").read_bytes())
    require(
        summary["development_only"] is True and summary["final_evaluation"] is False,
        "summary research scope",
    )
    require(summary["source_file_sha256"] == source_hashes(), "capture/analysis source identities")
    for name, digest in summary["file_sha256"].items():
        require(sha(folder / name) == digest, f"saved file identity: {name}")
    _, suite, reference = design()
    require(
        json.loads((folder / "contract.json").read_bytes())
        == {"suite_sha256": suite.sha256, "reference": reference.to_dict()},
        "independent contract record",
    )
    before, before_result = verify_capture(folder, "before", "before", suite, reference)
    after, _ = verify_capture(folder, "after", "after", suite, reference)
    identifiers = {row["query"] for row in before_result["compiled"]["witness"]}
    reduced = Suite(suite.axes, suite.outputs, tuple(p for p in suite.probes if p.id in identifiers))
    witness, witness_result = verify_capture(folder, "before-witness", "before", reduced, reference)
    require(
        witness_result["compiled"]["first_divergence"]["obligation"]
        == before_result["compiled"]["first_divergence"]["obligation"],
        "fresh witness preserves obligation",
    )
    require(
        witness["request_sha256"] != before["request_sha256"]
        and all(close(witness["outputs"][key], before["outputs"][key]) for key in identifiers),
        "fresh reduced measurements",
    )
    require(summary["outcomes"] == outcomes(before, after, len(reduced.probes)), "summary outcomes")
    require(not any(name in sys.modules for name in ("torch", "openpi", "jax")), "data-only replay")
    return summary


def outcomes(before, after, witness_queries):
    return {
        "before_configured_legacy_contract": "FAIL",
        "after_configured_legacy_contract": "PASS",
        "round_trip": {"before": "PASS", "after": "PASS"},
        "independent_oracles": "Pointwise and strong relational match compiler on supplied queries",
        "full_checkpoint_physical_conformance": "UNKNOWN",
        "full_queries_per_phase": len(before["outputs"]),
        "after_queries": len(after["outputs"]),
        "fresh_before_witness_queries": witness_queries,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify", type=Path)
    mode.add_argument("--output", type=Path)
    parser.add_argument("--before-python", type=Path)
    parser.add_argument("--after-python", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
        print("Verified genuine historical OpenPI factory captures, direct expectations and fresh witness")
        return
    if not args.before_python or not args.after_python:
        parser.error("Capture requires both pinned historical interpreters")
    args.output.mkdir(parents=True, exist_ok=False)
    initial_sources = source_hashes()
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        JAX_PLATFORMS="cpu",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        AWS_EC2_METADATA_DISABLED="true",
        PYTHONUTF8="1",
        PYTHONNOUSERSITE="1",
    )
    _, suite, reference = design()
    write_json(
        args.output / "contract.json", {"suite_sha256": suite.sha256, "reference": reference.to_dict()}
    )
    captures = {}

    def run(name, phase, query_suite):
        require(source_hashes() == initial_sources, "sources changed before capture")
        request = args.output / f"{name}.request.json"
        write_json(
            request,
            {
                "suite": query_suite.to_dict(),
                "suite_sha256": query_suite.sha256,
                "fixture_sha256": sha(PROFILE / "fixture.json"),
            },
        )
        python = args.before_python if phase == "before" else args.after_python
        process = subprocess.run(
            [
                str(python),
                "-I",
                str(WORKER),
                "--phase",
                phase,
                "--request",
                str(request),
                "--output",
                str(args.output / f"{name}.trace.json"),
                "--assets",
                str((args.output / f"empty-assets-{name}").resolve()),
            ],
            env=env,
            capture_output=True,
            timeout=180,
            check=False,
        )
        (args.output / f"{name}.stdout.txt").write_bytes(process.stdout)
        (args.output / f"{name}.stderr.txt").write_bytes(process.stderr)
        require(process.returncode == 0, f"worker {name}: {process.stderr.decode('utf-8', errors='replace')}")
        require(source_hashes() == initial_sources, "sources changed during capture")
        trace = json.loads((args.output / f"{name}.trace.json").read_bytes())
        result = analysis(query_suite, reference, trace)
        write_json(args.output / f"{name}.analysis.json", result)
        verify_capture(args.output, name, phase, query_suite, reference)
        captures[name] = trace
        print(f"{name}: {result['compiled']['status']} ({len(query_suite.probes)} queries)", flush=True)
        return result

    before_result = run("before", "before", suite)
    run("after", "after", suite)
    identifiers = {row["query"] for row in before_result["compiled"]["witness"]}
    reduced = Suite(suite.axes, suite.outputs, tuple(p for p in suite.probes if p.id in identifiers))
    run("before-witness", "before", reduced)
    summary = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "classification": "genuine_historical_data_factory_with_isolated_unused_model_preprocessing",
        "authority": "research/publication/openpi-history.md",
        "source_file_sha256": initial_sources,
        "file_sha256": {p.name: sha(p) for p in sorted(args.output.iterdir()) if p.is_file()},
        "outcomes": outcomes(captures["before"], captures["after"], len(reduced.probes)),
        "independent_unit": "One retrospectively inspected historical incident, not 63 independent cases",
        "scope": "No checkpoint, normalization statistics, tokenizer/model, controller or task evaluation",
    }
    write_json(args.output / "summary.json", summary)
    verify(args.output)


if __name__ == "__main__":
    main()
