"""Build/exercise stock ActionABI CPU library APIs, or verify development records offline."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from actionabi_fixtures import (
    authority,
    close,
    contract,
    expected_score,
    plans,
    prediction,
    probe_request,
    require,
    samples,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "integrations/actionabi"


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sources():
    paths = [
        Path(__file__),
        ROOT / "scripts/actionabi_fixtures.py",
        ROOT / "scripts/prepare_actionabi.py",
        *PROFILE.iterdir(),
    ]
    return {
        p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths) if p.is_file() and p.suffix != ".md"
    }


def upstream_identities(base):
    result = {}
    for name, record in read(PROFILE / "sources.json")["repositories"].items():
        root = base / name
        revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        require(revision == record["revision"], f"source revision: {name}")
        require(
            not subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"]),
            f"modified source: {name}",
        )
        raw_hashes, exact = {}, 0
        for path, expected in record["file_git_blobs"].items():
            data = (root / path).read_bytes()

            def blob(value):
                return hashlib.sha1(b"blob " + str(len(value)).encode() + b"\0" + value).hexdigest()

            matches = blob(data) == expected
            require(
                matches or blob(data.replace(b"\r\n", b"\n")) == expected, f"upstream bytes: {name}/{path}"
            )
            exact += matches
            raw_hashes[path] = hashlib.sha256(data).hexdigest()
        result[name] = {
            "revision": revision,
            "checked_files": len(raw_hashes),
            "exact_git_blob_files": exact,
            "crlf_to_lf_matches": len(raw_hashes) - exact,
            "raw_inventory_sha256": hashlib.sha256(
                json.dumps(raw_hashes, sort_keys=True).encode()
            ).hexdigest(),
        }
    return result


def invoke(command, folder, stem, env, timeout=180):
    started = time.perf_counter()
    process = subprocess.run(command, cwd=folder, env=env, capture_output=True, timeout=timeout, check=False)
    path = folder / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".stdout.txt").write_bytes(process.stdout)
    path.with_suffix(".stderr.txt").write_bytes(process.stderr)
    require(process.returncode == 0, f"native command {stem}: {process.stderr[-1500:]!r}")
    return process.stdout, time.perf_counter() - started


def build(args, env):
    require(not args.build.exists(), "use a fresh build directory")
    cmake = Path(sys.executable).parent / "cmake.exe"
    ctest = Path(sys.executable).parent / "ctest.exe"
    command = [
        str(cmake),
        "-S",
        str(PROFILE),
        "-B",
        str(args.build),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DCMAKE_CXX_COMPILER=" + str(args.toolchain / "bin/clang++.exe"),
        "-DACTIONABI_SOURCE_DIR=" + str(args.sources / "actionabi"),
        "-DFETCHCONTENT_SOURCE_DIR_CLI11=" + str(args.sources / "cli11"),
        "-DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=" + str(args.sources / "json"),
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        "-DPython3_EXECUTABLE=" + sys.executable,
    ]
    commands = [
        command,
        [str(cmake), "--build", str(args.build), "--parallel", "1"],
        [str(ctest), "--test-dir", str(args.build), "--output-on-failure"],
    ]
    write(args.output / "build/commands.json", commands)
    timings = {}
    for stem, command in (
        ("compiler-version", [str(args.toolchain / "bin/clang++.exe"), "--version"]),
        ("cmake-version", [str(cmake), "--version"]),
        ("ninja-version", [str(Path(sys.executable).parent / "ninja.exe"), "--version"]),
    ):
        invoke(command, args.output, "build/" + stem, env, timeout=30)
    for stem, command in zip(("configure", "compile", "upstream-tests"), commands, strict=True):
        _stdout, timings[stem] = invoke(command, args.output, "build/" + stem, env, timeout=300)
        print(f"ActionABI {stem} passed", flush=True)
    return args.build / "mltools_actionabi_controls.exe", timings


def capture(args):
    initial_sources = sources()
    upstream = upstream_identities(args.sources)
    ledger = read(PROFILE / "sources.json")
    archive = args.toolchain.parent / ledger["toolchain"]["archive"]
    require(digest(archive) == ledger["toolchain"]["sha256"], "official toolchain archive")
    env = os.environ.copy()
    env["PATH"] = (
        str(args.toolchain / "bin") + os.pathsep + str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    )
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    temporary = args.output / "runtime-temp"
    temporary.mkdir()
    env.update(TMP=str(temporary), TEMP=str(temporary), TMPDIR=str(temporary))
    executable, build_times = build(args, env)
    exe_hash = digest(executable)
    authority_path = args.output / "inputs/fixture-authority.json"
    write(authority_path, authority())
    for kind in ("excited", "constant", "inverted_last"):
        metadata = {
            "record_type": "metadata",
            "schema_version": "1.0",
            "source_filename": "inputs/fixture-authority.json",
            "source_sha256": digest(authority_path),
            "extraction_date": "2026-10-22",
            "state_columns": ["fixture_q0", "fixture_q1"],
            "state_units": ["rad", "rad"],
        }
        path = args.output / f"inputs/{kind}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row) for row in [metadata, *samples(kind)]) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    timings, prediction_counts = {}, {}

    def native(name, request):
        path = args.output / f"native/{name}.request.json"
        write(path, request)
        stdout, elapsed = invoke([str(executable), str(path)], args.output, "native/" + name, env, timeout=60)
        timings[name] = elapsed
        return json.loads(stdout)

    for name, plan in plans().items():
        relative = f"inputs/{plan['dataset']}.jsonl"
        request = {
            "operation": "score",
            "dataset": relative,
            "dataset_sha256": digest(args.output / relative),
            "hypotheses": plan["hypotheses"],
            "unsupported_fields": plan["unsupported_fields"],
            "huber_delta": 1.0,
            "train_episode_fraction": 0.5,
            "score_threshold": 0.0,
            "coverage_input": 0.0,
        }
        result = native(name + "-score", request)
        evidence_path = args.output / f"native/{name}-score.stdout.txt"
        native(
            name + "-convert",
            {
                "operation": "convert",
                "score_evidence": result,
                "evidence_sha256": digest(evidence_path),
                "score_threshold": 0.0,
                "coverage_input": 0.0,
                "unsupported_fields": plan["unsupported_fields"],
                "action": [0.25, -0.5],
            },
        )
    pulses = native(
        "pulses", {"operation": "pulses", "lower": [-1, -2], "upper": [1, 2], "amplitude": [0.5, 1]}
    )["pulses"]
    for name in ("separating", "first_unsafe", "all_unsafe", "inseparable", "greedy"):
        started = time.perf_counter()
        request = probe_request(name, pulses)
        prediction_counts[name] = {
            "calls": len(pulses) * len(request["hypotheses"]),
            "seconds": time.perf_counter() - started,
            "scope": "Caller-authored synthetic hypothesis predictions; not processor/robot queries",
        }
        native(name + "-select", request)
    native("calibration", {"operation": "calibration", "calibration_scores": [0, 0.1, 0.2, 0.3]})
    require(digest(executable) == exe_hash, "native executable changed")
    require(
        upstream_identities(args.sources) == upstream and sources() == initial_sources,
        "source changed during capture",
    )
    summary = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "source_file_sha256": initial_sources,
        "upstream_identities": upstream,
        "executable_sha256": exe_hash,
        "toolchain_archive_sha256": ledger["toolchain"]["sha256"],
        "build_seconds": build_times,
        "native_seconds": timings,
        "prediction_generation": prediction_counts,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "coverage_claim": False,
        "temporal_processor_comparison": (
            "NOT_APPLICABLE: native trajectory response model has no chunk-origin/queue/reset semantics"
        ),
        "file_sha256": {
            p.relative_to(args.output).as_posix(): digest(p)
            for p in sorted(args.output.rglob("*"))
            if p.is_file()
        },
    }
    write(args.output / "summary.json", summary)
    verify(args.output)


def verify(folder):
    summary = read(folder / "summary.json")
    require(summary["development_only"] is True and summary["final_evaluation"] is False, "development scope")
    require(summary["source_file_sha256"] == sources(), "project source identities")
    require(summary["coverage_claim"] is False, "no inferred calibration coverage")
    require(summary["temporal_processor_comparison"].startswith("NOT_APPLICABLE:"), "temporal applicability")
    require(
        "100% tests passed, 0 tests failed out of 2"
        in (folder / "build/upstream-tests.stdout.txt").read_text(),
        "actual upstream CPU package and CLI tests",
    )
    ledger = read(PROFILE / "sources.json")
    require(summary["toolchain_archive_sha256"] == ledger["toolchain"]["sha256"], "toolchain identity")
    require(set(summary["upstream_identities"]) == set(ledger["repositories"]), "complete upstream inventory")
    for name, info in summary["upstream_identities"].items():
        require(info["revision"] == ledger["repositories"][name]["revision"], "pinned upstream revision")
        require(
            info["checked_files"] == len(ledger["repositories"][name]["file_git_blobs"]),
            "complete checked inventory",
        )
    for name, sha in summary["file_sha256"].items():
        path = (folder / name).resolve()
        require(path.is_relative_to(folder.resolve()) and digest(path) == sha, f"saved identity: {name}")
    require(read(folder / "inputs/fixture-authority.json") == authority(), "independent fixture authority")
    for kind in ("excited", "constant", "inverted_last"):
        rows = [json.loads(row) for row in (folder / f"inputs/{kind}.jsonl").read_text().splitlines()]
        require(rows[1:] == samples(kind), "independently authored sample identity")
        require(
            rows[0]["source_sha256"] == digest(folder / "inputs/fixture-authority.json"), "fixture provenance"
        )
    for name, plan in plans().items():
        request = read(folder / f"native/{name}-score.request.json")
        require(
            request
            == {
                "operation": "score",
                "dataset": f"inputs/{plan['dataset']}.jsonl",
                "dataset_sha256": digest(folder / f"inputs/{plan['dataset']}.jsonl"),
                "hypotheses": plan["hypotheses"],
                "unsupported_fields": plan["unsupported_fields"],
                "huber_delta": 1.0,
                "train_episode_fraction": 0.5,
                "score_threshold": 0.0,
                "coverage_input": 0.0,
            },
            "complete fixed native scoring request",
        )
        require(
            request["hypotheses"] == plan["hypotheses"]
            and request["unsupported_fields"] == plan["unsupported_fields"],
            "fixed grammar/authority input",
        )
        require(request["dataset"] == f"inputs/{plan['dataset']}.jsonl", "fixed dataset input")
        require(request["dataset_sha256"] == digest(folder / request["dataset"]), "native dataset binding")
        result = read(folder / f"native/{name}-score.stdout.txt")
        hypotheses = result["native_evidence"]["ranked_hypotheses"]
        require(len(hypotheses) == len(plan["hypotheses"]), "native hypothesis inventory")
        for hypothesis in plan["hypotheses"]:
            native_rows = [row for row in hypotheses if row["contract"] == hypothesis]
            require(len(native_rows) == 1, "native scored hypothesis identity")
            expected = expected_score(samples(plan["dataset"]), hypothesis)
            actual = native_rows[0]["score"]
            require(
                close(actual["train_loss"], expected["train_loss"])
                and close(actual["heldout_loss"], expected["heldout_loss"]),
                "independent Huber losses",
            )
            require(
                actual["residual_count"] == expected["residual_count"]
                and all(
                    close(a, b)
                    for a, b in zip(actual["per_dimension_loss"], expected["per_dimension_loss"], strict=True)
                ),
                "independent residual counts/component losses",
            )
        equivalence = result["equivalence"]
        require(equivalence["caller_supplied_calibrated_coverage"] == 0, "coverage is caller supplied")
        retained = equivalence["retained"]
        require(
            len(retained) == (2 if name in ("unexcited", "structural_ambiguity") else 1),
            "native equivalence set",
        )
        if name in ("identified", "unsupported"):
            require(retained == [contract()], "true fixture retained")
        elif name == "misspecified_grammar":
            require(
                retained == [contract(sign=[-1, 1])], "relative grammar retention is not absolute acceptance"
            )
        fields = equivalence["fields"]
        require(
            fields
            == {
                field: "unsupported"
                if field in plan["unsupported_fields"]
                else "identified"
                if all(h[field] == retained[0][field] for h in retained)
                else "ambiguous"
                for field in (
                    "target",
                    "space",
                    "frame",
                    "permutation",
                    "sign",
                    "scale",
                    "lag_steps",
                    "gripper_inverted",
                )
            },
            "complete native field status",
        )
        if name == "unexcited":
            require(fields["permutation"] == "ambiguous", "excitation ambiguity")
        if name == "structural_ambiguity":
            require(fields["sign"] == fields["gripper_inverted"] == "ambiguous", "structural ambiguity")
        if name == "unsupported":
            require(fields["space"] == fields["frame"] == "unsupported", "unsupported physical authority")
        conversion_request = read(folder / f"native/{name}-convert.request.json")
        evidence_sha = digest(folder / f"native/{name}-score.stdout.txt")
        require(
            conversion_request
            == {
                "operation": "convert",
                "score_evidence": result,
                "evidence_sha256": evidence_sha,
                "score_threshold": 0.0,
                "coverage_input": 0.0,
                "unsupported_fields": plan["unsupported_fields"],
                "action": [0.25, -0.5],
            },
            "complete fixed native converter request",
        )
        require(
            conversion_request["score_evidence"] == result
            and conversion_request["evidence_sha256"] == digest(folder / f"native/{name}-score.stdout.txt"),
            "actual evidence-report digest binding",
        )
        conversion = read(folder / f"native/{name}-convert.stdout.txt")
        expected_decision = "EMIT" if name in ("identified", "misspecified_grammar") else "REFUSE"
        require(conversion["decision"] == expected_decision, "native converter decision")
        if expected_decision == "EMIT":
            require(conversion["bad_evidence_hash_rejected"] is True, "native converter hash refusal")
            require(
                conversion["converter"]["contract"] == retained[0]
                and conversion["converter"]["evidence_report_sha256"] == evidence_sha,
                "emitted contract and evidence binding",
            )
            require(
                conversion["converted"] == prediction([0.25, -0.5], retained[0]),
                "instantaneous conversion scope",
            )
    pulses = read(folder / "native/pulses.stdout.txt")["pulses"]
    require(
        read(folder / "native/pulses.request.json")
        == {
            "operation": "pulses",
            "lower": [-1, -2],
            "upper": [1, 2],
            "amplitude": [0.5, 1],
        },
        "fixed native pulse request",
    )
    require(pulses == [[0.5, 0], [-0.5, 0], [0, 1], [0, -1]], "bounded native pulses")
    for name in ("separating", "first_unsafe", "all_unsafe", "inseparable", "greedy"):
        request = read(folder / f"native/{name}-select.request.json")
        require(request == probe_request(name, pulses), "complete fixed selector request")
        require(
            summary["prediction_generation"][name]["calls"] == len(pulses) * len(request["hypotheses"]),
            "candidate prediction budget",
        )
        for candidate in request["candidates"]:
            require(
                candidate["predicted_observations"]
                == [prediction(candidate["action"], h) for h in request["hypotheses"]],
                "caller prediction identity",
            )
        result = read(folder / f"native/{name}-select.stdout.txt")
        expected_selection = (
            [] if name in ("all_unsafe", "inseparable") else [1] if name == "first_unsafe" else [0]
        )
        require(result["selected_candidate_indices"] == expected_selection, "native separating selection")
        require(
            result["inseparable_pairs"] == ([[0, 1]] if name in ("all_unsafe", "inseparable") else []),
            "native inseparability",
        )
        require(result["exact"] == (name != "greedy"), "native exact/greedy scope")
    calibration = read(folder / "native/calibration.stdout.txt")
    require(
        read(folder / "native/calibration.request.json")
        == {
            "operation": "calibration",
            "calibration_scores": [0, 0.1, 0.2, 0.3],
        },
        "fixed native calibration request",
    )
    require(
        calibration["threshold"] == 0.3 and calibration["insufficient_calibration_rejected"] is True,
        "native calibration finite-sample refusal",
    )
    require(not any(m in sys.modules for m in ("torch", "lerobot", "openpi")), "data-only verification")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    parser.add_argument("--sources", type=Path)
    parser.add_argument("--toolchain", type=Path)
    parser.add_argument("--build", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        require(
            args.sources and args.toolchain and args.build,
            "explicit local sources/toolchain/new build required",
        )
        for key in ("output", "sources", "toolchain", "build"):
            setattr(args, key, getattr(args, key).resolve())
        args.output.mkdir(parents=True, exist_ok=False)
        capture(args)
    print("Verified stock ActionABI CPU scoring, equivalence/refusal and separating-probe controls")


if __name__ == "__main__":
    main()
