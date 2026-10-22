"""Acquire/replay fixed development inputs for the prospective action-only runner.

This version deliberately cannot execute prospective cells. Its inputs are
authored arithmetic controls and the previously executed ACT move-cube lineage.
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import query_designs as qd
from action_designs import generate_all
from action_study_common import CONDITIONS, ROOT, development_inputs, fingerprint, sha, sources, write
from normalization_certificate import certificate
from normalization_profiles import affine_reference, finite, require, transform

from mltools._openpi_worker import BLOBS as OPENPI_BLOBS
from mltools._openpi_worker import REVISION as OPENPI_REVISION
from mltools.lerobot import BLOBS as LEROBOT_BLOBS
from mltools.lerobot import REVISION as LEROBOT_REVISION
from mltools.obligations import Axis, Reference, compile_suite, evaluate
from mltools.research_oracles import pointwise, relational

TOLERANCES = {"atol": 1e-7, "rtol": 1e-7}


def suite(cell):
    n = len(next(iter(cell["intended"]["statistics"].values())))
    return compile_suite(
        tuple(Axis("action", str(i), "normalized_output_index_" + str(i), -0.8, 0.8) for i in range(n)),
        tuple("raw_action_" + str(i) for i in range(n)),
    )


def designs():
    return {cell: generate_all(suite(value)) for cell, value in development_inputs().items()}


def reference(cell, *, erased=None):
    units = ["raw_action_index_" + str(i) for i in range(len(suite(cell).outputs))]
    return Reference(
        **affine_reference(
            cell["intended"],
            suite(cell).to_dict()["axes"],
            authority=None if erased == "authority" else cell["authority"],
            output_units=[None] * len(units) if erased == "units" else units,
        )
    )


def request(framework, selected=None):
    cells = {k: v for k, v in development_inputs().items() if v["framework"] == framework}
    return {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "source_file_sha256": sources(),
        "framework": framework,
        "phase": "main" if selected is None else "witness",
        "cells": cells,
        "designs": {k: {n: d.to_dict() for n, d in designs()[k].items()} for k in cells},
        "selected_queries": selected,
    }


def near(left, right, *, atol=1e-12, rtol=1e-12):
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
        and all(
            finite(a) and finite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
            for a, b in zip(left, right, strict=True)
        )
    )


def validate_capture(data, req):
    require(
        data["schema_version"] == 1
        and data["format"] == "mltools-action-normalization-native-development"
        and data["development_only"] is True
        and data["final_evaluation"] is False
        and data["phase"] == req["phase"]
        and data["framework"] == req["framework"]
        and data["source_file_sha256"] == sources(),
        "native capture scope/binding",
    )
    revision, blobs = (
        (LEROBOT_REVISION, LEROBOT_BLOBS)
        if req["framework"] == "lerobot"
        else (OPENPI_REVISION, OPENPI_BLOBS)
    )
    require(
        data["upstream_identity"]["revision"] == revision and data["upstream_identity"]["git_blobs"] == blobs,
        "actual upstream source identities",
    )
    require(data["upstream_identity"]["source_newlines"] == "canonical LF", "native source representation")
    if req["framework"] == "openpi":
        identity = data["upstream_identity"]
        require(
            identity["client_image_tools_blob"] == "7a971b9d5f6b1495fd6cdea202ffa607d8b34bf0"
            and set(identity["direct_urls"]) == {"openpi", "openpi-client"}
            and all(d["vcs_info"]["commit_id"] == revision for d in identity["direct_urls"].values()),
            "native OpenPI client and both installed package identities",
        )
    require(set(data["measurements"]) == set(req["cells"]), "all native cells")
    total, auxiliary = 0, 0
    for cell_id, cell in req["cells"].items():
        ds = designs()[cell_id]
        observed = data["measurements"][cell_id]
        selected = req["selected_queries"]
        conditions = set(CONDITIONS) if selected is None else set(selected.get(cell_id, {}))
        require(set(observed) == conditions, "actual condition inventory")
        for condition, capture in observed.items():
            candidate = cell["conditions"][condition]["specification"]
            require(capture["deployment_sha256"] == fingerprint(candidate), "actual deployment statistics")
            config = capture["configuration"]
            if req["framework"] == "lerobot":
                mode = "MEAN_STD" if "mean_std" in candidate["profile"] else "QUANTILES"
                expected = {
                    "eps": 1e-8,
                    "features": {"action": {"type": "ACTION", "shape": [len(suite(cell).outputs)]}},
                    "norm_map": {"ACTION": mode},
                }
                require(
                    config == {"encoder": expected, "decoder": expected, "device": "cpu", "dtype": "float64"},
                    "actual native LeRobot configuration",
                )
            else:
                require(
                    config
                    == {
                        "encoder": "openpi.transforms.Normalize",
                        "decoder": "openpi.transforms.Unnormalize",
                        "use_quantiles": True,
                        "device": "cpu",
                        "dtype": "float64",
                    },
                    "actual native OpenPI configuration",
                )
            names = set(ds) if selected is None else {"canonical"}
            require(set(capture["designs"]) == names, "complete native designs")
            for name, entry in capture["designs"].items():
                d = ds[name]
                queries = [p for p in d.probes if selected is None or p.id in selected[cell_id][condition]]
                require(
                    entry["design_sha256"] == d.sha256
                    and entry["decoder_calls"] == len(queries)
                    and len(entry["rows"]) == len(queries),
                    "one actual decoder call per selected query",
                )
                for row, p in zip(entry["rows"], queries, strict=True):
                    require(
                        set(row) == {"query", "input", "output"}
                        and row["query"] == p.id
                        and row["input"] == list(p.values),
                        "actual ordered native query input",
                    )
                    require(
                        near(row["output"], transform(candidate, row["input"], inverse=True)),
                        "native response agrees with its actual called statistics/profile",
                    )
                total += len(queries)
    if req["phase"] == "witness":
        require(data["roundtrips"] == {}, "no invented witness baseline acquisition")
    else:
        require(set(data["roundtrips"]) == set(req["cells"]), "round-trip cell inventory")
        for cell_id, cell in req["cells"].items():
            require(set(data["roundtrips"][cell_id]) == set(CONDITIONS), "round-trip condition inventory")
            for condition, rows in data["roundtrips"][cell_id].items():
                primary = data["measurements"][cell_id][condition]["designs"]["canonical"]["rows"]
                require(len(rows) == len(primary), "round-trip query count")
                candidate = cell["conditions"][condition]["specification"]
                for row, main in zip(rows, primary, strict=True):
                    require(
                        set(row)
                        == {"query", "self_encoded", "intended_decoded", "cross_encoded", "cross_decoded"}
                        and row["query"] == main["query"],
                        "round-trip row identity",
                    )
                    expected = {
                        "self_encoded": transform(candidate, main["output"], inverse=False),
                        "intended_decoded": transform(cell["intended"], main["input"], inverse=True),
                    }
                    # Compare every native boundary against its actual previous
                    # output, rather than assuming an exact round trip.
                    expected["cross_encoded"] = transform(
                        cell["intended"], row["intended_decoded"], inverse=False
                    )
                    expected["cross_decoded"] = transform(candidate, row["cross_encoded"], inverse=True)
                    require(
                        all(near(row[k], v) for k, v in expected.items()), "native round-trip boundary parity"
                    )
                    auxiliary += 4
    runtime = data["runtime"]
    require(
        runtime["main_decoder_calls"] == total
        and runtime["auxiliary_native_calls"] == auxiliary
        and runtime["network_events"] == []
        and type(runtime["peak_rss_bytes"]) is int
        and 0 < runtime["peak_rss_bytes"] <= 2 * 1024**3,
        "actual runtime counts/limits",
    )
    require(
        all(
            finite(runtime[k]) and runtime[k] >= 0
            for k in ("setup_seconds", "main_seconds", "roundtrip_seconds", "wall_seconds")
        ),
        "finite measured timings",
    )


def evidence(s, rows):
    return {"suite_sha256": s.sha256, "outputs": {r["query"]: r["output"] for r in rows}}


def comparisons(s, ref, obs):
    return {
        "compiled": evaluate(s, ref, obs, **TOLERANCES),
        "independent_pointwise": pointwise(s, ref, obs, **TOLERANCES),
        "strong_relational": relational(s, ref, obs, **TOLERANCES),
    }


def certificate_adapter(cell, condition, rows, ref):
    spec, candidate = cell["intended"], cell["conditions"][condition]["specification"]
    native_profile = spec["profile"]
    profile = "lerobot_mean_std_extension" if "mean_std" in native_profile else "openpi_quantile_extension"
    inputs = [r["input"] for r in rows]
    intended, deployed = copy.deepcopy(spec["statistics"]), copy.deepcopy(candidate["statistics"])
    adapter = "native normalized coordinates"
    if native_profile == "lerobot_quantiles_f64_v1":
        profile = "paper_quantile_equations"
        inputs = [[(v + 1) / 2 for v in row] for row in inputs]
        for stats in (intended, deployed):
            stats["q99"] = [
                a + (b - a if b != a else 1e-8) for a, b in zip(stats["q01"], stats["q99"], strict=True)
            ]
            stats["mask"] = [True] * len(stats["q01"])
        adapter = (
            "Explicit LeRobot quantile coordinate extension: "
            "z_paper=(z_native+1)/2; effective zero-width 1e-8"
        )
    intended["authority"] = deployed["authority"] = ref.authority
    selected = cell["conditions"][condition]["selected_coordinate"]
    payload = {
        "schema_version": 1,
        "profile": profile,
        "coordinates": [
            {"name": name, "unit": u} for name, u in zip(suite(cell).outputs, ref.output_units, strict=True)
        ],
        "intended": intended,
        "candidate": deployed,
        "calibration": {
            "kind": "designed",
            "space": "normalized",
            "authority": ref.authority,
            "values": inputs,
        },
        "metric": {"indices": [selected], "unit": ref.output_units[selected]},
        "gripper_indices": [],
        "thresholds": [],
        "percentiles": [0, 0.5, 1],
    }
    result = certificate(payload)
    require(
        all(
            near(actual["output"], predicted)
            for actual, predicted in zip(rows, result["candidate_decoded"], strict=True)
        ),
        "declared certificate profile/coordinate extension matches native decoding",
    )
    return {
        "adapter": adapter,
        "request": payload,
        "report": result,
        "scope": (
            "Prior-paper equations or explicit numerical extension; "
            "finite designed actions, no conformance decision/safety claim"
        ),
    }


def analyze(captures):
    reports, selected = {}, {"lerobot": {}, "openpi": {}}
    for cell_id, cell in development_inputs().items():
        s, ds = suite(cell), designs()[cell_id]
        capture = captures[cell["framework"]]
        reports[cell_id] = {}
        for condition in CONDITIONS:
            native = capture["measurements"][cell_id][condition]
            rows = native["designs"]["canonical"]["rows"]
            obs, ref = evidence(s, rows), reference(cell)
            methods = comparisons(s, ref, obs)
            geometry = {}
            for name, design in ds.items():
                measured = native["designs"][name]["rows"]
                geometry[name] = qd.analyze(
                    design,
                    ref.to_dict(),
                    {
                        "design_sha256": design.sha256,
                        "outputs": {r["query"]: r["output"] for r in measured},
                    },
                    **TOLERANCES,
                    measurement_allowance=1e-12,
                )
            roundtrips = capture["roundtrips"][cell_id][condition]
            roundtrip_report = {
                "candidate_self": {
                    "status": "WITHIN_TOLERANCE"
                    if all(
                        near(r["self_encoded"], p["input"], **TOLERANCES)
                        for r, p in zip(roundtrips, rows, strict=True)
                    )
                    else "DIFFERS",
                    "scope": "Native decode/encode consistency; zero std is lossy; no conformance decision",
                },
                "intended_to_candidate": {
                    "numeric_status": "PASS"
                    if all(near(r["cross_decoded"], r["intended_decoded"], **TOLERANCES) for r in roundtrips)
                    else "FAIL",
                    "scope": "Intended decode/encode then candidate decode on the same canonical inputs",
                },
                "auxiliary_native_calls": 4 * len(rows),
            }
            partial = ref.to_dict()
            partial["offset"] = [None] * len(s.outputs)
            erased = ref.to_dict()
            erased.update(
                coefficients=[[None] * len(s.axes) for _ in s.outputs],
                offset=[None] * len(s.outputs),
                authority=None,
                output_units=[None] * len(s.outputs),
                affine_fragment_verified=False,
            )
            witness_ids = []
            witness_methods = {}
            for name, method in methods.items():
                if method["status"] == "FAIL":
                    ids = method["first_divergence"]["queries"]
                    witness_ids.extend(ids)
                    witness_methods[name] = {
                        "query_ids": ids,
                        "original_relation": method["first_divergence"],
                    }
            if witness_ids:
                selected[cell["framework"]].setdefault(cell_id, {})[condition] = list(
                    dict.fromkeys(witness_ids)
                )
            reports[cell_id][condition] = {
                "methods": methods,
                "geometry_shared_oracle": geometry,
                "static_artifact_comparison": {
                    "status": "PASS"
                    if cell["intended"] == cell["conditions"][condition]["specification"]
                    else "FAIL",
                    "scope": (
                        "Supplied independent intended statistics versus actual deployment artifact; "
                        "no executed behavior proof"
                    ),
                },
                "native_api": {"status": "EXECUTED_WITHOUT_ERROR", "conformance_status": "NOT_ASSERTED"},
                "roundtrips": roundtrip_report,
                "certificate": certificate_adapter(cell, condition, rows, ref),
                "ablations": {
                    "offset_authority_erased": comparisons(s, Reference(**partial), obs),
                    "independent_reference_erased": comparisons(s, Reference(**erased), obs),
                    "provenance_erased": {
                        "reduced_numeric_status": methods["compiled"]["status"],
                        "full_binding_status": "UNKNOWN",
                        "scope": "A reduced check set cannot certify absent provenance",
                    },
                    "state": {
                        "applicability": "N/A",
                        "reason": "Action-only operations have no temporal state ports",
                    },
                    "without_witness_reduction": {
                        "queries_retained": len(s.probes),
                        "detection_unchanged": True,
                    },
                },
                "fresh_witness_selection": witness_methods,
                "checkpoint_physical_status": "UNKNOWN",
            }
        for erased_name, erased_kind in (
            ("reference_authority_erased", "authority"),
            ("output_units_erased", "units"),
        ):
            rows = capture["measurements"][cell_id]["numeric_control"]["designs"]["canonical"]["rows"]
            ref = reference(cell, erased=erased_kind)
            reports[cell_id][erased_name] = {
                "derived_from": "numeric_control; same execution, not another native/independent sample",
                "methods": comparisons(s, ref, evidence(s, rows)),
                "certificate": certificate_adapter(cell, "numeric_control", rows, ref),
                "checkpoint_physical_status": "UNKNOWN",
            }
    return {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "tolerances": TOLERANCES,
        "cells": reports,
        "limitations": [
            "Four development cells, not prospective results",
            "Injected variants/seeds/erasures are repeated measurements",
            "Numeric authority and raw-index units do not establish physical checkpoint conformance",
            "Stateless action checks are the independent pointwise workflow in this stratum",
            "External native adapters/applicability and committed final protocol remain unfinished",
        ],
    }, selected


def verify_witnesses(captures, fresh, report, selections):
    results = {}
    for framework, cells in selections.items():
        results[framework] = {}
        for cell_id, conditions in cells.items():
            cell, s = development_inputs()[cell_id], suite(development_inputs()[cell_id])
            results[framework][cell_id] = {}
            for condition in conditions:
                rows = fresh[framework]["measurements"][cell_id][condition]["designs"]["canonical"]["rows"]
                original = captures[framework]["measurements"][cell_id][condition]["designs"]["canonical"][
                    "rows"
                ]
                require(
                    all(row in original for row in rows),
                    "fresh native witness response reproduces actual original query",
                )
                retained = {}
                for method, item in report["cells"][cell_id][condition]["fresh_witness_selection"].items():
                    subset = [r for r in rows if r["query"] in item["query_ids"]]
                    result = comparisons(s, reference(cell), evidence(s, subset))[method]
                    findings = result.get("findings", [])
                    if not findings:
                        # Research oracle reports retain the first divergence;
                        # compare that complete relation when no finding list is exposed.
                        findings = [result["first_divergence"]]
                    original_relation = item["original_relation"]
                    require(
                        result["status"] == "FAIL" and original_relation in findings,
                        "same original failed relation in fresh native witness",
                    )
                    retained[method] = {
                        "status": "FAIL",
                        "query_ids": item["query_ids"],
                        "native_calls_in_method_view": len(subset),
                        "relation": original_relation,
                    }
                results[framework][cell_id][condition] = {
                    "methods": retained,
                    "actual_shared_fresh_decoder_calls": len(rows),
                }
    return results


def acquire(folder, framework, python, selected=None):
    prefix = framework + (".witness" if selected is not None else ".main")
    req = request(framework, selected)
    path = folder / (prefix + ".request.json")
    write(path, req)
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUTF8="1",
    )
    native = subprocess.run(
        [str(python.resolve()), "-I", str(ROOT / "scripts/action_study_worker.py")],
        input=path.read_bytes(),
        capture_output=True,
        timeout=180,
        env=env,
        check=False,
    )
    (folder / (prefix + ".stdout.txt")).write_bytes(native.stdout)
    (folder / (prefix + ".stderr.txt")).write_bytes(native.stderr)
    require(native.returncode == 0, f"native action-only capture: {native.stderr[-2500:]!r}")
    data = json.loads(native.stdout)
    write(folder / (prefix + ".capture.json"), data)
    require(data["request_sha256"] == sha(path), "native exact request bytes")
    validate_capture(data, req)
    return data


def expected_files():
    return {
        "inputs.json",
        "designs.json",
        "analysis.json",
        "witnesses.json",
        *(
            f"{fw}.{phase}.{suffix}"
            for fw in ("lerobot", "openpi")
            for phase in ("main", "witness")
            for suffix in ("request.json", "stdout.txt", "stderr.txt", "capture.json")
        ),
    }


def run(folder, pythons):
    initial = sources()
    write(folder / "inputs.json", development_inputs())
    write(
        folder / "designs.json", {k: {n: d.to_dict() for n, d in ds.items()} for k, ds in designs().items()}
    )
    captures = {fw: acquire(folder, fw, pythons[fw]) for fw in ("lerobot", "openpi")}
    start = time.perf_counter()
    report, selections = analyze(captures)
    evaluation_seconds = time.perf_counter() - start
    write(folder / "analysis.json", report)
    fresh = {fw: acquire(folder, fw, pythons[fw], selections[fw]) for fw in ("lerobot", "openpi")}
    write(folder / "witnesses.json", verify_witnesses(captures, fresh, report, selections))
    require(sources() == initial, "stable development producer")
    write(
        folder / "summary.json",
        {
            "schema_version": 1,
            "development_only": True,
            "final_evaluation": False,
            "prospective_execution_outcomes_observed": False,
            "source_file_sha256": initial,
            "evaluation_seconds": evaluation_seconds,
            "file_sha256": {n: sha(folder / n) for n in sorted(expected_files())},
        },
    )
    verify(folder)


def verify(folder):
    def read(n):
        return json.loads((folder / n).read_bytes())

    summary = read("summary.json")
    require(
        summary["schema_version"] == 1
        and summary["development_only"] is True
        and summary["final_evaluation"] is False
        and summary["prospective_execution_outcomes_observed"] is False,
        "development study scope",
    )
    require(
        summary["source_file_sha256"] == sources() and set(summary["file_sha256"]) == expected_files(),
        "closed evidence/producer manifest",
    )
    require(
        all(sha(folder / n) == h for n, h in summary["file_sha256"].items()), "exact evidence file identities"
    )
    require(read("inputs.json") == development_inputs(), "fixed development-only inputs")
    require(
        read("designs.json") == {k: {n: d.to_dict() for n, d in ds.items()} for k, ds in designs().items()},
        "declared complete action-only designs",
    )
    require(finite(summary["evaluation_seconds"]) and summary["evaluation_seconds"] >= 0, "evaluation cost")
    captures = {}
    for fw in ("lerobot", "openpi"):
        req = read(fw + ".main.request.json")
        require(req == request(fw), "complete main request")
        data = read(fw + ".main.capture.json")
        require(
            data == read(fw + ".main.stdout.txt")
            and data["request_sha256"] == sha(folder / (fw + ".main.request.json")),
            "raw native request/output",
        )
        validate_capture(data, req)
        captures[fw] = data
    report, selections = analyze(captures)
    require(read("analysis.json") == report, "deterministic independent numerical analysis")
    fresh = {}
    for fw in ("lerobot", "openpi"):
        req = read(fw + ".witness.request.json")
        require(req == request(fw, selections[fw]), "original relation's fresh witness request")
        data = read(fw + ".witness.capture.json")
        require(
            data == read(fw + ".witness.stdout.txt")
            and data["request_sha256"] == sha(folder / (fw + ".witness.request.json")),
            "raw native witness binding",
        )
        validate_capture(data, req)
        fresh[fw] = data
    require(
        read("witnesses.json") == verify_witnesses(captures, fresh, report, selections),
        "fresh witness analysis",
    )
    require(
        not any(n in sys.modules for n in ("torch", "numpy", "lerobot", "openpi")), "data-only parent replay"
    )


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
            args.lerobot_python is not None and args.openpi_python is not None, "explicit locked interpreters"
        )
        folder = args.output.resolve()
        require(folder.is_relative_to(ROOT), "workspace-contained evidence")
        folder.mkdir(parents=True, exist_ok=False)
        run(folder, {"lerobot": args.lerobot_python, "openpi": args.openpi_python})
    print("Verified action-only development execution, matched numerical workflows and fresh witnesses")


if __name__ == "__main__":
    main()
