"""Fixed numerical analysis over file-backed decoder captures and fresh witnesses.

This module acquires no observations. Physical/checkpoint authority is never
inferred from numerical agreement. Query/condition rows are repeated measures.
"""

import copy
import hashlib
import json
from collections import Counter

import query_designs as qd
import run_action_study as numerical
from action_designs import ActionDesign
from action_study_common import ROOT, fingerprint
from normalization_profiles import require, transform
from study_assets import action_statistics

from mltools.obligations import Reference


def identity(data, framework):
    native = data["upstream_identity"]
    revision, blobs = (
        (numerical.LEROBOT_REVISION, numerical.LEROBOT_BLOBS)
        if framework == "lerobot"
        else (numerical.OPENPI_REVISION, numerical.OPENPI_BLOBS)
    )
    require(
        data["framework"] == framework
        and native["revision"] == revision
        and native["git_blobs"] == blobs
        and native["source_newlines"] == "canonical LF",
        "pinned actual decoder identity",
    )
    if framework == "openpi":
        require(
            native["client_image_tools_blob"] == "7a971b9d5f6b1495fd6cdea202ffa607d8b34bf0"
            and set(native["direct_urls"]) == {"openpi", "openpi-client"}
            and all(
                value["url"] == "https://github.com/Physical-Intelligence/openpi.git"
                and value["vcs_info"]["commit_id"] == revision
                for value in native["direct_urls"].values()
            )
            and native["direct_urls"]["openpi-client"]["subdirectory"] == "packages/openpi-client",
            "pinned OpenPI client and both installed distributions",
        )


def configuration(cell):
    if cell["framework"] == "openpi":
        return {
            "encoder": "openpi.transforms.Normalize",
            "decoder": "openpi.transforms.Unnormalize",
            "use_quantiles": True,
            "device": "cpu",
            "dtype": "float64",
        }
    n = len(numerical.suite(cell).outputs)
    profile = {
        "eps": 1e-8,
        "features": {"action": {"type": "ACTION", "shape": [n]}},
        "norm_map": {"ACTION": "MEAN_STD" if "mean_std" in cell["intended"]["profile"] else "QUANTILES"},
    }
    return {"encoder": profile, "decoder": profile, "device": "cpu", "dtype": "float64"}


def validate_decoder(data, plan, folder, framework, selected=None):
    """Validate all native boundaries, with exact full/subset inventories."""
    identity(data, framework)
    cells = {k: v for k, v in plan["cells"].items() if v["framework"] == framework}
    require(cells and set(data["cells"]) == set(cells), "exact per-framework native cell inventory")
    require(selected is None or set(selected) <= set(cells), "fresh subset cell inventory")
    primary, auxiliary = 0, 0
    for cell_id, cell in cells.items():
        conditions = cell["conditions"] if selected is None else selected.get(cell_id, {})
        require(set(data["cells"][cell_id]) == set(conditions), "exact native condition subset")
        for condition in conditions:
            entry, actual = cell["conditions"][condition], data["cells"][cell_id][condition]
            raw = (folder / entry["directory"] / entry["post_state_file"]).read_bytes()
            require(
                hashlib.sha256(raw).hexdigest() == entry["file_sha256"][entry["post_state_file"]],
                "exact single-snapshot deployed statistics",
            )
            spec = entry["specification"]
            fields = tuple(spec["statistics"])
            n = len(numerical.suite(cell).outputs)
            independently_decoded = action_statistics(raw, n, fields)
            header_size = int.from_bytes(raw[:8], "little")
            header = json.loads(raw[8 : 8 + header_size])
            require(
                fingerprint(actual["configuration"]) == fingerprint(configuration(cell))
                and actual["loaded_artifact"]
                == {
                    "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                    "statistics": independently_decoded,
                    "tensor_dtypes": {
                        k: "torch.float" + ("32" if header["action." + k]["dtype"] == "F32" else "64")
                        for k in fields
                    },
                }
                and independently_decoded == spec["statistics"],
                "native loaded artifact values/dtypes and explicit runtime profile",
            )
            names = set(cell["designs"]) if selected is None else {"canonical"}
            require(set(actual["designs"]) == names, "exact full/fresh design inventory")
            for name, rows in actual["designs"].items():
                design = ActionDesign.from_dict(cell["designs"][name])
                queries = list(design.probes)
                if selected is not None:
                    ids = selected[cell_id][condition]
                    require(
                        isinstance(ids, list)
                        and ids
                        and len(set(ids)) == len(ids)
                        and set(ids) <= {q.id for q in queries},
                        "bounded unique fresh witness IDs",
                    )
                    queries = [q for q in queries if q.id in ids]
                require(len(rows) == len(queries), "one actual native call per retained query")
                for row, query in zip(rows, queries, strict=True):
                    require(
                        set(row) == {"query", "input", "output"}
                        and row["query"] == query.id
                        and fingerprint(row["input"]) == fingerprint(list(query.values))
                        and numerical.near(row["output"], transform(spec, row["input"], inverse=True)),
                        "actual file-backed response for the exact ordered input",
                    )
                    primary += 1
            if selected is not None:
                require("roundtrips" not in actual, "no hidden witness roundtrip acquisition")
                continue
            main = actual["designs"]["canonical"]
            require(len(actual["roundtrips"]) == len(main), "native canonical roundtrip denominator")
            for row, original in zip(actual["roundtrips"], main, strict=True):
                require(row["query"] == original["query"], "roundtrip input identity")
                expected = {
                    "self_encoded": transform(spec, original["output"], inverse=False),
                    "intended_decoded": transform(cell["intended"], original["input"], inverse=True),
                    "cross_encoded": transform(cell["intended"], row["intended_decoded"], inverse=False),
                    "cross_decoded": transform(spec, row["cross_encoded"], inverse=True),
                }
                require(all(numerical.near(row[k], v) for k, v in expected.items()), "all native boundaries")
                auxiliary += 4
    require(
        type(data["runtime"]["primary_decoder_calls"]) is int
        and data["runtime"]["primary_decoder_calls"] == primary
        and type(data["runtime"]["auxiliary_native_calls"]) is int
        and data["runtime"]["auxiliary_native_calls"] == auxiliary,
        "actual primary and auxiliary native call accounting",
    )


def merge_decoders(plan, captures):
    require(set(captures) == {cell["framework"] for cell in plan["cells"].values()}, "all native frameworks")
    merged = {}
    for framework, data in captures.items():
        expected = {k for k, cell in plan["cells"].items() if cell["framework"] == framework}
        require(set(data["cells"]) == expected and not set(merged) & expected, "disjoint framework cells")
        merged.update(data["cells"])
    return {"cells": merged}


def methods(cell, rows, erased=None):
    s = numerical.suite(cell)
    ref = numerical.reference(cell, erased=erased)
    return numerical.comparisons(s, ref, numerical.evidence(s, rows))


def geometry(design, ref, rows, role):
    raw = qd.analyze(
        design,
        ref.to_dict(),
        {"design_sha256": design.sha256, "outputs": {r["query"]: r["output"] for r in rows}},
        **numerical.TOLERANCES,
        measurement_allowance=1e-12,
    )
    original_tag = raw.pop("development_only")
    return {
        **raw,
        "role": role,
        "implementation_scope": {
            "source": "scripts/query_designs.py",
            "original_implementation_development_tag": original_tag,
            "meaning": "Frozen research implementation evaluated under the explicit current input role",
        },
    }


def select_witnesses(plan, decoder):
    """Canonical selection only; geometry/certificate work is not needed here."""
    result = {cell["framework"]: {} for cell in plan["cells"].values()}
    for cell_id, cell in plan["cells"].items():
        for condition in cell["conditions"]:
            rows = decoder["cells"][cell_id][condition]["designs"]["canonical"]
            ids = [
                query
                for method in methods(cell, rows).values()
                if method["status"] == "FAIL"
                for query in method["first_divergence"]["queries"]
            ]
            if ids:
                result[cell["framework"]].setdefault(cell_id, {})[condition] = list(dict.fromkeys(ids))
    return result


def analyze(plan, decoder):
    reports, selected = {}, {v["framework"]: {} for v in plan["cells"].values()}
    for cell_id, cell in plan["cells"].items():
        reports[cell_id] = {}
        s, ref = numerical.suite(cell), numerical.reference(cell)
        for condition, entry in cell["conditions"].items():
            capture = decoder["cells"][cell_id][condition]
            rows = capture["designs"]["canonical"]
            values = methods(cell, rows)
            witness, ids = {}, []
            for name, result in values.items():
                if result["status"] == "FAIL":
                    divergence = result["first_divergence"]
                    retained = divergence["queries"]
                    witness[name] = {"query_ids": retained, "original_relation": divergence}
                    ids.extend(retained)
            if ids:
                selected[cell["framework"]].setdefault(cell_id, {})[condition] = list(dict.fromkeys(ids))
            partial = ref.to_dict()
            partial["offset"] = [None] * len(s.outputs)
            erased = copy.deepcopy(partial)
            erased.update(
                coefficients=[[None] * len(s.axes) for _ in s.outputs],
                authority=None,
                output_units=[None] * len(s.outputs),
                affine_fragment_verified=False,
            )
            roundtrips = capture["roundtrips"]
            reports[cell_id][condition] = {
                "numerical_methods": values,
                "geometry_shared_oracle": {
                    name: geometry(
                        ActionDesign.from_dict(cell["designs"][name]),
                        ref,
                        observed,
                        plan["role"],
                    )
                    for name, observed in capture["designs"].items()
                },
                "static_numeric_statistics": {
                    "status": "PASS" if entry["specification"] == cell["intended"] else "FAIL",
                    "scope": "Independently decoded intended and actual deployed numbers, not byte identity",
                },
                "native_api": {"status": "EXECUTED_WITHOUT_ERROR", "conformance_status": "NOT_ASSERTED"},
                "roundtrips": {
                    "candidate_self": "WITHIN_TOLERANCE"
                    if all(
                        numerical.near(r["self_encoded"], p["input"], **numerical.TOLERANCES)
                        for r, p in zip(roundtrips, rows, strict=True)
                    )
                    else "DIFFERS",
                    "independent_intended_to_candidate": "PASS"
                    if all(
                        numerical.near(r["cross_decoded"], r["intended_decoded"], **numerical.TOLERANCES)
                        for r in roundtrips
                    )
                    else "FAIL",
                    "auxiliary_native_calls": 4 * len(rows),
                },
                "certificate": numerical.certificate_adapter(cell, condition, rows, ref),
                "ablations": {
                    "offset_authority_erased": numerical.comparisons(
                        s, Reference(**partial), numerical.evidence(s, rows)
                    ),
                    "independent_reference_erased": numerical.comparisons(
                        s, Reference(**erased), numerical.evidence(s, rows)
                    ),
                    "provenance_erased": {
                        "reduced_numeric_status": values["compiled"]["status"],
                        "full_binding_status": "UNKNOWN",
                    },
                    "state": {
                        "applicability": "N/A",
                        "reason": "No temporal state port in action-only operations",
                    },
                    "without_witness_reduction": {
                        "queries_retained": len(s.probes),
                        "detection_unchanged": True,
                    },
                },
                "fresh_witness_selection": witness,
                "checkpoint_physical_status": "UNKNOWN",
            }
        for name, kind in (("reference_authority_erased", "authority"), ("output_units_erased", "units")):
            rows = decoder["cells"][cell_id]["numeric_control"]["designs"]["canonical"]
            reports[cell_id][name] = {
                "derived_from": "numeric_control; no additional execution or independent unit",
                "numerical_methods": methods(cell, rows, kind),
                "certificate": numerical.certificate_adapter(
                    cell, "numeric_control", rows, numerical.reference(cell, erased=kind)
                ),
                "checkpoint_physical_status": "UNKNOWN",
            }
    return {
        "schema_version": 1,
        "role": plan["role"],
        "final_evaluation": plan["role"] == "prospective",
        "tolerances": numerical.TOLERANCES,
        "cells": reports,
    }, selected


def verify_witnesses(plan, decoder, fresh, report, selections):
    result = {}
    for framework, selected_cells in selections.items():
        result[framework] = {}
        for cell_id, conditions in selected_cells.items():
            cell = plan["cells"][cell_id]
            result[framework][cell_id] = {}
            for condition in conditions:
                rows = fresh[framework]["cells"][cell_id][condition]["designs"]["canonical"]
                original = decoder["cells"][cell_id][condition]["designs"]["canonical"]
                require(
                    all(row in original for row in rows),
                    "fresh exact-artifact response repeats original query",
                )
                retained = {}
                for method, item in report["cells"][cell_id][condition]["fresh_witness_selection"].items():
                    subset = [r for r in rows if r["query"] in item["query_ids"]]
                    checked = methods(cell, subset)[method]
                    findings = checked.get("findings") or [checked["first_divergence"]]
                    require(
                        checked["status"] == "FAIL" and item["original_relation"] in findings,
                        "same original failed relation, not merely another failure",
                    )
                    retained[method] = {
                        "status": "FAIL",
                        "query_ids": item["query_ids"],
                        "native_calls_in_method_view": len(subset),
                        "relation": item["original_relation"],
                    }
                result[framework][cell_id][condition] = {
                    "methods": retained,
                    "actual_shared_fresh_decoder_calls": len(rows),
                }
    return result


def metrics(plan, report, witnesses, phases):
    """Finite counts and descriptive artifact-weighted sensitivity, no population CI."""
    names = ("compiled", "independent_pointwise", "strong_relational")
    cells, clusters = {}, {}
    for cell_id, cell in plan["cells"].items():
        rows = report["cells"][cell_id]
        faults = [k for k, v in cell["conditions"].items() if v["specification"] != cell["intended"]]
        controls = [k for k in cell["conditions"] if k not in faults]
        cell_metrics = {}
        for name in names:
            counts = Counter(row["numerical_methods"][name]["status"] for row in rows.values())
            cell_metrics[name] = {
                "fault_fail": sum(rows[k]["numerical_methods"][name]["status"] == "FAIL" for k in faults),
                "declared_faults": len(faults),
                "control_fail": sum(rows[k]["numerical_methods"][name]["status"] == "FAIL" for k in controls),
                "numeric_controls": len(controls),
                "numeric_status_counts_including_derived_views": {
                    k: counts[k] for k in ("PASS", "FAIL", "UNKNOWN")
                },
            }
        cells[cell_id] = {
            "cluster": cell["cluster"],
            "methods": cell_metrics,
            "paired_oracle_verdict_disagreements": sum(
                len({row["numerical_methods"][name]["status"] for name in names}) > 1 for row in rows.values()
            ),
            "physical_status_counts_including_derived_views": {"UNKNOWN": len(rows)},
            "applicable_numeric_workflows": {
                name: {
                    "fault_fail": sum(getter(rows[k]) == "FAIL" for k in faults),
                    "declared_faults": len(faults),
                    "control_fail": sum(getter(rows[k]) == "FAIL" for k in controls),
                    "numeric_controls": len(controls),
                }
                for name, getter in (
                    ("static_statistics", lambda r: r["static_numeric_statistics"]["status"]),
                    ("independent_roundtrip", lambda r: r["roundtrips"]["independent_intended_to_candidate"]),
                )
            },
            "self_roundtrip_consistency_counts": dict(
                Counter(rows[k]["roundtrips"]["candidate_self"] for k in cell["conditions"])
            ),
            "geometry_shared_oracle_counts": {
                design: {
                    key: dict(
                        Counter(rows[k]["geometry_shared_oracle"][design][key] for k in cell["conditions"])
                    )
                    for key in ("finite_status", "full_affine_request_status")
                }
                for design in cell["designs"]
            },
        }
        clusters.setdefault(cell["cluster"], []).append(cell_id)
    artifact = {}
    for cluster, ids in clusters.items():
        artifact[cluster] = {"execution_cells": ids, "methods": {}}
        for method in names:
            counts = {
                key: sum(cells[i]["methods"][method][key] for i in ids)
                for key in ("fault_fail", "declared_faults", "control_fail", "numeric_controls")
            }
            artifact[cluster]["methods"][method] = {
                **counts,
                "finite_fault_detection_fraction": counts["fault_fail"] / counts["declared_faults"],
                "finite_control_fail_fraction": counts["control_fail"] / counts["numeric_controls"],
            }
    leave_one = {}
    if len(artifact) > 1:
        for omitted in artifact:
            leave_one[omitted] = {
                method: sum(
                    v["methods"][method]["finite_fault_detection_fraction"]
                    for key, v in artifact.items()
                    if key != omitted
                )
                / (len(artifact) - 1)
                for method in names
            }
    fresh_calls = sum(
        row["actual_shared_fresh_decoder_calls"]
        for framework in witnesses.values()
        for cell in framework.values()
        for row in cell.values()
    )
    return {
        "schema_version": 1,
        "role": plan["role"],
        "final_evaluation": plan["role"] == "prospective",
        "cells": cells,
        "artifact_groups": artifact,
        "leave_one_artifact_out_macro_detection_fraction": leave_one,
        "uncertainty": (
            "Descriptive equal-artifact-weight sensitivity only; "
            "no independent clusters, CI or population accuracy claim"
        ),
        "counts": {
            "execution_cells": len(cells),
            "artifact_groups": len(artifact),
            "actual_conditions": sum(len(v["conditions"]) for v in plan["cells"].values()),
            "derived_views": 2 * len(cells),
            "primary_decoder_calls": sum(
                v["runtime"]["primary_decoder_calls"] for k, v in phases.items() if k in ("lerobot", "openpi")
            ),
            "auxiliary_native_calls": sum(
                v["runtime"]["auxiliary_native_calls"]
                for k, v in phases.items()
                if k in ("lerobot", "openpi")
            ),
            "actual_shared_fresh_decoder_calls": fresh_calls,
            "policy_space_replay_queries": phases["policy_space"]["runtime"]["accepted_replay_queries"],
            "policy_space_structural_controls": phases["policy_space"]["runtime"][
                "rejected_structural_controls"
            ],
        },
        "phase_costs": {k: v["runtime"] for k, v in phases.items()},
        "tracked_implementation_lines": {
            name: len((ROOT / "scripts" / name).read_text(encoding="utf-8").splitlines())
            for name in (
                "publication_analysis.py",
                "publication_protocol.py",
                "publication_assets.py",
                "publication_network.py",
                "publication_worker.py",
                "run_publication_study.py",
                "matched_study_worker.py",
                "policy_space_numeric.py",
                "policy_space_numeric_worker.py",
            )
        },
        "effort_scope": (
            "Tracked physical source-line counts and explicit commands, "
            "not human setup times or equivalent conceptual effort"
        ),
        "cost_scope": (
            "Actual fixed-order implementation costs, not isolated method runtime rankings; "
            "fresh method views share one charged acquisition"
        ),
    }
