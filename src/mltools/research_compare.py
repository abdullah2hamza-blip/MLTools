"""Matched-evidence development oracle comparisons and explicitly scoped ablations."""

from dataclasses import replace
from pathlib import Path

from .io import read_json
from .obligations import evaluate
from .probe_cases import check, load
from .report import aggregate
from .research_oracles import fingerprint, pointwise, relational, stateless


def _compiled(suite, reference, evidence):
    report = evaluate(suite, reference, evidence)
    return {
        "method": report["method"],
        "status": report["status"],
        "applicability": "APPLICABLE",
        "suite_sha256": report["suite_sha256"],
        "reference_sha256": report["reference_sha256"],
        "evidence_sha256": report["evidence_sha256"],
        "first_divergence": report["first_divergence"],
        "coverage": report["coverage"],
        "comparison_count": len(report["findings"]),
        "witness_queries": [q["query"] for q in report["witness"]],
        "scope": "Compiled finite affine obligations plus design-rank diagnostic",
    }


def _methods(suite, reference, evidence, common):
    methods = {
        "compiled": _compiled(suite, reference, evidence),
        "pointwise": pointwise(suite, reference, evidence),
        "strong_relational": relational(suite, reference, evidence),
        "stateless_diagonal": stateless(suite, reference, evidence),
    }
    for result in methods.values():
        statuses = [*common.values(), result["status"]]
        if result.get("full_design_coverage_status") == "UNKNOWN":
            statuses.append("UNKNOWN")
        # Subdomain PASS cannot clear the unobserved temporal part of the request.
        if result.get("excluded_queries", 0):
            statuses.append("UNKNOWN")
        result["full_requested_status"] = aggregate(statuses)
    return methods


def compare(path, trace):
    """Analyze an immutable capture; altered references are not new executed cases.

    Baselines share authority, outputs, source/artifact checks and tolerance defaults.
    Their independent algebra and domains are reported without a superiority score.
    """
    if isinstance(trace, (str, Path)):
        trace = read_json(Path(trace))
    baseline = check(path, trace)
    _case, _root, suite, reference = load(path)
    bound = baseline["checks"]["binding"] == "PASS"
    evidence = trace if bound else {"suite_sha256": suite.sha256, "outputs": {}}
    common = {name: status for name, status in baseline["checks"].items() if name != "obligations"}
    methods = _methods(suite, reference, evidence, common)
    partial = replace(reference, offset=(None,) * len(suite.outputs))
    erased = replace(
        reference,
        coefficients=tuple((None,) * len(suite.axes) for _ in suite.outputs),
        offset=(None,) * len(suite.outputs),
        output_units=(None,) * len(suite.outputs),
        authority=None,
        affine_fragment_verified=False,
    )
    numeric_erased = _compiled(suite, erased, evidence)
    partial_methods = _methods(suite, partial, evidence, common)
    retained = {name: status for name, status in common.items() if name not in ("producer", "artifacts")}
    witness = methods["compiled"]["witness_queries"]
    return {
        "schema_version": 1,
        "format": "mltools-oracle-comparison-development",
        "development_only": True,
        "final_evaluation": False,
        "case_sha256": baseline["case_sha256"],
        "capture_trace_sha256": baseline["trace_sha256"],
        "numeric_observations_withheld": not bound,
        "common_checks": common,
        "methods": methods,
        "ablations": {
            "offset_authority_erased": {
                "operation": "Erase reference offsets; preserve original case and capture identities",
                "reference_sha256": fingerprint(partial.to_dict()),
                "methods": partial_methods,
            },
            "independent_reference_erased": {
                "operation": "Erase independent coefficients, offsets, units and authority",
                "numeric_status": numeric_erased["status"],
                "full_requested_status": aggregate([*common.values(), numeric_erased["status"]]),
                "reference_sha256": fingerprint(erased.to_dict()),
                "scope": "Missing independent reference remains UNKNOWN; no self-fitted expectations",
            },
            "provenance_checks_omitted": {
                "removed_checks": ["producer", "artifacts"],
                "retained_common_checks": retained,
                "reduced_scope_status": aggregate([*retained.values(), methods["compiled"]["status"]]),
                "original_full_requested_status": baseline["status"],
                "scope": "Reduced check set, not certification of original provenance requirements",
            },
            "witness_reduction_disabled": {
                "full_requested_status": baseline["status"],
                "original_queries": len(suite.probes),
                "compiled_local_witness_queries": len(witness) if witness else None,
                "retained_without_reduction": len(suite.probes),
                "scope": "Same detection; diagnostic size only. Fresh replay and reduction cost are separate",
            },
        },
        "limitations": [
            "Supplied canonical design only; manual/random query selection comparisons remain required",
            "Relational oracle includes pointwise, all-pairs differences and geometric midpoint checks",
            "Stateless eligibility uses saved values and declared units; no synthetic diagonal observations",
            "Reference ablations are data-only analyses, not newly captured or re-bound execution evidence",
            "Numerical relations have different conditioning; no universal detection advantage is implied",
            "Original v0.1 stateful checker and applicable external-tool comparisons remain required",
        ],
    }
