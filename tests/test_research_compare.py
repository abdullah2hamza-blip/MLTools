"""Scope and binding regressions using genuine saved development processor traces."""

import copy
from pathlib import Path

from mltools.io import digest, read_json
from mltools.research_compare import compare

ROOT = Path(__file__).resolve().parents[1]


def saved(name):
    case = ROOT / "examples/probes" / f"{name}.json"
    trace = ROOT / "validation/probe-cases-002" / f"{name.replace('/', '_')}.trace.json"
    return case, read_json(trace)


def test_reference_ablation_preserves_original_capture_identity_and_common_checks():
    case, trace = saved("lerobot/moving_anchor")
    original, case_hash = copy.deepcopy(trace), digest(case)
    result = compare(case, trace)
    assert result["case_sha256"] == case_hash
    assert result["common_checks"] == {"binding": "PASS", "producer": "PASS", "artifacts": "PASS"}
    partial = result["ablations"]["offset_authority_erased"]["methods"]
    assert partial["pointwise"]["status"] == "UNKNOWN"
    assert partial["compiled"]["status"] == partial["strong_relational"]["status"] == "FAIL"
    assert trace == original and digest(case) == case_hash


def test_stateless_pass_cannot_clear_the_full_temporal_request():
    case, trace = saved("lerobot/moving_anchor")
    stateless = compare(case, trace)["methods"]["stateless_diagonal"]
    assert stateless["status"] == "PASS"
    assert stateless["excluded_queries"] > 0
    assert stateless["full_requested_status"] == "UNKNOWN"


def test_genuine_published_artifact_numeric_pass_keeps_physical_unknown_for_every_oracle():
    case, trace = saved("act/metadata")
    result = compare(case, trace)
    for oracle in result["methods"].values():
        assert oracle["status"] == "PASS"
        assert oracle["full_requested_status"] == "UNKNOWN"
    assert result["common_checks"]["physical_semantics"] == "UNKNOWN"


def test_missing_producer_retains_numeric_pass_and_marks_only_reduced_scope_clear():
    case, trace = saved("lerobot/good")
    del trace["identity"]  # Explicit analysis-only metadata erasure, not a new capture.
    result = compare(case, trace)
    assert result["methods"]["compiled"]["status"] == "PASS"
    assert result["methods"]["compiled"]["full_requested_status"] == "UNKNOWN"
    assert result["ablations"]["provenance_checks_omitted"]["reduced_scope_status"] == "PASS"
    assert result["ablations"]["independent_reference_erased"]["numeric_status"] == "UNKNOWN"


def test_foreign_capture_binding_withholds_numerical_evidence_for_every_oracle():
    case, trace = saved("lerobot/moving_anchor")
    trace["case_sha256"] = "wrong"
    result = compare(case, trace)
    assert result["numeric_observations_withheld"]
    assert result["common_checks"]["binding"] == "FAIL"
    for method in result["methods"].values():
        assert method["status"] == "UNKNOWN"
        assert method["full_requested_status"] == "FAIL"


def test_witness_ablation_changes_retention_without_changing_detection():
    case, trace = saved("lerobot/moving_anchor")
    result = compare(case, trace)
    ablation = result["ablations"]["witness_reduction_disabled"]
    assert (
        ablation["full_requested_status"] == result["methods"]["compiled"]["full_requested_status"] == "FAIL"
    )
    assert ablation["compiled_local_witness_queries"] == 2
    assert ablation["retained_without_reduction"] == 37
