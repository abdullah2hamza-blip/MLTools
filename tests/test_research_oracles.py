"""Hand-computed oracle controls; these are algebra tests, not research results."""

import copy
from dataclasses import replace

import pytest

from mltools.io import InputError
from mltools.obligations import Axis, Reference, Suite, compile_suite, evaluate
from mltools.research_oracles import pointwise, relational, stateless


def fixture():
    suite = compile_suite(
        (
            Axis("action", "j", "normalized", -1, 1),
            Axis("origin", "j", "degree", -10, 10),
            Axis("current", "j", "degree", -10, 10),
        ),
        ("j.pos",),
    )
    reference = Reference(((2, 10, 0),), (5,), ("degree",), "hand-computed unit fixture", True)
    outputs = {
        "center": [5],
        "plus:action:j": [7],
        "minus:action:j": [3],
        "plus:origin:j": [15],
        "minus:origin:j": [-5],
        "plus:current:j": [5],
        "minus:current:j": [5],
    }
    return suite, reference, {"suite_sha256": suite.sha256, "outputs": outputs}


def moving(evidence):
    evidence = copy.deepcopy(evidence)
    evidence["outputs"].update(
        {"plus:origin:j": [5], "minus:origin:j": [5], "plus:current:j": [15], "minus:current:j": [-5]}
    )
    return evidence


def test_complete_control_and_fault_do_not_claim_compiler_superiority():
    suite, reference, evidence = fixture()
    for oracle in (pointwise, relational, evaluate):
        assert oracle(suite, reference, evidence)["status"] == "PASS"
        assert oracle(suite, reference, moving(evidence))["status"] == "FAIL"


def test_unknown_offset_cancels_in_strong_manual_and_compiler_only():
    suite, reference, evidence = fixture()
    partial = replace(reference, offset=(None,))
    fault = moving(evidence)
    assert pointwise(suite, partial, fault)["status"] == "UNKNOWN"
    manual = relational(suite, partial, fault)
    assert manual["status"] == evaluate(suite, partial, fault)["status"] == "FAIL"
    assert manual["first_divergence"]["kind"] == "paired"
    assert len(manual["first_divergence"]["queries"]) == 2
    # Correct effects cannot establish the missing absolute offset.
    correct = relational(suite, partial, evidence)
    assert correct["groups"]["paired"]["status"] == "PASS"
    assert correct["status"] == "UNKNOWN"


def test_unknown_coefficient_at_zero_does_not_hide_known_action_failure():
    suite, reference, evidence = fixture()
    reference = replace(reference, coefficients=((2, None, 0),))
    evidence["outputs"].update({"plus:action:j": [8], "minus:action:j": [2]})
    result = pointwise(suite, reference, evidence)
    assert result["status"] == "FAIL"
    assert result["first_divergence"]["expected"] == 7
    assert result["first_divergence"]["actual"] == 8


def test_manual_baseline_catches_known_offset_and_symmetric_curvature():
    suite, reference, evidence = fixture()
    bias = copy.deepcopy(evidence)
    bias["outputs"] = {key: [value[0] + 2] for key, value in bias["outputs"].items()}
    assert relational(suite, reference, bias)["first_divergence"]["queries"] == ["center"]
    curvature = copy.deepcopy(evidence)
    curvature["outputs"].update({"plus:action:j": [10], "minus:action:j": [6]})
    result = relational(
        suite, replace(reference, coefficients=((None, None, None),), offset=(None,)), curvature
    )
    assert result["status"] == "FAIL"
    assert result["first_divergence"]["kind"] == "midpoint"
    assert result["first_divergence"]["actual"] == 3


def test_stateless_subdomain_matches_both_maps_but_excludes_the_temporal_interventions():
    suite, reference, evidence = fixture()
    for data in (evidence, moving(evidence)):
        result = stateless(suite, reference, data)
        assert result["status"] == "PASS"
        assert result["coverage"]["queried_ids"] == ["center", "plus:action:j", "minus:action:j"]
        assert result["excluded_queries"] == 4


def test_stateless_empty_domain_is_not_a_pass():
    suite, reference, _ = fixture()
    suite = compile_suite((*suite.axes[:2], replace(suite.axes[2], lower=30, upper=50)), suite.outputs)
    result = stateless(suite, reference, {"suite_sha256": suite.sha256, "outputs": {}})
    assert result["applicability"] == "N/A"
    assert result["status"] == "UNKNOWN"


def test_stateless_membership_uses_physical_values_not_normalized_coordinates():
    suite, reference, _ = fixture()
    suite = compile_suite((*suite.axes[:2], replace(suite.axes[2], lower=0, upper=20)), suite.outputs)
    evidence = {"suite_sha256": suite.sha256, "outputs": {}}
    result = stateless(suite, reference, evidence)
    assert result["coverage"]["queried_ids"] == ["plus:origin:j", "minus:current:j"]


@pytest.mark.parametrize("oracle", [pointwise, relational, stateless])
def test_missing_authority_and_observations_never_become_ground_truth(oracle):
    suite, reference, evidence = fixture()
    original = copy.deepcopy(evidence)
    assert oracle(suite, replace(reference, authority=None), moving(evidence))["status"] == "UNKNOWN"
    assert oracle(suite, reference, {"suite_sha256": suite.sha256, "outputs": {}})["status"] == "UNKNOWN"
    assert evidence == original
    with pytest.raises(InputError, match="bound"):
        oracle(suite, reference, {**evidence, "suite_sha256": "wrong"})


def test_large_offset_paired_resolution_is_unknown():
    suite = compile_suite((Axis("action", "j", "unit", -1, 1),), ("j.pos",))
    reference = Reference(((1,),), (1e16,), ("unit",), "hand-computed unit fixture", True)
    evidence = {"suite_sha256": suite.sha256, "outputs": {p.id: [1e16] for p in suite.probes}}
    result = relational(suite, reference, evidence)
    assert result["groups"]["paired"]["status"] == "UNKNOWN"
    assert result["status"] == "UNKNOWN"


def test_contrast_cancellation_does_not_impose_a_stricter_relative_tolerance():
    suite = compile_suite(
        tuple(Axis(port, "j", "unit", -1, 1) for port in ("action", "origin", "current")),
        ("j.pos",),
    )
    reference = Reference(((1000, 1000, 0),), (0,), ("unit",), "hand-computed unit fixture", True)
    evidence = {
        "suite_sha256": suite.sha256,
        "outputs": {
            "center": [0],
            "plus:action:j": [1000.00001],
            "minus:action:j": [-1000.00001],
            "plus:origin:j": [1000],
            "minus:origin:j": [-1000],
            "plus:current:j": [0],
            "minus:current:j": [0],
        },
    }
    for oracle in (evaluate, pointwise, relational):
        assert oracle(suite, reference, evidence)["status"] == "PASS"
    # The same observation error is still rejected outside the component budget.
    evidence["outputs"]["plus:action:j"] = [1000.01]
    evidence["outputs"]["minus:action:j"] = [-1000.01]
    assert relational(suite, reference, evidence)["status"] == "FAIL"


def test_pointwise_cancellation_propagates_offset_and_coefficient_budgets():
    suite, _, _ = fixture()
    reference = Reference(((1000, 0, 0),), (1000,), ("degree",), "hand-computed unit fixture", True)
    evidence = {
        "suite_sha256": suite.sha256,
        "outputs": {p.id: [1000] for p in suite.probes},
    }
    evidence["outputs"].update({"plus:action:j": [2000.00001], "minus:action:j": [-0.00001]})
    for oracle in (evaluate, pointwise, relational):
        assert oracle(suite, reference, evidence)["status"] == "PASS"


def test_finite_agreement_does_not_imply_observability_of_a_full_affine_map():
    suite, reference, _ = fixture()
    subset = Suite(suite.axes, suite.outputs, (suite.probes[0],))
    evidence = {"suite_sha256": subset.sha256, "outputs": {"center": [5]}}
    for oracle in (pointwise, relational):
        report = oracle(subset, reference, evidence)
        assert report["status"] == "PASS"  # Finite-query claim only.
        assert report["full_design_coverage_status"] == "UNKNOWN"
        assert report["coverage"]["rank_by_output"] == {"j.pos": 1}
    from mltools.research_compare import _methods

    assert all(
        r["full_requested_status"] == "UNKNOWN" for r in _methods(subset, reference, evidence, {}).values()
    )


def test_entirely_unknown_rows_do_not_count_as_observed():
    suite, reference, evidence = fixture()
    evidence["outputs"] = {p.id: [None] for p in suite.probes}
    for oracle in (pointwise, relational, stateless):
        report = oracle(suite, reference, evidence)
        assert report["coverage"]["observed_eligible_queries"] == 0
        assert report["coverage"]["rank_by_output"] == {"j.pos": 0}


@pytest.mark.parametrize("offset", [None, 0])
def test_unverified_closure_cannot_ground_pointwise_or_cross_pair_contradictions(offset):
    suite = compile_suite((Axis("action", "j", "unit", -1, 1),), ("j.pos",))
    reference = Reference(((0,),), (offset,), ("unit",), "hand-computed unit fixture", False)
    evidence = {
        "suite_sha256": suite.sha256,
        "outputs": {"center": [0], "plus:action:j": [1], "minus:action:j": [1]},
    }
    for oracle in (evaluate, pointwise, relational):
        assert oracle(suite, reference, evidence)["status"] == "UNKNOWN"
    # A known center offset is testable without closure authority.
    if offset == 0:
        evidence["outputs"]["center"] = [5]
        assert pointwise(suite, reference, evidence)["status"] == "FAIL"


@pytest.mark.parametrize("bad", [True, float("inf"), "1", None])
def test_invalid_output_rows_are_operational_errors(bad):
    suite, reference, evidence = fixture()
    evidence["outputs"]["center"] = bad
    if bad is None:
        assert relational(suite, reference, evidence)["status"] == "UNKNOWN"
    else:
        with pytest.raises(InputError):
            relational(suite, reference, evidence)
