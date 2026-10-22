import math
from dataclasses import replace

import pytest

from mltools.io import InputError
from mltools.obligations import Axis, Reference, compile_suite, evaluate


def setup_case():
    axes = (
        Axis("action", "joint", "normalized_action", -1, 1),
        Axis("origin", "joint", "degree", -10, 10),
        Axis("current", "joint", "degree", -10, 10),
    )
    suite = compile_suite(axes, ("joint.pos",))
    reference = Reference(((2.0, 10.0, 0.0),), (3.0,), ("degree",), "independent-test-spec", True)
    return suite, reference


def observe(suite, function):
    return {"suite_sha256": suite.sha256, "outputs": {p.id: [function(*p.values)] for p in suite.probes}}


def test_compiler_builds_bounded_full_rank_design_without_adapter_access():
    suite, ref = setup_case()
    assert len(suite.probes) == 7
    result = evaluate(suite, ref, observe(suite, lambda z, origin, current: 2 * z + origin + 3))
    assert result["status"] == "PASS"
    assert result["coverage"]["rank_by_output"] == {"joint.pos": 4}


def test_moving_anchor_fails_with_two_query_witness():
    suite, ref = setup_case()
    result = evaluate(suite, ref, observe(suite, lambda z, origin, current: 2 * z + current + 3))
    assert result["status"] == "FAIL"
    assert result["first_divergence"]["obligation"] == "effect:origin:joint"
    assert len(result["witness"]) == 2


def test_partial_reference_preserves_known_failure_despite_unknown_bias():
    suite, ref = setup_case()
    partial = replace(ref, coefficients=((None, 10.0, 0.0),), offset=(None,))
    result = evaluate(suite, partial, observe(suite, lambda z, origin, current: 7 * z + current + 99))
    assert result["status"] == "FAIL"
    assert result["first_divergence"]["obligation"] == "effect:origin:joint"


def test_unknown_bias_cannot_be_inferred_as_ground_truth_from_matching_effects():
    suite, ref = setup_case()
    result = evaluate(
        suite, replace(ref, offset=(None,)), observe(suite, lambda z, origin, current: 2 * z + origin + 99)
    )
    assert result["status"] == "UNKNOWN"
    assert all(f["status"] == "PASS" for f in result["findings"] if f["obligation"].startswith("effect:"))


@pytest.mark.parametrize("erase", ["authority", "output_units", "affine_fragment_verified"])
def test_missing_grounding_never_passes(erase):
    suite, ref = setup_case()
    value = {"authority": None, "output_units": (None,), "affine_fragment_verified": False}[erase]
    result = evaluate(
        suite, replace(ref, **{erase: value}), observe(suite, lambda z, origin, current: 2 * z + origin + 3)
    )
    assert result["status"] == "UNKNOWN"


def test_constant_state_observations_cannot_certify_anchor_effects():
    suite, ref = setup_case()
    evidence = observe(suite, lambda z, origin, current: 2 * z + origin + 3)
    for key in list(evidence["outputs"]):
        if "origin:" in key or "current:" in key:
            evidence["outputs"][key] = None
    report = evaluate(suite, ref, evidence)
    assert report["status"] == "UNKNOWN"
    assert report["coverage"]["rank_by_output"] == {"joint.pos": 2}


def test_equal_central_derivative_does_not_hide_quadratic_difference():
    suite, ref = setup_case()
    report = evaluate(suite, ref, observe(suite, lambda z, origin, current: 2 * z + origin + 3 + z * z))
    assert report["status"] == "FAIL"
    assert report["first_divergence"]["obligation"] == "affinity:action:joint"
    assert len(report["witness"]) == 3


def test_probe_results_are_bound_to_exact_bounds_and_labels():
    suite, ref = setup_case()
    other = compile_suite((replace(suite.axes[0], upper=2), *suite.axes[1:]), suite.outputs)
    with pytest.raises(InputError, match="exact compiled suite"):
        evaluate(other, ref, observe(suite, lambda z, origin, current: 2 * z + origin + 3))


def test_unknown_axis_units_prevent_semantic_pass():
    suite, ref = setup_case()
    other = compile_suite((replace(suite.axes[0], unit=None), *suite.axes[1:]), suite.outputs)
    assert (
        evaluate(other, ref, observe(other, lambda z, origin, current: 2 * z + origin + 3))["status"]
        == "UNKNOWN"
    )


def test_empty_evidence_and_unplanned_query_cannot_pass():
    suite, ref = setup_case()
    empty = {"suite_sha256": suite.sha256, "outputs": {}}
    assert evaluate(suite, ref, empty)["status"] == "UNKNOWN"
    empty["outputs"]["unplanned"] = [0]
    with pytest.raises(InputError):
        evaluate(suite, ref, empty)


@pytest.mark.parametrize("value", [True, 1, {}, " "])
def test_nontext_authority_and_units_cannot_ground_a_result(value):
    suite, ref = setup_case()
    with pytest.raises(InputError):
        replace(ref, authority=value)
    with pytest.raises(InputError):
        replace(ref, output_units=(value,))
    with pytest.raises(InputError):
        replace(suite.axes[0], unit=value)


def test_unknown_cells_do_not_count_as_observed_queries():
    suite, ref = setup_case()
    report = evaluate(suite, ref, {"suite_sha256": suite.sha256, "outputs": {"center": [None]}})
    assert report["coverage"]["observed_queries"] == 0
    assert report["coverage"]["rank_by_output"] == {"joint.pos": 0}
    assert report["status"] == "UNKNOWN"


def test_nonobject_observations_have_explicit_input_error():
    suite, ref = setup_case()
    with pytest.raises(InputError, match="JSON object"):
        evaluate(suite, ref, [])


def test_compiled_suite_and_reference_do_not_share_mutable_caller_lists():
    suite, ref = setup_case()
    axes, outputs = list(suite.axes), list(suite.outputs)
    compiled = compile_suite(axes, outputs)
    coefficients, offset, units = [list(ref.coefficients[0])], list(ref.offset), list(ref.output_units)
    frozen_ref = replace(ref, coefficients=coefficients, offset=offset, output_units=units)
    digest = compiled.sha256
    axes.clear()
    outputs.clear()
    coefficients[0][0] = 999
    offset[0] = 999
    units.clear()
    assert compiled.sha256 == digest
    assert frozen_ref == ref


def test_missing_one_output_does_not_erase_independently_observed_failure():
    suite, ref = setup_case()
    suite = compile_suite(suite.axes, ("known.pos", "unknown.pos"))
    ref = replace(
        ref, coefficients=(ref.coefficients[0], (None,) * 3), offset=(3, None), output_units=("degree", None)
    )
    evidence = {
        "suite_sha256": suite.sha256,
        "outputs": {p.id: [2 * p.values[0] + p.values[2] + 3, None] for p in suite.probes},
    }
    report = evaluate(suite, ref, evidence)
    assert report["status"] == "FAIL"
    assert report["coverage"]["rank_by_output"] == {"known.pos": 4, "unknown.pos": 0}


def test_repeated_inputs_cannot_claim_unexecuted_design_directions():
    suite, _ = setup_case()
    collapsed = tuple(replace(p, values=(0.0,) * 3) for p in suite.probes)
    with pytest.raises(InputError, match="canonical bounded design"):
        replace(suite, probes=collapsed)


def test_duplicate_query_identity_and_changed_coordinates_are_rejected():
    suite, _ = setup_case()
    with pytest.raises(InputError):
        replace(suite, probes=(suite.probes[0], suite.probes[0]))
    with pytest.raises(InputError):
        replace(suite, probes=(replace(suite.probes[1], coordinates=(0.0,) * 3),))


def test_adjacent_float_bounds_cannot_claim_three_distinct_interventions():
    with pytest.raises(InputError, match="distinct representable center"):
        Axis("action", "joint", "unit", 1.0, math.nextafter(1.0, math.inf))


@pytest.mark.parametrize("coefficient", [0.0, 1.0])
def test_large_offset_output_quantization_is_unknown_not_structural_failure_or_pass(coefficient):
    suite = compile_suite((Axis("action", "joint", "unit", -1, 1),), ("command",))
    ref = Reference(((coefficient,),), (1e16,), ("unit",), "independent test fixture", True)
    evidence = observe(suite, lambda x: 1e16 + x)
    report = evaluate(suite, ref, evidence)
    assert report["status"] == "UNKNOWN"
    effect = next(f for f in report["findings"] if f["obligation"].startswith("effect:"))
    assert effect["status"] == "UNKNOWN"
    assert effect["output_resolution_floor"] > 1


def test_overflowing_comparison_tolerance_cannot_pass_everything():
    suite, ref = setup_case()
    with pytest.raises(InputError, match="Comparison exceeds"):
        evaluate(suite, ref, observe(suite, lambda z, origin, current: 2 * z + origin + 3), rtol=1e308)


def test_serialized_suite_cannot_change_queries_or_claim_uncompiled_method():
    suite, _ = setup_case()
    from mltools.obligations import Suite

    assert Suite.from_dict(suite.to_dict()) == suite
    value = suite.to_dict()
    value["probes"][1]["coordinates"] = [0, 0, 0]
    with pytest.raises(InputError):
        Suite.from_dict(value)
    value = suite.to_dict()
    value["method"] = "different-method"
    with pytest.raises(InputError):
        Suite.from_dict(value)
