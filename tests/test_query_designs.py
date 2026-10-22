"""Hand-calculated design/oracle controls, including exact nuisance cancellation."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/query_designs.py"
SPEC = importlib.util.spec_from_file_location("research_query_designs_tests", PATH)
qd = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qd
SPEC.loader.exec_module(qd)


def fixture(method="manual_axis", seed=7):
    axes = [
        dict(port="action", name="0", unit="fixture", lower=-1, upper=1),
        dict(port="origin", name="0", unit="fixture", lower=-1, upper=1),
    ]
    design = qd.generate({"axes": axes, "outputs": ["command"]}, method, seed)
    reference = {
        "coefficients": [[2.0, 3.0]],
        "offset": [5.0],
        "output_units": ["fixture"],
        "authority": "Independent algebra fixture",
        "affine_fragment_verified": True,
    }
    return design, reference


def evidence(design, function):
    return {
        "design_sha256": design.sha256,
        "outputs": {p.id: [function(*p.coordinates)] for p in design.probes},
    }


@pytest.mark.parametrize("method", ["manual_axis", "random_dense", "random_axis"])
def test_grounded_linear_control_and_fault(method):
    design, ref = fixture(method)
    good = qd.analyze(design, ref, evidence(design, lambda z, q: 5 + 2 * z + 3 * q))
    bad = qd.analyze(design, ref, evidence(design, lambda z, q: 5 + 2 * z + 4 * q))
    assert good["full_affine_request_status"] == "PASS"
    assert good["rank_by_output"] == [3]
    assert bad["finite_status"] == bad["full_affine_request_status"] == "FAIL"


def test_random_continuity_seed_and_manual_geometry():
    manual, _ = fixture()
    random, _ = fixture("random_axis")
    assert [p.coordinates for p in manual.probes] == [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    assert len(random.probes) == len(manual.probes) == 5
    assert any(0 < abs(c) < 1 for p in random.probes for c in p.coordinates)
    assert random.sha256 == fixture("random_axis")[0].sha256
    assert random.sha256 != fixture("random_axis", seed=8)[0].sha256
    assert qd.Design.from_dict(random.to_dict()) == random


def test_unknown_offset_cancels_without_becoming_authority():
    design, ref = fixture("random_dense")
    ref["offset"] = [None]
    result = qd.analyze(design, ref, evidence(design, lambda z, q: 99 + 2 * z + 3 * q))
    assert result["counts"]["antithetic"] == {"PASS": 2, "FAIL": 0, "UNKNOWN": 0}
    assert result["full_affine_request_status"] == "UNKNOWN"
    faulty = qd.analyze(design, ref, evidence(design, lambda z, q: 99 + 2 * z + 4 * q))
    assert faulty["finite_status"] == "FAIL"


def test_structured_random_preserves_known_effect_with_unknown_nuisance():
    design, ref = fixture("random_axis")
    ref["coefficients"][0][0] = None
    ref["offset"] = [None]
    bad = qd.analyze(design, ref, evidence(design, lambda z, q: 77 + 10 * z + 4 * q))
    assert bad["full_affine_request_status"] == "FAIL"
    assert bad["first_divergence"]["relation"] == "antithetic"


def test_tiny_uncancelled_unknown_is_not_zero():
    design, ref = fixture()
    ref["coefficients"][0][0] = None
    point = qd.Query("tiny", (1e-100, 1.0), (1e-100, 1.0))
    design = replace(design, probes=(point,))
    result = qd.analyze(design, ref, evidence(design, lambda z, q: 999))
    assert result["finite_status"] == "UNKNOWN"


def test_missing_affine_authority_blocks_extrapolation_and_closure():
    design, ref = fixture("random_dense")
    ref["affine_fragment_verified"] = False
    result = qd.analyze(design, ref, evidence(design, lambda z, q: 5 + 100 * z * z + 3 * q))
    assert result["finite_status"] == result["full_affine_request_status"] == "UNKNOWN"
    assert result["counts"]["midpoint"]["UNKNOWN"] == 2


def test_rank_deficient_finite_agreement_stays_incomplete():
    design, ref = fixture()
    design = replace(design, probes=design.probes[:3])
    result = qd.analyze(design, ref, evidence(design, lambda z, q: 5 + 2 * z + 3 * q))
    assert result["finite_status"] == "PASS"
    assert result["full_affine_request_status"] == "UNKNOWN"
    assert result["rank_by_output"] == [2]


@pytest.mark.parametrize("missing", ["absent", "unknown_cells"])
def test_missing_observations_are_unknown(missing):
    design, ref = fixture()
    outputs = {} if missing == "absent" else {p.id: [None] for p in design.probes}
    result = qd.analyze(design, ref, {"design_sha256": design.sha256, "outputs": outputs})
    assert result["finite_status"] == result["full_affine_request_status"] == "UNKNOWN"
    assert result["rank_by_output"] == [0]


def test_exact_physical_rounding_residual_cannot_cancel_unknown_coefficient():
    axis = qd.Axis("action", "0", "fixture", 0.0, 1.0)
    points = tuple(qd.Query(f"q{i}", (c,), (axis.at(c),)) for i, c in enumerate((0.0, 0.1, -0.1)))
    design = qd.Design((axis,), ("out",), points, "random_axis", 7, qd.SCHEDULE)
    ref = {
        "coefficients": [[None]],
        "offset": [0.0],
        "output_units": ["fixture"],
        "authority": "Independently affine physical-input fixture",
        "affine_fragment_verified": True,
    }
    observed = {
        "design_sha256": design.sha256,
        "outputs": {p.id: [2**40 * (p.values[0] - 0.5)] for p in points},
    }
    result = qd.analyze(design, ref, observed)
    assert result["finite_status"] == result["full_affine_request_status"] == "UNKNOWN"
    assert result["counts"]["midpoint"] == {"PASS": 0, "FAIL": 0, "UNKNOWN": 1}


def test_full_rank_tiny_design_does_not_certify_endpoint_conformance():
    axis = qd.Axis("action", "0", "fixture", -1.0, 1.0)
    points = tuple(qd.Query(f"q{i}", (c,), (c,)) for i, c in enumerate((0.0, 1e-14, -1e-14)))
    design = qd.Design((axis,), ("out",), points, "random_axis", 7, qd.SCHEDULE)
    ref = {
        "coefficients": [[1.0]],
        "offset": [0.0],
        "output_units": ["fixture"],
        "authority": "Hand-calculated observability counterexample",
        "affine_fragment_verified": True,
    }
    observed = {"design_sha256": design.sha256, "outputs": {p.id: [10 * p.values[0]] for p in points}}
    result = qd.analyze(design, ref, observed)
    assert result["rank_by_output"] == [2]
    assert result["finite_status"] == "PASS"
    assert result["full_affine_request_status"] == "UNKNOWN"
    assert len(result["identification_basis_queries"][0]) < result["required_rank"]


def test_exact_inverse_with_nonmonotone_basis_insertion():
    assert qd.inverse_square([[1, 0, 0], [1, 0, 1], [1, 1, 0]]) == [[1, 0, 0], [-1, 0, 1], [-1, 1, 0]]
    rows = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    indices, inverse = qd.basis_inverse(rows)
    assert len(set(indices)) == 3
    for i in range(3):
        for j in range(3):
            assert sum(inverse[i][k] * rows[indices[k]][j] for k in range(3)) == (i == j)


def test_observed_nonaffinity_can_fail_without_filling_unknown_reference():
    axis = qd.Axis("action", "0", "fixture", -1.0, 1.0)
    points = tuple(qd.Query(f"q{i}", (c,), (c,)) for i, c in enumerate((0.0, 1.0, -1.0, 0.5, -0.5)))
    design = qd.Design((axis,), ("out",), points, "random_axis", 7, qd.SCHEDULE)
    ref = {
        "coefficients": [[None]],
        "offset": [None],
        "output_units": ["fixture"],
        "authority": "Independent affine-fragment authority only",
        "affine_fragment_verified": True,
    }
    observed = {"design_sha256": design.sha256, "outputs": {p.id: [p.values[0] ** 3] for p in points}}
    result = qd.analyze(design, ref, observed)
    assert result["finite_status"] == "FAIL"
    assert result["first_divergence"]["relation"] == "affine_residual"
    assert ref["coefficients"] == [[None]] and ref["offset"] == [None]


def test_interior_curvature_is_not_bounded_by_endpoint_closure():
    manual, ref = fixture()
    random, _ = fixture("random_axis")

    def nonlinear(z, q):
        return 5 + 2 * z + 3 * q + z * z * (1 - z * z)

    assert qd.analyze(manual, ref, evidence(manual, nonlinear))["finite_status"] == "PASS"
    assert qd.analyze(random, ref, evidence(random, nonlinear))["finite_status"] == "FAIL"
    # The finite PASS above deliberately demonstrates the admitted-fragment assumption.
    ref["affine_fragment_verified"] = False
    assert qd.analyze(manual, ref, evidence(manual, nonlinear))["full_affine_request_status"] == "UNKNOWN"


@pytest.mark.parametrize("tamper", ["values", "binding", "collapse"])
def test_reject_invalid_evidence_or_query(tamper):
    design, ref = fixture()
    with pytest.raises(ValueError):
        if tamper == "values":
            replace(design, probes=(qd.Query("wrong", (1, 0), (0, 0)),))
        elif tamper == "collapse":
            axis = qd.Axis("action", "0", "fixture", 1.0, 1.0000000000000004)
            qd.Design(
                (axis,), ("out",), (qd.Query("x", (1e-100,), (axis.center,)),), "random_axis", 1, qd.SCHEDULE
            )
        else:
            qd.analyze(design, ref, {"design_sha256": "foreign", "outputs": {}})
