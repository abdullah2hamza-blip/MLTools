"""Hand calculations and independent numerical-reference boundaries."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import normalization_profiles as profiles  # noqa: E402


def spec(name, **statistics):
    return {"schema_version": 1, "profile": name, "statistics": statistics}


def test_distinct_quantile_denominators_and_outside_range():
    lr = spec("lerobot_quantiles_f64_v1", q01=[2.0, 4.0], q99=[6.0, 4.0])
    op = spec("openpi_quantiles_f64_v1", q01=[2.0, 4.0], q99=[6.0, 4.0])
    assert profiles.transform(lr, [3.0, 1.0], inverse=True) == [10.0, 4.00000001]
    assert profiles.transform(op, [3.0, 1.0], inverse=True) == [10.000002, 4.000001]
    # Nondegenerate LeRobot width receives no epsilon, unlike OpenPI.
    assert profiles.transform(lr, [6.0, 4.0], inverse=False) == [1.0, -1.0]
    assert profiles.transform(op, [6.0, 4.0], inverse=False)[0] < 1.0


def test_mean_std_forward_is_not_an_exact_inverse_at_zero_variance():
    value = spec("lerobot_mean_std_f64_v1", mean=[1.0, 5.0], std=[2.0, 0.0])
    assert profiles.transform(value, [3.0, 7.0], inverse=True) == [7.0, 5.0]
    normalized = profiles.transform(value, [3.0, 7.0], inverse=False)
    assert normalized == [2 / (2 + 1e-8), 2 / 1e-8]
    assert profiles.transform(value, normalized, inverse=True)[1] == 5.0


def test_reference_coefficients_use_actual_bounded_coordinates_without_outputs():
    value = spec("lerobot_quantiles_f64_v1", q01=[2.0], q99=[6.0])
    axes = [dict(port="action", name="0", unit="normalized", lower=-0.5, upper=1.0)]
    reference = profiles.affine_reference(value, axes, authority="fixture", output_units=["raw_index_0"])
    assert reference["offset"] == [4.5]
    assert reference["coefficients"] == [[1.5]]
    assert reference["affine_fragment_verified"] is True
    erased = profiles.affine_reference(value, axes, authority=None, output_units=[None])
    assert erased["authority"] is None and erased["output_units"] == [None]
    assert erased["coefficients"] == reference["coefficients"]


@pytest.mark.parametrize("bad", ["negative_std", "infinite", "bool", "inverted", "shape", "profile", "extra"])
def test_invalid_reference_never_silently_becomes_another_profile(bad):
    value = spec("lerobot_mean_std_f64_v1", mean=[1.0], std=[2.0])
    if bad == "negative_std":
        value["statistics"]["std"] = [-1.0]
    elif bad == "infinite":
        value["statistics"]["mean"] = [float("inf")]
    elif bad == "bool":
        value["statistics"]["mean"] = [True]
    elif bad == "inverted":
        value = spec("lerobot_quantiles_f64_v1", q01=[2], q99=[1])
    elif bad == "shape":
        value["statistics"]["mean"] = [1.0, 2.0]
    elif bad == "profile":
        value["profile"] = "paper_quantile_equations"
    else:
        value["eps"] = 0.0
    with pytest.raises(ValueError):
        profiles.transform(value, [0.0], inverse=True)


def test_reference_requires_explicit_index_mapping_and_preserves_inputs():
    value = spec("openpi_quantiles_f64_v1", q01=[-1], q99=[1])
    before = copy.deepcopy(value)
    axes = [dict(port="current", name="0", unit="raw", lower=-1, upper=1)]
    with pytest.raises(ValueError):
        profiles.affine_reference(value, axes, authority="fixture", output_units=["raw0"])
    assert value == before
