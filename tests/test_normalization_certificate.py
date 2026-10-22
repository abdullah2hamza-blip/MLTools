"""Independent hand arithmetic for the explicitly scoped certificate profiles."""

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from normalization_certificate import certificate, load_request  # noqa: E402


def request():
    return {
        "schema_version": 1,
        "profile": "paper_quantile_equations",
        "coordinates": [{"name": "arm", "unit": "synthetic"}, {"name": "gripper", "unit": "synthetic"}],
        "intended": {
            "authority": "Independent algebra fixture",
            "q01": [0, 0],
            "q99": [2, 1],
            "mask": [True, False],
        },
        "candidate": {
            "authority": "Designed alternative decoder",
            "q01": [1, -99],
            "q99": [5, 99],
            "mask": [True, False],
        },
        "calibration": {
            "kind": "designed",
            "space": "normalized",
            "authority": "Hand-calculated test inputs",
            "values": [[0, 0], [0.5, 0.25], [1, 1]],
        },
        "metric": {"indices": [0], "unit": "synthetic"},
        "gripper_indices": [1],
        "thresholds": [0, 2, 3, 4],
        "percentiles": [0, 0.5, 0.9, 1],
    }


def test_hand_calculated_mean_quantile_tail_and_gripper():
    result = certificate(request())
    assert result["differences"] == [[1, 0], [2, 0], [3, 0]]
    assert result["metric"]["mean_displacement"] == 2
    assert [x["fraction_greater_or_equal"] for x in result["metric"]["tails"]] == [1, 2 / 3, 1 / 3, 0]
    assert [x["value"] for x in result["metric"]["percentiles"]] == pytest.approx([1, 2, 2.8, 3])
    assert result["gripper_axis_displacement"][0]["absolute_per_sample"] == [0, 0, 0]
    assert result["axis_mean_absolute_displacement"] == [2, 0]


def test_identity_tail_at_zero_includes_every_sample():
    value = request()
    value["candidate"] = copy.deepcopy(value["intended"])
    result = certificate(value)
    assert result["metric"]["mean_displacement"] == 0
    assert [x["fraction_greater_or_equal"] for x in result["metric"]["tails"]] == [1, 0, 0, 0]


def test_mask_change_uses_two_direct_decoders():
    value = request()
    value["candidate"]["mask"] = [False, False]
    result = certificate(value)
    assert result["normalization_mask_mismatches"] == [0]
    assert [row[0] for row in result["differences"]] == [0, -0.5, -1]
    assert result["metric"]["mean_displacement"] == 0.5


def test_outside_normalized_range_is_not_clipped():
    value = request()
    value["calibration"]["values"] = [[-1, -3], [2, 4]]
    result = certificate(value)
    assert result["intended_decoded"] == [[-2, -3], [4, 4]]
    assert result["candidate_decoded"] == [[-3, -3], [9, 4]]


def test_physical_input_is_intended_normalized_first():
    value = request()
    value["calibration"].update(space="physical", values=[[0, 0], [1, 0.25], [2, 1]])
    result = certificate(value)
    assert result["normalized_actions"] == [[0, 0], [0.5, 0.25], [1, 1]]
    assert result["metric"]["mean_displacement"] == 2


def test_gripper_identity_is_explicit_not_last_coordinate():
    value = request()
    value["gripper_indices"] = [0]
    result = certificate(value)
    assert result["gripper_axis_displacement"][0]["index"] == 0
    assert result["gripper_axis_displacement"][0]["absolute_per_sample"] == [1, 2, 3]


@pytest.mark.parametrize("missing", ["intended", "candidate", "calibration"])
def test_missing_authority_withholds_aggregate_interpretation(missing):
    value = request()
    value[missing]["authority"] = None
    result = certificate(value)
    assert result["metric"]["status"] == "UNKNOWN"
    assert "mean_displacement" not in result["metric"]
    assert result["differences"] == [[1, 0], [2, 0], [3, 0]]


def test_mixed_or_missing_metric_units_stay_unknown():
    value = request()
    value["coordinates"][1]["unit"] = "radian"
    value["metric"]["indices"] = [0, 1]
    result = certificate(value)
    assert result["metric"]["status"] == "UNKNOWN"
    assert "per_sample_l2" not in result["metric"]


def test_lerobot_extension_uses_decode_baseline_after_lossy_normalization():
    value = request()
    value["profile"] = "lerobot_mean_std_extension"
    value["intended"] = {"authority": "Designed zero-variance metadata", "mean": [1, 0], "std": [0, 2]}
    value["candidate"] = copy.deepcopy(value["intended"])
    value["calibration"].update(space="physical", values=[[2, 1]])
    result = certificate(value)
    assert result["normalized_actions"][0][0] == 1e8
    assert result["intended_decoded"][0][0] == 1
    assert result["metric"]["mean_displacement"] == 0
    assert "extension" in result["implementation"]


def test_openpi_profile_has_its_native_domain_and_epsilon():
    value = request()
    value["profile"] = "openpi_quantile_extension"
    value["intended"] = {"authority": "Designed OpenPI metadata", "q01": [-2, -1], "q99": [2, 1]}
    value["candidate"] = copy.deepcopy(value["intended"])
    value["calibration"].update(values=[[-3, 0], [3, 0]])
    result = certificate(value)
    assert [row[0] for row in result["intended_decoded"]] == pytest.approx([-6.000001, 6.000002])
    assert result["metric"]["mean_displacement"] == 0


@pytest.mark.parametrize("invalid", ["empty", "duplicate", "mask", "threshold", "nonfinite", "extra"])
def test_invalid_requests_cannot_produce_certificates(invalid):
    value = request()
    if invalid == "empty":
        value["calibration"]["values"] = []
    elif invalid == "duplicate":
        value["coordinates"][1]["name"] = "arm"
    elif invalid == "mask":
        value["intended"]["mask"] = [1, 0]
    elif invalid == "threshold":
        value["thresholds"] = [-1]
    elif invalid == "nonfinite":
        value["candidate"]["q01"][0] = float("nan")
    else:
        value["clip"] = True
    with pytest.raises(ValueError, match="Normalization certificate:"):
        certificate(value)


@pytest.mark.parametrize("raw", ['{"schema_version": 1, "schema_version": 2}', '{"value": NaN}'])
def test_json_loader_rejects_duplicate_and_nonfinite_fields(tmp_path, raw):
    path = tmp_path / "request.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="Normalization certificate:"):
        load_request(path)
