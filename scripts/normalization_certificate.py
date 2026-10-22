"""Versioned finite normalization-drift certificates, without ML dependencies.

The paper-equation profile implements Tai, arXiv:2606.03724v1, Eqs. 7/13/14.
Framework profiles are explicitly labeled extensions, not author code.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

PROFILES = {"paper_quantile_equations", "openpi_quantile_extension", "lerobot_mean_std_extension"}


def require(condition, message):
    if not condition:
        raise ValueError("Normalization certificate: " + message)


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), label + " fields")


def vector(value, n, label):
    require(isinstance(value, list) and len(value) == n and all(map(finite, value)), label)


def indices(value, n, label, allow_empty=False):
    require(
        isinstance(value, list)
        and (allow_empty or value)
        and all(type(i) is int and 0 <= i < n for i in value)
        and len(set(value)) == len(value),
        label,
    )


def validate(request):
    keys(
        request,
        (
            "schema_version",
            "profile",
            "coordinates",
            "intended",
            "candidate",
            "calibration",
            "metric",
            "gripper_indices",
            "thresholds",
            "percentiles",
        ),
        "request",
    )
    require(type(request["schema_version"]) is int and request["schema_version"] == 1, "schema version")
    profile = request["profile"]
    require(isinstance(profile, str) and profile in PROFILES, "supported explicit decoder profile")
    coordinates = request["coordinates"]
    require(isinstance(coordinates, list) and 1 <= len(coordinates) <= 256, "bounded coordinates")
    names = []
    for coordinate in coordinates:
        keys(coordinate, ("name", "unit"), "coordinate")
        require(isinstance(coordinate["name"], str) and bool(coordinate["name"]), "coordinate name")
        require(
            coordinate["unit"] is None or isinstance(coordinate["unit"], str) and bool(coordinate["unit"]),
            "coordinate unit",
        )
        names.append(coordinate["name"])
    require(len(set(names)) == len(names), "unique coordinate names")
    n = len(coordinates)
    for key in ("intended", "candidate"):
        metadata = request[key]
        fields = (
            ("authority", "mean", "std")
            if profile == "lerobot_mean_std_extension"
            else ("authority", "q01", "q99")
        )
        if profile == "paper_quantile_equations":
            fields += ("mask",)
        keys(metadata, fields, key)
        require(
            metadata["authority"] is None
            or isinstance(metadata["authority"], str)
            and bool(metadata["authority"]),
            key + " authority",
        )
        for field in fields:
            if field not in ("authority", "mask"):
                vector(metadata[field], n, key + " " + field)
        if profile == "lerobot_mean_std_extension":
            require(all(x >= 0 for x in metadata["std"]), "nonnegative standard deviations")
        else:
            require(
                all(b >= a for a, b in zip(metadata["q01"], metadata["q99"], strict=True)),
                "ordered quantiles",
            )
        if profile == "paper_quantile_equations":
            require(
                isinstance(metadata["mask"], list)
                and len(metadata["mask"]) == n
                and all(type(v) is bool for v in metadata["mask"]),
                "explicit normalization mask",
            )
    calibration = request["calibration"]
    keys(calibration, ("kind", "space", "authority", "values"), "calibration")
    require(calibration["kind"] in ("designed", "demonstration", "model_outputs"), "calibration kind")
    require(calibration["space"] in ("physical", "normalized"), "calibration space")
    require(
        calibration["authority"] is None
        or isinstance(calibration["authority"], str)
        and bool(calibration["authority"]),
        "calibration authority",
    )
    require(
        isinstance(calibration["values"], list) and 1 <= len(calibration["values"]) <= 10000,
        "nonempty bounded calibration",
    )
    for row in calibration["values"]:
        vector(row, n, "calibration row")
    keys(request["metric"], ("indices", "unit"), "metric")
    indices(request["metric"]["indices"], n, "metric indices")
    unit = request["metric"]["unit"]
    require(unit is None or isinstance(unit, str) and bool(unit), "metric unit")
    indices(request["gripper_indices"], n, "explicit gripper indices", allow_empty=True)
    require(
        isinstance(request["thresholds"], list)
        and len(request["thresholds"]) <= 100
        and all(finite(x) and x >= 0 for x in request["thresholds"]),
        "nonnegative reporting thresholds",
    )
    require(
        isinstance(request["percentiles"], list)
        and len(request["percentiles"]) <= 100
        and all(finite(x) and 0 <= x <= 1 for x in request["percentiles"]),
        "percentiles in [0, 1]",
    )
    return n


def transform(profile, metadata, row, *, inverse):
    result = []
    for i, value in enumerate(row):
        if profile == "lerobot_mean_std_extension":
            mean, std = metadata["mean"][i], metadata["std"][i]
            value = value * std + mean if inverse else (value - mean) / (std + 1e-8)
        elif profile == "paper_quantile_equations":
            if metadata["mask"][i]:
                lower = metadata["q01"][i]
                width = metadata["q99"][i] - lower
                require(inverse or width > 0, "physical normalization needs nonzero paper quantile width")
                value = lower + value * width if inverse else (value - lower) / width
        else:
            lower = metadata["q01"][i]
            width = metadata["q99"][i] - lower + 1e-6
            value = (value + 1) / 2 * width + lower if inverse else (value - lower) / width * 2 - 1
        require(finite(value), "finite transformed value")
        result.append(value)
    return result


def percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def certificate(request):
    n = validate(request)
    profile = request["profile"]
    intended, candidate = request["intended"], request["candidate"]
    calibration = request["calibration"]
    z = [
        row[:] if calibration["space"] == "normalized" else transform(profile, intended, row, inverse=False)
        for row in calibration["values"]
    ]
    correct = [transform(profile, intended, row, inverse=True) for row in z]
    substituted = [transform(profile, candidate, row, inverse=True) for row in z]
    differences = [
        [b - a for a, b in zip(left, right, strict=True)]
        for left, right in zip(correct, substituted, strict=True)
    ]
    require(all(finite(v) for row in differences for v in row), "finite displacement")
    chosen = request["metric"]["indices"]
    reasons = []
    if not intended["authority"] or not candidate["authority"] or not calibration["authority"]:
        reasons.append("Missing independently supplied metadata/calibration authority")
    unit = request["metric"]["unit"]
    if not unit or any(request["coordinates"][i]["unit"] != unit for i in chosen):
        reasons.append("Metric requires explicit common units for every selected coordinate")
    distances = [math.hypot(*(row[i] for i in chosen)) for row in differences]
    require(all(map(finite, distances)), "finite metric distance")
    metric = {
        "status": "UNKNOWN" if reasons else "COMPUTED",
        "reasons": reasons,
        "unit": unit,
        "indices": chosen,
    }
    if not reasons:
        metric.update(
            per_sample_l2=distances,
            mean_displacement=math.fsum(distances) / len(distances),
            percentiles=[
                {"probability": p, "value": percentile(distances, p)} for p in request["percentiles"]
            ],
            tails=[
                {"threshold": t, "fraction_greater_or_equal": sum(d >= t for d in distances) / len(distances)}
                for t in request["thresholds"]
            ],
        )
    raw = json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "schema_version": 1,
        "method": "finite-normalization-drift-certificate",
        "profile": profile,
        "implementation": "paper-equation reimplementation"
        if profile == "paper_quantile_equations"
        else "framework-specific extension; not the paper decoder",
        "request_canonical_sha256": hashlib.sha256(raw).hexdigest(),
        "calibration_kind": calibration["kind"],
        "sample_count": len(z),
        "normalized_actions": z,
        "intended_decoded": correct,
        "candidate_decoded": substituted,
        "differences": differences,
        "axis_mean_absolute_displacement": [
            math.fsum(abs(row[i]) for row in differences) / len(z) for i in range(n)
        ],
        "normalization_mask_mismatches": [i for i in range(n) if intended["mask"][i] != candidate["mask"][i]]
        if profile == "paper_quantile_equations"
        else [],
        "gripper_axis_displacement": [
            {
                "index": i,
                "name": request["coordinates"][i]["name"],
                "unit": request["coordinates"][i]["unit"],
                "absolute_per_sample": [abs(row[i]) for row in differences],
            }
            for i in request["gripper_indices"]
        ],
        "metric": metric,
        "decision": "Reporting only; no inferred conformance or safety decision threshold",
        "choices": {
            "clipping": "none",
            "percentile_estimator": "linear interpolation at (n-1)*p",
            "tail_comparison": ">=",
            "arithmetic": "binary64; no universal upstream error bound",
        },
        "limits": [
            "Finite supplied actions only; not deployment support or task success",
            "Authority/provenance labels are caller assertions, not execution attestations",
            "Designed actions are not demonstration-derived calibration",
            "Drift is between the two decoded sequences, not against the original physical inputs",
            "No controller, chunk-state, safety-margin or policy-performance certificate",
        ],
    }


def load_request(path):
    require(path.stat().st_size <= 16 * 1024**2, "request file size")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("Normalization certificate: nonfinite JSON constant " + value)

    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=reject)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = certificate(load_request(args.request))
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
