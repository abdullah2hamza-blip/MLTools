"""Independent binary64 numerical references for the selected decoder study.

These explicitly named profiles describe pinned runtime formulas. They do not
attest a checkpoint's training runtime, physical units or learned behavior.
No existing normalization-certificate profile is changed or relabeled.
"""

import math

PROFILES = {
    "lerobot_mean_std_f64_v1": ("mean", "std"),
    "lerobot_quantiles_f64_v1": ("q01", "q99"),
    "openpi_quantiles_f64_v1": ("q01", "q99"),
}


def require(condition, message):
    if not condition:
        raise ValueError("Normalization profile: " + message)


def finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate(spec):
    require(
        isinstance(spec, dict) and set(spec) == {"schema_version", "profile", "statistics"},
        "exact profile fields",
    )
    require(type(spec["schema_version"]) is int and spec["schema_version"] == 1, "profile version")
    name = spec["profile"]
    require(isinstance(name, str) and name in PROFILES, "explicit supported numerical profile")
    stats = spec["statistics"]
    require(isinstance(stats, dict) and set(stats) == set(PROFILES[name]), "profile statistics")
    values = list(stats.values())
    require(isinstance(values[0], list) and 1 <= len(values[0]) <= 32, "bounded statistics")
    n = len(values[0])
    require(
        all(isinstance(v, list) and len(v) == n and all(map(finite, v)) for v in values),
        "finite equal-shaped statistics",
    )
    if name == "lerobot_mean_std_f64_v1":
        require(all(v >= 0 for v in stats["std"]), "nonnegative standard deviation")
    else:
        require(
            all(b >= a and finite(b - a) for a, b in zip(stats["q01"], stats["q99"], strict=True)),
            "ordered finite quantile widths",
        )
    return n


def transform(spec, row, *, inverse):
    n = validate(spec)
    require(isinstance(row, list) and len(row) == n and all(map(finite, row)), "finite action row")
    name, stats = spec["profile"], spec["statistics"]
    result = []
    for i, v in enumerate(row):
        if name == "lerobot_mean_std_f64_v1":
            scale = stats["std"][i]
            origin = stats["mean"][i]
            value = v * scale + origin if inverse else (v - origin) / (scale + 1e-8)
        else:
            origin = stats["q01"][i]
            width = stats["q99"][i] - origin
            # LeRobot substitutes only an exactly zero denominator; OpenPI adds
            # epsilon to every width. Neither selected profile clips actions.
            scale = width + 1e-6 if name == "openpi_quantiles_f64_v1" else width if width else 1e-8
            if name == "lerobot_quantiles_f64_v1":
                value = (v + 1) * scale / 2 + origin if inverse else 2 * (v - origin) / scale - 1
            else:
                value = (v + 1) / 2 * scale + origin if inverse else (v - origin) / scale * 2 - 1
        require(finite(value), "finite transformed coordinate")
        result.append(value)
    return result


def affine_reference(spec, axes, *, authority, output_units):
    """Compile a separately supplied numeric contract, without seeing outputs.

    Only a declared action-only coordinate alignment is accepted. The resulting
    reference is numerical, not physical authority for a published policy.
    """
    n = validate(spec)
    require(
        isinstance(axes, list)
        and len(axes) == n
        and all(a["port"] == "action" and a["name"] == str(i) for i, a in enumerate(axes)),
        "explicit action index alignment",
    )
    centers, radii = [], []
    for a in axes:
        lower, upper = a["lower"], a["upper"]
        require(finite(lower) and finite(upper) and lower < upper, "finite increasing bounds")
        center, radius = lower / 2 + upper / 2, upper / 2 - lower / 2
        require(finite(radius) and lower < center < upper, "representable bounded intervention")
        centers.append(center)
        radii.append(radius)
    origin = transform(spec, centers, inverse=True)
    stats, name = spec["statistics"], spec["profile"]
    coefficients = []
    for i in range(n):
        if name == "lerobot_mean_std_f64_v1":
            scale = stats["std"][i]
        else:
            width = stats["q99"][i] - stats["q01"][i]
            denominator = width + 1e-6 if name == "openpi_quantiles_f64_v1" else width if width else 1e-8
            scale = denominator / 2
        coefficient = scale * radii[i]
        require(finite(coefficient), "finite normalized-coordinate effect")
        coefficients.append([coefficient if i == j else 0.0 for j in range(n)])
    require(authority is None or isinstance(authority, str) and bool(authority.strip()), "authority")
    require(
        isinstance(output_units, list)
        and len(output_units) == n
        and all(u is None or isinstance(u, str) and bool(u.strip()) for u in output_units),
        "output units",
    )
    return {
        "coefficients": coefficients,
        "offset": origin,
        "output_units": output_units.copy(),
        "authority": authority,
        "affine_fragment_verified": True,
    }
