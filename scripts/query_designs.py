"""Research-only bounded query designs and a shared exact-contrast oracle.

Independent of canonical Suite validation. No framework imports or execution.
Component tolerances and a declared measurement allowance are an empirical
numerical profile, not a proof of all upstream floating-point error.
"""

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction as F


def require(condition, message):
    if not condition:
        raise ValueError(f"Research design: {message}")


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Axis:
    port: str
    name: str
    unit: str | None
    lower: float
    upper: float

    def __post_init__(self):
        require(
            self.port in ("action", "origin", "current")
            and isinstance(self.name, str)
            and bool(self.name.strip()),
            "axis identity",
        )
        require(self.unit is None or isinstance(self.unit, str) and bool(self.unit.strip()), "axis unit")
        require(number(self.lower) and number(self.upper) and self.lower < self.upper, "axis bounds")
        require(self.lower < self.center < self.upper and number(self.radius), "representable axis center")

    @property
    def center(self):
        return self.lower / 2 + self.upper / 2

    @property
    def radius(self):
        return self.upper / 2 - self.lower / 2

    def at(self, coordinate):
        return (
            self.upper
            if coordinate == 1
            else self.lower
            if coordinate == -1
            else self.center + self.radius * coordinate
        )


@dataclass(frozen=True)
class Query:
    id: str
    coordinates: tuple[float, ...]
    values: tuple[float, ...]


@dataclass(frozen=True)
class Design:
    axes: tuple[Axis, ...]
    outputs: tuple[str, ...]
    probes: tuple[Query, ...]
    generator: str
    seed: int | None
    schedule: dict

    def __post_init__(self):
        require(1 <= len(self.axes) <= 96 and 1 <= len(self.outputs) <= 32, "port count")
        require(len({(a.port, a.name) for a in self.axes}) == len(self.axes), "duplicate axes")
        require(
            len(set(self.outputs)) == len(self.outputs)
            and all(isinstance(v, str) and bool(v.strip()) for v in self.outputs),
            "output labels",
        )
        require(1 <= len(self.probes) <= 193, "query budget")
        require(len({p.id for p in self.probes}) == len(self.probes), "duplicate query identifiers")
        require(self.generator in ("compiled", "manual_axis", "random_dense", "random_axis"), "generator")
        require(self.seed is None or type(self.seed) is int and 0 <= self.seed < 2**32, "seed")
        require(self.schedule == SCHEDULE, "reviewed temporal schedule")
        for point in self.probes:
            require(
                bool(point.id) and len(point.coordinates) == len(point.values) == len(self.axes),
                "point shape",
            )
            require(
                all(number(c) and -1 <= c <= 1 for c in point.coordinates), "normalized coordinate bounds"
            )
            require(
                all(
                    number(v) and v == a.at(c) and a.lower <= v <= a.upper
                    for a, c, v in zip(self.axes, point.coordinates, point.values, strict=True)
                ),
                "coordinates do not identify the supplied bounded values",
            )
            require(
                all(
                    c == 0 or v != a.center
                    for a, c, v in zip(self.axes, point.coordinates, point.values, strict=True)
                ),
                "intervention collapsed at input resolution",
            )
        require(
            len({p.values for p in self.probes}) == len({p.coordinates for p in self.probes}),
            "distinct normalized queries collapsed to identical physical inputs",
        )

    def to_dict(self):
        return {
            "schema_version": 1,
            "format": "mltools-research-query-design",
            "axes": [asdict(a) for a in self.axes],
            "outputs": list(self.outputs),
            "probes": [asdict(p) for p in self.probes],
            "generator": self.generator,
            "seed": self.seed,
            "schedule": self.schedule,
        }

    @property
    def sha256(self):
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        require(
            type(value["schema_version"]) is int
            and value["schema_version"] == 1
            and value["format"] == "mltools-research-query-design",
            "format/version",
        )
        return cls(
            tuple(Axis(**a) for a in value["axes"]),
            tuple(value["outputs"]),
            tuple(Query(p["id"], tuple(p["coordinates"]), tuple(p["values"])) for p in value["probes"]),
            value["generator"],
            value["seed"],
            value["schedule"],
        )


SCHEDULE = {
    "reset_each_query": True,
    "chunk_size": 2,
    "origin_before_chunk": True,
    "current_before_second_dequeue": True,
    "record_action_index": 1,
    "policy_inference": False,
}


def generate(suite_dict, method, seed=0):
    axes = tuple(Axis(**a) for a in suite_dict["axes"])
    n = len(axes)
    rng = random.Random(seed)
    coordinates = [(0.0,) * n]
    if method == "compiled":
        coordinates = [tuple(p["coordinates"]) for p in suite_dict["probes"]]
    else:
        allocation = list(range(n))
        if method == "random_axis":
            rng.shuffle(allocation)
        for axis in allocation:
            if method == "random_dense":
                direction = tuple(rng.uniform(-1.0, 1.0) for _ in axes)
            else:
                require(method in ("manual_axis", "random_axis"), "unknown generator")
                amplitude = rng.uniform(0.25, 1.0) if method == "random_axis" else 1.0
                direction = tuple(amplitude if j == axis else 0.0 for j in range(n))
            coordinates.extend((direction, tuple(-c for c in direction)))
    probes = tuple(
        Query(f"q{i:03}", tuple(coords), tuple(a.at(c) for a, c in zip(axes, coords, strict=True)))
        for i, coords in enumerate(coordinates)
    )
    return Design(
        axes,
        tuple(suite_dict["outputs"]),
        probes,
        method,
        seed if method.startswith("random") else None,
        SCHEDULE.copy(),
    )


def rank(rows):
    rows = [[F(x) for x in row] for row in rows]
    pivot = 0
    for column in range(len(rows[0]) if rows else 0):
        selected = next((i for i in range(pivot, len(rows)) if rows[i][column]), None)
        if selected is None:
            continue
        rows[pivot], rows[selected] = rows[selected], rows[pivot]
        divisor = rows[pivot][column]
        rows[pivot] = [v / divisor for v in rows[pivot]]
        for i in range(pivot + 1, len(rows)):
            scale = rows[i][column]
            if scale:
                rows[i] = [a - scale * b for a, b in zip(rows[i], rows[pivot], strict=True)]
        pivot += 1
        if pivot == len(rows):
            break
    return pivot


def basis_inverse(rows):
    """Propose spread rows without outputs, then invert their executed geometry exactly."""
    if not rows:
        return [], None
    selected, orthogonal = [], []
    width = len(rows[0]) if rows else 0
    for _ in range(width):
        candidates = []
        for index, row in enumerate(rows):
            if index in selected:
                continue
            value = [float(v) for v in row]
            for _repeat in range(2):
                for vector in orthogonal:
                    scale = math.fsum(a * b for a, b in zip(value, vector, strict=True))
                    value = [a - scale * b for a, b in zip(value, vector, strict=True)]
            norm = math.sqrt(math.fsum(v * v for v in value))
            candidates.append((norm, -index, value))
        norm, negative_index, vector = max(candidates, default=(0.0, 0, []))
        if norm <= 1e-12:
            return selected, None
        selected.append(-negative_index)
        orthogonal.append([v / norm for v in vector])
    return selected, inverse_square([rows[i] for i in selected])


def inverse_square(rows):
    width = len(rows)
    require(all(len(row) == width for row in rows), "square interpolation basis")
    augmented = [[F(x) for x in row] + [F(j == k) for j in range(width)] for k, row in enumerate(rows)]
    for column in range(width):
        pivot = next((i for i in range(column, width) if augmented[i][column]), None)
        if pivot is None:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [v / scale for v in augmented[column]]
        for i in range(width):
            if i != column:
                scale = augmented[i][column]
                augmented[i] = [a - scale * b for a, b in zip(augmented[i], augmented[column], strict=True)]
    return [row[width:] for row in augmented]


def analyze(design, reference, evidence, *, atol=1e-7, rtol=1e-7, measurement_allowance=1e-12):
    """Exact binary-rational contrasts; no interpolation of unobserved outputs.

    All designs use this oracle and the same numerical profile. Continuous
    predictions/contrasts require independent affine-fragment authority.
    The measurement allowance must be calibrated/frozen; it is not a derived
    universal error bound. ULP floors are additional abstention diagnostics.
    """
    require(all(number(v) and v >= 0 for v in (atol, rtol, measurement_allowance)), "numerical profile")
    require(evidence.get("design_sha256") == design.sha256, "evidence binding")
    outputs = evidence["outputs"]
    ids = {p.id for p in design.probes}
    require(isinstance(outputs, dict) and set(outputs) <= ids, "observation identifiers")
    n, m = len(design.axes), len(design.outputs)
    coefficients, offsets, units = (reference[k] for k in ("coefficients", "offset", "output_units"))
    require(
        len(coefficients) == len(offsets) == len(units) == m and all(len(r) == n for r in coefficients),
        "reference dimensions",
    )
    require(
        all(v is None or number(v) for r in coefficients for v in r)
        and all(v is None or number(v) for v in offsets),
        "reference values",
    )
    for row in outputs.values():
        require(
            row is None
            or isinstance(row, list)
            and len(row) == m
            and all(v is None or number(v) for v in row),
            "output shape/values",
        )
    # Actual binary64 input values may not be nominal antipodes after mapping.
    # Reference coefficients use the frozen binary64 center/radius basis.
    executed = {
        p.id: tuple((F(v) - F(a.center)) / F(a.radius) for a, v in zip(design.axes, p.values, strict=True))
        for p in design.probes
    }
    counts, first, comparisons = Counter(), None, 0

    def relation(kind, points, weights):
        nonlocal first, comparisons
        weights = tuple(F(w) for w in weights)
        offset_weight = sum(weights, F(0))
        axis_weights = [
            sum((w * executed[p.id][j] for p, w in zip(points, weights, strict=True)), F(0)) for j in range(n)
        ]
        needs_fragment = any(axis_weights) or kind == "midpoint" or any(any(p.coordinates) for p in points)
        for i in range(m):
            comparisons += 1
            expected, budget = F(0), F(0)
            known = bool(reference["authority"] and units[i])
            known &= not needs_fragment or reference["affine_fragment_verified"] is True
            for value, weight in (
                (offsets[i], offset_weight),
                *zip(coefficients[i], axis_weights, strict=True),
            ):
                if weight:
                    if value is None:
                        known = False
                    else:
                        expected += weight * F(value)
                        budget += abs(weight) * (F(atol) + F(rtol) * abs(F(value)))
            known &= all(a.unit for a, weight in zip(design.axes, axis_weights, strict=True) if weight)
            observed = [outputs.get(p.id) for p in points]
            present = all(row is not None and row[i] is not None for row in observed)
            status, actual, discrepancy, resolution = "UNKNOWN", None, None, None
            if known and present:
                actual = sum((w * F(row[i]) for w, row in zip(weights, observed, strict=True)), F(0))
                budget += sum(abs(w) for w in weights) * F(measurement_allowance)
                resolution = sum(
                    (abs(w) * F(math.ulp(row[i])) for w, row in zip(weights, observed, strict=True)), F(0)
                )
                discrepancy = abs(actual - expected)
                status = (
                    "FAIL"
                    if discrepancy > budget + resolution
                    else "UNKNOWN"
                    if discrepancy > budget or resolution > budget
                    else "PASS"
                )
            counts[(kind, status)] += 1
            if first is None and status == "FAIL":
                first = {
                    "relation": kind,
                    "output": design.outputs[i],
                    "queries": [p.id for p in points],
                    "expected": float(expected),
                    "actual": float(actual),
                    "budget": float(budget),
                    "resolution_floor": float(resolution),
                }

    for point in design.probes:
        relation("pointwise", [point], [1])
    # Antithetic contrasts and actual measured midpoint closure. No prediction
    # of interior curvature is borrowed from canonical endpoint measurements.
    by_coords = {p.coordinates: p for p in design.probes}
    center = by_coords.get((0.0,) * n)
    visited = set()
    for point in design.probes:
        opposite = by_coords.get(tuple(-c for c in point.coordinates))
        if opposite is None or opposite.id == point.id or point.id in visited:
            continue
        visited.update((point.id, opposite.id))
        relation("antithetic", [point, opposite], [F(1, 2), F(-1, 2)])
        if center:
            relation("midpoint", [point, opposite, center], [F(1, 2), F(1, 2), -1])
    ranks, basis_cache, component_bounds, basis_queries = [], {}, [], []
    component_statuses = []
    for i in range(m):
        present_ids = tuple(
            p.id for p in design.probes if outputs.get(p.id) is not None and outputs[p.id][i] is not None
        )
        if present_ids not in basis_cache:
            rows = [[1, *executed[identifier]] for identifier in present_ids]
            selected, inverse = basis_inverse(rows)
            projections = {}
            if inverse is not None:
                for identifier, row in zip(present_ids, rows, strict=True):
                    projections[identifier] = [
                        sum((v * inverse[j][k] for j, v in enumerate(row)), F(0)) for k in range(n + 1)
                    ]
            basis_cache[present_ids] = rank(rows), selected, inverse, projections
        structural_rank, selected, inverse, projections = basis_cache[present_ids]
        ranks.append(structural_rank)
        basis_ids = [present_ids[k] for k in selected]
        basis_queries.append(basis_ids)
        bounds = []
        grounded = bool(reference["authority"] and units[i] and reference["affine_fragment_verified"] is True)
        if inverse is not None:
            for identifier in present_ids:
                weights = projections[identifier]
                residual = sum(
                    (w * F(outputs[key][i]) for w, key in zip(weights, basis_ids, strict=True)), F(0)
                ) - F(outputs[identifier][i])
                limit = F(measurement_allowance) + F(math.ulp(outputs[identifier][i]))
                limit += sum(
                    (
                        abs(w) * (F(measurement_allowance) + F(math.ulp(outputs[key][i])))
                        for w, key in zip(weights, basis_ids, strict=True)
                    ),
                    F(0),
                )
                status = "UNKNOWN" if not grounded else "FAIL" if abs(residual) > limit else "PASS"
                counts[("affine_residual", status)] += 1
                comparisons += 1
                if first is None and status == "FAIL":
                    first = {
                        "relation": "affine_residual",
                        "output": design.outputs[i],
                        "queries": list(dict.fromkeys([*basis_ids, identifier])),
                        "expected": 0.0,
                        "actual": float(residual),
                        "budget": float(limit),
                    }
        for j, expected in enumerate((offsets[i], *coefficients[i])):
            status = "UNKNOWN"
            detail = {
                "component": "offset" if j == 0 else f"effect:{j - 1}",
                "status": status,
                "reference": expected,
                "estimate": None,
                "measurement_radius": None,
                "weight_l1": None,
            }
            if inverse is not None:
                weights = inverse[j]
                estimate = sum(
                    (w * F(outputs[key][i]) for w, key in zip(weights, basis_ids, strict=True)), F(0)
                )
                radius = sum(
                    (
                        abs(w) * (F(measurement_allowance) + F(math.ulp(outputs[key][i])))
                        for w, key in zip(weights, basis_ids, strict=True)
                    ),
                    F(0),
                )
                detail.update(
                    estimate=float(estimate),
                    measurement_radius=float(radius),
                    weight_l1=float(sum(abs(w) for w in weights)),
                )
                if grounded and expected is not None and (j == 0 or design.axes[j - 1].unit):
                    limit = F(atol) + F(rtol) * abs(F(expected))
                    distance = abs(estimate - F(expected))
                    status = (
                        "FAIL"
                        if distance > limit + radius
                        else "PASS"
                        if distance + radius <= limit
                        else "UNKNOWN"
                    )
                    if first is None and status == "FAIL":
                        first = {
                            "relation": "identified_component",
                            "output": design.outputs[i],
                            "queries": basis_ids,
                            "expected": expected,
                            "actual": float(estimate),
                            "budget": float(limit),
                            "resolution_floor": float(radius),
                            "component": detail["component"],
                        }
            detail["status"] = status
            bounds.append(detail)
            counts[("identified_component", status)] += 1
            component_statuses.append(status)
            comparisons += 1
        component_bounds.append(bounds)
    finite = (
        "FAIL"
        if first
        else "UNKNOWN"
        if any(kind != "identified_component" and s == "UNKNOWN" and c for (kind, s), c in counts.items())
        else "PASS"
    )
    complete = (
        all(a.unit for a in design.axes)
        and reference["authority"]
        and reference["affine_fragment_verified"] is True
        and all(units)
        and all(v is not None for r in coefficients for v in r)
        and all(v is not None for v in offsets)
    )
    full = (
        "FAIL"
        if first
        else "PASS"
        if finite == "PASS" and complete and all(s == "PASS" for s in component_statuses)
        else "UNKNOWN"
    )
    return {
        "schema_version": 1,
        "format": "mltools-research-design-analysis",
        "development_only": True,
        "design_sha256": design.sha256,
        "reference_sha256": fingerprint(reference),
        "evidence_sha256": fingerprint(evidence),
        "finite_status": finite,
        "full_affine_request_status": full,
        "first_divergence": first,
        "counts": {
            kind: {status: counts[(kind, status)] for status in ("PASS", "FAIL", "UNKNOWN")}
            for kind in ("pointwise", "antithetic", "midpoint", "identified_component", "affine_residual")
        },
        "comparisons": comparisons,
        "rank_by_output": ranks,
        "required_rank": n + 1,
        "component_bounds": component_bounds,
        "identification_basis_queries": basis_queries,
        "basis_selection": "Output-independent pivoted rows, threshold 1e-12; exact inverse/intervals",
        "identification_scope": (
            "Component intervals assume the declared measurement allowance plus one output ULP; "
            "unknown references stay unknown"
        ),
        "numeric_profile": {
            "atol": atol,
            "rtol": rtol,
            "measurement_allowance": measurement_allowance,
            "authority": "Unfrozen empirical development allowance; not a universal execution-error bound",
        },
        "scope": "Executed-coordinate contrasts/conditional intervals; no inferred reference or affinity",
    }
