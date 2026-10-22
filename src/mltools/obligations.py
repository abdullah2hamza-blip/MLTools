"""Development research: compile bounded probes for partially specified affine interfaces.

This module is not a final benchmark or a universal equivalence checker. Coefficients use
normalized probe coordinates; physical labels and reference values require independent authority.
"""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

from .io import InputError, ensure_json, number
from .report import Status, aggregate


def _label(value, *, optional=False):
    return (optional and value is None) or (isinstance(value, str) and bool(value.strip()))


def _hash_json(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Axis:
    port: str
    name: str
    unit: str | None
    lower: float
    upper: float

    def __post_init__(self):
        if self.port not in ("action", "origin", "current") or not _label(self.name):
            raise InputError("Probe axis requires action/origin/current port and a name")
        if not _label(self.unit, optional=True):
            raise InputError("Axis unit must be a nonempty string or unknown")
        if not number(self.lower) or not number(self.upper) or self.lower >= self.upper:
            raise InputError("Probe bounds must be finite and increasing")
        if not number(self.radius) or self.radius <= 0 or not number(self.center):
            raise InputError("Probe bounds exceed supported numeric range")
        if not self.lower < self.center < self.upper:
            raise InputError("Probe bounds must admit a distinct representable center")

    @property
    def key(self):
        return f"{self.port}:{self.name}"

    @property
    def center(self):
        return self.lower / 2 + self.upper / 2

    @property
    def radius(self):
        return self.upper / 2 - self.lower / 2


@dataclass(frozen=True)
class Probe:
    id: str
    coordinates: tuple[float, ...]
    values: tuple[float, ...]

    def __post_init__(self):
        if not _label(self.id) or not all(number(x) for x in (*self.coordinates, *self.values)):
            raise InputError("Probes require a nonempty identifier and finite coordinates/values")
        object.__setattr__(self, "coordinates", tuple(self.coordinates))
        object.__setattr__(self, "values", tuple(self.values))


def _ports(axes, outputs):
    axes, outputs = tuple(axes), tuple(outputs)
    if not 1 <= len(axes) <= 96 or not all(isinstance(a, Axis) for a in axes):
        raise InputError("Provide 1..96 uniquely named port axes")
    if len({a.key for a in axes}) != len(axes):
        raise InputError("Provide 1..96 uniquely named port axes")
    if not 1 <= len(outputs) <= 32 or not all(_label(output) for output in outputs):
        raise InputError("Provide 1..32 unique output names")
    if len(set(outputs)) != len(outputs):
        raise InputError("Provide 1..32 unique output names")
    return axes, outputs


def _probes(axes):
    n = len(axes)
    probes = [Probe("center", (0.0,) * n, tuple(a.center for a in axes))]
    for j, axis in enumerate(axes):
        for sign, name in ((1.0, "plus"), (-1.0, "minus")):
            coordinates = tuple(sign if i == j else 0.0 for i in range(n))
            values = tuple(
                a.upper if i == j and sign == 1 else a.lower if i == j else a.center
                for i, a in enumerate(axes)
            )
            probes.append(Probe(f"{name}:{axis.key}", coordinates, values))
    return tuple(probes)


@dataclass(frozen=True)
class Suite:
    axes: tuple[Axis, ...]
    outputs: tuple[str, ...]
    probes: tuple[Probe, ...]

    def __post_init__(self):
        axes, outputs = _ports(self.axes, self.outputs)
        probes = tuple(self.probes)
        expected = {p.id: p for p in _probes(axes)}
        if not probes or not all(isinstance(p, Probe) for p in probes):
            raise InputError("Suite requires compiled probes or a nonempty subset for witness replay")
        if len({p.id for p in probes}) != len(probes) or any(expected.get(p.id) != p for p in probes):
            raise InputError(
                "Probe identities, coordinates and values must match the canonical bounded design"
            )
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "probes", probes)

    def to_dict(self):
        return {
            "schema_version": 1,
            "method": "affine-obligations-development",
            "axes": [asdict(a) for a in self.axes],
            "outputs": list(self.outputs),
            "probes": [
                {"id": p.id, "coordinates": list(p.coordinates), "values": list(p.values)}
                for p in self.probes
            ],
        }

    @classmethod
    def from_dict(cls, value):
        ensure_json(value)
        try:
            if (
                type(value["schema_version"]) is not int
                or value["schema_version"] != 1
                or value["method"] != "affine-obligations-development"
            ):
                raise InputError("Unsupported compiled suite version or method")
            return cls(
                tuple(Axis(**a) for a in value["axes"]),
                tuple(value["outputs"]),
                tuple(Probe(**p) for p in value["probes"]),
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise InputError("Invalid compiled probe suite") from exc

    @property
    def sha256(self):
        return _hash_json(self.to_dict())


@dataclass(frozen=True)
class Reference:
    coefficients: tuple[tuple[float | None, ...], ...]
    offset: tuple[float | None, ...]
    output_units: tuple[str | None, ...]
    authority: str | None
    affine_fragment_verified: bool = False

    def __post_init__(self):
        if not _label(self.authority, optional=True):
            raise InputError("Reference authority must be a nonempty string or unknown")
        if not isinstance(self.affine_fragment_verified, bool):
            raise InputError("Affine fragment evidence must be explicitly true or false")
        if not all(_label(unit, optional=True) for unit in self.output_units):
            raise InputError("Output units must be nonempty strings or unknown")
        object.__setattr__(self, "coefficients", tuple(tuple(row) for row in self.coefficients))
        object.__setattr__(self, "offset", tuple(self.offset))
        object.__setattr__(self, "output_units", tuple(self.output_units))

    def to_dict(self):
        return {
            "coefficients": [list(row) for row in self.coefficients],
            "offset": list(self.offset),
            "output_units": list(self.output_units),
            "authority": self.authority,
            "affine_fragment_verified": self.affine_fragment_verified,
        }


def compile_suite(axes: tuple[Axis, ...], outputs: tuple[str, ...]) -> Suite:
    """Generate center and paired bounded interventions without inspecting an adapter."""
    axes, outputs = _ports(axes, outputs)
    return Suite(axes, outputs, _probes(axes))


def _rank(rows: list[list[float]]) -> int:
    if not rows:
        return 0
    rows = [row[:] for row in rows]
    pivot = 0
    for column in range(len(rows[0])):
        found = next((i for i in range(pivot, len(rows)) if abs(rows[i][column]) > 1e-10), None)
        if found is None:
            continue
        rows[pivot], rows[found] = rows[found], rows[pivot]
        divisor = rows[pivot][column]
        rows[pivot] = [x / divisor for x in rows[pivot]]
        for i in range(pivot + 1, len(rows)):
            factor = rows[i][column]
            rows[i] = [a - factor * b for a, b in zip(rows[i], rows[pivot], strict=True)]
        pivot += 1
        if pivot == len(rows):
            break
    return pivot


def evaluate(suite: Suite, reference: Reference, observations: dict, *, atol=1e-7, rtol=1e-7) -> dict:
    """Check finite obligations, retaining UNKNOWN for absent reference/observation cells.

    Paired differences cancel unspecified offsets. Rank is an observability diagnostic,
    not evidence of physical correctness or of arbitrary black-box affinity.
    """
    ensure_json(observations)
    if not isinstance(observations, dict):
        raise InputError("Probe results must be a JSON object")
    if observations.get("suite_sha256") != suite.sha256:
        raise InputError("Probe results are not bound to this exact compiled suite")
    if not number(atol) or not number(rtol) or atol < 0 or rtol < 0:
        raise InputError("Tolerances must be finite and nonnegative")
    width, height = len(suite.axes), len(suite.outputs)
    if (
        len(reference.coefficients) != height
        or len(reference.offset) != height
        or len(reference.output_units) != height
        or any(len(row) != width for row in reference.coefficients)
    ):
        raise InputError("Reference dimensions do not match compiled ports")
    if any(x is not None and not number(x) for row in reference.coefficients for x in row):
        raise InputError("Reference coefficients must be finite numbers or unknown")
    if any(x is not None and not number(x) for x in reference.offset):
        raise InputError("Reference offsets must be finite numbers or unknown")
    values = observations.get("outputs")
    if not isinstance(values, dict) or set(values) - {p.id for p in suite.probes}:
        raise InputError("Probe outputs must use only compiled query identifiers")
    for row in values.values():
        if row is not None and (
            not isinstance(row, list)
            or len(row) != height
            or any(x is not None and not number(x) for x in row)
        ):
            raise InputError("Probe output rows must match output ports with finite/unknown values")
    findings = []

    def cell(query, i):
        row = values.get(query)
        return row[i] if row is not None else None

    def add(kind, output, expected, actual, ids, grounded=True, resolution=0.0):
        if actual is not None and not number(actual):
            raise InputError("Computed obligation exceeds supported numeric range")
        known = bool(reference.authority and reference.output_units[output] and grounded)
        if not known or expected is None or actual is None:
            status = Status.UNKNOWN
        else:
            limit, discrepancy = atol + rtol * abs(expected), abs(expected - actual)
            if not number(limit) or not number(discrepancy):
                raise InputError("Comparison exceeds supported numeric range")
            if discrepancy > limit + resolution:
                status = Status.FAIL
            elif discrepancy > limit or resolution > limit:
                status = Status.UNKNOWN
            else:
                status = Status.PASS
        findings.append(
            {
                "obligation": kind,
                "output": suite.outputs[output],
                "status": status,
                "expected": expected,
                "actual": actual,
                "queries": ids,
                "output_resolution_floor": resolution,
            }
        )

    for i, _output in enumerate(suite.outputs):
        center = cell("center", i)
        add(
            "offset",
            i,
            reference.offset[i],
            center,
            ["center"],
            resolution=math.ulp(center) if center is not None else 0.0,
        )
        for j, axis in enumerate(suite.axes):
            plus_id, minus_id = f"plus:{axis.key}", f"minus:{axis.key}"
            plus, minus = cell(plus_id, i), cell(minus_id, i)
            slope = (plus / 2 - minus / 2) if plus is not None and minus is not None else None
            add(
                f"effect:{axis.key}",
                i,
                reference.coefficients[i][j],
                slope,
                [plus_id, minus_id],
                grounded=bool(axis.unit),
                resolution=(math.ulp(plus) + math.ulp(minus) + math.ulp(slope) if slope is not None else 0.0),
            )
            closure = (
                plus / 2 + minus / 2 - center
                if plus is not None and minus is not None and center is not None
                else None
            )
            add(
                f"affinity:{axis.key}",
                i,
                0.0,
                closure,
                ["center", plus_id, minus_id],
                grounded=bool(axis.unit and reference.affine_fragment_verified),
                resolution=(
                    math.ulp(plus) + math.ulp(minus) + math.ulp(center) + math.ulp(closure)
                    if closure is not None
                    else 0.0
                ),
            )
    ranks = {}
    for i, output in enumerate(suite.outputs):
        design = [list(p.coordinates) + [1.0] for p in suite.probes if cell(p.id, i) is not None]
        ranks[output] = _rank(design)
        if ranks[output] < width + 1:
            findings.append(
                {
                    "obligation": "design_rank",
                    "output": output,
                    "status": Status.UNKNOWN,
                    "expected": width + 1,
                    "actual": ranks[output],
                    "queries": [],
                }
            )
    first = next((f for f in findings if f["status"] == Status.FAIL), None)
    selected = set(first["queries"]) if first else set()
    return {
        "schema_version": 1,
        "method": "affine-obligations-development",
        "status": aggregate(f["status"] for f in findings),
        "suite_sha256": suite.sha256,
        "reference_sha256": _hash_json(reference.to_dict()),
        "evidence_sha256": _hash_json(observations),
        "coverage": {
            "rank_by_output": ranks,
            "required_rank": width + 1,
            "planned_queries": len(suite.probes),
            "observed_queries": sum(v is not None and any(x is not None for x in v) for v in values.values()),
        },
        "findings": findings,
        "first_divergence": first,
        "witness": [
            {"query": p.id, "inputs": list(p.values), "outputs": values[p.id]}
            for p in suite.probes
            if p.id in selected
        ],
        "limitations": [
            "Finite affine-fragment obligations only; not arbitrary functional equivalence",
            "Reference authority, units, fragment membership and execution producer are trusted inputs",
            "Development method; no final benchmark evidence is supplied by this module",
            "Binary64 output-resolution floor is a diagnostic, not a bound on upstream numeric error",
        ],
    }
