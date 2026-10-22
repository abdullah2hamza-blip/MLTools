"""Independent development oracles over the same supplied, executed probe design.

These are oracle comparisons, not manual/random test-generation baselines. No ML
imports or adapter execution occur here. Missing expectations are never fitted.
"""

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict

from .io import InputError, ensure_json, number
from .obligations import Reference, Suite
from .report import aggregate


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _validate(suite, reference, evidence, atol, rtol):
    ensure_json(evidence)
    if not isinstance(suite, Suite) or not isinstance(reference, Reference):
        raise InputError("Oracles require a validated Suite and Reference")
    if not isinstance(evidence, dict) or evidence.get("suite_sha256") != suite.sha256:
        raise InputError("Oracle evidence is not bound to this exact suite")
    if not number(atol) or not number(rtol) or atol < 0 or rtol < 0:
        raise InputError("Oracle tolerances must be finite and nonnegative")
    n, m = len(suite.axes), len(suite.outputs)
    if (
        len(reference.coefficients) != m
        or len(reference.offset) != m
        or len(reference.output_units) != m
        or any(len(row) != n for row in reference.coefficients)
    ):
        raise InputError("Oracle reference dimensions do not match ports")
    if any(x is not None and not number(x) for row in reference.coefficients for x in row):
        raise InputError("Oracle coefficients must be finite or unknown")
    if any(x is not None and not number(x) for x in reference.offset):
        raise InputError("Oracle offsets must be finite or unknown")
    outputs = evidence.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) - {p.id for p in suite.probes}:
        raise InputError("Oracle outputs require supplied query identifiers")
    for row in outputs.values():
        if row is not None and (
            not isinstance(row, list) or len(row) != m or any(x is not None and not number(x) for x in row)
        ):
            raise InputError("Oracle output rows must contain finite/unknown port values")
    return outputs


def _finite(value):
    if not number(value):
        raise InputError("Oracle arithmetic exceeds supported finite range")
    return value


def _sum(values):
    try:
        return _finite(math.fsum(values))
    except (OverflowError, ValueError) as exc:
        raise InputError("Oracle arithmetic exceeds supported finite range") from exc


def _expectation(coefficients, coordinates, axes):
    terms = []
    for coefficient, coordinate, axis in zip(coefficients, coordinates, axes, strict=True):
        if coordinate == 0:
            continue  # An unknown coefficient at exactly zero cannot affect this prediction.
        if coefficient is None or axis.unit is None:
            return None
        terms.append(_finite(coefficient * coordinate))
    return _sum(terms)


def _budget(coefficients, weights, atol, rtol):
    terms = []
    for coefficient, weight in zip(coefficients, weights, strict=True):
        if weight != 0:
            if coefficient is None:
                return None
            terms.append(abs(weight) * _finite(atol + rtol * abs(coefficient)))
    return _sum(terms)


def _ranks(suite, outputs, selected):
    """Independent Gram-Schmidt observability diagnostic; no reference is fitted."""
    ranks = {}
    eligible = set(selected)
    for i, name in enumerate(suite.outputs):
        basis = []
        for probe in suite.probes:
            row = outputs.get(probe.id)
            if probe.id not in eligible or row is None or row[i] is None:
                continue
            vector = [*probe.coordinates, 1.0]
            for unit in basis:
                dot = _sum(a * b for a, b in zip(vector, unit, strict=True))
                vector = [a - dot * b for a, b in zip(vector, unit, strict=True)]
            norm = math.sqrt(_sum(v * v for v in vector))
            if norm > 1e-10:
                basis.append([v / norm for v in vector])
        ranks[name] = len(basis)
    return ranks


class _Record:
    def __init__(self, suite, reference, evidence, method, atol, rtol):
        self.suite, self.reference, self.evidence = suite, reference, evidence
        self.method, self.atol, self.rtol = method, atol, rtol
        self.groups = defaultdict(Counter)
        self.first = None
        self.used = set()

    def compare(self, kind, output, expected, actual, queries, *, grounded=True, resolution=0.0, budget=None):
        known = bool(self.reference.authority and self.reference.output_units[output] and grounded)
        if not known or expected is None or actual is None:
            status = "UNKNOWN"
        else:
            expected, actual, resolution = map(_finite, (expected, actual, resolution))
            limit = _finite(self.atol + self.rtol * abs(expected) if budget is None else budget)
            error = _finite(abs(actual - expected))
            status = (
                "FAIL"
                if error > limit + resolution
                else "UNKNOWN"
                if error > limit or resolution > limit
                else "PASS"
            )
        self.groups[kind][status] += 1
        self.used.update(queries)
        if status == "FAIL" and self.first is None:
            self.first = {
                "kind": kind,
                "output": self.suite.outputs[output],
                "expected": expected,
                "actual": actual,
                "queries": list(queries),
                "output_resolution_floor": resolution,
                "comparison_budget": limit,
            }

    def result(self, selected):
        # Affine class and axis-unit evidence are required for full numerical PASS.
        grounded = bool(
            self.reference.affine_fragment_verified and all(axis.unit for axis in self.suite.axes)
        )
        groups = {
            kind: {"status": aggregate(counts), "counts": dict(counts)}
            for kind, counts in self.groups.items()
        }
        ranks = _ranks(self.suite, self.evidence["outputs"], selected)
        complete = all(rank == len(self.suite.axes) + 1 for rank in ranks.values())
        return {
            "schema_version": 1,
            "method": self.method,
            "status": aggregate(
                [*(g["status"] for g in groups.values()), "PASS" if grounded and selected else "UNKNOWN"]
            ),
            "applicability": "APPLICABLE" if selected else "N/A",
            "suite_sha256": self.suite.sha256,
            "reference_sha256": fingerprint(self.reference.to_dict()),
            "evidence_sha256": fingerprint(self.evidence),
            "groups": groups,
            "comparison_count": sum(sum(counts.values()) for counts in self.groups.values()),
            "full_design_coverage_status": "PASS" if complete else "UNKNOWN",
            "first_divergence": self.first,
            "coverage": {
                "supplied_queries": len(self.suite.probes),
                "eligible_queries": len(selected),
                "queried_ids": list(selected),
                "observed_eligible_queries": sum(
                    self.evidence["outputs"].get(q) is not None
                    and any(value is not None for value in self.evidence["outputs"][q])
                    for q in selected
                ),
                "rank_by_output": ranks,
                "required_full_rank": len(self.suite.axes) + 1,
            },
            "scope": "Finite numerical oracle on the supplied design; not new probe generation",
            "limitations": [
                "Reference/fragment/producer evidence is trusted; no global or physical safety proof",
                "Binary64 resolution diagnostic is not an upstream error bound",
                "Oracle runtime excludes acquiring the shared adapter observations",
                "Contrast budgets propagate component tolerances and sampled closure residuals",
            ],
        }


def _pointwise(record, probes, outputs):
    reference, suite = record.reference, record.suite
    for probe in probes:
        observed = outputs.get(probe.id)
        for i, coefficients in enumerate(reference.coefficients):
            effect = _expectation(coefficients, probe.coordinates, suite.axes)
            expected = (
                _sum((reference.offset[i], effect))
                if reference.offset[i] is not None and effect is not None
                else None
            )
            actual = observed[i] if observed is not None else None
            resolution = (
                math.ulp(actual) + math.ulp(expected) if actual is not None and expected is not None else 0
            )
            # Canonical y(+/-j) = center +/- effect_j + closure_j. Propagate
            # component budgets instead of a stricter cancelled-output rtol.
            effect_budget = _budget(coefficients, probe.coordinates, record.atol, record.rtol)
            budget = (
                _finite(
                    record.atol
                    + record.rtol * abs(reference.offset[i])
                    + effect_budget
                    + record.atol * sum(abs(x) for x in probe.coordinates)
                )
                if reference.offset[i] is not None and effect_budget is not None
                else None
            )
            record.compare(
                "pointwise",
                i,
                expected,
                actual,
                [probe.id],
                grounded=reference.affine_fragment_verified or not any(probe.coordinates),
                resolution=resolution,
                budget=budget,
            )


def pointwise(suite, reference, evidence, *, atol=1e-7, rtol=1e-7):
    """Direct independent affine prediction; unknown zero-weight terms cancel exactly."""
    outputs = _validate(suite, reference, evidence, atol, rtol)
    record = _Record(suite, reference, evidence, "pointwise-reference-development", atol, rtol)
    _pointwise(record, suite.probes, outputs)
    return record.result([p.id for p in suite.probes])


def relational(suite, reference, evidence, *, atol=1e-7, rtol=1e-7):
    """Strong reference oracle: pointwise checks plus all pairs and geometric midpoints.

    All pairs receive the available reference, so unknown offsets cancel without
    giving the compiler an information advantage. This implementation does not
    call its evaluator or extract expectations from observed adapter outputs.
    """
    outputs = _validate(suite, reference, evidence, atol, rtol)
    record = _Record(suite, reference, evidence, "relational-reference-development", atol, rtol)
    _pointwise(record, suite.probes, outputs)
    coordinates = {p.coordinates: p for p in suite.probes}
    for left, right in itertools.combinations(suite.probes, 2):
        delta = tuple((a - b) / 2 for a, b in zip(left.coordinates, right.coordinates, strict=True))
        closure_weights = tuple(
            (abs(a) - abs(b)) / 2 for a, b in zip(left.coordinates, right.coordinates, strict=True)
        )
        middle = coordinates.get(
            tuple((a + b) / 2 for a, b in zip(left.coordinates, right.coordinates, strict=True))
        )
        lrow, rrow = outputs.get(left.id), outputs.get(right.id)
        for i, coefficients in enumerate(reference.coefficients):
            lvalue, rvalue = (row[i] if row is not None else None for row in (lrow, rrow))
            actual = _sum((lvalue / 2, -rvalue / 2)) if lvalue is not None and rvalue is not None else None
            expected = _expectation(coefficients, delta, suite.axes)
            resolution = math.ulp(lvalue) + math.ulp(rvalue) + math.ulp(actual) if actual is not None else 0
            coefficient_budget = _budget(coefficients, delta, atol, rtol)
            budget = (
                _finite(coefficient_budget + atol * sum(abs(value) for value in closure_weights))
                if coefficient_budget is not None
                else None
            )
            record.compare(
                "paired",
                i,
                expected,
                actual,
                [left.id, right.id],
                grounded=reference.affine_fragment_verified or not any(closure_weights),
                resolution=resolution,
                budget=budget,
            )
            if middle is not None:
                mrow = outputs.get(middle.id)
                mvalue = mrow[i] if mrow is not None else None
                closure = (
                    _sum((lvalue / 2, rvalue / 2, -mvalue))
                    if lvalue is not None and rvalue is not None and mvalue is not None
                    else None
                )
                floor = (
                    math.ulp(lvalue) + math.ulp(rvalue) + math.ulp(mvalue) + math.ulp(closure)
                    if closure is not None
                    else 0
                )
                record.compare(
                    "midpoint",
                    i,
                    0,
                    closure,
                    [left.id, right.id, middle.id],
                    grounded=reference.affine_fragment_verified
                    and all(a.unit for a, value in zip(suite.axes, delta, strict=True) if value != 0),
                    resolution=floor,
                )
    return record.result([p.id for p in suite.probes])


def stateless(suite, reference, evidence, *, atol=1e-7, rtol=1e-7):
    """Pointwise checks only on actually observed equal-origin/current physical values."""
    outputs = _validate(suite, reference, evidence, atol, rtol)
    ports = {
        port: {a.name: (i, a) for i, a in enumerate(suite.axes) if a.port == port}
        for port in ("origin", "current")
    }
    pairs = []
    if ports["origin"] and ports["origin"].keys() == ports["current"].keys():
        for name, (i, axis) in ports["origin"].items():
            j, other = ports["current"][name]
            if axis.unit is None or axis.unit != other.unit:
                break
            pairs.append((i, j))
        else:
            selected = [p for p in suite.probes if all(p.values[i] == p.values[j] for i, j in pairs)]
            record = _Record(suite, reference, evidence, "stateless-diagonal-development", atol, rtol)
            _pointwise(record, selected, outputs)
            result = record.result([p.id for p in selected])
            result["scope"] = "Only supplied queries with origin=current in matching declared units"
            result["excluded_queries"] = len(suite.probes) - len(selected)
            if not selected:
                result["reason"] = "No supplied query lies on the equal-origin/current subdomain"
            return result
    record = _Record(suite, reference, evidence, "stateless-diagonal-development", atol, rtol)
    result = record.result([])
    result["reason"] = "No matched origin/current coordinate layout with known common units"
    result["excluded_queries"] = len(suite.probes)
    return result
