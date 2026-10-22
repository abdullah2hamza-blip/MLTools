"""Versioned action-only designs; no temporal capture schedule is implied.

This research adapter reuses independent geometry/oracle primitives without
altering the existing source-bound temporal design format or its evidence.
"""

import random
from dataclasses import asdict, dataclass

from query_designs import Axis, Query, fingerprint, number, require

from mltools.obligations import Axis as CoreAxis
from mltools.obligations import compile_suite

FORMAT = "mltools-action-normalization-design"
METHODS = {"canonical", "manual_axis", "random_axis", "random_dense"}
SCHEDULE = {
    "operation": "action_unnormalization",
    "native_decoder_calls_per_query": 1,
    "batch_queries": False,
    "state_inputs": False,
    "queue_operations": False,
    "policy_inference": False,
}


def points(axes, method, seed):
    """Generate geometry from inputs only, before any processor observations."""
    n = len(axes)
    if method == "canonical":
        suite = compile_suite(tuple(CoreAxis(**asdict(a)) for a in axes), ("geometry",))
        return tuple(Query(p.id, p.coordinates, p.values) for p in suite.probes)
    rng = random.Random(seed) if method.startswith("random") else None
    coordinates = [(0.0,) * n]
    allocation = list(range(n))
    if method == "random_axis":
        rng.shuffle(allocation)
    for axis in allocation:
        if method == "random_dense":
            direction = tuple(rng.uniform(-1.0, 1.0) for _ in axes)
        else:
            amplitude = rng.uniform(0.25, 1.0) if method == "random_axis" else 1.0
            direction = tuple(amplitude if i == axis else 0.0 for i in range(n))
        coordinates.extend((direction, tuple(-v for v in direction)))
    return tuple(
        Query(f"q{i:03}", c, tuple(a.at(v) for a, v in zip(axes, c, strict=True)))
        for i, c in enumerate(coordinates)
    )


@dataclass(frozen=True)
class ActionDesign:
    axes: tuple[Axis, ...]
    outputs: tuple[str, ...]
    probes: tuple[Query, ...]
    generator: str
    seed: int | None

    def __post_init__(self):
        require(
            isinstance(self.axes, tuple)
            and 1 <= len(self.axes) <= 32
            and all(isinstance(a, Axis) and a.port == "action" for a in self.axes),
            "action-only axes",
        )
        require(len({a.name for a in self.axes}) == len(self.axes), "duplicate action axis")
        require(
            isinstance(self.outputs, tuple)
            and 1 <= len(self.outputs) <= 32
            and len(set(self.outputs)) == len(self.outputs)
            and all(isinstance(v, str) and bool(v.strip()) for v in self.outputs),
            "normalization output labels",
        )
        require(isinstance(self.generator, str) and self.generator in METHODS, "action generator")
        require(
            type(self.seed) is int and 0 <= self.seed < 2**32
            if self.generator.startswith("random")
            else self.seed is None,
            "explicit random seed or deterministic design",
        )
        require(isinstance(self.probes, tuple), "immutable action queries")
        expected = points(self.axes, self.generator, self.seed)
        require(self.probes == expected, "queries differ from the declared input-only generator")
        require(
            all(
                number(c)
                and number(v)
                and -1 <= c <= 1
                and a.lower <= v <= a.upper
                and (c == 0 or v != a.center)
                for p in self.probes
                for a, c, v in zip(self.axes, p.coordinates, p.values, strict=True)
            ),
            "bounded noncollapsed action inputs",
        )
        require(
            len({p.values for p in self.probes}) == len(self.probes) == 2 * len(self.axes) + 1,
            "distinct fixed-budget action queries",
        )

    def to_dict(self):
        return {
            "schema_version": 1,
            "format": FORMAT,
            "axes": [asdict(a) for a in self.axes],
            "outputs": list(self.outputs),
            "probes": [
                {"id": p.id, "coordinates": list(p.coordinates), "values": list(p.values)}
                for p in self.probes
            ],
            "generator": self.generator,
            "seed": self.seed,
            "schedule": SCHEDULE.copy(),
        }

    @property
    def sha256(self):
        return fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        require(
            isinstance(value, dict)
            and set(value)
            == {"schema_version", "format", "axes", "outputs", "probes", "generator", "seed", "schedule"},
            "action design fields",
        )
        require(
            type(value["schema_version"]) is int
            and value["schema_version"] == 1
            and value["format"] == FORMAT
            and isinstance(value["schedule"], dict)
            and fingerprint(value["schedule"]) == fingerprint(SCHEDULE),
            "action-only format/version/schedule",
        )
        require(
            all(isinstance(value[k], list) for k in ("axes", "outputs", "probes"))
            and 1 <= len(value["axes"]) <= 32
            and 1 <= len(value["outputs"]) <= 32
            and len(value["probes"]) == 2 * len(value["axes"]) + 1,
            "bounded action design arrays",
        )
        try:
            for p in value["probes"]:
                require(
                    isinstance(p, dict)
                    and set(p) == {"id", "coordinates", "values"}
                    and all(isinstance(p[k], list) for k in ("coordinates", "values")),
                    "action query fields",
                )
            return cls(
                tuple(Axis(**a) for a in value["axes"]),
                tuple(value["outputs"]),
                tuple(Query(p["id"], tuple(p["coordinates"]), tuple(p["values"])) for p in value["probes"]),
                value["generator"],
                value["seed"],
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("Research design: invalid action-only data") from exc


def generate(suite, method, seed=None):
    axes = tuple(Axis(**asdict(a)) for a in suite.axes)
    require(method in METHODS, "action generator")
    return ActionDesign(axes, tuple(suite.outputs), points(axes, method, seed), method, seed)


def generate_all(suite):
    return {
        "canonical": generate(suite, "canonical"),
        "manual_axis": generate(suite, "manual_axis"),
        **{
            f"{method}_{seed}": generate(suite, method, seed)
            for method in ("random_axis", "random_dense")
            for seed in (7, 23, 101)
        },
    }
