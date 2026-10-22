# Development affine probe API

This method is a research primitive; see `research/publication/method.md` for assumptions and pending scientific validation. It is not a replacement for the existing bundle checker or its lifecycle/camera checks.

```python
from mltools.obligations import Axis, Reference, Suite, compile_suite, evaluate

suite = compile_suite(
    (Axis("action", "joint", "normalized", -1, 1),
     Axis("origin", "joint", "degree", -10, 10),
     Axis("current", "joint", "degree", -10, 10)),
    ("joint.pos",),
)
reference = Reference(
    coefficients=((2.0, 10.0, 0.0),), offset=(3.0,),
    output_units=("degree",), authority="independently supplied development specification",
    affine_fragment_verified=True,
)
# Obtain actual observations from an explicitly executed, reviewed adapter.
# Never synthesize these observations from the reference to claim validation.
report = evaluate(suite, reference, observations)
```

Here the specification is `u=2*z+q_origin+3`; coefficients use normalized probe coordinates, so the origin coefficient is 10 for an axis whose radius is 10 degrees. `None` means an unspecified coefficient, offset, unit or authority. Known failing obligations can yield FAIL despite other unknowns; otherwise all required evidence must pass for PASS. A missing affine-fragment declaration prevents a complete PASS. Authority text is a trusted assertion, not an automatically authenticated physical reference.

`suite.to_dict()` returns a version-1 JSON object with method `affine-obligations-development`, `axes`, `outputs`, and `probes`. `Suite.from_dict()` validates this exact method/version and canonical intervention identities, coordinates and values. Witness replay may retain a nonempty subset of canonical probes; changing their inputs or labels is rejected. Bounds must have a distinct representable center. `Reference.to_dict()` returns coefficient rows, offsets, output units, authority and the explicit boolean fragment declaration.

Observations are a JSON object with `suite_sha256` and `outputs`, mapping canonical query IDs to output lists. Missing rows or per-output `null` values are unknown. Unplanned queries, wrong dimensions, nonfinite values and mismatched suite identities are errors. A trusted producer may include further inspectable trace/provenance data; its canonical JSON is included in the report's evidence fingerprint.

Reports identify the suite, reference and complete observation object by canonical SHA-256. They include per-output nominal design rank, planned/observed query counts, each offset/effect/closure obligation, its expected/actual values, a conservative binary64 output-resolution diagnostic, and a two- or three-query witness for the first failing local relation. Rank establishes input-design coverage, not numerical effect resolution, physical authority or arbitrary functional equivalence. Output-resolution ambiguity returns UNKNOWN; the diagnostic does not bound upstream arithmetic, sensor noise or lower-precision producer errors.

All formats and API names here remain explicitly developmental. A schema does not prove execution, source identity or scope. The OpenPI worker records selected source checks; those checks are not automatically promoted to conformance obligations by `evaluate()`. The versioned case wrapper and `mltools probe` CLI add explicit binding, source and artifact checks; see `probe-cases.md`. Producer metadata remains trusted rather than cryptographic execution attestation.
