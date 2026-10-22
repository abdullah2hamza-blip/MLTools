# Matched-evidence development oracles

`mltools.research_compare.compare(case, trace)` analyzes a saved versioned probe capture with four numerical methods. It imports neither framework and executes no adapter. These are development comparators on a **supplied design**, not a final benchmark or a manual/random test-generation comparison.

```python
from mltools.research_compare import compare

report = compare("examples/probes/lerobot/moving_anchor.json",
                 "validation/probe-cases-002/lerobot_moving_anchor.trace.json")
print(report["methods"]["strong_relational"]["full_requested_status"])
```

| Method | Actual checks | Scope |
| --- | --- | --- |
| Compiled | Existing center, paired effect, midpoint closure and design-rank obligations | Original method, unchanged |
| Pointwise | Independently compute each known affine prediction from reference coefficients and normalized query coordinates | An unknown coefficient times an exactly zero coordinate does not erase a known prediction |
| Strong relational | Pointwise checks, all supplied query-pair half-differences, and geometric midpoint checks | Independent implementation; unknown offsets cancel; includes bias and curvature checks rather than slopes alone |
| Stateless diagonal | Pointwise checks only on supplied queries whose origin/current **values** match by name in common declared units | No manufactured diagonal observations; excluded temporal queries remain outside its claim |

No baseline calls the compiled numerical evaluator internally. The wrapper uses the existing case loader and provenance report to give every method the same input authority, artifacts, observations and source/binding checks. Observations with an incomplete or failed measurement binding are withheld from every numerical method.

`status` describes the finite numerical checks actually performed. `full_requested_status` additionally includes common provenance/semantic checks and sufficient full-design coverage. An independent Gram-Schmidt calculation reports rank per output using only known cells. Center-only agreement cannot clear a full affine request; rows containing only UNKNOWN values do not count as observed. The stateless diagonal does not need full temporal rank to report its own subdomain agreement, but that agreement cannot clear excluded temporal obligations. Empty applicable domains are N/A with UNKNOWN, never PASS.

## Numerical acceptance budgets

Using the same relative-tolerance constants on arbitrary contrasts would impose different acceptance regions: two large expected effects may cancel to zero. The independent oracles therefore propagate the compiled basis component budgets rather than applying a new relative threshold to that cancellation.

For normalized canonical coordinates `c`, let the reference offset be `b`, effects `A_j`, and define `t_b = atol + rtol*abs(b)`, `t_j = atol + rtol*abs(A_j)`. The sampled midpoint residual `h_j` has budget `atol`. At a canonical endpoint the finite response decomposes as `y(c) = center + c·effect + abs(c)·h`. This is an algebraic decomposition of this supplied center/axis design, not a novel identification theorem or a proof of global affinity.

Pointwise budgets propagate `t_b + sum(abs(c_j)*t_j) + sum(abs(c_j)*atol)`. For a half-difference between queries `l` and `r`, put `w_j=(c_lj-c_rj)/2` and `v_j=(abs(c_lj)-abs(c_rj))/2`. Its budget is `sum(abs(w_j)*t_j) + sum(abs(v_j)*atol)`. Opposite axis endpoints give exactly `t_j`; their closure nuisance cancels. Cross-pair predictions with nonzero closure nuisance, and noncenter pointwise predictions, require verified closure/affine authority. Otherwise they remain UNKNOWN. Center checks and opposite-pair effects can still use their independently known expectations.

These envelopes are conservative propagation of admitted component deviations. They are not a joint feasibility solver, an upstream floating-point error bound, or a universal equivalence guarantee. Binary64 output-resolution floors remain separate and can force UNKNOWN. Development defaults are `atol=rtol=1e-7`; final numerical profiles must be frozen before evaluation. Hand-calculated cancellation, missing-authority, missing-output, curvature and rank controls test these distinctions.

## Ablations and evidence identity

- Offset erasure removes expected offsets at the numerical layer. It preserves the original case/capture identity and records a separate reference fingerprint. It can retain a grounded coefficient failure, but a correct coefficient map still has UNKNOWN absolute offset semantics.
- Independent-reference erasure removes coefficients, offsets, units and authority. The tool abstains rather than fitting a deployment-derived reference. Other known provenance failures can still make the full request FAIL.
- Provenance omission explicitly reports a **reduced check set**. It cannot certify the original provenance requirements. Measurement binding remains required.
- Witness removal preserves the verdict and retains the full supplied design. It changes diagnostic size; real replay, resets, dequeues and generation cost must be measured separately. Earlier genuinely executed reduced witnesses are identified as saved evidence, not rerun by this analysis.

Changing a reference inside the case and relabeling the old trace as a capture of that new case would confound ablation with binding. The comparison API never does that. It neither mutates inputs nor changes missing physical metadata into a numeric convention.

## Reproduce and verify

After installing the current package, run in an environment without ML dependencies:

```powershell
python scripts/analyze_oracle_development.py --output .test-tmp/oracles-NEW
python scripts/verify_oracle_development.py .test-tmp/oracles-NEW
```

The runner first verifies the immutable source capture matrix, original exports and saved freshly executed witnesses. It checks installed-package bytes against workspace sources, then records implementation/input/output hashes and actual analysis time. The verifier independently checks identities and deterministic reanalysis, including under Python `-O`. Timing covers all oracles and ablations together; it is not a per-method speed comparison or adapter acquisition cost.

This development comparison does not complete the required v0.1 stateful checker, optimized/manual query-design, random-query, native round-trip, certificate or external-tool evaluations. The all-pairs implementation's quadratic comparison count must not be used to claim a speed advantage over an optimized manual basis oracle. Different first failing relations are valid witnesses; they are not localization errors merely because their order differs.
