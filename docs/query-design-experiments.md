# Research query-design comparisons

`scripts/compare_query_designs.py` acquires new observations for equal-budget query designs on the six existing development cases. It does not execute the unobserved checkpoint candidates or a frozen final benchmark. The canonical public `Suite` format and its validation remain unchanged.

The research-only format, `mltools-research-query-design` version 1, binds named axes/bounds, binary64 input values, nominal normalized coordinates, ordered query identifiers, generator/seed and a two-action temporal schedule. The worker binds the original case, statistics snapshot and design. It verifies genuine upstream sources and project source stability, then executes actual processors. `Design` never invents canonical probe IDs or interpolates old captured outputs.

## Designs and equal evidence

For d input coordinates, every design acquires 2d+1 queries, counting the measured center:

- **Compiled geometry:** the existing compiler's center and positive/negative axis endpoints.
- **Manual axis:** independently authored center and antithetic axis endpoints. Its geometry intentionally matches the compiler; this is the strongest straightforward manual design, not a weakened baseline.
- **Random axis:** random axis order and continuous amplitude uniformly in [0.25,1], with both signs for every axis. This structured random design isolates effects under partial reference evidence.
- **Random dense:** d continuous directions sampled coordinatewise uniformly in [-1,1], each with its negative and the measured center. Dense mixing is not the only random baseline, because it can obscure effects when nuisance reference coefficients are unknown.

Random seeds are 7, 23 and 101. They are repeated measurements within a development case, not independent incidents. No seed is selected by execution outcome. The generator uses the recorded Python environment; the exact generated inputs are preserved in the request for replay. One worker runs at a time. Each framework uses one acquisition implementation for all eight designs, with fresh query state/queues. LeRobot uses the existing genuine processor capture; OpenPI uses the same deployed-input, unnormalization, absolute conversion and output operators, with an explicitly authored fixture queue. The OpenPI research capture does not acquire training-transform evidence.

All designs use the same independent reference, provenance checks and **shared research oracle**. “Compiled” in these results denotes query geometry. This is not an end-to-end runtime or detection comparison between the original compiled evaluator and a different oracle. Existing supplied-observation oracle comparisons remain separately recorded in `development/oracles-002/`.

## Executed geometry and numerical scope

Let the frozen binary64 axis center/radius be c and r, with executed input v. The oracle forms `(v-c)/r` using exact binary-rational arithmetic. Nominal opposite coordinates can round into asymmetric actual inputs. Unknown coefficients cancel only when their **executed-coordinate** contrast weights are exactly zero; small residual weights are not rounded away.

For a linear contrast with observation weights w, the offset weight is sum(w) and each coefficient weight is the corresponding weighted executed coordinate. Known reference component tolerances propagate through those weights. Unspecified reference values remain unspecified. Continuous predictions and affinity checks require independent affine-fragment authority; measured center assertions can remain grounded separately.

Exact design rank is structural coverage, not sufficient numerical observability. A design can have full rank and still hide a large coefficient error if its input displacement is tiny. The oracle selects a spread interpolation basis using output-independent pivoted row geometry with two reorthogonalization passes and a 1e-12 proposal threshold. It then inverts the actual selected binary-rational matrix exactly. A numerical basis proposal never replaces the exact inverse or interval checks, and an inadequate basis retains UNKNOWN.

For an admitted underlying affine map, write measured basis outputs as `y = X theta + e`. Conditional on per-measurement bounds epsilon, exact inversion gives `theta_hat = inverse(X)y` and component radius `R_j = sum_q abs(inverse(X)[j,q]) epsilon_q`. The report records those radii and weight norms. A known component passes only if its entire interval lies inside its independent reference tolerance; a disjoint interval fails; overlap is UNKNOWN. Additional observed rows check affine residual consistency. Estimated unknown components are diagnostics, never reference authority.

The development profile uses component `atol=rtol=1e-7` and per-measurement allowance 1e-12, plus one recorded-output ULP for interval propagation. These are **unfrozen empirical allowances**, not a derived bound on every upstream operation. The conditional interval statement is elementary linear algebra, not a new theorem or unconditional executable-equivalence guarantee. Source fragment membership, reference authority and the numerical model remain assumptions. New or unsupported numerical regimes require evidence/calibration before interpretation.

`finite_status` distinguishes observed agreement/contradiction from `full_affine_request_status`, which additionally requires complete reference authority and sufficiently tight component intervals. The latter concerns the admitted affine model under the declared numerical profile. It does not certify arbitrary nonlinear behavior, future floating-point errors, checkpoint training identity, physical calibration or safety. The enclosing report preserves UNKNOWN for unsupported requested checks and withholds numerical observations when binding/provenance/schedule guards fail.

## Capture, replay and cost

With the documented pinned environments already installed:

```powershell
uv run --no-sync python scripts/compare_query_designs.py --lerobot-python .venv/Scripts/python.exe --openpi-python .venv-openpi-clean/Scripts/python.exe --output .test-tmp/query-designs-NEW
uv run --no-sync python -O scripts/compare_query_designs.py --verify research/publication/development/query-designs-001
```

The output must be a new directory. Initial environment setup is documented in the repository and `integrations/openpi/README.md`. Acquisition constructs no policy/model/tokenizer, fetches no weights or data, contacts no robot, and rejects Python socket/DNS audit events. The guard is not an OS network sandbox. Peak working set is observed and checked after execution, not enforced as a hard ceiling.

Requests, native stdout/stderr, structured captures, analyses, analysis timings and file/source identities are preserved. Replay checks actual recorded inputs, second actions, queue exhaustion and LeRobot reset observations against the requested schedule. Original artifact/case hashes, upstream source identities and exact design hashes apply equally to all methods. Recorded producer assertions are not cryptographic execution attestation.

Framework import/setup time is separated from per-design acquisition; analysis timings are saved separately before replay verification. Shared-process caches and fixed development order remain disclosed. Single executions are not timing estimates or evidence of a speed advantage. Identical geometry still receives real independent acquisition, rather than being counted as unobserved measurements. Counterexample acquisition/reduction cost, original stateful/manual workflow comparison, external tools and final held-out evaluation remain separate required work.
