# Development specification: partially grounded temporal obligations

This is a proposed research method, not a completed novelty or benchmark claim. It builds on established affine identification, metamorphic testing and state-machine conformance. The existing v0.1 checker remains a required strong manual stateful baseline, rather than being omitted to make the new compiler appear better.

## System and evidence

Treat a postprocessing adapter as a transducer receiving reset, observation, chunk-creation and dequeue events. A chunk has an episode identifier, origin observation and queue position. A physical contract states separately grounded input/output identities, units, frames, representations, normalization artifacts and lifecycle rules. Learned inference is replaced by explicit representative outputs only when the claim is about the processor boundary. No resulting claim concerns policy competence or task success.

Within a reviewed, deterministic additive/affine fragment, write a controller boundary as

`u = A z + B q_origin + C q_current + b`.

Named permutations, affine normalizers, scalar masks and additive anchor conversions compose within this fragment. Rotations, clipping branches, opaque callbacks, model inference and arbitrary nonlinear transforms are excluded unless independently handled. A legacy checkpoint may require a counterintuitive conversion: the reference follows its grounded training/deployment convention, not an assumed universal representation.

Bounds and coordinate definitions come from the contract. Center/scale each probe coordinate to [-1,1]. Generate the center and positive/negative intervention on each of the n coordinates, giving 2n+1 queries. To make origin/current independently observable in a real queued adapter, a capture worker must reset for each trial, create a chunk against the origin, and evaluate the selected action later against the controlled current observation. These are software probes; they never command hardware. The compiler alone does not establish that a worker executed this schedule.

For paired responses, `(u_plus - u_minus)/2` estimates a coefficient in the normalized probe coordinates and cancels the offset and fixed nuisance terms. Unknown reference coefficients remain unknown; they are never filled from deployment observations. Known state coefficients can therefore reveal a state violation even when a normalization offset is unavailable. The aggregate remains UNKNOWN if any required independent evidence is missing, unless another grounded obligation already fails.

The closure relation `(u_plus + u_minus)/2 = u_center` checks the sampled affine assumption. Matching it at finitely many points does not establish that an arbitrary function is affine. Exact class membership must come from a reviewed operator fragment and its guards, or the broader equivalence scope remains UNKNOWN. A nonlinear function can agree on every finite probe, and an unknown or wrong physical reference can invalidate all interpretations.

## Conditional identification statement

In exact arithmetic, an affine map on n independent coordinates has n+1 coefficients per output. If the augmented observed design has rank n+1, agreement on a spanning design determines that affine map. This is elementary linear algebra, not a new theorem. It provides no guarantee outside the admitted fragment, probe bounds, coordinate/authority assumptions, numeric tolerances, or executed state schedule. Rank deficiency means the observations cannot uniquely identify the full map; matching such observations cannot certify unobserved directions.

The development implementation reports design rank per output, missing paired evidence, unknown reference/units/authority and sampled non-affinity. Canonical probe validation binds coordinates and query identifiers to the actual declared endpoint/center values, including reduced witness suites. Bounds whose center rounds to an endpoint are rejected. In finite precision, differences can lose accuracy. Comparisons now include a conservative binary64 output-resolution floor derived from ULPs of observed terms; unresolved comparisons return UNKNOWN, including zero effects hidden by a large offset. This is a diagnostic for output quantization, not an upper bound on all upstream arithmetic error, unknown producer dtypes, conditioning or sensor noise. Tolerances and admissible numerical profiles must be calibrated on development controls and frozen before evaluation. Further numerical analysis is required before publishing a tolerance-based identification guarantee.

## Witnesses and reset

A violated coefficient has a two-query behavioral witness; sampled non-affinity has a three-query witness. These are minimal for those local relations, not globally minimal causal trajectories. Each real query must retain the episode/chunk context required for replay. Reset obligations also require a preceding populated state; a fresh process followed by reset is insufficient evidence of state cleanup. The existing episode-prefix exporter is a starting point, not a proven generic minimizer.

## Comparisons and falsification

ExecSpec-style metadata checks are prior art for static normalization drift. Full independent stateful reference replay already exists in the prototype. The new method must be compared with both, as well as native checks, static declarations, round trips, stateless independent checks and equal-budget random/manual probes. If equal-budget manual tests cover the same relations with equal effort, report that result. If most public artifacts lack enough authority to ground the contract, report coverage failure rather than relabeling numeric agreement as physical correctness.
