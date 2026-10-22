# Action-only study primitives

The selected prospective metadata study needs a different execution schedule from the existing temporal query experiments. `scripts/action_designs.py` defines `mltools-action-normalization-design`, schema version 1: exactly one native action decoder invocation per query, no batched queries, state inputs, queue operations or policy inference. The format describes a required execution schedule, not evidence that a worker has executed it. The future native producer must record and verify actual invocation counts separately.

Each full design contains `2n+1` input-only queries. The canonical design retains the compiler's exact query IDs, coordinates and values; it can therefore bind to the actual core Suite rather than renaming arbitrary random observations. Manual axis, random axis and random dense designs have separate identities. Seeds 7, 23 and 101 produce eight equal-budget designs in total. Deserialization regenerates the declared geometry and rejects altered values, IDs, seeds, schedules, state ports and budgets. Old temporal formats are not reinterpreted or edited.

The adapter reuses the existing research oracle's exact geometry/rank calculations without constructing its temporal `Design`. Shared-oracle comparisons measure query geometry only; they are not independent end-to-end method comparisons. Canonical/manual geometry is intentionally identical here, so equivalent results would not establish an advantage. Repeated seeds and queries are not independent artifact samples.

`scripts/normalization_profiles.py` supplies three separately named independent numerical references:

| Profile | Forward denominator | Decoder |
| --- | --- | --- |
| `lerobot_mean_std_f64_v1` | `std + 1e-8` | `z * std + mean` |
| `lerobot_quantiles_f64_v1` | Quantile width; replace exactly zero by `1e-8` | `(z + 1) * denominator / 2 + q01` |
| `openpi_quantiles_f64_v1` | Quantile width plus `1e-6` for every coordinate | `(z + 1) / 2 * denominator + q01` |

These describe the inspected pinned current-runtime operations in explicit binary64 research experiments. They do not infer a publisher's training source, production dtype, physical units or checkpoint/controller mapping. They preserve the different operation order and zero-width behavior; none clips out-of-range inputs. The existing paper-equation certificate and earlier extension profiles remain unchanged. No new certificate decision rule is introduced.

The independent reference compiles declared action-index alignment and bounds into normalized-coordinate coefficients before observations. It retains absent authority/units rather than filling them from outputs. Source-grounded affinity is restricted to the selected fixed-statistics numerical formulas. Floating-point allowances and native parity still require development validation and protocol freeze; this implementation does not prove global numerical error bounds.

Hand-calculated tests cover zero variance, equal quantiles, out-of-range input, asymmetric bounds, nonfinite/invalid statistics, canonical Suite binding and erased authority. All eight designs pass an independent affine fixture and distinguish its separate slope fault. These are software-development checks, not prospective native measurements. None of the selected prospective metadata revisions has been executed by these primitives.

## Native development runner

`scripts/run_action_study.py` now drives actual isolated native acquisition and deterministic data-only replay. Its current worker accepts exactly the fixed development input inventory in `action_study_common.py`: authored arithmetic fixtures and the already inspected/executed ACT move-cube artifact. It rejects prospective cell substitution. This development gate is separate from the complete protocol/commit gate still required for the main study.

```powershell
.venv-smoke/Scripts/python.exe scripts/run_action_study.py --lerobot-python .venv/Scripts/python.exe --openpi-python .venv-openpi/Scripts/python.exe --output validation/action-study-development-NEW
.venv-smoke/Scripts/python.exe -O scripts/run_action_study.py --verify validation/action-study-development-001
```

Use a new output directory. The parent needs the installed MLTools core. LeRobot uses the existing full pinned base environment and OpenPI the documented separately locked CPU subset; see `integrations/openpi/README.md` and `uv.lock`. Workers run sequentially with explicit interpreters, Python `-I`, trusted project source paths, CPU/thread settings, offline flags, a network-event audit hook, a 180-second subprocess timeout and a post-measurement 2 GiB peak-RSS check. These are not OS memory/network limits. Native source identities are checked before and after execution. The published development statistics additionally pass genuine Safetensors decoding against the independently decoded source bytes. No weights, learned policy, task rollout or hardware are loaded.

Primary acquisition calls the candidate decoder once per query. A separately measured baseline phase makes four auxiliary native calls per canonical query: candidate encoding of the observed candidate output, intended decoding, intended encoding of that actual result, then candidate decoding. Self-round-trip consistency is retained as WITHIN_TOLERANCE or DIFFERS, with no automatic conformance interpretation. Intended-to-candidate round trips use independent intended metadata, rather than silently pairing the same wrong metadata on both sides. Actual boundary arrays and auxiliary call counts are available in the same evidence packet as the primary observations.

The compiled, independent pointwise and strongest relational comparators share the canonical observations, independently supplied reference and component tolerances. The pointwise workflow is stateless in this action-only domain; the older equal-origin/current restriction is inapplicable. Static comparison receives the same intended and deployment artifacts. API execution success is labeled EXECUTED_WITHOUT_ERROR, not physical conformance. Shared-oracle geometry comparisons remain distinct from end-to-end comparator comparisons.

The prior certificate uses its exact existing mean/std or OpenPI extension where applicable. For LeRobot QUANTILES, an explicitly labeled coordinate adapter supplies `z_paper=(z_native+1)/2`, all coordinate masks true and an effective `1e-8` zero-width interval. Its finite decoded arrays are checked against actual native outputs within the recorded binary64 allowance. This is a framework coordinate extension of the paper equations, with possible rounding at effective endpoints; it is not the paper's native decoder, author implementation or a global numerical bound. The metric selects only the intervention coordinate with its declared raw-index unit. No gripper axis or shared physical Euclidean unit is invented. No certificate detection/safety threshold is fitted.

Offset-reference and full-reference erasures are derived numerical analyses, with no replacements fitted from observed outputs. Provenance erasure retains full binding UNKNOWN, witness-disable changes diagnostic size, and state ablation is N/A for action-only operations. Missing reference authority/units reuse the actual control execution and retain UNKNOWN. These derived views must not be counted as new native samples or as fair comparisons with an unerased reference.

Fresh witness workers execute the union of the methods' selected query IDs from their original first failed relations. Each method is evaluated only on its selected, actually re-executed observations, and that same original relation must still fail. Shared physical execution cost and logical method-view cost are both retained. Reduction-disabled size is descriptive; all full-design acquisition costs remain charged. Witness selection does not establish minimality or an advantage over pointwise/relational diagnostics.

`validation/action-study-development-001/` records the reviewed workflow: 896 primary calls, 448 auxiliary calls and 16 fresh calls. All three numerical comparators agree on its controls/faults/erasures. All geometries agree as well. The static and independent cross-round-trip workflows distinguish the deliberate metadata changes. Native self-round trips do not supply independent conformance, and zero variance can make even the correct control lossy. The compiler's two-call slope witnesses are larger than the independent oracles' one-call witnesses. These are development findings and negative comparative evidence, not final benchmark rates.

Final prospective inputs/native gate, applicable external stock-tool adapters, complete frozen analysis, committed protocol identity and main-result reproduction remain unfinished. The earlier primitive-verification JSON preserves its historical pre-native status; this new evidence establishes development parity without rewriting that record.
