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

Hand-calculated tests cover zero variance, equal quantiles, out-of-range input, asymmetric bounds, nonfinite/invalid statistics, canonical Suite binding and erased authority. All eight designs pass an independent affine fixture and distinguish its separate slope fault. These are software-development checks, not prospective native measurements. Native acquisition, matched baseline/ablation adapters, fresh witnesses and the complete committed freeze remain unfinished. None of the selected prospective metadata revisions has been executed by these primitives.
