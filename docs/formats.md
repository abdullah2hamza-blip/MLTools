# Format reference: version 1

Every document is a JSON object with integer `schema_version: 1`. The executable examples in `examples/so101/` are the canonical complete examples. Unknown requested check names return UNKNOWN; unknown version numbers are operational errors.

## Bundle

A bundle contains `contract`, `adapter`, `artifacts` and `scenario`. The reference contract and runtime adapter are intentionally separate. Do not generate the reference by copying whatever deployment currently does.

| Contract field | Meaning |
| --- | --- |
| `embodiment` | Human-readable robot/configuration identity |
| `authority.source`, `authority.description` | Source and rationale for the independent reference; no automatic retrieval/authentication |
| `checks` | Nonempty unique list: `artifacts`, `cameras`, `joint_order`, `normalization`, `actions`, `reset`, or explicitly requested unsupported checks |
| `joints` | Ordered objects with `name`, `unit`, `mode`; `mode` is `relative` or `absolute`; absent unit/mode makes affected semantics UNKNOWN |
| `cameras` | Objects with physical `role`, raw `source` key and expected model `target` key; missing role makes camera evidence UNKNOWN |
| `normalization` | `{ "kind": "mean_std", "artifact": "training_stats" }`; other kinds are unsupported |
| `horizon` | Expected predicted chunk length, 1–64; one command is consumed per tick |

Supported scalar units are `degree`, `radian`, `meter`, `range_0_100`. These are labels to compare, not automatic unit conversions. Relative reconstruction is scalar addition; it must not be applied to quaternion/pose/rotation representations. Frame conventions require a separate, currently unsupported check.

The LeRobot `adapter` declares `state_order`, `action_names`, `motor_names`, `camera_map`, `relative`, `exclude_joints`, `bind_queue`, `reset`, `stats_artifact`, and `units` (joint-name to unit). Its configurable behavior is fixed built-in wiring; fields cannot name Python classes, callbacks or shell commands. Null/missing output units remain UNKNOWN.

Each artifact is `{ "id": "...", "path": "artifacts/file.json", "sha256": "64 lowercase hex digits" }`. Paths use relative POSIX syntax, resolve inside the bundle directory, and cannot point outside through symlinks. All listed artifacts are required evidence. A null/missing digest or missing file yields UNKNOWN; a changed digest yields FAIL. Hashes identify bytes, not correctness, authorship or authenticity. Line endings matter; `.gitattributes` and the writer preserve LF for reproducibility.

The live integration requires `checkpoint_config` plus the declared training and deployment statistics artifacts. Statistics are bounded JSON containing action `mean` and positive `std` vectors matching the joint count. It never calls a saved pipeline's arbitrary registry or reads a checkpoint pickle. The calibration `checkpoint_config` explicitly declares state/action order, relative conversion, excluded joints, camera mapping, horizon and `use_degrees`; it is not represented as a trained checkpoint.

Scenario events are:

```json
{ "kind": "reset" }
```

```json
{
  "kind": "tick",
  "state": { "shoulder_pan": 10, "shoulder_lift": 20, "elbow_flex": 30,
             "wrist_flex": 40, "wrist_roll": 50, "gripper": 60 },
  "cameras": { "observation.images.front": [[11,12],[13,14]],
               "observation.images.wrist": [[21,22],[23,24]] },
  "prediction": [[0.5,1,1.5,2,2.5,0.3], [1,1.5,2,2.5,3,0.4], [1.5,2,2.5,3,3.5,0.5]],
  "training_targets": [[11,22,33,44,55,56], [12,23,34,45,56,58], [13,24,35,46,57,60]]
}
```

`prediction` is a known representative normalized model-output chunk, not model inference. `training_targets` supplies independently specified physical training actions for exercising the real training transform; omit it and training normalization evidence is UNKNOWN. Both are omitted while consuming a previous chunk. Replacing an unconsumed chunk is invalid. Reset interrupts and discards it. Camera fixtures must distinguish roles; indistinguishable camera values cannot demonstrate routing and yield UNKNOWN.

The reference uses the observation at chunk creation, not subsequent observations: `target[k,j] = normalized[k,j] * std[j] + mean[j] + anchor[j]` on relative dimensions; absolute dimensions omit the anchor. Training uses `(target - anchor - mean)/(std + 1e-8)` on relative dimensions. LeRobot's inverse deliberately omits epsilon; this asymmetry is documented rather than used as a roundtrip oracle.

An explicitly declared `training_action_transform: "identity"` tests numerical normalization of raw training vectors without asserting their physical action mode. The published example uses this because the saved processor has no relative-action step. Physical commands remain UNKNOWN and `adapter.emit_commands: false` suppresses command construction. This does not establish whether a dataset's raw actions themselves encode deltas, absolute positions or another convention.

## Captured trace

A trace contains `schema_version`, `bundle_sha256`, `execution`, and one `events` entry per bundle event. Any change to the bundle requires recapture. `execution` identifies the producer, source revision/blobs, dtype/device and scope. Missing scope is UNKNOWN. It is a producer assertion; replay cannot verify that the producer actually ran that code.

A tick records `cameras`, `training_cameras`, `state_names`, `state_values`, `training_state_names`, `training_state_values`, `queue_remaining` and, when predicting, `prediction_horizon`, `training_horizon`, `training_units`, `training_normalized`. State values are captured at the ordered raw-state boundary, before any observation-state normalization. A consumed action records `denormalized`, named `.pos` `commands`, and declared `units`. Reset records `anchor_cleared` and `queue_remaining`. Missing evidence is UNKNOWN. Structurally invalid input is an operational error.

## Report

Reports contain overall `status`, per-check `checks`, `findings`, `first_divergence`, `counterexample`, `execution`, `bundle_sha256`, `tolerances`, and `limitations`. Each finding names its check/boundary, optional event index, expected/actual evidence, message and status. Numeric comparisons use absolute and relative tolerance `1e-7`; categorical names/units must match exactly.

Overall FAIL outranks UNKNOWN; all applicable required evidence must pass for PASS. Artifact identity is always required even if omitted from `checks`. No exercised fixtures means UNKNOWN. `first_divergence` is the first observed failing boundary, not a proven root cause. Artifact mismatches precede runtime findings.

Counterexamples retain one event for input mismatches or the required episode prefix for stateful mismatches. A failing reset retains the preceding episode so stale state remains reproducible. `--counterexample NEW_DIRECTORY` exports that context and narrows requested semantic checks to the failing check, preserving artifact checks. It is a useful regression, not a globally minimal trajectory.

Input limits: 2 MiB per JSON/artifact, 100,000 JSON nodes, depth 40, 32 artifacts/joints, 16 cameras, 500 events, and 64 actions per chunk. Duplicate JSON keys, non-finite or out-of-range numeric values, escaping paths and unsafe serialization formats are rejected. Weight files are outside this metadata/fixture format. Unknown extra descriptive keys do not add coverage; use `checks` to request it.
