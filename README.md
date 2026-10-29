# MLTools

The [Kaggle release workflow](releases/kaggle/README.md) builds a licensed,
privacy-reviewed offline replay dataset and CPU notebook from the frozen study.
Original code/results use [Apache-2.0](LICENSE); third-party terms remain in force.
The current local release candidate is [release-005](releases/kaggle/artifacts/release-005/).
Its [review record](releases/kaggle/validation/review-005-readiness.json) covers
local validation; the previous hosted Kaggle proof applies to release-004.

Check whether a robot policy adapter preserves the physical meaning of its training interface. Matching tensor shapes and successful encode/decode round trips can conceal swapped cameras, permuted joints, stale statistics, or a changing reference state inside an action chunk.

MLTools compares executed boundary traces with a separately supplied physical contract. It reports **PASS**, **FAIL**, or **UNKNOWN**, with the first observed divergence and an exportable regression fixture.

## Try it

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/). Run from this repository. The core has no runtime dependencies and can replay the committed evidence without importing Torch or LeRobot:

```sh
uv sync --locked
uv run mltools check examples/so101/good.json --trace examples/so101/good.trace.json
uv run mltools check examples/so101/moving_anchor.json --trace examples/so101/moving_anchor.trace.json
```

The first command passes. The second fails at event 2: the shoulder target should be **12 degrees**, but the executed adapter produces **13 degrees** because it re-anchors an in-flight chunk. Exit code 1 is expected for that example.

Execute the actual pinned LeRobot processors locally on CPU:

```sh
uv sync --locked --extra lerobot
uv run --extra lerobot mltools check examples/so101/good.json --execute lerobot
uv run --extra lerobot mltools check examples/so101/moving_anchor.json --execute lerobot --json report.json --counterexample regression
```

`regression` must be a new directory. It contains a reduced bundle, its artifacts and captured trace, and replay instructions. From that directory, `mltools check bundle.json --trace trace.json` reproduces the failure offline. Re-execute with the optional integration installed using `mltools check bundle.json --execute lerobot`.

`uv run` synchronizes the environment: retain `--extra lerobot` for live execution or it may remove the optional integration. Initial installation needs network access and installs LeRobot's full base dependencies. Subsequent checker execution performs no downloads or uploads; `uv run --offline --extra lerobot ...` also prevents uv from fetching dependencies.

## What is implemented

| Boundary | Evidence checked |
| --- | --- |
| Artifact identity | SHA-256 of explicitly pinned local configuration/statistics files |
| Cameras | Actual LeRobot training/deployment rename results against independently labeled input fixtures |
| Joints and units | Model-facing state order/content, training unit convention, named controller commands and declared units |
| Normalization | Real training normalization and deployment unnormalization against a separate scalar reference |
| Action chunks | Relative/absolute modes, absolute gripper, horizon, originating anchor, queue consumption and re-anchoring |
| Episodes | Cached anchor and queue cleared across reset, including an interrupted chunk |

The integration uses [LeRobot revision 5517a326](https://github.com/huggingface/lerobot/tree/5517a326eeb9b457cfe07d59b9bfdf291799c8d9), verifies eight relevant source files, and composes its real processor classes. The SO-101 calibration scenarios use fabricated observations, statistics and known policy outputs. They do **not** load trained weights, run a robot, or establish policy performance. The moving-anchor case injects the failure class described in [LeRobot issue 3312](https://github.com/huggingface/lerobot/issues/3312) by omitting queue binding; it does not claim that the pinned upstream implementation itself is broken.

Run the full example matrix and tests:

```sh
uv run --extra lerobot python scripts/validate_examples.py
uv run --extra lerobot pytest -q
uv run ruff check src tests scripts examples/so101/build_cases.py
uv build
```

The matrix requires the correct control to pass, eight injected faults to fail at their intended boundaries, and two insufficient-evidence cases to return UNKNOWN. [Recorded integration evidence](validation/integration.json) includes the environment, source identities and individual findings. Without the extra, integration tests skip; core tests still replay genuine recorded processor output.

## Use your own evidence

A second example uses the actual configuration and normalization artifacts of [a public SO-101 SmolVLA checkpoint](https://huggingface.co/yutofujita/smolvla-so101-move-cube/tree/0220b37dbcf1e0b2f6f5e21c64c75bf16fac6b69), pinned by revision and file digests:

```sh
uv run mltools check examples/published_so101/metadata.json --trace examples/published_so101/metadata.trace.json
uv run --extra lerobot mltools check examples/published_so101/metadata.json --execute lerobot
```

Normalization and declared front/wrist routing pass, while the overall result is **UNKNOWN** (exit 2): physical units/action modes, the exact training dataset revision, camera3's role and model preprocessing/inference are unverified. The `wrong_stats.json` variant fails its numeric normalization check using those same published artifacts. No weights are downloaded. [Source manifest](examples/published_so101/source_manifest.json), [results](validation/published_checkpoint.json), and [scope](docs/evidence.md#published-checkpoint-metadata).

```python
from mltools import check

report = check("bundle.json", trace="trace.json")  # Data only; no adapter execution.
print(report["status"], report["first_divergence"])
```

Keep the reference independent of the deployment adapter. Supply the authority for joint/camera labels, checkpoint conventions and statistics; a self-consistent but wrong reference can still pass. A trace is bound to the exact bundle bytes. Recorded traces are producer assertions, not cryptographic execution attestations.

Exit codes: **0 PASS**, **1 FAIL**, **2 UNKNOWN**, **3 operational/input error**. A CI gate should accept only 0. FAIL takes precedence over UNKNOWN, and missing boundaries never silently pass.

- [Bundle, trace and report formats](docs/formats.md)
- [Architecture, compatibility and extension guide](docs/architecture.md)
- [Evidence scope and competing workflows](docs/evidence.md)
- [Third-party provenance](docs/provenance.md)
- [Original opportunity research](research/2026-10-21-opportunities.md), [early synthetic probe](research/probes/policy_interface.py), and [limited baseline-module experiment](research/probes/baseline-results.json)

## Reproducible research artifact

The [publication project](research/publication/README.md) now includes a committed protocol, the complete prospective native acquisition, clean main reproduction, generated figures/tables and a [twelve-page manuscript with editable sources](research/publication/paper/README.md). The [original](research/publication/runs/prospective-001/README.md) executes three published metadata groups in four LeRobot/OpenPI cells; the [clean comparison](research/publication/runs/clean-reproduction-001/reproduction-comparison.json) reproduces semantic JSON/CSV byte for byte in fresh exact-profile environments.

The finite main study records 1,792 primary calls, 896 auxiliary calls and sixteen fresh witness calls. Full-reference numerical checks agree; the compiler shows no detection or witness-size advantage. Offset-authority erasure distinguishes pointwise from relational checks. All selected physical-authority views remain UNKNOWN. Published family labels describe processor metadata; no weights, learned inference, task performance or hardware safety are measured. Retrospective incidents and [Panda software semantics](docs/panda-software.md) remain separate strata.

[Reproduction instructions](research/publication/reproduction/README.md) preserve exact inputs, complete source/environment identities, attempt order, raw failures and reporting. The [probe API/CLI](docs/probe-cases.md) supports explicit execution, data-only replay and regression exports with an isolated [OpenPI CPU profile](integrations/openpi/README.md). Human scientific/authorship/licensing review and separate approval are required before public release or submission. This private artifact establishes no acceptance or generalization guarantee.

## Limits

PASS applies to the requested checks, pinned artifacts and finite fixtures. Missing identity, physical semantics or observations produce UNKNOWN; unsupported checks such as frame/rotation conversion also produce UNKNOWN when requested. The `mltools check` bundle format supports scalar additive joints and MEAN_STD action normalization. It does not verify arbitrary transforms, coordinate frames, dynamics, sensor calibration, image preprocessing, stochastic inference, latency, padding masks, or collision/safety constraints. Request unsupported checks explicitly instead of inferring their coverage from PASS. Development OpenPI capture records additional boundaries without making them required checks in the affine evaluator.

The tool is local and diagnostic. It never repairs deployment behavior, controls hardware, executes callbacks from a bundle, loads pickle/weights, or sends telemetry. The optional published-example preparation script can explicitly download its fixed, pinned metadata with `--download`; ordinary capture/check commands do not. No claim of global novelty, detection accuracy, customer demand or improved task success follows from these examples.
