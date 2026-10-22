# MLTools

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

## Publication research in progress

The active [publication project](research/publication/README.md) includes a [development API](docs/affine-development.md) for bounded affine probes, partial-reference checking, design-rank diagnostics and query witnesses. The versioned [`mltools probe` workflow](docs/probe-cases.md) adds explicit execution, data-only replay, provenance checks and regression export for LeRobot and an isolated [OpenPI CPU integration](integrations/openpi/README.md). It is separate from the existing `mltools check` bundle format.

[Recorded development evidence](research/publication/development/README.md) distinguishes synthetic controls, injected faults and a source-backed legacy composition reconstruction. [New CLI evidence](validation/probe-cases-001/summary.json) also uses actual published ACT statistics: numeric normalization passes while missing physical authority keeps the overall result UNKNOWN. These are development evidence, not a frozen benchmark or publication result. The literature audit identified close prior art; novelty and the proposed contribution remain under investigation. The full benchmark, final comparisons and paper are unfinished.

The optional [Panda software profile](integrations/panda/README.md) executes genuine controller operators and populated gripper reset in a renderer-free MuJoCo environment, with a [clean-environment reproduction](validation/panda-reproduction-verification.json). [Its contract and limits](docs/panda-software.md) distinguish supported translation from nonlinear rotation/gripper behavior, for which the affine evaluator retains UNKNOWN. The [selected finite study design](research/publication/study-design.json) still requires implementation and a committed protocol freeze before prospective metadata outcomes.

## Limits

PASS applies to the requested checks, pinned artifacts and finite fixtures. Missing identity, physical semantics or observations produce UNKNOWN; unsupported checks such as frame/rotation conversion also produce UNKNOWN when requested. The `mltools check` bundle format supports scalar additive joints and MEAN_STD action normalization. It does not verify arbitrary transforms, coordinate frames, dynamics, sensor calibration, image preprocessing, stochastic inference, latency, padding masks, or collision/safety constraints. Request unsupported checks explicitly instead of inferring their coverage from PASS. Development OpenPI capture records additional boundaries without making them required checks in the affine evaluator.

The tool is local and diagnostic. It never repairs deployment behavior, controls hardware, executes callbacks from a bundle, loads pickle/weights, or sends telemetry. The optional published-example preparation script can explicitly download its fixed, pinned metadata with `--download`; ordinary capture/check commands do not. No claim of global novelty, detection accuracy, customer demand or improved task success follows from these examples.
