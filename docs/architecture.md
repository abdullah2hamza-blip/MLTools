# Architecture and extension guide

The independent checker imports only the standard library. `io.py` bounds and validates JSON/files; `checker.py` verifies identities and replays the reference state machine; `report.py` aggregates findings; `reproduce.py` exports regression fixtures. `cli.py` exposes those operations without requiring the optional framework.

`lerobot.py` is a separate explicit execution boundary. It verifies the installed distribution's pinned Git revision and eight relevant source blobs before importing processor modules. Git blob verification canonicalizes Windows CRLF to upstream LF and reports that convention. This protects reproducibility, not against a compromised Python environment or malicious dependency installation. Source identities and the uv lock are complementary evidence.

Training executes actual LeRobot camera renaming and `RelativeActionsProcessorStep -> NormalizerProcessorStep`. Deployment executes `RenameObservationsProcessorStep`, caches the observation using the real relative step, then runs `UnnormalizerProcessorStep -> AbsoluteActionsProcessorStep -> PolicyActionToRobotActionProcessorStep`. A small queue models one-action-per-tick policy chunk consumption. This queue is harness code, not a learned policy implementation.

Before preprocessing each observation, the real relative step asks the queue how many old actions remain. Nonempty queues preserve the originating anchor. An empty queue permits re-anchoring before a new prediction. Reset clears both queue and cached state. The fixture interrupts a chunk and begins a second episode to test both transitions. Omitting queue binding is an explicit injected deployment fault.

The reference scalar arithmetic in `checker.py` never imports or calls LeRobot. Physical targets and mappings are specified separately from runtime settings. Tests assert known numbers (12 vs 13 degrees; absolute gripper 56 vs faulty 116), not just agreement between two implementations sharing configuration. Hashes cannot establish the truth of reference metadata; the integrator remains responsible for that authority.

## Compatibility

The core targets Python 3.13+ with no runtime dependencies. Live integration is intentionally pinned to LeRobot 0.6.2 at commit `5517a326eeb9b457cfe07d59b9bfdf291799c8d9`; a different install is an operational error requiring deliberate compatibility review. Windows Python 3.13.7 (tests) and 3.13.5 (fresh locked installation), with Torch 2.11.0+cpu, have been exercised. Committed integration evidence records the producing interpreter. Other environments require their own integration run; dependency metadata advertising support is not proof of execution.

LeRobot is optional because its base install includes Torch/TorchVision, NumPy, OpenCV and utilities even for these CPU-only processors. The base checker remains usable without them. Use the locked environment for reproducibility; initial setup and uv synchronization may use the network, while checker execution is local. Installing dependencies is explicit code installation; bundles themselves cannot download/import arbitrary adapters.

## Extend responsibly

For a new framework, first obtain a permitted immutable fixture, independent reference and correct control. Add an explicitly registered built-in capture adapter that emits v1 boundary traces. Verify source/artifact identities, record execution scope, and call only reviewed transforms. Keep heavy dependencies optional. A bundle must not be able to select an arbitrary import path or executable callback.

Add a new semantic check only with a defined mathematical convention and an independent oracle. Add an UNKNOWN path before adding PASS. Cover both valid checkpoint-specific conventions and known faults, including composed state when relevant. Frames and rotations require explicit conventions and group operations; stochastic inference requires a separately designed comparison protocol. Do not generalize scalar addition to either.

The regression exporter preserves episode prefixes and does not optimize an arbitrary controller's state machine. Extending it to new event types requires retaining enough context to reproduce the fault. Every live fault case should assert its intended boundary, so unrelated failures cannot satisfy validation.

## Development

```sh
uv sync --locked --extra lerobot
uv run --extra lerobot pytest -q
uv run ruff check src tests scripts examples/so101/build_cases.py
uv run ruff format --check src tests scripts examples/so101/build_cases.py
uv run --extra lerobot python scripts/validate_examples.py
uv build
```

Use a unique writable `--basetemp PATH` if a sandbox restricts the system temporary directory. Core-only runs intentionally skip live integration tests. Do not report those skips as evidence that the integration works. Regenerate calibration bundles with `python examples/so101/build_cases.py`; regenerate committed traces/reports with `scripts/validate_examples.py` after any bundle change. Never hand-rebind production traces to changed bundles.

The opt-in `Conformance validation` GitHub Actions workflow provides Linux/Windows core jobs and a Windows live integration job using pinned official actions. Hosted execution requires an explicit workflow dispatch; no hosted run is claimed by local validation evidence.
