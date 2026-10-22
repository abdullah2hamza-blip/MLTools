# OpenPI CPU processor integration

This is a reviewed **processor subset**, using actual OpenPI and openpi-client packages at Git revision `215abfb217dbac7d5f1273282331b9b1866c0479`. It does not provide the full upstream training/inference dependency environment. The audit and execution target is Windows x64, Python 3.11.13, CPU. Upstream's documented supported platform is Ubuntu 22.04; these checks do not establish general Windows support. The core MLTools environment remains Python 3.13.

From the repository root, using the installed `uv` executable:

```powershell
uv venv .venv-openpi --python 3.11.13
uv pip install --python .venv-openpi/Scripts/python.exe --torch-backend cpu --require-hashes -r integrations/openpi/requirements-cpu.lock
uv pip install --python .venv-openpi/Scripts/python.exe --no-deps --build-constraints integrations/openpi/build.lock -r integrations/openpi/frameworks.txt
uv sync --locked --group dev
$env:MLTOOLS_OPENPI_PYTHON = (Resolve-Path .venv-openpi/Scripts/python.exe).Path
uv run --no-sync pytest tests/test_openpi.py -q
uv run --no-sync python scripts/validate_openpi_development.py --python .venv-openpi/Scripts/python.exe --output .test-tmp/openpi-evidence-NEW
```

The output directory must be new. These commands fetch pinned packages and source, then execute genuine transforms. They download no model weights, construct no models/tokenizers, contact no robot, and request no policy inference. Cold imports can take much longer than the tiny transforms; each capture has a 180-second timeout. Imports pull in model-class definitions, JAX and PyTorch even though no model is constructed. Worker environments set CPU, offline Hugging Face and single-thread BLAS options. Those environment variables are not an OS network sandbox. Peak RSS is measured and checked after execution, not enforced as a hard memory ceiling.

`requirements.in` records audited direct import pins, mostly selected from upstream's lock. `requirements-cpu.in` adds the observed CPU PyTorch dependency set. `requirements-cpu.lock` pins and hashes the complete selected subset; installation **also requires `--torch-backend cpu`**. `build.lock` constrains the upstream build tools. `frameworks.txt` pins both genuine Git packages. `--no-deps` deliberately avoids the upstream CUDA/training dependency graph. A successful subset capture does not imply that upstream's complete declared dependency set is installed or passes `pip check`.

The worker verifies both direct-URL Git revisions, eight reviewed OpenPI module blobs and the client image module before ML imports. It checks canonical LF Git source identities. This verifies selected boundary/import sources; it is not attestation of the complete dependency chain. Recorded traces are assertions by a trusted execution producer, not cryptographic proof that a remote producer executed the code.

The API is explicit:

```python
from mltools.openpi import capture
from mltools.obligations import evaluate

trace = capture(suite, stats_json, python=worker_python)
report = evaluate(suite, independently_authored_reference, trace)
```

The suite has action axes `0..6`, origin/current state axes `0..7`, and seven output labels. The compiler supports named units and independent references, but these labels do not authenticate physical authority. The worker captures `LiberoInputs`, `DeltaActions`, `Normalize`, `PadStatesAndActions`, `Unnormalize`, `AbsoluteActions` and `LiberoOutputs`. Each query starts fresh, creates two representative outputs, queues a converted chunk, and observes its second action under the controlled current observation. OpenPI's processor objects are pure here; the queue is the reviewed deployment composition. `anchor="current"` and alternative masks are explicitly configured intervention conditions, not claims of an upstream defect.

Camera routing and masks are recorded using distinct synthetic images. Training-side arithmetic reuses the numerical action probes as synthetic targets; the trace field `training_targets` does not assert independent physical training data or checkpoint training parity. These are inspectable evidence, but the affine evaluator checks only the declared numerical output obligations. It does not evaluate image identity, general episode reset, inference, simulator control, geometric rotations, clipping or task success. The original LeRobot checker continues to cover its separately documented lifecycle/camera obligations. The versioned `mltools probe` CLI now wraps this worker with case/artifact/source checks and regression export; see `docs/probe-cases.md`. A final research benchmark remains unfinished.

The development runner persists synthetic controls, a moving-anchor injection and a source-backed reconstruction of PR351's legacy wiring change. Its legacy statistics and representative model outputs are synthetic. See `research/publication/openpi-history.md` for the exact historical evidence and limits. These artifacts do not constitute a final benchmark or held-out experiment.
