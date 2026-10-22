# Historical OpenPI data-factory environment

This separate CPU subset installs the complete unmodified OpenPI and client Python packages at the actual first parent and merge revision of PR351. It is not the original full training environment. It uses Python 3.11.13 and the previously tested current CPU processor lock, with additional import versions from the historical upstream lock. JAX 0.5.3, Orbax 0.11.13 and Transformers 4.53.2 therefore differ from the historical declared versions. The audit target is Windows x64; upstream's general supported platform is not changed by this experiment.

From the repository root, with `uv` available:

```powershell
uv venv .venv-openpi-history-before --python 3.11.13
uv venv .venv-openpi-history-after --python 3.11.13
uv pip install --python .venv-openpi-history-before/Scripts/python.exe --torch-backend cpu --require-hashes -r integrations/openpi-history/requirements.lock
uv pip install --python .venv-openpi-history-after/Scripts/python.exe --torch-backend cpu --require-hashes -r integrations/openpi-history/requirements.lock
uv pip install --python .venv-openpi-history-before/Scripts/python.exe --no-deps --build-constraints integrations/openpi/build.lock -r integrations/openpi-history/before.txt
uv pip install --python .venv-openpi-history-after/Scripts/python.exe --no-deps --build-constraints integrations/openpi/build.lock -r integrations/openpi-history/after.txt
uv run --no-sync python scripts/reproduce_openpi_history.py --before-python .venv-openpi-history-before/Scripts/python.exe --after-python .venv-openpi-history-after/Scripts/python.exe --output .test-tmp/openpi-history-NEW
```

The output directory must be new. The runner executes one worker at a time with a 180-second timeout per capture. Setup fetches pinned source and dependencies. Capture requests no weights, tokenizer assets, dataset download, model construction, inference, simulator or robot. The worker rejects Python socket connection/DNS audit events; this is an execution tripwire, not an OS network sandbox. CPU/offline/thread environment variables and observed peak working set are recorded; no hard memory limit is claimed.

The genuine named `pi0_libero` configuration's `LeRobotLiberoDataConfig.create` executes unchanged. Exactly one method is temporarily replaced in memory: unused `ModelTransformFactory.__call__` returns a genuine empty `transforms.Group()`. It is restored in `finally`. Without this explicit isolation, the factory would construct a tokenizer and may fetch its asset. Repacking, `LiberoInputs`, padding, legacy delta/absolute transforms and `LiberoOutputs` are genuine historical code. The dataset identifier is retained. An empty relative local assets path takes the historical downloader's native local-file branch; an absolute Windows drive path is misclassified as a URL scheme by that code. No statistics are loaded or normalization claimed.

`sources.json` pins the complete installed Python module inventory: 42 OpenPI and 13 client files per revision. The worker checks all those Git blobs and both installation revisions before imports and after execution. It binds read-once request/fixture/ledger bytes, snapshots worker identity, and rejects source/input changes during capture. The runner also checks its complete recorded source set before and after each dispatch. These checks assume a trusted local producer and do not cryptographically attest execution or the complete dependency chain.

`fixture.json` supplies independent numerical targets for the pre-removal configured legacy convention. It does not establish the training history of released weights. Native LIBERO commands already represent controller deltas; the historical extra conversion is a separate compatibility layer. Do not interpret additive conversion of configuration coordinates as physical frame/rotation semantics.

The runner checks both rows of training targets and representative deployment outputs directly, records native round trips, applies the compiled evaluator and independent pointwise/strong relational oracles to the same 31 measurements per revision, then executes a fresh reduced witness in another worker. This remains one retrospective incident, not 63 independent samples. The shared versioned public `mltools probe` producer profile does not yet support these historical revisions; this is an explicit research runner.

Data-only replay needs neither historical environment nor an ML framework:

```powershell
uv run --no-sync python -O scripts/reproduce_openpi_history.py --verify validation/openpi-history-001
```

The verifier checks source/data identities, known raw observations, actual transform inventories, saved reports and fresh-witness consistency. Saved measurements remain trusted-producer assertions. [Authority and scope](../../research/publication/openpi-history.md) explain the independent earlier training configuration and distinguish this capture from the older current-class reconstruction.
