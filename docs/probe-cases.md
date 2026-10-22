# Versioned affine research workflow

This development workflow wraps the numerical method in `affine-development.md` with artifact/source checks, explicit execution, offline replay and portable regression exports. It supports reviewed LeRobot and OpenPI processor compositions. It does not load model weights or saved Python registries.

```powershell
mltools probe examples/probes/lerobot/good.json --execute lerobot
mltools probe examples/probes/lerobot/moving_anchor.json --execute lerobot --save-trace trace.json --json report.json --counterexample regression-NEW
mltools probe examples/probes/openpi/good.json --execute openpi --python .venv-openpi/Scripts/python.exe
mltools probe examples/probes/act/metadata.json --execute lerobot
```

Run with the appropriate environments from `README.md` and `integrations/openpi/README.md`. The first and third cases pass; the second fails; the published ACT metadata case returns UNKNOWN overall. These are development outcomes, not final experimental results. The ACT numerical obligations pass, while its explicitly requested physical semantics remain unsupported. Exit codes are 0=PASS, 1=FAIL, 2=UNKNOWN, 3=input/execution error.

`mltools probe CASE --trace TRACE` performs data-only replay without importing either ML framework. Without execution or a trace it reports UNKNOWN. `--save-trace` requires explicit execution, and `--python` is only accepted for OpenPI. Case inputs, statistics, output reports and regression destinations cannot overwrite one another.

Verify the saved development matrix and exported/fresh witnesses with `python scripts/verify_probe_evidence.py validation/probe-cases-002`. This data-only command also works from the wheel-installed environment without Torch, LeRobot or OpenPI. It checks the recorded identities and replay consistency; it does not authenticate execution by an untrusted producer.

## Case format

Every case is bounded JSON with integer `schema_version: 1`, `format: "mltools-affine-case"`, a canonical compiled `suite`, an independently supplied `reference`, an `artifacts` list and an `adapter`. `examples/probes/` contains complete instances. Descriptive fields do not add verification coverage.

Artifacts have unique `id`, contained relative POSIX `path`, and expected SHA-256 or `null`. Missing files or expected hashes remain UNKNOWN. Mismatching bytes fail; observed files never supply their own expected identity. The adapter declares `framework`, `stats_artifact` and a fixed reviewed `configuration`. JSON fields cannot select modules, callbacks or shell commands.

OpenPI requires seven action axes and eight origin/current axes named by index, and seven output labels. Its configuration declares `model_type`, `use_quantiles`, seven-element boolean `mask`, and `anchor`. The current-anchor mode is an explicit deployment intervention. LeRobot requires equal action/origin/current dimensions, action/motor names, `relative`, `exclude_joints` and `bind_queue`. Its output labels match named motor `.pos` keys. Those keys can also label uninterpreted checkpoint coordinates; their spelling does not establish physical position semantics.

LeRobot statistics may be bounded mean/std JSON or a tiny Safetensors statistics file. Capture reads a bounded byte snapshot once and records its exact SHA-256. Native `safetensors.torch.load` decodes that snapshot, validating finite action mean/std dimensions and positive standard deviations; JSON is parsed from the same hashed bytes. A file change during execution rejects the capture. The ACT reference is separately decoded from the published preprocessor bytes using the standard library. This avoids using the same decoder for expected and observed values; shared source authority can still be wrong. The published ACT case tests normalization only, with two representative actions. It does not execute ACT inference, its native 100-action queue, images, or its training data. Earlier immutable development captures record the previous native file loader in their producer metadata.

All four checks—binding, producer identity, artifacts and numerical obligations—are required. Optional `checks` can explicitly request additional boundaries. Unsupported names yield UNKNOWN and cannot be ignored to obtain PASS. A known failure still takes precedence over missing evidence. Missing physical authority is not inferred from tensor shapes, names or ranges.

## Evidence and export

The trace binds exact case bytes, canonical suite, framework, actual configuration, statistics and reviewed source blobs. Input changes during capture are rejected. Source metadata from recorded producers is trusted; hashes do not prove that a remote producer executed that code. The report includes per-check statuses, source/artifact findings, the complete numerical report and fingerprints. Numerical evidence bound to another suite is not reinterpreted as this suite's observations.

`--counterexample NEW_DIRECTORY` preserves the original case bytes, its contained artifacts and original trace. `export.json` identifies the offline replay paths. A separate reduced case retains the violated local relation's query context and requires fresh capture. No reduced execution trace is fabricated. After export, run the offline command in `README.txt`, or execute the witness path in `export.json` with the same framework environment. Witnesses are local behavioral relations, not globally minimal trajectories.

The original `mltools check` format retains its separately documented training, camera, joint/unit, action-chunk and episode-reset checks. The probe workflow adds generated numerical experiments and supports both frameworks; it does not imply that all original semantic boundaries are implemented in every framework.
