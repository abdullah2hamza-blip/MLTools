# Native ActionABI CPU library controls

This optional profile exercises the genuine CPU library at `Archerkattri/actionabi@5ad1d45a082860f5f16bb43f3c6bbb0758ec6c11`. It builds the complete stock CPU core and CLI, links a small research driver against `actionabi_core`, and runs the upstream CPU package/CLI tests. CUDA, Pinocchio, Python bindings and upstream experiment stacks are disabled. The main code, CLI11 and nlohmann JSON dependencies have pinned MIT license files and immutable source identities in `sources.json`.

The profile currently targets Windows x86-64. A portable official LLVM-MinGW archive is pinned by release, byte count and SHA-256; it lives in the ignored workspace and is not globally installed or redistributed. CMake 4.1.3 and Ninja 1.13.0 have a hash-locked tools environment. Building uses one compiler process and local dependency checkouts with CMake FetchContent disconnected. CMake's declared minimum-policy override supports older dependency build metadata without patching upstream sources.

## Setup, native execution and offline verification

Setup explicitly downloads source and the approximately 182 MiB compiler archive. This dependency installation is separate from benchmark metadata and experiment cost. Choose a new destination:

```powershell
uv venv --python 3.13 .venv-actionabi
uv pip sync --python .venv-actionabi/Scripts/python.exe --require-hashes integrations/actionabi/tools.lock
.venv-actionabi/Scripts/python.exe scripts/prepare_actionabi.py --destination .test-tmp/actionabi-setup-NEW
.venv-actionabi/Scripts/python.exe scripts/validate_actionabi.py --sources .test-tmp/actionabi-setup-NEW/sources --toolchain .test-tmp/actionabi-setup-NEW/llvm-mingw-20261020-ucrt-x86_64 --build .test-tmp/actionabi-build-NEW --output validation/actionabi-native-NEW
```

Both build and output directories must be new. No ML framework, dataset, trained weights, simulator or robot is needed. After setup, the native build uses the supplied local dependencies. The runner checks all 138 main-repository, 250 CLI11 and 1,090 JSON source files before/after execution. Exact upstream blob matches and explicit Windows CRLF-to-LF matches are distinguished; raw checked-inventory fingerprints are retained. These producer checks are not cryptographic execution attestation or an OS network sandbox.

Data-only verification needs only stdlib Python and the repository:

```powershell
.venv-smoke/Scripts/python.exe -O scripts/validate_actionabi.py --verify validation/actionabi-native-001
```

It verifies identities, independently authored fixtures, hand-calculable Huber scores, retained hypotheses, field status, converter evidence bindings, refusals and probe-selection controls. It does not rebuild or rerun the native algorithms. A fresh native run is the separate explicit command above.

The documented setup command was executed end to end with a new `.venv-actionabi-clean`, fresh source checkouts, a newly downloaded/verified compiler archive and a new build. `validation/actionabi-native-reproduction-001/` preserves that run. All 55 fixture/request/native output files match the first run byte for byte, and all 1,478 clean checkout files match their Git blobs exactly. Build logs/paths and raw source inventory fingerprints are recorded separately. This is one development workflow reproduced, not an additional independent benchmark unit or clean reproduction of unfinished main research results.

## Evidence and native semantics

The scorer receives two independently authored four-row synthetic episodes with equal two-dimensional state/action coordinates and positive time intervals. The fixture defines lag-zero delta response in named synthetic joint/radian coordinates. Correct and separately specified wrong sign, permutation, target, velocity, lag, scale and gripper hypotheses have genuine CPU scores. An independent direct Huber calculation checks the results. The first sign-flip control has train loss 1.75 and held-out loss 25/12; the intended mapping has zero loss. Hypotheses with lag have different usable residual counts.

Native equivalence controls cover an excited unique mapping, insufficient excitation, a structurally indistinguishable sign/gripper pair, explicitly unsupported space/frame authority and an intentionally misspecified grammar. Retention is relative to the supplied grammar's best held-out score. Native “identified” means retained hypotheses agree; it can include fields fixed by the grammar. It is not an absolute conformance verdict, and a wrong sole hypothesis can be retained and produce a converter. Distinguish empirically varied fields from independently fixed declarations. The CPU scorer does not itself identify spatial frames.

The deterministic development threshold is zero, not a calibrated research threshold. `calibrated_coverage` is a caller-supplied field; this profile passes zero and makes no coverage claim. Separate controls exercise a hand-calculated split-conformal threshold and native rejection when the requested finite-sample rank is unavailable.

The actual converter APIs emit for the identified fixture, refuse ambiguous or unsupported fields, serialize/reload using the actual native evidence report digest, and reject the wrong digest. Instantaneous conversion only applies permutation, sign, scale and last-coordinate inversion. It does not integrate delta/velocity actions, implement lag, transform spatial frames or execute a temporal adapter.

Native bounded-axis generation produces four synthetic pulses. The selector receives explicitly modeled candidate predictions and fixture permission flags. Both exact and greedy branches, alternative permitted probes, complete refusal of unavailable candidates and an inseparable pair are exercised. Caller prediction generation/counts are recorded separately. Predicted observations are not measured processor responses, and permission flags are not hardware safety evidence.

## Final comparison boundary

These are development applicability controls, not a final comparison or published-performance reproduction. Existing arbitrary origin/current interventions are not observed robot trajectories and cannot be relabeled to satisfy the scorer. Chunk-origin anchors, queues and reset are outside this native trajectory grammar. Their processor-level comparison is N/A; adding names to an unsupported-field set does not extend the method.

A final applicable comparison must freeze the hypothesis grammar, independent authority, calibration, evidence access, prediction-generation budget and response model before observing outcomes. Credit retained equivalence, ambiguity, converter refusal and separating-probe capabilities. Do not count a relative retained hypothesis as accepted physical conformance, a synthetic fixture as a checkpoint sample, or inapplicability as a miss.
