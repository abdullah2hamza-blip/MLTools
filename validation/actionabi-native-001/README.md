# Stock ActionABI CPU development evidence

This records genuine native library execution at `5ad1d45a082860f5f16bb43f3c6bbb0758ec6c11`. The complete stock CPU core/CLI was built with a separate CMake 4.1.3/Ninja 1.13.0 environment, pinned CLI11/JSON source checkouts and the official SHA-verified LLVM-MinGW `20261020` toolchain (Clang 23.1.4). No upstream source was patched. Optional CUDA, Pinocchio and Python bindings were disabled. The upstream installation smoke test and CLI contract test group both passed.

All 138 main-repository, 250 CLI11 and 1,090 JSON files were checked before/after execution. Main-repository bytes comprise 16 exact Git-blob matches and 122 explicit CRLF-to-LF matches; dependencies match exactly. Raw checked-inventory fingerprints are preserved. The custom driver links `actionabi_core` and calls actual stock library APIs; it does not reimplement the native scorer or selector.

| Independently authored control | Native retained set/status | Native converter |
| --- | --- | --- |
| Excited delta fixture, eight hypotheses | Intended mapping alone | EMIT |
| Constant state/zero action, identity versus permutation | Two mappings; permutation ambiguous | REFUSE |
| Equivalent last-sign and gripper-inversion mappings | Two mappings; sign/gripper ambiguous | REFUSE |
| Excited fixture, space/frame authority unsupported | Intended mapping; space/frame unsupported | REFUSE |
| Misspecified grammar containing only the wrong sign | Wrong sole mapping retained with positive loss | EMIT |

The last control demonstrates relative-grammar retention, not an absolute native conformance verdict or an MLTools win. Native Identified fields include prior-fixed values; only varied hypotheses support empirical discrimination. The CPU scorer does not use spatial frame/space semantics. `calibrated_coverage` is explicitly caller supplied at zero, with no empirical coverage claim. All records are synthetic, deterministic development controls. The native train/held-out episode split is not the project's held-out research partition.

Independent direct Huber calculations verify every scored hypothesis's train/held-out and per-component losses and usable residual count. The intended excited mapping has zero loss with 12 residuals; the first-sign reversal has train loss 1.75 and held-out loss 25/12. Lag-one hypotheses have eight residuals. Instantaneous converters bind the actual native score report digest, reload successfully and reject the wrong digest. They do not implement lag, action integration or frame transformation.

Native pulse generation yields four bounded signed axis pulses. Selector controls exercise the exact and greedy branches, a replacement when the first candidate is unavailable, all candidates unavailable and structural inseparability. Candidate predictions are authored model forecasts, not measurements. There are 116 caller prediction calculations across selector controls, recorded separately from 17 native driver invocations. No physical safety inference follows from fixture permission flags. A separate split-conformal control returns the hand-calculated threshold and refuses unavailable finite-sample rank.

`inputs/` preserves independently authored authority and three two-episode JSONL fixtures. `native/` preserves every complete request and exact native stdout/stderr. `build/` preserves commands, compiler/tool versions, configure/compile logs and upstream tests. `summary.json` binds all 68 experiment files, project/profile bytes, upstream inventories, executable/archive hashes and timing records. Single-run timings are not performance estimates; preparation/install costs are separate. No trained policy, checkpoint, processor, robot, simulator, GPU or network service was executed.

Review fixed a replay gap where altered hypotheses/permission flags could preserve expected selector outcomes while skipping the intended control. Complete selector, scorer and converter requests now bind to fixed definitions; tests include that rehashed counterexample and converter/evidence/score/scope/source tampering. Earlier command-quoting and sandbox-temp trials remain ignored under `.test-tmp`, outside this recorded run. Upstream temporary files were placed inside the writable workspace.

Data-only optimized verification:

```powershell
.venv-smoke/Scripts/python.exe -O scripts/validate_actionabi.py --verify validation/actionabi-native-001
```

This checks recorded evidence without rebuilding or executing the native library. `integrations/actionabi/README.md` documents setup and fresh acquisition. The trajectory scorer's delayed single-step response assumptions do not hold for arbitrary origin/current processor interventions; chunk-origin/queue/reset comparison is N/A. Final equal-evidence adapters/calibration, remaining baselines, eligible corpus, frozen evaluation, clean main-result reproduction and paper remain unfinished.

The full documented setup was independently repeated with a new tools environment, fresh source/portable-compiler installation and a new build. `validation/actionabi-native-reproduction-001/` records byte-identical reproduction of all 55 fixture, request and native stdout/stderr files. This is replication of these controls, not extra independent samples.
