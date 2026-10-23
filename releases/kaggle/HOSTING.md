# Verified private Kaggle artifacts

- Dataset: [ML Boundary Conformance: Saved Trace Replay](https://www.kaggle.com/datasets/abdullah2hamza/boundary-conformance-replay), version 1, private, ready, Apache-2.0.
- Notebook: [Boundary Conformance Replay Notebook](https://www.kaggle.com/code/abdullah2hamza/boundary-conformance-replay-notebook), version 2, private, completed with CPU and internet disabled.
- Reviewed local upload snapshot: `artifacts/release-004`, with exact byte hashes in `release-004-lock.json`.
- Hosted source, image, settings and version proof: `validation/kaggle-hosted-002/verification.json`.
- Download inventory and strict numerical output checks: `validation/kaggle-hosted-002-output-verification.json`.

The received notebook cell sources equal the reviewed sources. All five research
products match their predeclared hashes: analysis, metrics, witnesses, workflow
analysis and the 24-row conditions CSV. The report-bound output directory passes
the unchanged strict verifier. Raw downloads remain preserved separately with
their complete hash inventory; the CLI-generated log is explicitly recorded as
a sidecar. No downloaded report was edited or rehashed to force agreement.

Hosting required three infrastructure corrections:

1. Kaggle expands the uploaded ZIP. Bootstrap now accepts both the original ZIP
   and an expanded directory, pins the manifest hash, checks the exact member
   inventory and hashes, and imports source only after those checks.
2. Kaggle rejected the original notebook name because it collided with the
   dataset. Dataset and notebook now have distinct checked identities. The
   failed requests, read-only reconciliation evidence and staged inputs remain
   in ignored private attempt state.
3. Kaggle injects a runtime `__notebook__.ipynb` which it omits from downloadable
   output. Version 1 completed and matched all five numerical hashes, but its
   report included that unavailable file. Version 2 records the platform file's
   producer-reported hash separately and lists only produced research outputs
   in the downloadable closure. Version 1 is preserved as historical evidence,
   with no claim of full output verification for that report.

The dataset bytes, frozen sources, observations, statistical interpretation and
expected numerical products are unchanged. This is a hosted **saved-trace
replay**, not another native framework acquisition, policy evaluation, hardware
experiment or independent runtime attestation. The original single-use native
acquisition records were not retried or modified.

Validation: **59 tests and 22 subtests pass**, including distinct identifiers,
per-predecessor notebook version guards, credential redaction, archive/expanded
inputs, unknown sidecars and rehashed substitute results. Ruff lint and format
pass for the release sources. The local fresh-kernel simulation covers
both expanded inputs and a platform-managed notebook file. Hosted execution
records its actual container image digest. The configured CI job was not run.

The supplied token was used only for the authorized private Kaggle workflow,
through hidden input and process memory/environment. It was absent from command
arguments, notebooks, upload files, committed files and saved state/logs in the
credential-pattern scan. Authentication processes were closed after verification.
Private attempt state and raw downloads are ignored by Git. No public visibility
change or third-party notification was performed. Public release still requires
the owner's separate approval after reviewing the links.
