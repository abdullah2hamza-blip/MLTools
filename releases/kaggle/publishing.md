# Private Kaggle publication steps

`publish.py` is standard-library-only infrastructure. Its default operation is a
read-only plan. It does not inspect credentials, read environment variables,
import Kaggle, discover a CLI, access the network, or create attempt state during
planning. Tests use a stub runner; no Kaggle operation has been performed by them.

The later execution path requires the official `kaggle` distribution **2.2.5** in
the publisher's Python environment. No installation happens automatically. The
CLI source/documentation revision is
[`8b88a6aeffed5acae0b1e43137d946afd5c43351`](https://github.com/Kaggle/kaggle-cli/tree/8b88a6aeffed5acae0b1e43137d946afd5c43351).
The inspected command references are
[`docs/datasets.md`](https://github.com/Kaggle/kaggle-cli/blob/8b88a6aeffed5acae0b1e43137d946afd5c43351/docs/datasets.md)
(blob `4a48befa67da6dd598effac300c95b2b2e09038c`) and
[`docs/kernels.md`](https://github.com/Kaggle/kaggle-cli/blob/8b88a6aeffed5acae0b1e43137d946afd5c43351/docs/kernels.md)
(blob `7a5c3ded416c1d306a22f38c00258de392f1373b`). Installed version checking
does not constitute a source or credential attestation.

## Build and configuration contract

The build root contains exactly two directories:

```text
dataset/
  dataset-metadata.json
  payload.zip
  conditions.csv
  README.md
  release-manifest.json
  LICENSE
  THIRD_PARTY_NOTICES.md
notebook/
  kernel-metadata.json
  replay.ipynb
```

The separate configuration requires these fields:

```json
{
  "schema_version": 1,
  "owner": "abdullah2hamza",
  "dataset_slug": "boundary-conformance-replay",
  "notebook_slug": "boundary-conformance-replay",
  "license": "apache-2.0",
  "license_approved": true,
  "dataset_title": "ML Boundary Conformance: Saved Trace Replay",
  "notebook_title": "Boundary Conformance Replay"
}
```

The two title fields are optional; if supplied they must equal the shown fixed
values. All other unrecognized configuration keys are refused.

`license: null` or `license_approved: false` permits a blocked plan but prevents
external operations. A non-null license must match the dataset metadata exactly.
License approval is a human decision; matching configuration strings does not
establish ownership or satisfy third-party obligations by itself.

`release-manifest.json` has integer `schema_version: 1`, a stable `release_id`,
`payload_sha256`, and `files`, a mapping of every other dataset filename to its
lowercase SHA256. Its `expected_replay_outputs` mapping contains exactly
`conditions.csv`, `analysis.json`, `metrics.json`, `witnesses.json` and
`workflow-analysis.json`, with their expected exported byte hashes. The conditions
hash must equal the top-level dataset CSV hash. Additional manifest provenance
fields may describe export transformations; the publisher does not rewrite them.

The notebook must be clean (no saved cell output), pin the exact payload SHA256
in its source, attach only `abdullah2hamza/boundary-conformance-replay`, and remain
private with CPU/offline settings. Numeric-ID overrides, extra input sources,
accelerator settings and unknown kernel metadata fields are refused. Archive
checks reject traversal, links, duplicate names, encrypted members, runtime/model
binaries and known local-path/credential patterns. These are bounded inspection
guards, not a Python sandbox or an exhaustive secret/license audit. The notebook
and payload source still require human review. Public upstream publisher
`/home/...` values remain in byte-exact metadata; local Windows user paths and
the known researcher's Unix/Mac home path are rejected. The only allowed notebook
environment reads are `os.environ.get("MLTOOLS_INPUT", "/kaggle/input")` and
`os.environ.get("MLTOOLS_OUTPUT", "/kaggle/working")`, for path isolation in local
smoke tests. Other environment access remains refused.

## Plan now; execute only after explicit authorization

Replace `BUILD_ROOT` and `STATE_DIR` below with actual paths. `STATE_DIR` must be
outside the upload root and must be retained privately across all attempts. Never
upload it. The publisher writes no metadata, source or evidence files in the
build root. Private staged snapshots are made only after explicit execution.
`releases/kaggle/private-state` is ignored by Git and is a suitable local state
location. Retain a private backup; deleting state loses duplicate-attempt guards.

```sh
python releases/kaggle/publish.py --root BUILD_ROOT --config releases/kaggle/config.json
```

Every remote step defaults to a plan as well. Only adding both `--execute` and an
explicit `--authorization` invokes the CLI. Do not add those flags until the
human has authorized the particular operation. The actual child uses the same
interpreter's pinned CLI; it may then use credentials configured by that human.
This infrastructure never reads or configures credentials itself.

Perform and inspect these steps separately:

1. `--step dataset-create`: private creation, `--keep-tabular --dir-mode skip`.
   The flat allowlist prevents skipped nested payload files. No public flag is
   generated. The CLI exit only records `SUBMITTED`, not dataset readiness.
2. `--step dataset-status`: inspect the native result before proceeding. Status
   output does not automatically authorize another step or resolve a mutation.
3. `--step notebook-push`: upload the validated private notebook. The official
   CLI also starts its run. The notebook performs saved-trace replay only.
4. `--step notebook-status --notebook-version N`: inspect a specific positive
   version, not an unbound latest run.
5. `--step output-pull --notebook-version N --output NEW_OUTPUT_DIR`: retrieve
   outputs of that exact notebook version into a new directory. No overwrite
   flag is used.
6. `--step output-verify --output NEW_OUTPUT_DIR`: a local read-only verification;
   do not supply `--execute`. See the report contract below.

For a remote step, the explicit execution form is:

```sh
python releases/kaggle/publish.py --root BUILD_ROOT --config releases/kaggle/config.json --step dataset-create --state STATE_DIR --execute --authorization "Human approved this private dataset upload"
```

This is documentation for a later authorized action, not a request to execute it.

## Attempts, unknown outcomes and reconciliation

Before a mutation can reach the CLI, the publisher exclusively creates
`STATE_DIR/STEP/NNNN/claim.json`, flushes it, and records release/build/payload
identity, command, CLI pin and a hash of the authorization text. Dataset/notebook
uploads use a separately validated private staged snapshot, so editing the build
folder after staging does not change the uploaded files.

Each result has separate immutable `result.json`, `stdout.txt` and `stderr.txt`
records. Nonzero exits, timeouts, exceptions and incomplete claims leave an
unknown outcome. A zero exit is merely `SUBMITTED`. Repeating either mutation is
refused, including after a successful submission. There is no automatic retry,
dataset versioning, deletion or visibility change. Keep timeout partial output.

After independently inspecting the native account/status/files, a human may
explicitly reconcile the exact attempt using a local evidence file:

```sh
python releases/kaggle/publish.py --root BUILD_ROOT --config releases/kaggle/config.json --step reconcile --target dataset-create --outcome CONFIRMED_NO_REMOTE_MUTATION --evidence MANUAL_EVIDENCE.txt --state STATE_DIR --execute --authorization "Human checked remote state and authorizes reconciliation"
```

`CONFIRMED_NO_REMOTE_MUTATION` permits one new attempt while preserving the old
one. `CONFIRMED_REMOTE_MUTATION` keeps duplicate creation blocked. Neither a
nonzero exit nor an HTTP error by itself establishes that nothing was created.
Reconciliation stores the supplied evidence hash and remains a human assertion.
Do not discard state or choose a new state directory to bypass these controls.

## Hosted output verification and public review

The notebook leaves a root `replay-report.json` containing integer
`schema_version: 1`, `status: VERIFIED_SAVED_TRACE_REPLAY`, exact `release_id`,
`payload_sha256`, and `files`, a SHA256 mapping covering **every other hosted
output**. Required outputs are the five predeclared semantic files plus
`replay-summary.json`; additional figure files must also be listed. All files
and directories must be free of links and known privacy patterns. Extraction
directories, temporary files and logs must not remain as unlisted hosted output.

The publisher checks the closed inventory, every report hash, release/payload
binding, and the five predeclared semantic output byte hashes. A rehashed but
substituted analysis is refused. A successful report establishes saved-trace
output agreement under trusted-producer assumptions, not fresh native framework
execution, model inference, hardware safety, or cryptographic execution proof.

`--step public-plan --output NEW_OUTPUT_DIR` is read-only and first verifies
hosted outputs. It never generates or executes a public command. After reviewing
the dataset, hosted notebook, output bytes, licenses, scope and privacy, obtain
separate human approval and change dataset/notebook visibility manually in the
Kaggle UI. Public release is not implied by private upload authorization.

Run the local infrastructure tests without installing Kaggle:

```sh
python -m unittest discover -s releases/kaggle -p test_publish.py -v
```
