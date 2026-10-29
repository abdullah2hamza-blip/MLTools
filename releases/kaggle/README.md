# Kaggle release

The approved owner is `abdullah2hamza`; original code/results use Apache-2.0.
The first release is a focused **saved-trace replay dataset and offline CPU
notebook**, not another native acquisition. It preserves all frozen research
files and records the separate public projection's provenance. The manuscript,
mixed-license TeX assets and development strata are excluded from this dataset.
See [the dataset card](DATASET_CARD.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

The current local release candidate is [artifacts/release-005](artifacts/release-005/),
bound by [release-005-lock.json](release-005-lock.json). Its card, notice and
notebook text have been revised; numerical inputs, observations and the five
expected research products retain their original bytes. The notebook supports
Kaggle's automatic ZIP expansion and uses distinct dataset/notebook identities.
Fresh-kernel local checks cover the ZIP and expanded-input/platform-file forms
in [validation/review-005-zip](validation/review-005-zip/) and
[validation/review-005-expanded](validation/review-005-expanded/).

Release-005 has not been uploaded or executed on Kaggle. The existing private
dataset version 1 and notebook version 2 were verified using **release-004** on
23 October 2026; that proof does not verify the new candidate. See
[HOSTING.md](HOSTING.md) and
[the historical hosted evidence](validation/kaggle-hosted-002/verification.json).
All earlier preparation/upload snapshots and their proofs are retained unchanged.
The [candidate review record](validation/review-005-readiness.json) records
67 passing tests and 39 subtests, lint/format checks, a byte-identical second
build, both fresh-kernel checks and bounded credential scans. Historical frozen
gates pass; these checks establish local readiness for the revised package.

## Rebuild and inspect without credentials

Use Python 3.13+ from a complete Git checkout including the freeze commit. Core
replay/build/preflight use the standard library; do not install/import Kaggle.
Run from the repository root, with a new output directory for each build:

```sh
python releases/kaggle/build.py --output releases/kaggle/build/new-release --record releases/kaggle/build/new-release-lock.json
python releases/kaggle/publish.py --root releases/kaggle/artifacts/release-005 --config releases/kaggle/config.json
python -m pytest releases/kaggle -q --basetemp .test-tmp/kaggle-tests
```

The builder first runs both original full verifiers: frozen Git ancestry,
406-file source/input closure, native receipts and all regenerated products.
It refuses source/input drift. It then builds a closed allowlist from the frozen
source map, exact prepared plan, three selected publisher manifests and two
saved acquisitions. No recursive repository mirror is uploaded.

The flat dataset root contains a deterministic ZIP, CSV, metadata, card,
license/notices and release manifest. ZIP names, timestamps, permissions and
ordering are fixed. `release-lock.json` binds every final dataset/notebook byte;
the notebook pins the payload hash and the payload manifest hash. Kaggle expands
uploaded ZIPs; in that representation, every expanded member is checked against
the pinned manifest before copying or importing source. The payload's own manifest checks the
exported closure. The ledger retains original/exported SHA256 and exact redacted
JSON pointers. Local runtime prefixes and extracted `policy/source_path` fields
are redacted in **separate projections**; original observations remain unchanged.
Unexpected sensitive fields fail the build. Hashes are not attestation.

The original protocol and source files remain unchanged. In particular,
`pyproject.toml` is frozen, so its license metadata was not edited; root LICENSE
and NOTICE supply the now-approved first-party terms for this release.

## Run the real notebook locally

Authoring dependencies are separately hash-locked for Python 3.13 on Windows.
They exclude the Kaggle client and are not needed by the released numerical
replay. The generated notebook uses optional Matplotlib already supplied by the
current Kaggle image. Install the authoring environment using uv:

```sh
uv venv .venv-kaggle --python 3.13
uv pip sync releases/kaggle/testing-requirements.lock --python .venv-kaggle/Scripts/python.exe --require-hashes
.venv-kaggle/Scripts/python.exe releases/kaggle/smoke.py --root releases/kaggle/artifacts/release-005 --work .test-tmp/kaggle-smoke --record releases/kaggle/build/new-smoke-validation --expanded-input --platform-runtime-file
```

The smoke test executes every code cell in a fresh Jupyter kernel with copied
read-only dataset files and a new output directory. The isolated replay process
denies socket connection/DNS/bind events, loads no native ML packages, validates
all saved numerical boundaries and re-derives five products for both runs. It
checks exact product/output hashes and input stability. Hosted Kaggle validation
is recorded separately from the local Windows test in the linked evidence.

## Publishing later

[publishing.md](publishing.md) documents the credential-free planner, separately
authorized private upload, version-specific status/output retrieval, exact output
verification and unknown-outcome reconciliation. The CLI dependency lock is
prepared but **not installed/imported during preparation**, because importing
Kaggle can authenticate. Only install it into a separate publishing environment
after explicit credential-use authorization. Configure credentials locally;
never add tokens to arguments, notebooks, configs, committed files or upload
folders. Keep the publisher's attempt/state directory private and outside the
upload root. Public visibility requires a separate review and authorization.

The existing dataset must receive a deliberate `dataset-version` update for
release-005; its predecessor and meaningful notes are required. Plan that
operation against recorded dataset version 1, inspect readiness and exact files,
then plan the matching notebook revision against predecessor 2. The
[release-005 update instructions](publishing.md#plan-the-revised-release-005)
explain the local duplicate guards and unversioned dataset attachment limit.
No update is executed by a plan and no server predecessor condition is implied.

Official requirements and inspected revisions are recorded in
[requirements-reviewed.json](requirements-reviewed.json). Review hosted outputs
and the exact dataset version before public visibility. A draft artifact does
not imply a peer-reviewed paper, accepted submission or assigned DOI.
