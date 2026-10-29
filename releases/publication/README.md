# Complete research publication package

The focused Kaggle replay remains an executable saved-trace analysis. This tool
adds the manuscript and a source/reproducibility supplement to the **same**
dataset, using an explicit optional distribution profile. It creates no account,
duplicate dataset, model, upload, Git tag or public visibility change.

Build after committing all final edits and checking out the intended existing
release tag. The tracked checkout must be clean. The replay root must be a new,
reviewed focused build from the matching sources. Choose unused output/lock
paths outside the upload folder, normally under the ignored `.test-tmp`:

```powershell
python releases/publication/bundle.py --ref RELEASE_TAG --replay-root FOCUSED_BUILD --output .test-tmp/public-release --record .test-tmp/public-release-lock.json
python releases/kaggle/publish.py --root .test-tmp/public-release --config releases/kaggle/config.json
```

The builder reads regular Git blobs from that exact tag, verifies the current
PDF against its successful stable compilation record, and checks every source
file recorded in that compilation. It retains all 406 protected frozen files
with their exact original bytes. A clean source snapshot has no `.git` ancestry;
the strict original verifier still requires a full public clone at the recorded
commit. The portable notebook verifies its separate saved-trace export. These
scopes are stated in both source ZIPs and the public manifest.

The flat dataset has the original seven files plus exactly:

- `manuscript.pdf`: current compiled research preprint.
- `manuscript-source.zip`: editable TeX, bibliography, generated figures/tables,
  and exact external-style acquisition instructions.
- `research-source.zip`: tagged first-party code, tests, configurations, locks,
  provenance, original/clean saved evidence, reproduction/reporting tools and
  documentation; includes the complete protected byte closure.
- `public-release-manifest.json`: source commit/tag, file hashes, exact ZIP
  member hashes, frozen/research revisions, numerical replay binding and every
  omitted Git filename/reason. It excludes its own checksum to avoid a cycle.

Kaggle can expand every uploaded ZIP. The public manifest contains both archive
hashes and exact member inventories so expanded manuscript/research sources can
be checked individually. The numerical notebook continues to pin and verify its
own archive or expanded payload before importing source. Publishing must retain
the actual dataset/notebook versions and hosted input/output proof.

The source snapshot excludes historical release ZIP duplicates, historical
executed release notebooks and intermediate paper compilation folders. It keeps
the final PDF, current editable manuscript and all frozen inputs. Local
environments, caches, credentials/private attempt state and dependency/compiler
binaries are excluded. No Git history bundle is generated; the public Git clone
is the documented ancestry-preserving reproduction entry point.

Original authored contributions retain the owner's approved Apache-2.0 license.
Publisher metadata retains its pinned cards/licenses. The optional TMLR style
is Apache-2.0, `fancyhdr.sty` is LPPL1.3-or-later, and `tmlr.bst` is LPPL1-or-later.
These style files are omitted from the ZIPs, with exact revision/URL/SHA256
retrieval instructions and the official LPPL1.3c license text supplied. No mixed
file set is silently relicensed. The approved source snapshot contains only
small numeric statistics and explicitly authored synthetic identity fixtures,
with no learned model weights or robot recordings.

The publisher's original focused-file privacy checks remain unchanged. Only the
explicit `public_research_supplement_v1` profile admits the four fixed extras.
Those sources legitimately contain credential **variable names** and historical
producer paths. Separate guards reject credential values/private key bodies,
unsafe archive names, links, runtime/model binaries and unexpected ZIP members;
frozen input hashes and all existing replay guards still apply. Bounded scans
and producer hashes do not establish exhaustive secret detection or attestation.
The decoded PDF/notebook and reachable-history audit is recorded separately in
`research/publication/release/security-rights.json`; final source/bundle deltas
must be checked before the authorized public transition.

Run the meaningful packaging and existing publisher regressions together:

```powershell
python -m pytest releases/kaggle releases/publication -q --basetemp .test-tmp/publication-tests
```

No test imports Kaggle or reaches its API. Real hosted execution and anonymous
download/access checks remain separate release steps. Use the existing guarded
dataset-version operation for the already created dataset, then inspect that
version's exact files/readiness before the corresponding notebook revision.
