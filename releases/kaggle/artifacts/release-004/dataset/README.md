# ML Boundary Conformance: Saved Trace Replay

An offline CPU research artifact for checking whether normalization boundaries
agree with an explicitly supplied numerical reference. The accompanying Kaggle
notebook recomputes numerical verdicts, eight query-design comparisons, authority
ablations, fresh-witness relations, baseline interpretation, transport checks and
counts from two saved native acquisitions. It requires Python 3.13+, the standard
library and optional Matplotlib for the final plot. No downloads are needed.

This is **saved-trace replay**. It is not a new LeRobot/OpenPI/Robotruth/Policy
Space execution, learned-policy evaluation, robot experiment or independent
replication of the producer's environment. A separate original verifier checked
the full committed protocol and native records before export. The portable
verifier checks a smaller public export without Git ancestry or independent
attestation of the original producer.

## Contents and use

`conditions.csv` contains 24 rows: 16 actual conditions and 8 derived views that
reuse observations. `payload.zip` contains exact frozen analysis source, selected
publisher metadata, prepared intervention statistics/configurations, saved
observations from the original and clean acquisition, expected products, licenses,
and a hash-bound provenance/redaction ledger. It contains no learned weights.
The notebook pins the payload SHA256, unpacks safely into a temporary working
directory, recomputes all five products for both acquisitions in a fresh Python
process, checks agreement, and writes a closed output report.

Attach this dataset to the **Boundary Conformance Replay** notebook. Use CPU,
internet off. Run all cells. The expected final status is
`VERIFIED_SAVED_TRACE_REPLAY`. For local use follow the release README. Source
hashes, original artifact revisions, source manifests and component license
declarations are retained inside the payload; `release-provenance.json` records
each original/exported SHA256 and any permitted JSON-pointer redaction.

## Experimental scope and findings

Three published metadata groups provide four execution cells: ACT/MEAN_STD,
SmolVLA/MEAN_STD, PI05/QUANTILES in LeRobot, and an explicit OpenPI transport of
the **same PI statistics**. The OpenPI cell is nested within that artifact, not
another independent checkpoint. The groups share ancestry and development
knowledge. Metadata and intervention recipes were previously inspected; this
does not establish blinded or unseen-root-cause generalization.

The original acquisitions made 1,792 primary decoder calls, 896 auxiliary native
calls and 16 shared fresh witness calls. There are four intended controls and
twelve injected changes. With full numerical authority, compiled, pointwise and
strong relational methods detect all twelve changes, fail none of the four
controls, and agree on eight UNKNOWN derived views. Static statistics and
independently referenced round trips match these finite detection counts.
Candidate self-round trips remain consistent for all sixteen actual conditions.

Offset-authority ablation produces **eight disagreements**: compiled and strong
relational methods retain slope evidence where pointwise comparison abstains.
The zero-disagreement full-reference result excludes this ablation. All eight
equal-budget query designs agree on the reported finite/full-affine statuses.
Compiled method witness views use 16 native calls versus 12 each for pointwise
and strong relational methods; overlapping acquisitions are charged once as
16 shared calls. These observations support no performance-superiority claim.

Robotruth finds known statistics identity changes in twelve conditions, while
full certification remains WITHHELD for all sixteen. These are not sixteen
semantic detections or four false alarms. Policy Space accepts 1,792 raw-vector
replays and rejects sixteen structural controls. Its float32 wire conversion is
reported separately from float64 numerical verdicts; transport acceptance is
not conformance. All 24 physical-authority views are UNKNOWN.

## Data dictionary

| Column | Type | Meaning |
| --- | --- | --- |
| cell | string | Execution cell; PI/OpenPI reuses PI artifact |
| cluster | string | Declared artifact ancestry group; not independent sampling |
| condition | string | Numeric control, injected change, or authority-erased view |
| derived | boolean | True means reused observations, no new acquisition |
| compiled | string | PASS / FAIL / UNKNOWN under supplied numerical authority |
| pointwise | string | Independent pointwise numerical status |
| strong | string | Strong relational numerical status |
| physical | string | UNKNOWN in every row; no physical calibration authority |
| robotruth_identity_difference | boolean | Known union statistics hash differs |
| robotruth_certification | string | WITHHELD for missing required evidence |
| policy_space_transport | string | ACCEPT is structural raw-vector transport |

## Provenance, license and exclusions

Original code and results: Apache-2.0, approved by the repository owner.
Publisher metadata: publisher-declared Apache-2.0 at pinned source revisions;
original license texts/cards remain included. Intervention copies are explicit
research derivatives, with exact changed values in `plan.json`; they are not
publisher releases. See `THIRD_PARTY_NOTICES.md`. No rights to unprovided weights
are asserted. Local environment/source-path fields in public trace projections
are replaced with stable redaction labels, with exact JSON pointers and both
hashes recorded. Numerical observations, source pins and verdicts are unchanged.

Excluded: credentials, local environments, executables, dependency checkouts,
compiler/style assets, original private Git history, unresolved OpenPI GCS
assets, Tether sources and unrelated development archives. The manuscript and
retrospective/Panda/SO101 strata are outside this focused initial dataset. Their
results must not be inferred from the main table. The human-review manuscript
is neither submitted nor accepted. Extensive AI assistance was used in the
research and tooling. No population accuracy, hardware safety, policy task
success or cryptographic attestation is claimed.

## Citation

MLTools contributors (2026). *ML Boundary Conformance: Saved Trace Replay*.
Kaggle, `abdullah2hamza/boundary-conformance-replay`. Cite the dataset version and
the `release_id`/payload SHA256 printed by the notebook. This is an artifact
citation, not a claim of a peer-reviewed publication or an assigned DOI.
