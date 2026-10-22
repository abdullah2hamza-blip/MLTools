# OpenPI: audited and executed CPU processor subset

The source audit selected commit `215abfb217dbac7d5f1273282331b9b1866c0479`. Genuine `openpi` and `openpi-client` were installed and imported in a separate Windows Python 3.11.13 CPU environment. Nine initial tests passed, including actual mean/std and quantile transforms, PI0/PI05/FAST camera masks, and a two-query counterexample replay in a fresh worker. A clean environment installed from the CPU/build/source pins then passed ten checks, including an added raw-suite validation regression. These are development tests with synthetic fixtures, not benchmark performance measurements. Upstream officially supports Ubuntu 22.04; WSL is absent here. This must not be described as a supported full OpenPI installation. Setup locks and commands are in `integrations/openpi/README.md`; verification is recorded in `validation/publication-development.json`.

Install actual `openpi` and `openpi-client` from the same immutable Git revision without their full dependency graph, then provide the audited CPU import dependencies explicitly. No source extraction, fake module stubs or transformer patches are permitted as substitutes for genuine processor execution. Avoid constructing models/tokenizers or `create_trained_policy`.

Expected imports: `openpi.transforms`, `openpi.shared.normalize`, and actual `LiberoInputs`/`LiberoOutputs` if their transitive model-class imports are feasible. ALOHA's genuine input/output classes are a fallback integration path, not an excuse to invent a second embodiment. Audit/install requirements and exact resolved transitive versions before use.

Mean/std normalization uses epsilon `1e-6` in both forward and inverse. Quantile normalization uses quantile width plus `1e-6` and extrapolates. This differs from the LeRobot implementation. Relative/absolute transforms mutate arrays in place, so capture conditions require defensive copies. Padded dimensions may have zero standard deviations or different handling; do not force LeRobot's positive-std-only assumptions onto OpenPI.

## Immutable public metadata candidates

All paths are beneath `https://storage.googleapis.com/openpi-assets/checkpoints/` and refer only to small normalization artifacts. Generation-qualified URLs pin the stored object version; bytes/hashes must be independently reverified during retrieval.

| Family | Relative path | Generation | SHA-256 |
| --- | --- | --- | --- |
| PI05 | `pi05_libero/assets/physical-intelligence/libero/norm_stats.json` | `1757354341826612` | `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84` |
| PI0 | `pi0_libero/assets/physical-intelligence/libero/norm_stats.json` | `1748456806752855` | `dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1` |
| PI0 FAST | `pi0_fast_libero/assets/physical-intelligence/libero/norm_stats.json` | `1748456809464792` | `4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e` |

PI0 statistics are padded to 32 dimensions; PI05/FAST have state dimension eight and action dimension seven. These model versions do not automatically count as three independent policy families. LIBERO's software embodiment is Panda, with end-effector position, axis-angle and gripper coordinates; physical controller frames/units still need pinned authority.

## Historical authority and uncertainty

[Merged PR 351](https://github.com/Physical-Intelligence/openpi/pull/351), commit `92b10824421d6d810eb1e398330acd79dc7cd934`, is a candidate maintainer-confirmed legacy compatibility case: released LIBERO checkpoints required delta transforms. Removing an apparently redundant conversion is not universally correct.

[Issue 849](https://github.com/Physical-Intelligence/openpi/issues/849) and [open PR 971](https://github.com/Physical-Intelligence/openpi/pull/971) report a normalization-mode mismatch for PI0 FAST. Release-era serving/config source at `5869716a34ada84b883fe8d05751ab4744210f5b` corroborates a historical mean/std serving profile, while the newer source selects quantiles for FAST. The small checkpoint metadata does not attest training normalization mode, and statistics contain both kinds of summaries. Record this as reported and source-corroborated, not confirmed checkpoint training ground truth or our reproduced task-success result.

Actual second-framework processor execution is established. The benchmark gate remains unsatisfied: published artifact authority, licenses, adequate family/embodiment coverage, historical-case scope, splits and final evaluation still require completion. Public normalization artifact candidates above have not been incorporated into the benchmark or redistributed under an assumed license. No checkpoint inference or simulator execution was performed.
