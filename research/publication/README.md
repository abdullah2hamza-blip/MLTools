# Publication project: progress and evidence

Status: active research development. The existing package is a working software prototype; it is not yet a publication-ready research result. The complete objective is preserved in `objective.md`.

## Important novelty correction

[Tai, 2026](https://arxiv.org/abs/2606.03724) already studies executable VLA specifications and derives normalization-mismatch certificates. The original broad idea therefore cannot be claimed as our research novelty. [XPolicyLab, 2026](https://arxiv.org/abs/2608.09892) also covers standardized policy interfaces and isolated runtimes. Classical conformance testing and ML metamorphic/differential testing are established prior art. `literature.json` records inspected primary sources and unresolved questions; the audit is ongoing.

The current hypothesis is narrower: compile independently grounded, partially specified interface contracts into distinguishing temporal experiments, quantify what their observations identify, and produce small behavioral witnesses across real framework compositions. Neither novelty nor effectiveness of that hypothesis has been established. The mathematical basis is ordinary affine identification and model-based testing; the proposed contribution would have to come from the supported composition, cross-framework method and empirical corpus. The expanded audit also identified robotruth's executable-policy manifests and ActionABI's partial-evidence equivalence sets and separating probes, so those features are not independently novel either. `baseline-audit.md` records the comparison implications.

## Current implementation

`src/mltools/obligations.py` is a development research primitive. It generates bounded paired interventions over model-output, originating-state and current-state ports. It checks known coefficient obligations while canceling unspecified offsets, reports incomplete design rank, and emits query witnesses. Canonical design validation prevents mislabeled or collapsed probes from claiming unobserved directions. A binary64 output-resolution diagnostic retains UNKNOWN when quantization obscures an effect. Its algebra tests are development verification, not scientific detection results.

`src/mltools/openpi.py` and its isolated worker execute genuine pinned OpenPI processors using a separately locked CPU environment. `integrations/openpi/README.md` documents setup, API, scope and limits. Seven recorded development captures in `development/openpi-001/` exercise synthetic controls, an explicit composition injection, fresh-worker witness replay and a source-backed legacy reconstruction. They contain raw measurements and are never a held-out benchmark. `development/openpi-analysis-001/` replays those immutable observations through the strengthened evaluator; the initial mutable worktree capture limitation is recorded. Exact historical/controller authority is in `openpi-history.md`.

`method.md` states the proposed assumptions and scope. `protocol-draft.json` is explicitly **not frozen**. No final evaluation may run until the dataset inventory, authority labels, splits, methods, tolerances, budgets and protocol hashes are fixed in a committed revision. Existing examples, previously inspected incidents and their descendants are development evidence and cannot be relabeled held-out results.

The versioned `mltools probe` workflow now supports explicit execution, offline replay, provenance checks and original/reduced regression cases across both frameworks (`docs/probe-cases.md`). Development evidence under `validation/probe-cases-001/` includes actual published ACT statistics loaded through native Safetensors, separately decoded reference values, missing physical authority, and an explicitly injected wrong-statistics case. Fresh captures under `validation/probe-cases-002/` verify the reviewed implementation that binds execution to a single bounded statistics-byte snapshot. `scripts/verify_probe_evidence.py` checks saved file identities, original regression replay and fresh reduced witnesses without either ML framework; `validation/probe-workflow-verification.json` records package/test verification. `inventory-draft.json` and `selection/` record candidate revisions, licenses, dataset lineage and selection gaps. ACT boccia remains unexecuted; ACT move-cube is grouped with the existing SmolVLA development lineage.

`lerobot-history.md` and `validation/lerobot-history-001/` now record genuine installed parent/fix processor execution for PR4057's chunk-anchor drift, using an explicit fixture queue. The pre-existing documentation/training path independently grounds the intended anchor. Historical synchronous-rollout guards, RTC exclusions and missing ACT-relative support are preserved. This is retrospective processor evidence, not a full historical rollout or a benchmark result from the current probe checker. Integration into the frozen evaluation remains required.

`integrations/baselines/` now pins complete preflight and Policy Space packages in a hash-locked no-ML environment. `validation/native-baselines-002/` records genuine preflight CLI and Policy Space dispatcher/MessagePack/schema/lifecycle controls. These establish applicability and credit native validation behavior. They do not constitute final matched-evidence comparisons, full checkpoint/service runs or a superiority result. Tether's full paired-rollout verification and the additional close tools still require applicable comparison paths.

## Completion ledger

| Objective | Current evidence | Still required |
| --- | --- | --- |
| Literature/contribution | Close prior art identified; primary-source register started | Complete focused audit, defensible contribution and venue fit |
| Formal method | Development specification, compiler, numerical/canonical guards and actual OpenPI captures | Scope proofs/review, meaningful contribution validation, broader composition support |
| Tool | Versioned probe API/CLI for both frameworks, offline replay and real reduced regression execution; original boundary checker | Integration hardening and broader shared semantic coverage |
| Benchmark | Three-family candidate ledger; published ACT and SmolVLA metadata; pinned controller and incident source audits | Final eligible corpus, independent lineages, second embodiment artifact scope and real incident reproductions |
| Evaluation | Draft protocol, leakage/independence rules | Frozen inventory and splits before final runs; real experiments and uncertainty analysis |
| Baselines/ablations | Pinned full-package preflight CLI and Policy Space runtime controls; expanded Tether/source audit and close-tool overlap | Matched-evidence full applicable comparisons, strong stateful/manual paired baseline, ablations and attribution |
| Reproducibility | Existing prototype evidence, OpenPI subset locks, development capture/reanalysis runner | Final experiment runner, raw research results, figure/table pipeline and clean main-result reproduction |
| Paper | No research paper written | Verified citations, full source/bibliography, compiled PDF and visual inspection |

All previous prototype verification remains under `validation/`; it must not be described as evidence that this new objective is complete. Publishing code to the existing private origin is authorized. Public release, paid compute, outreach and submission require separate approval after a concrete reviewable result exists.
