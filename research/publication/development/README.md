# Development evidence only

These artifacts are deliberately outside any final evaluation partition. They include cases inspected during implementation, synthetic numerical fixtures, injected faults, and retrospective source-backed processor reconstructions. They do not establish unseen detection performance, three independent policy families, task success, or publication readiness.

`openpi-001/` records seven genuine OpenPI processor captures and their original reports, configurations, measurements and byte hashes. All use actual installed source revision `215abfb217dbac7d5f1273282331b9b1866c0479`, with source checks recorded per trace. No trained weights or simulator were used. Synthetic training-side numerical probes do not provide independent physical training data. The historical PR351 cases isolate a changed composition with synthetic identity statistics; they are not full historical policy executions.

The initial run occurred during an evolving uncommitted development worktree. Exact project worker hashes were not recorded. The selected upstream source identities and raw observations are recorded; a final frozen experiment must additionally identify the committed project code before execution. These limitations disqualify this run from final evaluation.

`openpi-analysis-001/` contains offline reanalysis through the evaluator after canonical-input and numerical-resolution safeguards were added. Original observations and original reports are preserved. The analysis summary identifies the evaluator bytes and links each input/output by SHA-256. This is verification of those observations, not an additional independent experiment.

Reproduce a new run with a new destination:

```powershell
uv run --no-sync python scripts/validate_openpi_development.py --python .venv-openpi/Scripts/python.exe --output .test-tmp/openpi-run-NEW
uv run --no-sync python scripts/reanalyze_openpi_development.py --input .test-tmp/openpi-run-NEW --output .test-tmp/openpi-analysis-NEW
```

The capture runner is explicit execution; reanalysis is data-only and imports no ML framework. Both assert the intended development outcomes and preserve labels. An incorrect outcome stops the run instead of silently editing its expected label. Setup instructions are in `integrations/openpi/README.md`.

`oracles-002/` records matched-evidence replay of the six versioned CLI captures using independent pointwise, strong all-pairs/midpoint and stateless-diagonal numerical comparators, plus authority/provenance/witness ablations. It was reproduced through the installed wheel without ML dependencies. The strong relational oracle matches the compiler on these development observations; no detection superiority is established. Missing offsets make a constant bias unidentifiable for every oracle. `docs/research-oracles.md` explains domains, propagated numerical budgets and finite-query versus full-design coverage.

`query-designs-001/` adds new processor acquisition for eight equal-budget canonical/manual/random designs per existing development case. All eight agree on each numerical verdict; missing physical authority remains UNKNOWN at full scope. Exact executed-coordinate contrasts and conditional component intervals address numerical counterexamples found during review. All geometries share one research oracle; this does not compare end-to-end original evaluators. See its README and `docs/query-design-experiments.md` for the empirical numerical profile and provenance/cost limits. The original stateful checker comparison, external workflows, final matched evaluation and fresh reduced witnesses for the research format remain required before evaluation can be frozen.
