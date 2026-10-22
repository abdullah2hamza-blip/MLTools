# Equal-budget query designs: development evidence

This is newly executed processor evidence on the six previously inspected development cases. It is not a frozen benchmark or a held-out evaluation. No unobserved SmolVLA candidate, ACT boccia candidate, trained model, tokenizer, simulator or physical robot was executed.

Each case acquired canonical geometry, independently authored manual-axis geometry, structured random-axis probes and dense random probes. Random methods use seeds 7, 23 and 101. Every design gets a measured center plus d antithetic pairs (2d+1 queries); canonical/manual geometry deliberately matches. All eight use the same native framework capture, independent reference, guards and research oracle.

| Existing development case | Queries per design | Numerical verdict, all eight designs | Full requested scope, all eight |
| --- | ---: | --- | --- |
| LeRobot good | 37 | PASS | PASS |
| LeRobot injected moving anchor | 37 | FAIL | FAIL |
| OpenPI good | 47 | PASS | PASS |
| OpenPI injected moving anchor | 47 | FAIL | FAIL |
| Published ACT metadata | 37 | PASS | UNKNOWN |
| ACT injected wrong statistics | 37 | FAIL | FAIL |

All 48 design conditions pass common provenance and recorded-schedule guards. There are 1,936 software queries with two fixture actions per query. These are repeated measurements within development integration/checkpoint/root-cause clusters, not 48 independent experiments or 1,936 incidents. “Compiled” labels query geometry, not an end-to-end comparison with the original compiled evaluator. Parity is the observed result; no superiority or task-success claim follows.

`summary.json` binds project source bytes, environment locks, case outcomes and every request, capture, native stdout/stderr, analysis and timing file. Each capture records upstream identities, original case/statistics identities, actual input/output rows, setup time, per-design acquisition time, packages and observed peak working set. Original native stdout parses identically to the corresponding structured capture. Python socket/DNS audit guards observed no network attempts; this is not an OS sandbox. Worker memory was observed below the declared 2 GiB post-execution check, not hard-limited.

The parent ran from the installed core wheel in `.venv-smoke`; genuine LeRobot used `.venv`, and genuine OpenPI used its separate Python 3.11 CPU-subset environment. Project/upstream sources and inputs were checked for stability around acquisition. OpenPI captures the deployed postprocessing path with an explicit fixture queue; it supplies no new training-transform or checkpoint evidence. Single ordered runs with shared caches are not timing estimates. Recorded timing separates import/setup, acquisition and initial analysis; verification also costs computation.

Review found two numerical pitfalls before this recorded run: nominal antipodes can have asymmetric executed coordinates, and full exact rank can coexist with poor numerical observability. The final research oracle uses actual binary-rational input geometry, exact cancellation requirements, a spread basis and exact-inverse component intervals. Regression tests include those hand-calculated counterexamples, unknown references, missing observations and nonlinear interior behavior. The interval claim remains conditional on admitted affine structure and an unfrozen empirical measurement allowance; it does not bound arbitrary framework arithmetic or certify physical correctness. Pre-review trials were retained only in ignored `.test-tmp` directories and excluded from this evidence.

With the documented environments installed, data-only replay is:

```powershell
.venv-smoke/Scripts/python.exe -O scripts/compare_query_designs.py --verify research/publication/development/query-designs-001
```

To acquire an independent new development run, choose a new destination:

```powershell
.venv-smoke/Scripts/python.exe scripts/compare_query_designs.py --lerobot-python .venv/Scripts/python.exe --openpi-python .venv-openpi-clean/Scripts/python.exe --output .test-tmp/query-designs-NEW
```

`docs/query-design-experiments.md` explains the format and conditional mathematics. Fresh reduced-witness acquisition for this separate format, original stateful and external comparisons, numerical-profile justification, eligible corpus, frozen protocol and final experiments remain unfinished.
