# Matched-evidence development replay

This is data-only reanalysis of the six genuine processor captures in `validation/probe-cases-002/`. It is not a held-out benchmark, an additional independent sample, or final evaluation. The source capture matrix includes synthetic controls/interventions and published ACT normalization artifacts; it does not include trained policy inference or physical training measurements.

The implementation was independently reviewed, corrected for numerical cancellation and incomplete coverage, tested, built, and installed into `.venv-smoke` without Torch, LeRobot or OpenPI. The analysis and optimized-Python verifier both ran in that wheel-installed environment. `summary.json` identifies source inputs, implementation bytes, analysis outputs and the actual runtime. Source capture verification also checks original exports and the earlier genuinely executed reduced witnesses.

## Recorded outcomes

These columns compare **numerical** oracle results using identical saved observations and reference information. Full-request statuses additionally retain missing authority/provenance and exclude unsupported temporal claims.

| Development case | Compiled | Pointwise | Strong relational | Stateless diagonal |
| --- | --- | --- | --- | --- |
| LeRobot correct calibration | PASS | PASS | PASS | PASS on 13 eligible queries |
| LeRobot moving anchor injection | FAIL | FAIL | FAIL | PASS on 13 eligible queries |
| Published ACT statistics | PASS | PASS | PASS | PASS on 13 eligible queries |
| ACT statistics offset injection | FAIL | FAIL | FAIL | FAIL on 13 eligible queries |
| OpenPI correct calibration | PASS | PASS | PASS | PASS on 15 eligible queries |
| OpenPI moving anchor injection, offsets already unknown | FAIL | UNKNOWN | FAIL | UNKNOWN on 15 eligible queries |

The strong independently implemented relational oracle matches the compiled detector on these reused development captures. These results do not establish improved detection over a capable manual stateful oracle. The pointwise limitation on unknown offsets must not be presented as the strongest manual baseline. Manual/random query generation and external tools remain separate required comparisons.

Stateless PASS does not clear the full temporal request. The full stateless request remains UNKNOWN on all rows above except the known ACT normalization failure. Published ACT statistics remain UNKNOWN overall despite numeric PASS because physical authority is missing. The original OpenPI moving-anchor reference already lacks offsets; this is disclosed rather than treating that partial-reference case as complete information.

Erasing offsets from the independently supplied reference retains moving-anchor failures for both compiled and strong relational checks, while the pointwise oracle abstains. It also makes the ACT constant-bias injection **UNKNOWN for every oracle**: identical paired effects cannot establish an unknown absolute offset. Correct controls with erased offsets likewise remain UNKNOWN overall. No deployment-derived offset is inserted as ground truth.

Witness-removal ablation preserves the verdict and retains the original 37 or 47 queries. The saved, previously freshly executed compiled witnesses retain 2 queries for LeRobot anchor drift, 1 for ACT bias and 2 for OpenPI anchor drift. This analysis verifies their saved replay; it does not claim new adapter execution, globally minimal trajectories, or replay-cost measurements for the comparator methods.

## Reproduce

Use the current installed package and new output directories:

```powershell
python scripts/analyze_oracle_development.py --output .test-tmp/oracles-NEW
python -O scripts/verify_oracle_development.py .test-tmp/oracles-NEW
```

The source files must match the saved analysis identities to verify this particular run. Changed implementations require a separately identified reanalysis. Analysis timing is actual elapsed time for all methods and ablations together, not a per-method speed comparison, compute-speedup result or acquisition-cost estimate. The pre-review analysis was retained outside the research artifact under ignored `.test-tmp/`; only this reviewed run is used here.
