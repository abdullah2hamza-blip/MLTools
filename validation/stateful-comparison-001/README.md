# Same-observation stateful development evidence

Genuine pinned LeRobot processor captures compare the unchanged original
stateful checker with the compiled evaluator. This is one previously inspected
SO101 software-fixture development cluster, not a final benchmark or policy run.

`summary.json` binds all 65 evidence files and the implementation sources.
`full.*` contains the eight-condition acquisition; `witness.*` contains eight
fresh reduced-bundle acquisitions. Original full reports, matched comparisons,
compiled reports, reduced suites, complete inputs and measured durations are
retained separately. Raw native stdout/stderr are preserved exactly.

Both matched methods agree on the eight expected outcomes. The original
witnesses require 1–2 native actions; compiled witnesses require 2–4. All
reproduce the same original failed relation. See `docs/stateful-comparison.md`
for denominators, assumptions, action counts and reproduction commands.

Data-only verification, also supported after copying this directory:

```sh
python -O scripts/compare_stateful_development.py --verify validation/stateful-comparison-001
```
