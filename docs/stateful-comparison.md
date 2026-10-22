# Original stateful checker comparison

`scripts/compare_stateful_development.py` compares the unmodified original
`mltools.checker.check` with the compiled affine evaluator on newly executed,
identical second-dequeue commands. This is a development comparison on one
previously inspected SO101 software fixture, not a held-out benchmark or model
inference experiment.

## Evidence and matched scope

The independent scalar contract and named units come from
`examples/so101/good.json`; the affine suite/reference and statistics come from
`examples/probes/lerobot/`. Exact rational arithmetic checks that these two
reference representations describe the same map, including centers, radii,
origin-relative body joints and the absolute gripper. References are never
inferred from the deployment outputs.

The existing native LeRobot capture executes all 37 canonical queries. Each
query resets the fixture, supplies an originating state and a two-action chunk,
then dequeues the second action at the current state. The actual first/second
commands, state inputs, declared units, queue counts, horizon and reset evidence
are retained. Each of eight conditions consumes 74 native actions: 592 actions
in the full batch. Native processors and source files are verified before and
after capture; a Python audit hook rejects attempted network connections.
This hook is not an operating-system sandbox or an execution attestation.

The original full report is saved unchanged. The explicit matched projection
selects its second-dequeue controller-command and unit findings. The compiled
evaluator uses those same named commands and independently supplied affine
reference. Both retain the same authority/unit guards. Missing or unsupported
units withhold the affected output cells; a grounded failure in another cell
still takes precedence. The full original report additionally checks first
actions, queues and horizons, so its timing is not an equal-scope speed baseline.
OpenPI end-effector coordinates and incomplete published metadata are not
force-mapped into this scalar fixture domain.

## Recorded development result

`validation/stateful-comparison-001/summary.json` records parity:

| Condition | Original matched | Compiled matched |
| --- | --- | --- |
| Correct control | PASS | PASS |
| Moving anchor | FAIL | FAIL |
| Wrong statistics | FAIL | FAIL |
| Relative gripper | FAIL | FAIL |
| Absolute body | FAIL | FAIL |
| Missing authority | UNKNOWN | UNKNOWN |
| Missing controller units | UNKNOWN | UNKNOWN |
| Wrong controller units | FAIL | FAIL |

The four numerical failures also have genuinely re-executed reduced bundles.
The original exporter keeps the episode prefix through the first failure;
the compiler keeps its existing local distinguishing queries, each requiring
two actions in this capture protocol.

| Numerical condition | Original witness actions | Compiled witness actions |
| --- | ---: | ---: |
| Moving anchor | 2 | 4 |
| Wrong statistics | 1 | 2 |
| Relative gripper | 1 | 2 |
| Absolute body | 1 | 4 |

All eight fresh witnesses reproduce their original failed relation and values.
Together they require 17 additional native actions, for 609 actions overall.
The compiled witnesses are larger under this action-count metric. Query counts
must not be substituted for native action counts to suggest a reduction.
Neither exporter claims global minimality. The conditions, queries and witnesses
are repeated measurements of one development cluster, not independent samples.

## Reproduction and verification

With an installed matching core and the pinned LeRobot integration available:

```sh
python scripts/compare_stateful_development.py --python /path/to/lerobot/python --output validation/stateful-comparison-new
python -O scripts/compare_stateful_development.py --verify validation/stateful-comparison-001
```

The parent interpreter requires only the installed core; only the explicitly
selected subprocess imports Torch/LeRobot. The saved evidence can be copied to
another directory and verified without those libraries. The verifier checks the
closed 65-file manifest, independently regenerated fixture inputs, complete
native requests, producer/source bindings, actual query schedules, method reports,
expected control outcomes, witness suites, reduced bundles and execution costs.
Fresh witnesses must reproduce the original relation, including the renumbered
event for the reset-prefix export. Rehashed unrelated failures are rejected.

Raw stdout/stderr are preserved alongside parsed captures and measured durations.
Durations are recorded observations, not reproducible constants. The worker has
a 180-second subprocess timeout and one Torch thread; no peak-RSS measurement or
hard operating-system memory cap is claimed for this development runner.
