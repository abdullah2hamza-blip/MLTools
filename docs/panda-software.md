# Source-grounded Panda software semantics

This development stratum supplies an independently specified second software embodiment alongside the SO101 processor contract. It runs actual robosuite/Panda code in a local MuJoCo `Lift` environment. It does not establish a published checkpoint's mapping to that controller. Every checkpoint-linkage status remains UNKNOWN.

## Authority and independent reference

The released robosuite 1.4.1 wheel is pinned by SHA-256. Eight exercised semantic files match the inspected [robosuite source revision](https://github.com/ARISE-Initiative/robosuite/tree/fbee5844ff5632f5b5698e204ec5357ca50be0df). `scripts/panda_reference.py` lists their Git blob identities and implements independent scalar/matrix/state arithmetic without importing the native package. The complete wheel's separate identity and Windows import compatibility are documented in `integrations/panda/README.md`.

The native [OSC configuration](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/controllers/config/osc_pose.json), [base scaling](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/controllers/base_controller.py), [goal setter](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/controllers/osc.py) and [control utilities](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/utils/control_utils.py) ground the translation contract. On the selected unclipped input domain, `goal = current + 0.05 * action`; originating state has zero coefficient. Position limits are unset. Units are explicitly named simulator coordinates, with no calibrated SI assertion.

The orientation reference independently uses Rodrigues' formula and matrix multiplication. Delta rotation pre-multiplies the current matrix. Two noncommuting cases distinguish the intended order from deliberately reversed caller composition. Componentwise addition of axis-angle vectors is not used as a reference. The [native Panda gripper](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/models/grippers/panda_gripper.py) integrates `0.01 * sign(command)` with opposite finger directions and clips to [-1, 1]. Native environment reset is checked on the same populated gripper object, with the reset action state and first subsequent command observed.

This is source-grounded software authority: the independent implementation can still share a source-reading error with the producer. Hand-calculated noncommuting rotations and gripper fixtures are regression checks, not independent physical calibration.

## Executed evidence and results

`validation/panda-software-001/` retains inputs, native request, raw stdout/stderr, parsed capture and analysis, bound by a closed six-file SHA-256 manifest. The translation design has nine ports/coordinates and 19 canonical queries. Each of six conditions executes actual native scaling and goal-position operators on explicitly controlled state vectors. Those vectors are operator fixtures; they are not claimed to be reachable simulator trajectories. Two separate full native controller `update`/`set_goal` calls use the actual simulator end-effector state and nonzero rotations.

| Translation condition | Compiled / pointwise / strong relational | Stateless full-design coverage |
| --- | --- | --- |
| Intended configuration | PASS | UNKNOWN |
| Deliberate doubled translation gain | FAIL | UNKNOWN |
| Deliberate originating-state anchor | FAIL | UNKNOWN |
| Deliberate action coordinate permutation | FAIL | UNKNOWN |
| Reference authority erased | UNKNOWN | UNKNOWN |
| Output units erased | UNKNOWN | UNKNOWN |

The stateless method's own finite subdomain contains seven diagonal origin/current queries (rank 4 versus the required 10). Its finite status is PASS for both the control and wrong-anchor condition, while full-design coverage remains UNKNOWN. This limited PASS must not be reported as certification of the complete stateful contract. Gain/order faults are detectable within that subdomain.

Both normal orientation cases pass the independent nonlinear reference; both deliberate reversed compositions fail. The populated gripper sequence passes with native reset and fails when reset is deliberately omitted. Each gripper variant makes 119 genuine `format_action` calls, including the first command after the reset boundary. The affine evaluator returns UNKNOWN for all orientation and gripper cases because those operations lie outside its admitted fragment. These results favor an explicit scope boundary and a strong independent reference; they establish no compiler superiority over the strong oracle.

There are 114 translation queries, two full goal setters, four orientation operator calls and 238 gripper formatting calls per capture. They are repeated measurements of one development software cluster. The original worker took 11.65474 seconds with measured peak RSS 290,242,560 bytes; the clean environment took 20.00594 seconds and 328,859,648 bytes. These measurements include setup/import and integrity checks within the worker. They are not isolated per-query execution costs.

Both captures have zero physics-time progression, environment steps, torque-controller calls and renders. Environment construction/reset/close are real native lifecycle operations. No task reward, success rate, model inference or hardware behavior was measured.

## Reproduction and review guards

`scripts/validate_panda.py` first checks each raw result against the actual called arguments and injected configuration, then compares it with the intended contract. This prevents an arbitrary altered response from passing verification merely because a fault condition still reports FAIL. Optimized-mode copied-folder tests cover that failure mode, orientation output tampering, reset state, unsupported affine PASS, manifest omissions, source/DLL identities and simulation/request changes.

The clean-environment run reproduces nine semantic fields exactly; three data-only files are byte-identical. `validation/panda-reproduction-verification.json` records the compared sources and summary hashes. Runtime and stderr are preserved separately rather than forced to match. Replay validates saved data and current producer identities; it does not re-execute native code or authenticate a trace against a malicious producer. See `integrations/panda/README.md` for explicit setup/native-run and offline-replay commands.

## Relationship to the main study

This is development evidence. The selected finite corpus in `research/publication/study-design.json` separately plans four metadata execution cells and six conditions per cell. The selected revisions are metadata-inspected and execution-outcome-unseen; they are not blinded or ancestry-independent. Implementation, equal-evidence adapters and a committed complete protocol freeze are still required before their execution. PI statistics transported into OpenPI will remain an explicitly scoped cross-framework numerical experiment, with no inferred native OpenPI checkpoint or Panda physical mapping.
