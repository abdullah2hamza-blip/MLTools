# OpenPI legacy compatibility and controller authority

This is a retrospective development case study, not an unseen benchmark incident or a policy task-success result.

## PR351: independently documented intended legacy wiring

[PR351](https://github.com/Physical-Intelligence/openpi/pull/351) merged on 2025-02-28 as `92b10824421d6d810eb1e398330acd79dc7cd934`. Its first merge parent is `f1b9f4ab715cb8c87f177cf44d0e132a4765dc16`, second parent/head `a4b1bf92f16853a9235c1c3a30d9b9d0a0cb57b2`. Use the actual first parent, rather than the PR API's base SHA, for the before/after comparison.

Only `src/openpi/training/config.py` changed: before Git blob `3b9dfb9f59677e9c52f91201b960aef898404279`, after `164d59be376a646bd82ecb35b96d5090e746bae4`. [The merged configuration](https://github.com/Physical-Intelligence/openpi/blob/92b10824421d6d810eb1e398330acd79dc7cd934/src/openpi/training/config.py#L305) restores `DeltaActions` and `AbsoluteActions` with the first six dimensions selected and gripper excluded. The added rationale ties this to existing LIBERO checkpoint compatibility pending checkpoint updates. The PR body is empty; no separate maintainer prose statement is inferred.

Historical `transforms.py` is unchanged across those two commits, blob `84a015e2b9c9967c6d5cf648f005de00ef57c3a2`. Its complete `DeltaActions` and `AbsoluteActions` class syntax trees match those installed at current pinned revision `215abfb217dbac7d5f1273282331b9b1866c0479`. Comparison used CPython 3.13 `ast.parse`/`ast.dump(include_attributes=False)` on complete source fetched from both Git revisions, without executing extracted source. SHA-256 of the identical class dumps:

- DeltaActions: `9df9c3c825c7b5b2f8d770f1fce184fec1cef70385e721bdf39e0e04e191e123`
- AbsoluteActions: `76b9a3df4095758e8a6fd50e643281c0afb45ffc1693cbe1091f52a166a43512`

The development runner reconstructs the old omitted-mask and restored-mask compositions using these genuine installed classes. It supplies an independently authored controller vector and corresponding legacy model-space residual, with synthetic identity inverse-normalization statistics to isolate the changed action conversion. Expected values are explicit; a round trip alone is not the oracle. This reproduces the processor-level mismatch supported by the source change. It does not execute the full historical installation, recover actual old training statistics, verify checkpoint weights, or measure task success. It must retain the label `retrospective_processor_reconstruction`.

## Genuine historical factory reproduction

The earlier active training configuration provides an independent source of the intended configured convention. [Commit `4a10482`](https://github.com/Physical-Intelligence/openpi/commit/4a10482dfbdc1d580c7eed4c9c444d675f679254), dated 2025-02-27, commented out the delta/absolute wiring while adding LIBERO documentation. Its actual parent [`bf25a4d`](https://github.com/Physical-Intelligence/openpi/blob/bf25a4d9c4dc3c0ba9b132ddec3ecf54b0fa0c07/src/openpi/training/config.py#L281), config blob `5b93cee815decfb1d91713612950c967bdd0bff5`, actively applies `make_bool_mask(6, -1)` to both transforms. That file assigns the data factory to named PI0 and PI0-FAST LIBERO training configurations. This supports the historical configured training convention beyond the restoration comment; it does not attest the actual training revision of any released checkpoint.

`scripts/reproduce_openpi_history.py` now installs and executes the genuine parent/fix data factories in separate pinned CPU environments. `integrations/openpi-history/sources.json` records the complete installed Python inventory, 42 OpenPI plus 13 client modules per revision, checked before imports and after execution. Both complete source packages are unmodified. The environment is a reviewed CPU subset with explicit version differences from the historical full training dependencies; it is not a recreation of that complete runtime.

Exactly one unused stage is isolated in memory: `ModelTransformFactory.__call__` returns a genuine empty `Group`, with call count and restoration recorded. Without this substitution the factory constructs a tokenizer and may fetch its asset. The unchanged named `pi0_libero` data factory, dataset repacking, image/padding transforms, and legacy action transforms execute. An empty relative assets path avoids the historical downloader's misinterpretation of Windows drive paths as URL schemes. No statistics, model, tokenizer, simulator or weights are loaded.

The independently authored fixture contains two different action rows and explicit expected legacy targets. The parent leaves those raw training actions unchanged and emits legacy residuals unchanged at deployment; the fixed factory subtracts/adds the first six originating-state coordinates and preserves the gripper. Native round trips pass in both versions, illustrating why independently grounded direct expectations matter. The runner also passes the same 31 executed output measurements per revision to the compiled evaluator and independent pointwise/strong relational oracles. Their verdicts agree. A fresh worker re-executes the compiled failure's one-query witness. These are one retrospective incident, not independent samples or a superiority result.

The saved evidence is `validation/openpi-history-001/`; its separate README and [environment instructions](../../integrations/openpi-history/README.md) document replay and limitations. Worker/request/fixture/ledger identities are snapshotted before execution and checked afterward. The parent runner pins its source set before dispatch, including equality between the installed MLTools package and recorded workspace modules. The data-only verifier uses explicit conditional checks, including under Python `-O`. Capture identities are trusted local producer assertions, not execution attestation.

This advances beyond the older current-class reconstruction without changing that artifact's label. It verifies the historical configured data boundary only. Checkpoint training identity, normalization statistics, model behavior and full physical conformance remain UNKNOWN. Integration into the common public historical producer profile and the frozen final evaluation is still required.

## Native LIBERO/Panda semantics are a separate reference

OpenPI pins LIBERO submodule `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`. Its [environment wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/libero/libero/envs/env_wrapper.py), Git blob `b5a732b0d7e5d67ba5428fdf0edef150e80f6296`, selects Panda, `OSC_POSE`, 20 Hz and agentview/wrist cameras. OpenPI's LIBERO requirements pin robosuite 1.4.1.

There is no assumed `v1.4.1` Git tag. A read-only audit compared selected source entries from the exact PyPI wheel, using bounded HTTP ranges and CRC verification, against immutable Git files. The following canonical-LF entries match Git revision `fbee5844ff5632f5b5698e204ec5357ca50be0df`:

| Source | Git blob |
| --- | --- |
| [OSC config](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/controllers/config/osc_pose.json) | `8dc645e44bb13ba6806e7a74af52d1efaefc79e8` |
| [OSC controller](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/controllers/osc.py) | `a45843d087ca2329456a53f5a86e87ded6e9c44a` |
| [Scaling](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/controllers/base_controller.py) | `895952a5a4e300ae58ec998cc379853d06fff689` |
| [Goal construction](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/utils/control_utils.py) | `18cc0a6b30cd8011f5267cc39542c86a01867e44` |
| [Panda gripper](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/models/grippers/panda_gripper.py) | `64650e33d40f6df02d48032c91df3c71213d1751` |
| [Rotation convention](https://github.com/ARISE-Initiative/robosuite/blob/fbee5844ff5632f5b5698e204ec5357ca50be0df/robosuite/utils/transform_utils.py) | `ac371c09da962dcceea8a376876fc1d46c0e52da` |

Native action order is three translational commands, three axis-angle rotational commands, then one gripper command. Arm commands are normalized/dimensionless and clipped/scaled from [-1,1]; translation scales to ±0.05 simulator position coordinates and rotation to ±0.5 radians. With delta control, the position goal adds the scaled translation. The orientation goal composes `R(delta_axis_angle) @ R_current`; componentwise axis-angle addition is not a valid substitute.

The Panda gripper uses −1=open, +1=closed, integrating the command sign into internal finger commands. Excluding it from OpenPI's extra delta transform does not make it an absolute hardware finger position. Checkpoint-level physical SI calibration for translation was not found; the source-defined simulator coordinate scale is the available authority.

[OpenPI's evaluation client](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/main.py#L133), blob `dc015a61740f2d3174152bebb60176fac52f3f40`, concatenates end-effector position[3], quaternion-to-axis-angle[3] and gripper joint positions[2]. The exact released robosuite `single_arm.py` source has canonical Git blob `598a168e0fdf9ba7e774bf75f349d5787448e287`; it differs from the older tag. It obtains position from grip-site world position, quaternion from end-effector body orientation converted to XYZW, and the two gripper joint positions from qpos.

These sources justify interface interpretation. No controller, simulator or live robot was executed in this audit. Native already-delta controller commands and a legacy checkpoint's extra training/deployment conversion must remain distinct contracts.
