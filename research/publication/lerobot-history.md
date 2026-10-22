# Historical LeRobot chunk-anchor incident: source audit

This includes a pinned source audit and an executed historical processor reproduction. The current-version omitted-binding development cases remain separate explicit injections.

[PR4057](https://github.com/huggingface/lerobot/pull/4057) merged on 2026-09-22 as `7c98c1b2ce309f94e542bd4b9ee3fe176301edbf`; its actual sole parent is `240ea44cf9f887df944019cf46e18a0d496e1a56`. The PR API's base SHA is different and must not be substituted for the parent. The merged implementation adds policy queue-depth access, processor queue binding and `bind_relative_anchor`; use its actual code rather than earlier naming/design described in discussion.

The intended prediction-time anchor has evidence independent of the fix. The [parent action-representation guide](https://github.com/huggingface/lerobot/blob/240ea44cf9f887df944019cf46e18a0d496e1a56/docs/source/action_representations.mdx#L78-L95), blob `4c54098a5e06ccbb106775ce1eb5b55220e10f78`, specifies a common prediction-time reference across a chunk. The [parent statistics computation](https://github.com/huggingface/lerobot/blob/240ea44cf9f887df944019cf46e18a0d496e1a56/src/lerobot/datasets/compute_stats.py#L677-L681), blob `849df2a9e4320e14ee99a9c66140b4700de1e65a`, selects chunk-start states and broadcasts subtraction across each chunk's horizon. The merged regression tests cover queue-held anchors, refill and a pre/select/post loop (`tests/policies/test_relative_actions.py`, blob `9490ad528f437d7e29e250f2c2388fa851559e29`).

| Source under src/lerobot | Parent Git blob | Fix Git blob |
| --- | --- | --- |
| processor/relative_action_processor.py | `4c1af387b21e203bf4f364c7f98be3bf7c12f757` | `346acd1b4c4906087467985cd89a483de2635de2` |
| policies/pretrained.py | `dbada7b0ad4d0dec63e3f3eb8007dde1f5f8d435` | `1bbed0378a43310113fc661241e7995fc2e0c548` |
| processor/__init__.py | `f8c14a6b7376490e3924764f4fe1cdaf51e4f7fe` | `455c90d672970fdd5da36f3965bd32d9e278b930` |
| rollout/context.py | `c25cf05188c018afe9dabd6212ceb1c8bbff9899` | `7fc3c0d7e402af745a1a6dfcf64da0aab1b0162a` |
| rollout/inference/sync.py | `8b905df6192d28232140418dad93c01c87d556c5` | `30631b7939cdd68595c7140f86924746f649226a` |
| scripts/lerobot_eval.py | `6dcc5de189f47b0e4217893d36b284e4b0382e15` | `6aa586ed032a775cc17470a5256d61a116f3c73d` |

## Historical path and exclusions

The parent's normal synchronous rollout already rejects enabled relative-action policies during context construction. Its sync source documents the issue and says RTC is unaffected. It would be incorrect to claim that this guarded path silently emitted wrong commands.

The [parent public evaluation loop](https://github.com/huggingface/lerobot/blob/240ea44cf9f887df944019cf46e18a0d496e1a56/src/lerobot/scripts/lerobot_eval.py#L297-L302) does preprocess, queued selection and postprocessing every tick without anchor binding. The fix adds binding to that path. A weight-free reproduction should execute each complete installed historical revision's genuine processors with synthetic queue outputs and disclose that boundary substitution.

[Issue3312](https://github.com/huggingface/lerobot/issues/3312) remains a related report rather than a closed ACT integration fix. A project response says relative actions were implemented for PI0/PI05 and invites ACT integration. This audit does not establish historical or current trained ACT-relative support. The published ACT normalization cases use relative=False and make no such claim.

## Executed reproduction and independent expected values

For scalar observations 1, 5, 9 and offsets 0.1, 0.2, 0.3 from one queued chunk, the documented reference expects 1.1, 1.2, 1.3. The complete installed parent source actually produced 1.1, 5.2, 9.3; the complete installed fixed source produced 1.1, 1.2, 1.3. Both adopted a new chunk origin, cleared the cached anchor across reset of an interrupted fixture queue, and adopted the next episode's origin. Raw observations, six verified source blobs per phase, direct URLs and the worker hash are recorded under `validation/lerobot-history-001/`. This is retrospective development evidence, never a held-out experiment.

The reproduction used current locked CPU dependencies shared with separate historical source installations. It is a processor-subset compatibility environment rather than the original full training runtime. No extracted function fragments or trained inference were executed. A declared fixture queue supplies known offsets; it is not a real policy. Reproduction and data-only verification commands are in `integrations/lerobot-history/README.md`.
