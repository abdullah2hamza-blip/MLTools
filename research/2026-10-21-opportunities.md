# ML tooling opportunities: evidence and decision

Research date: 21 October 2026. This memo separates observed capabilities and reports from proposed products and commercial hypotheses. It is a decision document, not evidence of customer demand or a global novelty claim.

## Recommendation

Pursue **executable interface contracts for robot policies**, starting with a conformance gate between a checkpoint's training pipeline and its deployment adapter on the same embodiment. The product answers: **does this deployed pipeline still give this policy the observations it was trained to interpret, and execute its outputs with the intended physical meaning?**

The first user is a robotics ML engineer integrating or updating a VLA or imitation-learning policy. The buyer is the robotics company's ML/platform lead, responsible for integration time and failed deployment experiments. The immediate trigger is a new checkpoint, framework version, camera configuration, normalization artifact, action chunking implementation, or controller adapter.

The initial product should produce a counterexample with an exact artifact/configuration identity and an executable regression case. Its value should be measured in avoided debugging and integration work. A generic dataset score or another visualization interface would not establish this wedge.

The most useful technical insight from this investigation is that **shape checks and even successful encode/decode round trips do not establish interface compatibility**. Two transforms can share the same wrong convention. The independent reference must be the checkpoint's declared semantics and the controller's declared command semantics, supported by representative recorded fixtures.

The proposed novelty is narrower than policy contracts or deployment validation: an independently specified, stateful comparison of what the pipeline actually does against the checkpoint's training semantics and the controller's command semantics. A late competition check found close products, including Policy Space and Tether, so a broad "policy release infrastructure" pitch is already occupied. The algorithms underneath include established type checking, differential testing and metamorphic testing. They should not be presented as new inventions. Public evidence supports investigating the remaining comparison gap; it does not establish a world-first product or a defensible company.

## Selection criteria

I looked for a recurring expensive failure, an identifiable user and budget owner, an independently checkable result, an entry point a small team could build, and an expansion path beyond a narrow patch. I treated direct existing implementations as disconfirming evidence. GitHub activity, papers and vendor claims establish technical context; none establishes willingness to pay.

The research covered agent replay/learning, RL environments and reward checking, numerical/kernel validation, train/rollout parity, robot data quality/synchronization, adaptive evaluation, and agent consistency testing. Two product theses remain worth targeted validation. The first is the recommended project; the second has substantially more competitive overlap.

| Candidate | Concrete entry point | Why it could support a company | Principal risk | Decision |
| --- | --- | --- | --- | --- |
| Robot-policy interface conformance | Fail a deployment change when an observation/action keeps its shape but changes meaning | Repeated checkpoint, framework and hardware-adapter integrations; potential release infrastructure and compatibility ecosystem | Becomes an upstream validation feature or bespoke integration service | Pursue a technical and customer validation sprint |
| Shared-state race testing for existing business agents | Find a bad interleaving involving a real customer's agent and multiple actors modifying the same records | Agents moving from advice to consequential business writes need repeatable release evidence | Close open-source simulators and established deterministic testing vendors already exist; service fidelity is expensive | Reserve candidate; require an integration advantage before building |

These are venture hypotheses, not two businesses with validated markets. I would concentrate effort on the first instead of building both.

## Why the recommended problem is concrete

### Same-shaped cameras can be silently swapped

LeRobot's 21 October commit corrects the RoboCasa camera mapping in documentation and CI. The updated explanation ties the mapping to the checkpoint's training configuration and describes why exchanging views can be silently accepted. This is unusually direct evidence for checking semantic camera identity against a training artifact. The specific error was corrected upstream; the recurring failure class is the opportunity. [Commit 5517a326](https://github.com/huggingface/lerobot/commit/5517a326eeb9b457cfe07d59b9bfdf291799c8d9).

### A correct-looking action can use the wrong state

[LeRobot issue 3312](https://github.com/huggingface/lerobot/issues/3312) describes relative actions from one predicted chunk being reconstructed against changing observations. It is a stateful interface concern: the conversion's reference state matters over the lifetime of the chunk. Current source contains chunk-anchor preservation logic, so an open issue alone is not proof that today's implementation remains broken. The product must check the actual pinned combination of policy, processor and execution loop. [Current processor snapshot](https://github.com/huggingface/lerobot/blob/5517a326eeb9b457cfe07d59b9bfdf291799c8d9/src/lerobot/processor/relative_action_processor.py).

### Joint identity can get lost at a representation boundary

[LeRobot issue 5011](https://github.com/huggingface/lerobot/issues/5011), filed 19 October, supplies a small reproduction concerning observation insertion order and motor order. Source inspected at commit `5517a326` still constructs the relevant IK input vector by iterating observation entries. This supports a permutation-based conformance test; I did not execute the upstream reproduction or establish its downstream physical consequences. [Inspected source](https://github.com/huggingface/lerobot/blob/5517a326eeb9b457cfe07d59b9bfdf291799c8d9/src/lerobot/robots/so_follower/robot_kinematic_processor.py#L323).

### Existing action schemas leave composition mistakes possible

NVIDIA's GR00T configuration guide already has action representation/type/format metadata. It also explains that positional misalignment between action configuration entries and modality keys can silently apply the wrong representation. It documents dependencies between temporal configuration and normalization statistics. This is evidence to build on existing schemas, rather than invent an incompatible replacement. [Pinned GR00T guide](https://github.com/NVIDIA/Isaac-GR00T/blob/dac54b313160c201c450f97df0fd74ebd379c892/getting_started/data_config.md).

### Reported deployment failures need disciplined interpretation

[GR00T issue 190](https://github.com/NVIDIA/Isaac-GR00T/issues/190) reports a gap between offline evaluation and on-robot behavior, with contributor suggestions involving normalization and execution settings. It supports the existence of a debugging workflow, not a confirmed root cause or a measured benefit for this proposed tool.

One tempting example was weaker after checking the discussion: the feature mismatch in [LeRobot issue 2259](https://github.com/huggingface/lerobot/issues/2259) was traced to a stale dataset cache and the discussion was closed. That is provenance evidence, not an unresolved proof of action semantics being broken. A product must distinguish those diagnoses.

## Closest alternatives and the remaining distinction

| Existing project | Capability observed in its public material | Implication for our scope |
| --- | --- | --- |
| [LeRobot processors](https://github.com/huggingface/lerobot/tree/5517a326eeb9b457cfe07d59b9bfdf291799c8d9/src/lerobot/processor) | Transform pipelines, normalization, relative/absolute actions and robot-policy bridges | Integrate these components. A new transform library alone is not differentiated. |
| [OpenPI transforms and tests](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms_test.py) | Existing tests for action transforms and data repacking | Show additional failures at composed training/deployment boundaries, beyond the existing component checks. |
| [GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/dac54b313160c201c450f97df0fd74ebd379c892) | Embodiment and modality configuration with action interpretation | Import its semantics; test their use against the controller and training artifacts. |
| [Policy Space](https://github.com/X-Square-Robot/policy-space/tree/4bdd7ed9b97131878fb745018ad898dfa06cc999) | Versioned observation/action/embodiment contracts, adaptation and an episode lifecycle | Contracts themselves are direct prior art. Distinguish declared compatibility from independently testing adapter behavior. |
| [Robot Policy Preflight](https://github.com/gbesse/robot-policy-preflight/tree/09137467bbf988bf5f4cc7418f8e300eaa6dd191) | Static comparison of dataset, policy and robot contracts | A schema/units checker is already implemented; it is not a novel product proposal. |
| [Tether](https://github.com/FastCrest/tether/tree/dcbcbb5a669fa18ae07b1c22d17bd5bd386a33da) | Export parity, deployment proof packets, action-execution checks and release decisions | This is the strongest adjacent product. Require evidence of additional semantic failures on actual integrations; do not claim release evidence or deployment validation is unoccupied. |
| [embodied-sync](https://github.com/anicut-ai/embodied-sync/tree/c5ade42a61ab595e55e175f8d80272582f4d8df0) | Dataset/live temporal alignment, multiple clocks, causal observations, calibration, replay and provenance | This directly occupies the basic timing/causal-synchronization idea. Use it as an input or integration where appropriate. |
| [Calibra](https://github.com/Calibra-Robotics/Calibra) | Dataset integrity, quality characterization, selection and export | Dataset quality and a passable policy interface are separate questions. Do not compete on another quality score. |
| [Forge](https://github.com/arpitg1304/forge) | Format conversion, data inspection, quality tooling and a catalog | Reuse conversion paths where suitable. File compatibility does not by itself answer checkpoint-to-controller compatibility. |
| [Rerun](https://rerun.io/) and [Foxglove](https://foxglove.dev/) | Established robot data visualization and debugging platforms | Export counterexamples to their workflows; do not spend the first months rebuilding a viewer. |
| [UMI](https://umi-gripper.github.io/) | Existing latency matching and action-interface design, including a latency ablation | Temporal/action interface design is established prior art, not our invention. |

This is a bounded comparison of published capabilities, documentation and selected source. I have not established the absence of equivalent internal tools. The practical novelty test is whether this layer finds a reproducible failure that these projects' existing checks miss, on a customer's actual integration, with less effort than adding a local test.

## What to build first

The first milestone is a **policy release conformance bundle** for one embodiment and one supported pair of training/deployment paths. A bundle contains:

1. A checkpoint and transform identity: weight/config digests, normalization statistics, dataset revision and the relevant training configuration. Missing identity remains explicitly unknown.
2. A physical interface contract: named joints and cameras, units, coordinate conventions, absolute/relative semantics, gripper convention, action horizon and chunk reference state. Import existing metadata and ask the integrator only for missing semantics.
3. Recorded boundary fixtures: raw observations, expected model-facing inputs, representative model-output actions, expected controller commands, reset events and chunk boundaries.
4. An executable adapter for each side. First support explicitly registered transforms; custom opaque transforms require a callback and meaningful reference fixtures.
5. A small set of stateful and metamorphic checks, followed by a report that identifies the first inconsistent boundary and writes a deterministic reproducer.

Initial checks should cover camera-role changes, joint permutations, normalization artifact mismatch, relative-action anchoring over a multi-step chunk, gripper exclusion from relative conversion, and state reset between episodes. Frame/rotation conversions and opaque GPU model equivalence should wait until there is real demand and an adequate reference.

Do not require model generation for the first checks. Substitute representative known model outputs to isolate postprocessing from model competence. Later, frozen-noise or otherwise appropriately controlled model runs can extend coverage. Replaying a single sampled trajectory does not prove a stochastic policy equivalent under all conditions.

For a chunk defined relative to state `s0`, expected reconstruction is `target[k] = s0 + offset[k]` on the declared additive dimensions. A moving-state implementation can still invert each individual transform perfectly while giving a different target sequence. Rotations need their own explicit group convention; the scalar formula must not be generalized to arbitrary poses.

The initial contract should reject incompatibility or report insufficient evidence. It should not silently rewrite a deployment. Once customers trust the checks, generating adapters from the same contract is a plausible expansion.

## Feasibility probe completed in this repository

[policy_interface.py](probes/policy_interface.py) constructs four faulty configurations and four correct controls. All faulty configurations pass their weaker local checks; the independently specified interfaces reject them. The controls pass. [Recorded output](probes/results.json).

| Constructed case | Weak check that passes | Independent comparison |
| --- | --- | --- |
| Two camera roles exchanged | Keys, image dimensions and types | Training camera identity versus deployment identity |
| Consistently permuted joint adapter | Encode/decode round trip | Checkpoint's declared joint order |
| Moving reference for a predicted chunk | Relative/absolute round trip at each tick | Full chunk against its originating state |
| Wrong normalization artifact used consistently | Normalize/denormalize round trip | Output interpreted with training statistics |

These examples establish why an independent contract can add information. They are deliberately constructed specification examples, not a detection benchmark, an upstream reproduction, an ML result, or evidence of real robot performance. The contract itself can be wrong. The hardest engineering work remains extracting trustworthy contracts and testing real integrations cheaply.

### Executed comparison with two real baseline modules

[compare_baselines.py](probes/compare_baselines.py) imports exact source snapshots only after verifying their Git blob identities. On a constructed runtime joint permutation and a correct control, the real static preflight module and Tether's action-execution certificate module both pass. An independent named-command comparison distinguishes the two. [Results and pinned source URLs](probes/baseline-results.json).

The Tether module checks numeric execution evidence; the preflight module compares declarations. This experiment shows a boundary between their checks and actual adapter semantics. It is not a failure of either module's stated contract, and it does not demonstrate that the full Tether product misses the case. Policy Space was inspected but not executed. This modest result is more informative than presenting the earlier four constructed examples as competitive validation.

To reproduce, obtain the two source files at the commit URLs in `baseline-results.json`, name them `tether_action_execution_cert.py` and `robot_policy_preflight.py` in a separate directory, then run:

```powershell
uv run python research/probes/compare_baselines.py --source-dir PATH_TO_SNAPSHOTS
```

## Why this could become a substantial business

The proposed commercial loop is: a robotics team repeatedly integrates checkpoints and runtime changes; a conformance failure shortens a costly debugging cycle; the bundle becomes a release artifact; the platform accumulates compatibility fixtures and adapters. The expansion would be organization-wide policy qualification, dataset/checkpoint procurement checks, framework migrations and controller integration testing.

The potential defensibility is a high-quality compatibility and counterexample corpus, maintained integrations, and position in a release workflow. A schema, a handful of validators, or the word "compiler" is not a moat. Tether and Policy Space could add these checks; unless extraction, integration or the corpus produces a substantial advantage, this belongs as an upstream contribution rather than a separate company. Private customer fixtures cannot be pooled without authorization; reuse would need public fixtures, permission, or carefully abstracted failure classes.

Start with robotics teams already deploying learned manipulation policies, rather than hobbyists or teams still collecting their first demonstrations. Interview the engineer who owns integration and the lead who pays for repeated failures. Ask for the last concrete incident, its time cost, existing test coverage and actual budget owner. A research paper or a GitHub complaint is not a substitute for this.

As a pricing experiment, test a fixed-scope paid integration audit, then an annual release-infrastructure subscription. No price or market-size estimate has been validated in this research. The crucial distinction is whether several customers need substantially the same contract/checking machinery; if every delivery is a new consulting project, the software-company thesis is weaker.

## Four-week decision sprint

The following are proposed gates, not achieved results or industry thresholds.

| Period | Work | Evidence needed to continue |
| --- | --- | --- |
| Week 1 | Review incidents with 8–12 deploying teams; obtain permission for at least three concrete integration examples | Repeated failure class, an owner who feels the cost, and access to artifacts sufficient for a meaningful test |
| Week 2 | Implement one public framework-pair adapter and a mutation corpus with clean controls; compare against current upstream checks, Policy Space and Tether's complete applicable workflow | Correctly distinguish injected faults from intended checkpoint-specific conventions; capture what the baselines already detect |
| Week 3 | Run on the permitted integrations; reproduce a historical incident before seeing its diagnosis where feasible | At least one useful counterexample that existing checks missed; setup time and false alarms recorded |
| Week 4 | Let customers run the gate themselves and test payment | Repeated use, low integration effort and paid commitments for a recurring workflow |

Kill or substantially change the direction if there are no repeated costly incidents, customers will not provide usable artifacts, upstream checks already catch the cases, false alarms require constant expert intervention, or customers want only one-time consulting. Any claim of improved task success needs controlled policy evaluation; interface checks alone cannot justify it.

## Reserve candidate: race testing existing business agents

The specific potential gap is a low-friction tester for an existing customer's agent operating against shared business state. Example: an agent checks an order's refund eligibility, a human changes the order, then the agent writes using stale assumptions. Another is an external write that commits but loses its response, followed by a retry. The buyer would be an agent platform or product team enabling business writes.

The tester would control tool-call interleavings and service behavior in an isolated test environment, check business invariants, and reduce a failing history to a replay. A changed observation must allow the real agent to make a new decision. Reusing an old language-model response after its input changes is not a valid counterfactual.

The competitive obstacle is substantial. [Deterministic Agent Simulator](https://github.com/rouviere-german/deterministic-agent-simulator/tree/bc0dfc843e9491271e02956fa1e1fe02de7b08f2) already describes seeded worlds, post-commit timeouts, invariants, shrinking and generated regression tests. Its documented closed loop uses a seeded stand-in for the language model. That leaves a possible integration-oriented distinction, not an untouched invention.

[agentic-chaos](https://github.com/DeepAgentLabs/agentic-chaos) and [Faultkit](https://github.com/faultkit/faultkit) already offer agent fault injection and contracts/invariants. [Antithesis](https://antithesis.com/product/) offers powerful deterministic exploration for systems, and [Jepsen](https://jepsen.io/analyses) is established prior art for checking concurrent histories under failures. A new product must win on connecting real agent decisions to faithful business-service semantics, not on claiming to invent deterministic testing.

[Temporal's activity documentation](https://docs.temporal.io/activity-definition) makes idempotency an explicit application concern. A tester must not claim universal exactly-once effects over arbitrary external APIs. Faithful tool/service models and a clear treatment of unknown outcomes are necessary.

Before choosing this direction, require two customer integrations achievable in roughly a day each and a reproducible failure absent from their current evals. If that needs a bespoke digital twin for every service, distribution and margins become the central problem. I rank it below policy conformance because the direct overlap is stronger.

## Attractive directions rejected in their broad form

| Direction | Disconfirming evidence | Remaining condition that could change the decision |
| --- | --- | --- |
| Production traces into continually improving agents | [oncell](https://oncell.ai/) and [Stash](https://github.com/Fergana-Labs/stash) already publicly position around this loop | A proprietary outcome signal or underserved vertical, not a generic trace pipeline |
| RL environment/reward QA | [YC's current RL directory](https://www.ycombinator.com/companies/industry/reinforcement-learning) includes direct environment and reward-hacking work | A genuinely different verifiable environment class and access to buyers/data |
| Generic GPU-kernel correctness CI | [gpuemu](https://github.com/Skelf-Research/gpuemu), [gpucheck](https://github.com/Akasxh/gpucheck) and [TritonFuzz](https://github.com/JueonPark/TritonFuzz) already describe the relevant testing approaches | An uncovered failure class validated against those baselines |
| FP8 debugging dashboard | [Transformer Engine's debug tools](https://docs.nvidia.com/deeplearning/transformer-engine/debug.html) already expose numerical inspection and intervention | Demonstrated cross-stack root-cause automation unavailable in the existing tooling |
| Generic train/rollout parity checker | [AMD's logprob debugging workflow](https://rocm.blogs.amd.com/software-tools-optimization/logprob-debug/README.html) and [vLLM's October consistency work](https://vllm-project.github.io/2026/10/14/train-inference-consistency.html) are direct, recent prior art | A materially distinct, measured benefit beyond another discrepancy report |
| Basic robot-data synchronization and quality | embodied-sync, Calibra, Forge, rdq and Kalanos have explicit overlapping capabilities | The independent end-to-end interface semantics proposed above, supported by real customer failures |
| Statistical significance for agent evals | [holdout-evals](https://pypi.org/project/holdout-evals/) already advertises paired testing, multiple-comparison handling and memorization checks; [LangChain](https://www.langchain.com/blog/better-harness-a-recipe-for-harness-hill-climbing-with-evals) already discusses holdouts for optimization | A proven buyer for a persistent adaptive-feedback control layer; basic significance checks alone are too small a distinction |

The rejected products' advertised capabilities were not independently benchmarked here. Their existence is enough to reject a "nobody does this" pitch; it is not evidence they solve every customer's problem.

## Research limits and completion status

Completed: broad candidate search, primary-source competition checks, inspection of selected current source, issue/discussion cross-checks, two concrete product theses, a preferred direction, synthetic feasibility examples and falsifiable next gates.

Unproven: exhaustive novelty, adoption, willingness to pay, production detection accuracy, integration effort, improved robot task success, or venture-scale outcomes. No customers were contacted, no paid resources were provisioned, and no robot was controlled. The next work is a validation sprint around the selected interface-conformance thesis, not a claim that desk research has already established a winning company.
