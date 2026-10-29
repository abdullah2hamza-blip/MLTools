"""Reanalyse saved evidence and build the public claim-to-evidence record.

Uses only the standard library. No native acquisition, model, simulator or
network operation is requested. Run from any directory with a full checkout.
"""

# Narrative evidence descriptions are kept as single JSON string literals.
# ruff: noqa: E501

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ORIGINAL = "research/publication/runs/prospective-001"
CLEAN = "research/publication/runs/clean-reproduction-001"
PLAN = "research/publication/prepared/action-study-001/plan.json"
FREEZE = "7513755e9627098c3529c46b3ae9a056d76434a5"
METHODS = ("compiled", "independent_pointwise", "strong_relational")


def read(path):
    return json.loads((ROOT / path).read_bytes())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    started = time.perf_counter()
    verifications = {}
    targets = {
        "main_original": ("scripts/run_publication_study.py", ORIGINAL),
        "main_clean": ("scripts/run_publication_study.py", CLEAN),
        "stateful": ("scripts/compare_stateful_development.py", "validation/stateful-comparison-001"),
        "lerobot_history": ("scripts/reproduce_lerobot_history.py", "validation/lerobot-history-001"),
        "openpi_history": ("scripts/reproduce_openpi_history.py", "validation/openpi-history-001"),
        "panda": ("scripts/validate_panda.py", "validation/panda-software-001"),
        "panda_clean": ("scripts/validate_panda.py", "validation/panda-software-reproduction-001"),
    }
    for name, (script, folder) in targets.items():
        process = subprocess.run(
            [sys.executable, "-O", script, "--verify", folder],
            cwd=ROOT,
            capture_output=True,
            timeout=180,
            check=False,
        )
        verifications[name] = {
            "command": ["python", "-O", script, "--verify", folder],
            "returncode": process.returncode,
            "stdout": process.stdout.decode("utf-8", errors="replace").strip(),
            "stderr": process.stderr.decode("utf-8", errors="replace").strip(),
            "category": "reanalysis_of_saved_outputs",
        }
        require(process.returncode == 0, name + " saved-output verification failed")

    plan = read(PLAN)
    analysis = read(ORIGINAL + "/analysis.json")
    metrics = read(ORIGINAL + "/metrics.json")
    facts = read("research/publication/paper/generated/facts.json")
    witness = read(ORIGINAL + "/witnesses.json")
    summary = read(ORIGINAL + "/summary.json")
    decoders = {name: read(ORIGINAL + "/" + name + ".native.json") for name in ("lerobot", "openpi")}
    fresh = {name: read(ORIGINAL + "/" + name + "_witness.native.json") for name in decoders}
    primary = auxiliary = fresh_calls = 0
    independent_pointwise = Counter()
    for framework, native in decoders.items():
        for cell_id, conditions in native["cells"].items():
            cell = plan["cells"][cell_id]
            for capture in conditions.values():
                primary += sum(len(rows) for rows in capture["designs"].values())
                auxiliary += 4 * len(capture["roundtrips"])
                stats = cell["intended"]["statistics"]
                profile = cell["intended"]["profile"]
                differences = []
                for row in capture["designs"]["canonical"]:
                    expected = []
                    for axis, value in enumerate(row["input"]):
                        if "mean_std" in profile:
                            result = stats["mean"][axis] + value * stats["std"][axis]
                        else:
                            lo, hi = stats["q01"][axis], stats["q99"][axis]
                            width = hi - lo
                            width = width + 1e-6 if framework == "openpi" else width if width else 1e-8
                            result = lo + (value + 1) * width / 2
                        expected.append(result)
                    differences.append(
                        all(
                            math.isclose(actual, intended, rel_tol=1e-7, abs_tol=1e-7)
                            for actual, intended in zip(row["output"], expected, strict=True)
                        )
                    )
                independent_pointwise["PASS" if all(differences) else "FAIL"] += 1
        fresh_calls += sum(
            len(rows)
            for conditions in fresh[framework]["cells"].values()
            for capture in conditions.values()
            for rows in capture["designs"].values()
        )
    require((primary, auxiliary, fresh_calls) == (1792, 896, 16), "raw call counts")
    require(dict(independent_pointwise) == {"PASS": 4, "FAIL": 12}, "independent finite reference check")
    with (ROOT / ORIGINAL / "conditions.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 24, "table row count")
    csv_counts = dict(Counter(row["compiled"] for row in csv_rows))
    require(csv_counts == {"PASS": 4, "FAIL": 12, "UNKNOWN": 8}, "CSV numerical statuses")
    require(
        all(row["compiled"] == row["pointwise"] == row["strong"] for row in csv_rows), "CSV oracle parity"
    )
    require(all(row["physical"] == "UNKNOWN" for row in csv_rows), "physical authority scope")
    actual_csv = [row for row in csv_rows if row["derived"] == "false"]
    require(len(actual_csv) == 16, "actual condition denominator")
    require(
        sum(row["robotruth_identity_difference"] == "true" for row in actual_csv) == 12
        and all(row["robotruth_certification"] == "WITHHELD" for row in actual_csv),
        "separate statistics identity and certification findings",
    )
    witness_views = {
        method: sum(
            item["methods"][method]["native_calls_in_method_view"]
            for framework in witness.values()
            for conditions in framework.values()
            for item in conditions.values()
        )
        for method in METHODS
    }
    require(
        witness_views == {"compiled": 16, "independent_pointwise": 12, "strong_relational": 12},
        "witness views",
    )
    require(
        witness_views == facts["witness_native_calls_in_overlapping_method_views"], "figure witness facts"
    )
    for cell, conditions in analysis["cells"].items():
        actual = plan["cells"][cell]["conditions"]
        for method in METHODS:
            offset_status = [
                conditions[name]["ablations"]["offset_authority_erased"][method]["status"] for name in actual
            ]
            require(
                offset_status
                == (
                    ["UNKNOWN"] * 4
                    if method == "independent_pointwise"
                    else ["UNKNOWN", "UNKNOWN", "FAIL", "FAIL"]
                ),
                "offset ablation",
            )
            require(
                offset_status == facts["offset_authority_erasure"][cell][method], "authority figure facts"
            )
            require(
                all(
                    conditions[name]["ablations"]["independent_reference_erased"][method]["status"]
                    == "UNKNOWN"
                    for name in actual
                ),
                "complete reference erasure",
            )
        for name in actual:
            require(
                conditions[name]["roundtrips"]["candidate_self"] == "WITHIN_TOLERANCE", "self round trips"
            )
        for result in metrics["cells"][cell]["geometry_shared_oracle_counts"].values():
            require(set(result) == {"finite_status", "full_affine_request_status"}, "geometry status scope")
            require(all(value == {"PASS": 1, "FAIL": 3} for value in result.values()), "geometry parity")
    semantic_products = ("analysis.json", "witnesses.json", "workflow-analysis.json", "conditions.csv")
    require(
        all(
            (ROOT / ORIGINAL / name).read_bytes() == (ROOT / CLEAN / name).read_bytes()
            for name in semantic_products
        ),
        "clean saved semantic equality",
    )
    stateful = read("validation/stateful-comparison-001/summary.json")
    require(
        len(stateful["cases"]) == 8
        and all(
            row["original_matched_status"] == row["compiled_matched_status"]
            for row in stateful["cases"].values()
        ),
        "stateful parity",
    )
    require(
        set(row["original_native_actions"] for row in stateful["fresh_witnesses"].values()) == {1, 2},
        "original stateful witness range",
    )
    require(
        set(row["compiled_native_actions"] for row in stateful["fresh_witnesses"].values()) == {2, 4},
        "compiled stateful witness range",
    )

    registry = {}

    def file(path, kind):
        require((ROOT / path).is_file(), "missing evidence file: " + path)
        value = {"sha256": digest(path), "kind": kind}
        if path in registry:
            require(registry[path]["sha256"] == value["sha256"], "input changed during audit")
        else:
            registry[path] = value
        return path

    def group(folder, names, kind):
        return [file(folder + "/" + name, kind) for name in names]

    def glob(folder, pattern, kind):
        return [
            file(path.relative_to(ROOT).as_posix(), kind)
            for path in sorted((ROOT / folder).glob(pattern))
            if path.is_file()
        ]

    common_code = [
        file(path, "frozen_result_producing_code")
        for path in (
            "scripts/study_assets.py",
            "scripts/action_study_common.py",
            "scripts/action_designs.py",
            "scripts/action_study_worker.py",
            "scripts/matched_study_worker.py",
            "scripts/normalization_profiles.py",
            "scripts/run_action_study.py",
            "scripts/query_designs.py",
            "scripts/normalization_certificate.py",
            "scripts/publication_analysis.py",
            "scripts/publication_worker.py",
            "scripts/publication_protocol.py",
            "scripts/publication_assets.py",
            "scripts/publication_network.py",
            "scripts/run_publication_study.py",
            "scripts/policy_space_numeric.py",
            "scripts/policy_space_numeric_worker.py",
            "src/mltools/obligations.py",
            "src/mltools/research_oracles.py",
            "src/mltools/checker.py",
            "src/mltools/lerobot.py",
            "src/mltools/_openpi_worker.py",
            "src/mltools/report.py",
        )
    ]
    require(
        all(summary["source_file_sha256"][path] == registry[path]["sha256"] for path in common_code),
        "frozen producer bytes",
    )
    source_artifacts = []
    for cell in plan["cells"].values():
        path = file(cell["artifact_manifest"], "publisher_metadata_provenance")
        if path not in source_artifacts:
            source_artifacts.append(path)
    main_inputs = [
        file(PLAN, "frozen_plan_parameters_and_seeded_queries"),
        file("research/publication/protocol-freeze-001.json", "frozen_protocol"),
        *source_artifacts,
    ]
    main_raw = glob(ORIGINAL, "*.native.json", "prior_native_observations")
    main_raw += glob(ORIGINAL, "*.stdout.txt", "prior_raw_subprocess_stream")
    main_raw += glob(ORIGINAL, "*.stderr.txt", "prior_raw_subprocess_stream")
    administrative = [file(ORIGINAL + "/summary.json", "closed_prior_run_manifest")]
    for pattern in ("*.request.json", "*.receipt.json", "*.claim.json", "attempt-start.json"):
        administrative += glob(ORIGINAL, pattern, "prior_run_request_claim_or_receipt")
    derived = group(
        ORIGINAL,
        ("analysis.json", "metrics.json", "witnesses.json", "workflow-analysis.json", "conditions.csv"),
        "derived_saved_semantic_result",
    )
    reporting_code = file(
        "research/publication/reporting/build_results.py", "post_acquisition_reporting_code"
    )
    reporting = group(
        "research/publication/paper/generated",
        (
            "facts.json",
            "claims.tex",
            "outcomes.tex",
            "authority-ablation.pdf",
            "authority-ablation.png",
            "witness-cost.pdf",
            "witness-cost.png",
            "manifest.json",
        ),
        "generated_publication_result",
    )
    evidence_sets = {
        "main": {
            "observation_type": "native software processor responses to designed numeric inputs; no trained-policy inference",
            "inputs": main_inputs,
            "code": common_code,
            "configuration": {
                "plan": PLAN,
                "dimensions": [6, 6, 7, 7],
                "native_action_bounds": [-0.8, 0.8],
                "random_seeds": [7, 23, 101],
                "designs_per_condition": 8,
                "atol": 1e-7,
                "rtol": 1e-7,
                "native_replay_and_geometry_allowance": 1e-12,
                "execution_dtype": "float64",
                "device": "cpu",
                "source_statistics_dtype": "F32 controls; F64 injected derivatives",
            },
            "raw_outputs": main_raw,
            "administrative_evidence": administrative,
            "derived_outputs": derived,
            "code_revision": FREEZE,
            "publisher_versions": {
                cell_id: {
                    "model": cell["model"],
                    "revision": cell["revision"],
                    "framework": cell["framework"],
                }
                for cell_id, cell in plan["cells"].items()
            },
            "framework_versions": {
                framework: native["upstream_identity"]["revision"] for framework, native in decoders.items()
            },
            "environment_profiles": "research/publication/protocol-freeze-001.json#/native_environments; package inventories in native raw outputs",
            "verification": ["main_original", "main_clean"],
            "original_acquisition_commands": "Exact prior executable arguments and environment identities are retained in per-phase request/receipt records; reacquisition in a NEW isolated freeze checkout is documented in research/publication/reproduction/README.md.",
            "reporting_command": "python research/publication/reporting/build_results.py",
            "reporting_environment": file(
                "research/publication/reporting/requirements.lock", "reporting_dependency_lock"
            ),
            "reporting_code": reporting_code,
            "publication_outputs": reporting,
        }
    }
    for name, folder, scripts, inputs, outputs in (
        (
            "stateful",
            "validation/stateful-comparison-001",
            (
                "scripts/compare_stateful_development.py",
                "scripts/stateful_fixtures.py",
                "scripts/stateful_comparison_worker.py",
                "src/mltools/checker.py",
                "src/mltools/obligations.py",
            ),
            ("inputs/plan.json",),
            ("summary.json", "full.capture.json", "witness.capture.json", "timings.json"),
        ),
        (
            "lerobot_history",
            "validation/lerobot-history-001",
            ("scripts/reproduce_lerobot_history.py", "scripts/lerobot_history_worker.py"),
            (),
            ("summary.json", "before.json", "after.json"),
        ),
        (
            "openpi_history",
            "validation/openpi-history-001",
            ("scripts/reproduce_openpi_history.py", "scripts/openpi_history_worker.py"),
            ("contract.json",),
            (
                "summary.json",
                "before.trace.json",
                "after.trace.json",
                "before-witness.trace.json",
                "before.analysis.json",
                "after.analysis.json",
                "before-witness.analysis.json",
            ),
        ),
        (
            "panda",
            "validation/panda-software-001",
            (
                "scripts/panda_reference.py",
                "scripts/panda_worker.py",
                "scripts/validate_panda.py",
                "scripts/compare_panda_reproduction.py",
            ),
            ("inputs.json", "request.json"),
            ("summary.json", "capture.json", "analysis.json", "native.stdout.txt", "native.stderr.txt"),
        ),
    ):
        evidence_sets[name] = {
            "observation_type": "retrospective/development native software execution with explicitly authored fixtures; repeated measurements",
            "inputs": group(folder, inputs, "declared_fixture_and_configuration"),
            "code": [file(path, "frozen_development_or_retrospective_code") for path in scripts],
            "outputs": group(folder, outputs, "traceable_prior_software_run"),
            "random_seeds": [],
            "seed_scope": "Deterministic authored fixture; no random training or independent test subjects",
            "verification": [name],
            "code_revision_scope": "Exact producer SHA-256s in prior summaries/requests; current files also belong to the committed main freeze. Do not relabel the earlier runs as prospectively acquired under that later freeze.",
        }
    evidence_sets["stateful"]["inputs"] += glob(
        "validation/stateful-comparison-001/inputs", "*.bundle.json", "declared_fixture_and_configuration"
    )
    evidence_sets["lerobot_history"]["inputs"] += [
        file("uv.lock", "shared_current_locked_compatibility_dependencies"),
        file(
            "integrations/lerobot-history/README.md",
            "historical_revision_installation_and_compatibility_instructions",
        ),
        file("research/publication/lerobot-history.md", "independent_historical_authority_sources"),
    ]
    evidence_sets["openpi_history"]["inputs"] += [
        file("integrations/openpi-history/fixture.json", "declared_fixture_and_configuration"),
        file("integrations/openpi-history/sources.json", "pinned_historical_python_source_inventory"),
        file("integrations/openpi-history/requirements.lock", "pinned_dependency_profile"),
        file("research/publication/openpi-history.md", "independent_historical_authority_sources"),
    ]
    evidence_sets["panda"]["inputs"] += [
        file("integrations/panda/distribution.json", "upstream_wheel_and_source_provenance"),
        file("integrations/panda/requirements.lock", "pinned_dependency_profile"),
    ]
    evidence_sets["panda"]["clean_evidence"] = group(
        "validation/panda-software-reproduction-001",
        ("summary.json", "inputs.json", "request.json", "capture.json", "analysis.json"),
        "traceable_prior_clean_software_run",
    )
    evidence_sets["panda"]["verification"].append("panda_clean")
    evidence_sets["panda"]["random_seeds"] = [2026]
    evidence_sets["panda"]["seed_scope"] = (
        "Environment construction seed 2026; deterministic authored controller/rotation/gripper fixtures. "
        "No policy training, physics stepping or independently sampled episodes."
    )
    evidence_sets["panda"]["configuration"] = {
        "fixture": "validation/panda-software-001/inputs.json",
        "robosuite": "1.4.1",
        "mujoco": "2.3.7",
        "source_semantics_revision": "fbee5844ff5632f5b5698e204ec5357ca50be0df",
    }

    def claim(identifier, text, locator, sets, pointers, value=None, caveat=None):
        item = {
            "id": identifier,
            "claim": text,
            "manuscript_locator": locator,
            "evidence_sets": sets,
            "result_locations": pointers,
            "empirical_evidence_category": "supported_by_traceable_prior_run",
            "current_task_verification_category": "reproduced_during_this_task_by_saved_output_reanalysis",
            "full_native_experiment_repeated_during_this_task": False,
        }
        if value is not None:
            item["value"] = value
        if caveat:
            item["interpretation_limit"] = caveat
        return item

    claims = [
        claim(
            "main_units_and_budget",
            "Three source-artifact groups, four cells, sixteen actual conditions, eight derived views and eight equal-budget geometries; 1792 primary, 896 auxiliary, 16 shared fresh native calls",
            "Abstract; Sections 4.2, 4.3; Appendix B",
            ["main"],
            [
                PLAN + "#/cells",
                ORIGINAL + "/metrics.json#/counts",
                "research/publication/paper/generated/facts.json#/counts",
            ],
            {"primary": primary, "auxiliary": auxiliary, "fresh": fresh_calls},
            "Nested PI transport, conditions, seeds and queries are repeated/dependency-linked measurements, not independent checkpoint trials.",
        ),
        claim(
            "main_table_1",
            "Each of four cells: three injected FAIL, zero control FAIL, two derived UNKNOWN, six physical UNKNOWN",
            "Table 1, label tab:outcomes; Section 5.1",
            ["main"],
            [
                ORIGINAL + "/conditions.csv",
                ORIGINAL + "/metrics.json#/cells",
                "research/publication/paper/generated/outcomes.tex",
            ],
            csv_counts,
            "All numerical methods agree under full reference. No population sensitivity/specificity estimate.",
        ),
        claim(
            "main_numerical_parity",
            "Compiled, pointwise and strong relational methods agree on 16 actual conditions and 8 derived views; static numerical/reference roundtrip counts match",
            "Abstract; Sections 5.1, 5.2; Conclusion",
            ["main"],
            [
                ORIGINAL + "/analysis.json#/cells/*/*/numerical_methods",
                ORIGINAL + "/metrics.json#/cells/*/applicable_numeric_workflows",
            ],
            {"independently_recomputed_pointwise": dict(independent_pointwise), "oracle_disagreements": 0},
        ),
        claim(
            "physical_authority",
            "Physical authority UNKNOWN on all 24 actual/derived requested views",
            "Table 1; Sections 5.1, 6",
            ["main"],
            [
                ORIGINAL + "/conditions.csv#physical",
                ORIGINAL + "/analysis.json#/cells/*/*/checkpoint_physical_status",
            ],
            {"UNKNOWN": 24},
            "Metadata lacks independently calibrated physical authority; this is an evidence gap, not measured hardware failure.",
        ),
        claim(
            "figure_1_offset_ablation",
            "Offset erasure: compiled/strong retain 8 slope/composition FAIL; pointwise UNKNOWN on 16; pure offsets and controls UNKNOWN",
            "Figure 1, label fig:authority; Section 5.1",
            ["main"],
            [
                ORIGINAL + "/analysis.json#/cells/*/*/ablations/offset_authority_erased",
                "research/publication/paper/generated/facts.json#/offset_authority_erasure",
                "research/publication/paper/generated/authority-ablation.pdf",
            ],
            {"compiled_FAIL": 8, "strong_relational_FAIL": 8, "pointwise_UNKNOWN": 16},
            "Relation-based advantage over pointwise abstention only; no advantage over strong matched relational baseline.",
        ),
        claim(
            "complete_reference_ablation",
            "Complete independent-reference erasure makes all 3 methods UNKNOWN on 16 actual conditions",
            "Section 5.1",
            ["main"],
            [
                ORIGINAL + "/analysis.json#/cells/*/*/ablations/independent_reference_erased",
                "research/publication/paper/generated/facts.json#/full_reference_erased_status_counts",
            ],
            {method: {"UNKNOWN": 16} for method in METHODS},
        ),
        claim(
            "units_and_provenance_ablations",
            "Missing units or authority withholds complete numerical PASS; removing provenance keeps reduced numeric verdict but full binding UNKNOWN",
            "Sections 4.3, 5.1",
            ["main"],
            [
                ORIGINAL + "/analysis.json#/cells/*/output_units_erased",
                ORIGINAL + "/analysis.json#/cells/*/reference_authority_erased",
                ORIGINAL + "/analysis.json#/cells/*/*/ablations/provenance_erased",
            ],
            caveat="Derived control views reuse observations. Provenance-erasure status is deliberately assigned evidence semantics, not an observed native failure.",
        ),
        claim(
            "temporal_state_ablation_scope",
            "Temporal-state ablation is unavailable for action-only main observations and evaluated separately",
            "Section 4.3; Section 5.4, Appendix D",
            ["main", "stateful", "panda"],
            [
                ORIGINAL + "/analysis.json#/cells/*/*/ablations/state",
                "validation/stateful-comparison-001/summary.json",
                "validation/panda-software-001/analysis.json",
            ],
            caveat="Do not score missing state ports as main-study detections or misses.",
        ),
        claim(
            "geometry_and_artifact_sensitivity",
            "All 8 designs yield 1 control PASS/3 injected FAIL per cell for finite/full-affine statuses; leave-one-artifact-out macro fractions all 1",
            "Section 5.1",
            ["main"],
            [
                ORIGINAL + "/metrics.json#/cells/*/geometry_shared_oracle_counts",
                ORIGINAL + "/metrics.json#/leave_one_artifact_out_macro_detection_fraction",
            ],
            caveat="Descriptive selected-design sensitivity only; structural rank and source-justified affine class do not justify nonlinear or population generalization.",
        ),
        claim(
            "self_roundtrip",
            "Native candidate self roundtrips consistent in all 16 actual conditions, including 12 changed maps",
            "Abstract; Section 5.2",
            ["main"],
            [
                ORIGINAL + "/lerobot.native.json#/cells/*/*/roundtrips",
                ORIGINAL + "/openpi.native.json#/cells/*/*/roundtrips",
                ORIGINAL + "/analysis.json#/cells/*/*/roundtrips",
            ],
            {"WITHIN_TOLERANCE": 16},
            "Internal inverse consistency cannot establish conformity to an independent intended map.",
        ),
        claim(
            "robotruth_identity_and_certification",
            "Stock extraction/checking retains 12 statistics identity differences; certification WITHHELD on 16",
            "Section 5.2",
            ["main"],
            [
                ORIGINAL + "/robotruth.native.json",
                ORIGINAL + "/workflow-analysis.json",
                ORIGINAL + "/conditions.csv",
            ],
            {"identity_differences": 12, "certification_withheld": 16},
            "Native severity semantics; withheld certification is neither 16 detections nor 4 false alarms. Shared union statistics digest counts once.",
        ),
        claim(
            "policy_space_transport",
            "Stock custom-registry dispatcher/MessagePack/lifecycle accepts 1792 raw-vector replays and rejects 16 structural controls",
            "Section 5.2",
            ["main"],
            [
                ORIGINAL + "/policy_space.native.json",
                ORIGINAL + "/metrics.json#/counts",
                ORIGINAL + "/workflow-analysis.json",
            ],
            {"ACCEPT": 1792, "REJECT_structural_controls": 16},
            "Structural raw-vector transport only. Float32 wire conversion is distinct from float64 decoder conformance; no physical embodiment asserted.",
        ),
        claim(
            "figure_2_witnesses",
            "Compiled 16 versus pointwise 12 and strong 12 native calls in overlapping method views; 16 actual shared fresh acquisitions",
            "Figure 2, label fig:witness; Section 5.3",
            ["main"],
            [
                ORIGINAL + "/witnesses.json",
                ORIGINAL + "/lerobot_witness.native.json",
                ORIGINAL + "/openpi_witness.native.json",
                "research/publication/paper/generated/witness-cost.pdf",
            ],
            witness_views,
            "Canonical local failed relations only; no globally minimal causal explanation or human effort advantage.",
        ),
        claim(
            "measured_costs",
            "Original parent receipts total 234.15 seconds and native intervals 137.34 process CPU seconds",
            "Section 5.3",
            ["main"],
            [
                ORIGINAL + "/*.receipt.json#subprocess_wall_seconds",
                ORIGINAL + "/metrics.json#/phase_costs",
                "research/publication/paper/generated/facts.json",
            ],
            {
                "parent_wall_seconds": facts["primary_parent_subprocess_wall_seconds"],
                "native_interval_cpu_seconds": facts["primary_native_interval_cpu_seconds"],
            },
            "Different coverage; no subtraction-based speedup or isolated method ranking. Human effort not measured.",
        ),
        claim(
            "main_clean_reproduction",
            "Four semantic products byte-identical and metrics equal excluding phase costs in separately installed profiles",
            "Section 6; Appendix A",
            ["main"],
            [
                CLEAN + "/reproduction-comparison.json",
                CLEAN + "/summary.json",
                "research/publication/reproduction/environment-build-003/setup.json",
            ],
            {"byte_identical_products": list(semantic_products)},
            "Prior fresh native reproduction on same machine/cache; current task verifies saved outputs rather than rerunning native acquisition.",
        ),
        claim(
            "historical_incidents",
            "LeRobot PR4057 and OpenPI PR351 parent/fix processor conventions distinguish documented mismatch",
            "Section 5.4; Appendix D",
            ["lerobot_history", "openpi_history"],
            [
                "validation/lerobot-history-001/summary.json#/phases",
                "validation/openpi-history-001/summary.json#/outcomes",
            ],
            caveat="Two previously inspected retrospective incidents with authored fixtures; no newly discovered held-out faults, complete historical rollout, weights or task-loss estimate.",
        ),
        claim(
            "stateful_comparison",
            "Original checker/compiler agree on 8 matched SO101 conditions; original 1-2 versus compiler 2-4 native actions for retained fresh witnesses",
            "Section 5.4; Appendix D",
            ["stateful"],
            [
                "validation/stateful-comparison-001/summary.json#/cases",
                "validation/stateful-comparison-001/summary.json#/fresh_witnesses",
            ],
            {"matched_conditions": 8, "original_action_range": [1, 2], "compiled_action_range": [2, 4]},
            "One reused development cluster; original full report has extra scope, so runtimes not equal-scope.",
        ),
        claim(
            "panda_software",
            "Native robosuite/MuJoCo translation, goal setters, rotation, gripper/reset software executes; independent references detect wrong composition/omitted reset; affine UNKNOWN for unsupported paths",
            "Section 5.4; Appendix D",
            ["panda"],
            [
                "validation/panda-software-001/capture.json",
                "validation/panda-software-001/analysis.json",
                "validation/panda-software-reproduction-001/analysis.json",
                "validation/panda-reproduction-verification.json",
            ],
            caveat="Controlled source-grounded software fixtures, 114 affine queries, 2 goal setters, 4 orientation calls, 238 gripper calls; zero physics steps/torque calls/renders, no PI-checkpoint/Panda hardware mapping.",
        ),
        claim(
            "freeze_and_resource_limits",
            "Protocol protects 406 files; source/input closure committed before acquisition; sequential CPU workers have 180-second timeouts, observed 2 GiB RSS limit and 7200 native-interval CPU-second total limit",
            "Sections 4.1, 4.3; Appendix A",
            ["main"],
            [
                "research/publication/protocol-freeze-001.json",
                ORIGINAL + "/summary.json",
                ORIGINAL + "/*.request.json",
                ORIGINAL + "/*.receipt.json",
            ],
            caveat="Observed RSS checks are not OS enforcement. Audit hooks are not an OS sandbox; producer records are not execution attestation.",
        ),
    ]
    quality = claim(
        "prior_implementation_tests",
        "390 tests passed with 5 skips before freezing, including 40 publication wrapper tests",
        "Section 6",
        ["main"],
        [
            "validation/publication-wrapper-development-tests.txt",
            "validation/publication-wrapper-development-checks.json",
            "tests/test_publication_study.py",
        ],
        {"passed": 390, "skipped": 5, "included_wrapper_tests": 40},
        "Prior software validation, not a research detection result or a statement of current release test counts.",
    )
    quality["current_task_verification_category"] = "inspection_of_traceable_prior_test_log_and_code"
    claims.append(quality)
    for path in (
        "validation/publication-wrapper-development-tests.txt",
        "validation/publication-wrapper-development-checks.json",
        "tests/test_publication_study.py",
        "research/publication/reproduction/machine.json",
    ):
        file(path, "prior_implementation_or_environment_evidence")
    for path in (
        CLEAN + "/summary.json",
        CLEAN + "/reproduction-comparison.json",
        "research/publication/reproduction/environment-build-003/setup.json",
        "validation/panda-reproduction-verification.json",
    ):
        file(path, "prior_reproduction_proof")
    for path in semantic_products:
        file(CLEAN + "/" + path, "prior_clean_semantic_result")
    verification = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "status": "SAVED_EVIDENCE_REANALYSIS_PASSED",
        "python_version": platform.python_version(),
        "audit_code_sha256": digest(Path(__file__).relative_to(ROOT).as_posix()),
        "commands": verifications,
        "independently_recomputed": {
            "primary_calls": primary,
            "auxiliary_calls": auxiliary,
            "fresh_calls": fresh_calls,
            "finite_pointwise_counts": dict(independent_pointwise),
            "csv_counts": csv_counts,
            "witness_method_calls": witness_views,
            "byte_identical_main_semantic_products": list(semantic_products),
            "offset_reference_ablations_and_geometry_counts_match": True,
            "stateful_parity_and_witness_ranges_match": True,
        },
        "full_native_acquisition_repeated": False,
        "wall_seconds": time.perf_counter() - started,
        "scope": "Code inspection, saved-evidence integrity verification and independent reanalysis; not a new native experimental reproduction or human peer review.",
    }
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "claim-audit-checks.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    file("research/publication/release/claim-audit-checks.json", "current_task_saved_output_reanalysis")
    file(Path(__file__).relative_to(ROOT).as_posix(), "current_task_claim_audit_code")
    result = {
        "schema_version": 1,
        "audited_at_utc": verification["checked_at_utc"],
        "scope": "Headline claims, one table and two figures, ablations and separate software/historical strata in current manuscript. Per-file identities below anchor actual inputs, code and outputs. File globs in result_locations are selectors within the explicitly enumerated evidence sets.",
        "research_code_revision": FREEZE,
        "release_documentation_revision": "Final release commit/tag is recorded by the release manifest; it must not replace the earlier frozen research-code revision.",
        "manuscript_sources": [
            "research/publication/paper/main.tex",
            "research/publication/paper/article.tex",
            "research/publication/paper/appendix.tex",
            "research/publication/paper/references.bib",
        ],
        "verification_categories": {
            "supported_by_traceable_prior_run": "Native observations were acquired earlier with saved producer/package/input/source evidence.",
            "reproduced_during_this_task_by_saved_output_reanalysis": "Current data-only verifiers and direct independent counts/reference calculations executed; no native observations reacquired.",
            "unverified": "Physical calibration, learned-policy performance, hardware safety, population performance, human effort and publication acceptance are not established.",
        },
        "current_reanalysis": "research/publication/release/claim-audit-checks.json",
        "evidence_sets": evidence_sets,
        "claims": claims,
        "formal_statement": {
            "location": "Proposition 1; Section 3.2",
            "category": "code_and_mathematical_inspection",
            "basis": "If X has full column rank and both maps are independently justified affine maps on the same coordinates, X*d=0 implies d=0. Finite matches do not establish affine class membership.",
            "finite_precision_scope": "Component radii depend on the declared measurement model; empirical 1e-12 allowances are not universal bounds on upstream arithmetic.",
        },
        "research_design_audit": {
            "training_split": "N/A: no training, learned inference or fitted preprocessing. References come from independently parsed source metadata and explicit runtime formulas, not deployment responses.",
            "selection_and_leakage": "Metadata and injection recipes were inspected before committed acquisition; selected native outcomes were unseen at selected revisions, not blinded/ancestry-disjoint/unseen root causes.",
            "duplicates_and_independence": "Canonical/manual endpoints overlap, PI transport shares same artifact, random seeds/interventions/queries are repeated measures, artifacts share base/corpus ancestry, clean runs repeat same inputs. No independent sample count is inferred.",
            "baselines": "Equal canonical query/reference access for three numerical oracles; all 8 geometries get 2n+1 native queries. Native workflows keep their distinct scopes. Unavailable rollout/physical/response workflows receive no scored wins/misses.",
            "statistics": "Finite counts and descriptive equal-artifact-weight leave-one-out sensitivity only; no population CI, p-values, multiple-testing or accuracy generalization claims.",
            "effect_size": "Large declared offset/width injections; no empirical sensitivity curve for naturally occurring small drift. Witness/cost findings are negative for superiority; human effort unmeasured.",
            "scientific_scope": "Engineering/reproducible evidence-comparison contribution; no new affine theorem or new policy/safety discovery asserted.",
            "reviewer_kind": "AI-assisted technical review; no independent human peer review or human-author attestation.",
        },
        "literature_audit": {
            "register": file(
                "research/publication/literature.json", "versioned_bounded_primary_source_audit"
            ),
            "scope": "Focused primary-source audit, not systematic/exhaustive. Principal citations previously verified against author/publisher/arXiv sources; immutable versions retained; Zeller/Bauer DOIs added. Tai ExecSpec, XPolicyLab, metamorphic/differential testing, partial-evidence tools and reduction credited.",
            "primary_sources": [
                "https://arxiv.org/html/2606.03724v1",
                "https://arxiv.org/abs/2608.09892v1",
                "https://ojs.aaai.org/index.php/ICAPS/article/view/27185",
                "https://www.cs.purdue.edu/homes/lintan/publications/cradle-icse19.pdf",
                "https://jiannanwang.github.io/files/eagle-icse22.pdf",
                "https://link.springer.com/chapter/10.1007/978-3-540-78917-8_1",
                "https://doi.org/10.1109/32.988498",
                "https://doi.org/10.1145/2000799.2000800",
            ],
        },
        "files": registry,
    }
    (HERE / "claim-evidence.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"Verified {len(claims)} claim groups, 1 table, 2 figures; {len(registry)} evidence/code files; no native acquisition"
    )


if __name__ == "__main__":
    main()
