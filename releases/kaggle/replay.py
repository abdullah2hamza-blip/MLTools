"""Offline saved-trace numerical replay, separate from the frozen Git/native gate."""

import argparse
import csv
import io
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import publication_analysis as analysis  # noqa: E402
import run_matched_study as matched  # noqa: E402
import run_policy_space_numeric as transport  # noqa: E402
from integrity import require, verify  # noqa: E402

PHASES = ("robotruth", "lerobot", "openpi", "lerobot_witness", "openpi_witness", "policy_space")
METHODS = ("compiled", "independent_pointwise", "strong_relational")
ASSETS = "research/publication/prepared/action-study-001"


def read(path):
    return json.loads(Path(path).read_bytes())


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def table(plan, numerical, workflows):
    out = io.StringIO(newline="")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(
        (
            "cell",
            "cluster",
            "condition",
            "derived",
            "compiled",
            "pointwise",
            "strong",
            "physical",
            "robotruth_identity_difference",
            "robotruth_certification",
            "policy_space_transport",
        )
    )
    for cell_id, cell in plan["cells"].items():
        for condition, row in numerical["cells"][cell_id].items():
            derived = condition not in cell["conditions"]
            source = "numeric_control" if derived else condition
            external = workflows["cells"][cell_id][source]["robotruth"]
            w.writerow(
                (
                    cell_id,
                    cell["cluster"],
                    condition,
                    str(derived).lower(),
                    *(row["numerical_methods"][k]["status"] for k in METHODS),
                    row["checkpoint_physical_status"],
                    str(external["known_statistics_identity_difference"]).lower(),
                    external["full_certification"],
                    workflows["transport"][cell_id][source]["policy_space_transport"],
                )
            )
    return out.getvalue().encode()


def products(folder, plan):
    phases = {p: read(folder / (p + ".native.json")) for p in PHASES}
    for framework in ("lerobot", "openpi"):
        analysis.validate_decoder(phases[framework], plan, ROOT / ASSETS, framework)
    decoder = analysis.merge_decoders(plan, {k: phases[k] for k in ("lerobot", "openpi")})
    numerical, selected = analysis.analyze(plan, decoder)
    fresh = {k: phases[k + "_witness"] for k in ("lerobot", "openpi")}
    for framework, data in fresh.items():
        analysis.validate_decoder(data, plan, ROOT / ASSETS, framework, selected[framework])
    witnesses = analysis.verify_witnesses(plan, decoder, fresh, numerical, selected)
    matched.validate_robotruth(phases["robotruth"], plan)
    shared = {"cells": {}}
    for cell_id, cell in plan["cells"].items():
        shared["cells"][cell_id] = {}
        base = matched.combined_statistics_hash(cell, cell["conditions"]["numeric_control"])
        for condition, entry in cell["conditions"].items():
            different = matched.combined_statistics_hash(cell, entry) != base
            shared["cells"][cell_id][condition] = {
                "numerical_methods": numerical["cells"][cell_id][condition]["numerical_methods"],
                "robotruth": {
                    "native_passed": phases["robotruth"]["cells"][cell_id][condition]["native_passed"],
                    "full_certification": "WITHHELD",
                    "applicability": "STATIC_SOURCE_ARTIFACT_COMPARISON",
                    "known_statistics_identity_difference": different,
                    "underlying_artifact_differences_counted": int(different),
                    "limits": (
                        "One union hash covers both normalizers; no separate observation-numeric finding"
                    ),
                },
                "same_numeric_statistics_as_control": entry["specification"] == cell["intended"],
                "physical_checkpoint_authority_status": "UNKNOWN",
            }
    transport.verify_contracts(folder / "contracts", plan)
    require(
        phases["policy_space"]["upstream_identity"] == read(ROOT / "integrations/baselines/sources.json"),
        "pinned native transport assertions",
    )
    transported = transport.analyze(phases["policy_space"], plan, decoder, shared)
    workflows = {
        "schema_version": 1,
        "role": plan["role"],
        "cells": shared["cells"],
        "transport": transported["cells"],
    }
    metrics = analysis.metrics(plan, numerical, witnesses, phases)
    return {
        "analysis.json": numerical,
        "witnesses.json": witnesses,
        "workflow-analysis.json": workflows,
        "metrics.json": metrics,
        "conditions.csv": table(plan, numerical, workflows),
    }


def replay(output):
    manifest = verify(ROOT)
    plan = read(ROOT / ASSETS / "plan.json")
    # The exported original plan and all statistics/configuration inputs retain exact bytes.
    from integrity import sha

    for name, digest in plan["file_sha256"].items():
        require(sha((ROOT / ASSETS / name).read_bytes()) == digest, "plan input hash: " + name)
    outputs = {}
    for run in ("prospective-001", "clean-reproduction-001"):
        folder = ROOT / "evidence" / run
        generated = products(folder, plan)
        for name, value in generated.items():
            expected = folder / name
            require(
                expected.read_bytes() == value if isinstance(value, bytes) else read(expected) == value,
                "recomputed saved product: " + run + "/" + name,
            )
        outputs[run] = generated
    a, b = outputs.values()
    for name in a:
        av, bv = a[name], b[name]
        if name == "metrics.json":
            av, bv = ({k: v for k, v in x.items() if k != "phase_costs"} for x in (av, bv))
        require(av == bv, "two saved acquisitions agree: " + name)
    require(
        not any(
            m in sys.modules for m in ("numpy", "torch", "lerobot", "openpi", "robotruth", "policy_space")
        ),
        "stdlib-only replay process",
    )
    output = Path(output).resolve()
    require(not output.is_relative_to(ROOT), "output must be outside read-only payload")
    require(not output.exists(), "output destination must be new")
    output.mkdir(parents=True)
    for name, value in a.items():
        (output / name).write_bytes(value if isinstance(value, bytes) else encoded(value))
    counts = a["metrics.json"]["counts"]
    summary = {
        "schema_version": 1,
        "status": "SAVED_TRACE_REPLAY_VERIFIED",
        "release_id": manifest["release_id"],
        "scope": (
            "Two saved acquisitions reanalyzed; "
            "no fresh native execution or Git ancestry verification on Kaggle"
        ),
        "counts": counts,
        "semantic_products_checked_per_run": len(a),
        "compiled_fault_fail": sum(
            c["methods"]["compiled"]["fault_fail"] for c in a["metrics.json"]["cells"].values()
        ),
        "compiled_control_fail": sum(
            c["methods"]["compiled"]["control_fail"] for c in a["metrics.json"]["cells"].values()
        ),
        "physical_unknown_rows": sum(
            row["checkpoint_physical_status"] == "UNKNOWN"
            for conditions in a["analysis.json"]["cells"].values()
            for row in conditions.values()
        ),
    }
    (output / "replay-summary.json").write_bytes(encoded(summary))
    verify(ROOT)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(replay(args.output), indent=2))
