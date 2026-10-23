"""Regenerate publication tables, figures and claim facts from verified captures."""

import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "research/publication/paper/generated"
METHODS = ("compiled", "independent_pointwise", "strong_relational")
LABELS = {
    "act_boccia_lerobot": "ACT / LeRobot",
    "smolvla_orange_lerobot": "SmolVLA / LeRobot",
    "pi05_libero_lerobot": "PI05 / LeRobot",
    "pi05_libero_openpi_transport": "PI05 / OpenPI transport",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
    from run_publication_study import verify

    original = ROOT / "research/publication/runs/prospective-001"
    clean = ROOT / "research/publication/runs/clean-reproduction-001"
    verify(original)
    verify(clean)
    metrics = read(original / "metrics.json")
    analysis = read(original / "analysis.json")
    witnesses = read(original / "witnesses.json")

    def require(condition):
        if not condition:
            raise ValueError("publication data scope")

    require(metrics["role"] == "prospective" and metrics["final_evaluation"] is True)
    require(set(metrics["cells"]) == set(LABELS))
    cells = list(metrics["cells"])
    methods = {
        method: {
            field: sum(c["methods"][method][field] for c in metrics["cells"].values())
            for field in ("fault_fail", "declared_faults", "control_fail", "numeric_controls")
        }
        for method in METHODS
    }
    witness_calls = {
        method: sum(
            row["methods"][method]["native_calls_in_method_view"]
            for framework in witnesses.values()
            for cell in framework.values()
            for row in cell.values()
        )
        for method in METHODS
    }
    conditions = (
        "numeric_control",
        "offset_intervention",
        "slope_intervention",
        "fixed_offset_slope_composition",
    )
    ablation = {
        cell: {
            method: [
                analysis["cells"][cell][c]["ablations"]["offset_authority_erased"][method]["status"]
                for c in conditions
            ]
            for method in METHODS
        }
        for cell in cells
    }
    facts = {
        "schema_version": 1,
        "scope": "Finite dependency-linked conditions; no population estimates or independent mutation count",
        "counts": metrics["counts"],
        "methods": methods,
        "witness_native_calls_in_overlapping_method_views": witness_calls,
        "offset_authority_erasure": ablation,
        "paired_oracle_disagreements": sum(
            c["paired_oracle_verdict_disagreements"] for c in metrics["cells"].values()
        ),
        "full_reference_erased_status_counts": {
            method: {
                status: sum(
                    analysis["cells"][cell][c]["ablations"]["independent_reference_erased"][method]["status"]
                    == status
                    for cell in cells
                    for c in conditions
                )
                for status in ("PASS", "FAIL", "UNKNOWN")
            }
            for method in METHODS
        },
        "query_geometry_counts": {
            cell: c["geometry_shared_oracle_counts"] for cell, c in metrics["cells"].items()
        },
        "primary_parent_subprocess_wall_seconds": sum(
            read(p)["subprocess_wall_seconds"] for p in original.glob("*.receipt.json")
        ),
        "primary_native_interval_cpu_seconds": sum(
            c["process_cpu_seconds"] for c in metrics["phase_costs"].values()
        ),
        "reproduction": read(clean / "reproduction-comparison.json"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT / "facts.json", facts)
    macros = {
        "MainPrimary": metrics["counts"]["primary_decoder_calls"],
        "MainAuxiliary": metrics["counts"]["auxiliary_native_calls"],
        "MainFresh": metrics["counts"]["actual_shared_fresh_decoder_calls"],
        "MainCells": metrics["counts"]["execution_cells"],
        "MainGroups": metrics["counts"]["artifact_groups"],
        "MainConditions": metrics["counts"]["actual_conditions"],
        "MainDerived": metrics["counts"]["derived_views"],
        "CompiledFaultFail": methods["compiled"]["fault_fail"],
        "DeclaredFaults": methods["compiled"]["declared_faults"],
        "ControlFail": methods["compiled"]["control_fail"],
        "NumericControls": methods["compiled"]["numeric_controls"],
        "OracleDisagreements": facts["paired_oracle_disagreements"],
        "CompiledWitnessCalls": witness_calls["compiled"],
        "PointwiseWitnessCalls": witness_calls["independent_pointwise"],
        "StrongWitnessCalls": witness_calls["strong_relational"],
        "ParentWall": f"{facts['primary_parent_subprocess_wall_seconds']:.2f}",
        "NativeCPU": f"{facts['primary_native_interval_cpu_seconds']:.2f}",
    }
    (OUTPUT / "claims.tex").write_text(
        "\n".join("\\newcommand{\\" + key + "}{" + str(value) + "}" for key, value in macros.items()) + "\n",
        encoding="utf-8",
    )
    table = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Execution cell & Fault FAIL & Control FAIL & Derived UNKNOWN & Physical UNKNOWN \\",
        r"\midrule",
    ]
    for cell in cells:
        c = metrics["cells"][cell]
        row = c["methods"]["compiled"]
        table.append(
            LABELS[cell]
            + f" & {row['fault_fail']}/{row['declared_faults']}"
            + f" & {row['control_fail']}/{row['numeric_controls']}"
            + f" & {row['numeric_status_counts_including_derived_views'].get('UNKNOWN', 0)}/2"
            + f" & {c['physical_status_counts_including_derived_views'].get('UNKNOWN', 0)}/6"
            + r" \\"
        )
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (OUTPUT / "outcomes.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".test-tmp/matplotlib-paper"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "svg.hashsalt": "mltools-001"})
    fig, ax = plt.subplots(figsize=(6.4, 2.9), layout="constrained")
    bars = ax.bar(
        ["Compiled", "Pointwise", "Strong relational"],
        list(witness_calls.values()),
        color=["#35618c", "#668ca6", "#8ca6b8"],
        width=0.58,
    )
    ax.bar_label(bars, padding=3)
    ax.set(ylabel="Native witness calls", ylim=(0, max(witness_calls.values()) * 1.25))
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(
        0.5,
        0.97,
        "Overlapping views; actual shared acquisition: "
        + str(metrics["counts"]["actual_shared_fresh_decoder_calls"])
        + " calls",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    for extension in ("pdf", "png"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(OUTPUT / ("witness-cost." + extension), dpi=200, **kwargs)
    plt.close(fig)
    codes = {"PASS": 0, "FAIL": 1, "UNKNOWN": 2}
    rows = [ablation[cell][method] for cell in cells for method in METHODS]
    fig, ax = plt.subplots(figsize=(6.4, 4.6), layout="constrained")
    ax.imshow(
        [[codes[v] for v in row] for row in rows],
        aspect="auto",
        vmin=0,
        vmax=2,
        cmap=ListedColormap(["#d9eee2", "#efd2c4", "#e4e7ec"]),
    )
    for i, row in enumerate(rows):
        for j, status in enumerate(row):
            ax.text(j, i, status, ha="center", va="center", fontsize=7)
    names = [
        LABELS[cell]
        + " | "
        + {"compiled": "C", "independent_pointwise": "P", "strong_relational": "S"}[method]
        for cell in cells
        for method in METHODS
    ]
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xticks(range(4), ["Control", "Offset", "Slope", "Combined"])
    ax.set_title("Offset authority erased; remaining slopes stay independently grounded", fontsize=9)
    for boundary in (2.5, 5.5, 8.5):
        ax.axhline(boundary, color="white", linewidth=2)
    ax.tick_params(length=0)
    for extension in ("pdf", "png"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
        fig.savefig(OUTPUT / ("authority-ablation." + extension), dpi=200, **kwargs)
    plt.close(fig)
    inputs = [original / n for n in ("summary.json", "metrics.json", "analysis.json", "witnesses.json")]
    inputs += [
        clean / "summary.json",
        clean / "reproduction-comparison.json",
        ROOT / "research/publication/protocol-freeze-001.json",
    ]
    write(
        OUTPUT / "manifest.json",
        {
            "schema_version": 1,
            "generator_sha256": sha(Path(__file__)),
            "input_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
            "output_sha256": {p.name: sha(p) for p in sorted(OUTPUT.iterdir()) if p.name != "manifest.json"},
            "plot_packages": {name: importlib.metadata.version(name) for name in ("matplotlib", "numpy")},
            "scope": "Descriptive reporting after acquisition; no experiments or method tuning",
        },
    )
    print("Regenerated verified facts, TeX table/macros and two publication figures")


if __name__ == "__main__":
    main()
