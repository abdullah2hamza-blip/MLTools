"""Generate an ordinary offline Kaggle notebook without importing Jupyter/Kaggle."""

import textwrap


def notebook(payload_sha256, release_id, dataset_slug):
    cells = []

    def add(kind, text):
        cell = {
            "cell_type": kind,
            "id": f"cell-{len(cells):02d}",
            "metadata": {},
            "source": textwrap.dedent(text).strip() + "\n",
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add(
        "markdown",
        """
        # Boundary Conformance Replay
        Recompute a bounded normalization-boundary study from **two saved native
        acquisitions**. CPU, internet off, no credentials, downloads or installs.
        This is saved-trace numerical replay, not fresh framework execution,
        learned inference, hardware calibration or a population accuracy estimate.

        Three dependency-linked metadata groups supply four cells; the OpenPI
        cell transports the same PI statistics. Sixteen actual conditions and
        eight derived views reuse observations. See the dataset card for the
        source/license inventory, prior inspection and experimental limits.
    """,
    )
    add(
        "code",
        f"""
        import hashlib, json, os, shutil, stat, subprocess, sys, tempfile, zipfile
        from pathlib import Path, PurePosixPath
        PAYLOAD_SHA256 = {payload_sha256!r}
        RELEASE_ID = {release_id!r}
        # The override is only for local smoke tests; Kaggle uses the attached dataset.
        input_root = Path(os.environ.get("MLTOOLS_INPUT", "/kaggle/input"))
        candidates = sorted(input_root.rglob("payload.zip"))
        matches = [p for p in candidates if hashlib.sha256(p.read_bytes()).hexdigest() == PAYLOAD_SHA256]
        assert len(matches) == 1, "Attach the exact dataset version matching this notebook's payload hash"
        payload = matches[0]
        working = Path(os.environ.get("MLTOOLS_OUTPUT", "/kaggle/working"))
        working.mkdir(parents=True, exist_ok=True)
        expected_names = {{"analysis.json", "metrics.json", "witnesses.json", "workflow-analysis.json",
                           "conditions.csv", "replay-summary.json", "replay-report.json", "outcomes.svg"}}
        assert not any((working / n).exists() for n in expected_names), "Use a clean output directory"
        scratch = Path(tempfile.mkdtemp(prefix="mltools-replay-"))
        unpacked = scratch / "payload"
        with zipfile.ZipFile(payload) as archive:
            entries = archive.infolist()
            assert 0 < len(entries) <= 2000
            assert sum(i.file_size for i in entries) <= 64 * 1024 * 1024
            names = [i.filename for i in entries]
            assert len(set(n.casefold() for n in names)) == len(names)
            for item in entries:
                p = PurePosixPath(item.filename)
                assert item.filename and "\\\\" not in item.filename and not p.is_absolute()
                assert p.as_posix() == item.filename
                assert all(x not in ("", ".", "..") and ":" not in x for x in p.parts)
                assert not item.is_dir() and stat.S_IFMT(item.external_attr >> 16) in (0, stat.S_IFREG)
                assert not item.flag_bits & 1
            for item in entries:
                target = unpacked.joinpath(*PurePosixPath(item.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(item))
        print("Pinned release:", RELEASE_ID)
    """,
    )
    add(
        "markdown",
        """
        The payload hash binds the notebook to this export version. The replay
        subprocess verifies every released file, revalidates saved native input/
        output relations, reconstructs all five products, checks each against
        its saved product and compares both acquisitions. Original Git/native
        verification was performed by the builder before export. Neither this
        notebook nor a producer hash independently attests the original runtime.
    """,
    )
    add(
        "code",
        '''
        # A fresh isolated process prevents notebook preloaded libraries from
        # contaminating the stdlib-only replay. The audit hook denies network.
        runner = """import runpy, sys
        def audit(event, args):
            if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.bind'):
                raise RuntimeError('offline replay: network forbidden')
        sys.addaudithook(audit)
        sys.dont_write_bytecode = True
        sys.path.insert(0, sys.argv[1])
        sys.argv = [sys.argv[1] + '/replay.py', '--output', sys.argv[2]]
        runpy.run_path(sys.argv[0], run_name='__main__')
        """
        try:
            result = subprocess.run([sys.executable, "-I", "-B", "-c", runner,
                                     str(unpacked), str(scratch / "results")],
                                    check=True, capture_output=True, text=True, timeout=180)
            print(result.stdout)
            for file in (scratch / "results").iterdir():
                shutil.copyfile(file, working / file.name)
        except subprocess.CalledProcessError as error:
            print(error.stderr)
            raise
        finally:
            shutil.rmtree(scratch)
        summary = json.loads((working / "replay-summary.json").read_bytes())
        assert summary["status"] == "SAVED_TRACE_REPLAY_VERIFIED"
        assert summary["release_id"] == RELEASE_ID
        assert summary["compiled_fault_fail"] == 12 and summary["compiled_control_fail"] == 0
        assert summary["counts"]["actual_conditions"] == 16 and summary["counts"]["derived_views"] == 8
    ''',
    )
    add(
        "markdown",
        """
        ## Inspect the finite results
        With full numerical authority, all three methods detect 12/12 injected
        changes, fail 0/4 controls and abstain on eight derived views. Every
        physical-authority view is UNKNOWN. Candidate self-round trips remain
        consistent for all sixteen actual conditions; they do not establish
        independent-reference conformance. These are correlated finite cases.
    """,
    )
    add(
        "code",
        """
        import csv
        from collections import Counter
        rows = list(csv.DictReader((working / "conditions.csv").open(encoding="utf-8", newline="")))
        assert len(rows) == 24 and Counter(r["physical"] for r in rows) == {"UNKNOWN": 24}
        for method in ("compiled", "pointwise", "strong"):
            print(method, dict(Counter(r[method] for r in rows)))
        metrics = json.loads((working / "metrics.json").read_bytes())
        print("Original acquisition call accounting:", metrics["counts"])
        analysis = json.loads((working / "analysis.json").read_bytes())
        for cell, conditions in analysis["cells"].items():
            print(cell, "offset-authority ablation:")
            for condition, row in conditions.items():
                if "ablations" in row:
                    view = row["ablations"]["offset_authority_erased"]
                    print(" ", condition, view)
    """,
    )
    add(
        "markdown",
        """
        Removing only offset authority leaves slope evidence for compiled and
        strong relational methods in eight changed cases; pointwise comparison
        abstains. Full-reference parity excludes these ablations. All eight
        equal-budget designs agree on reported finite/full-affine statuses.
        Compiled witness method views total 16 calls, versus 12 each for the two
        comparisons, with 16 shared acquired calls. No witness-size or detection
        superiority is claimed. Robotruth's missing-evidence certification is
        WITHHELD; structural Policy Space acceptance is not numerical conformance.
    """,
    )
    add(
        "code",
        """
        # Optional presentation only; the numerical replay uses no plotting dependency.
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib unavailable; the verified tables remain complete.")
        else:
            plt.rcParams["svg.hashsalt"] = RELEASE_ID
            fig, ax = plt.subplots(figsize=(8, 3.5))
            labels = ["Compiled", "Pointwise", "Strong relational"]
            ax.bar(labels, [12, 12, 12], color="#286b83", label="Injected changes: FAIL (n=12)")
            ax.set(ylim=(0, 14), ylabel="Finite case count",
                   title="Full numerical authority: paired finite results")
            caption = "0/4 control failures each; 24/24 physical views UNKNOWN"
            ax.text(0.02, 0.96, caption,
                    transform=ax.transAxes, va="top", fontsize=9)
            ax.legend(loc="lower right")
            fig.tight_layout()
            fig.savefig(working / "outcomes.svg", metadata={"Date": None})
            plt.show()
            plt.close(fig)
        files = {p.relative_to(working).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(working.rglob("*")) if p.is_file() and p.name != "replay-report.json"}
        report = {"schema_version": 1, "release_id": RELEASE_ID, "payload_sha256": PAYLOAD_SHA256,
                  "status": "VERIFIED_SAVED_TRACE_REPLAY", "files": files}
        (working / "replay-report.json").write_text(
            json.dumps(report, indent=2) + "\\n", encoding="utf-8", newline="\\n")
        print(report["status"])
    """,
    )
    add(
        "markdown",
        f"""
        Original contributions: Apache-2.0; retained publisher licenses and
        explicit derivative/redaction notices are in the dataset. Extensive AI
        assistance was used. Cite `abdullah2hamza/{dataset_slug}`, its Kaggle
        version, release ID and payload SHA256. No peer-review acceptance or DOI
        is asserted. See `README.md` and `THIRD_PARTY_NOTICES.md` in the dataset.
    """,
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        },
    }
