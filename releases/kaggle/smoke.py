"""Execute every notebook cell in a fresh local kernel with read-only input files."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoke(root, work, record):
    root, work = Path(root).resolve(), Path(work).resolve()
    if work.exists():
        raise ValueError("smoke work destination must be new")
    (work / "temp").mkdir(parents=True)
    # Jupyter's parent process creates runtime/spec files before kernel startup.
    os.environ["TMP"] = os.environ["TEMP"] = str(work / "temp")
    os.environ["JUPYTER_RUNTIME_DIR"] = str(work / "jupyter-runtime")
    os.environ["IPYTHONDIR"] = str(work / "ipython")
    import tempfile

    tempfile.tempdir = str(work / "temp")
    # Optional authoring dependencies, absent from the released replay itself.
    import nbformat
    import publish
    from nbclient import NotebookClient

    source = root / "dataset"
    inputs, outputs = work / "input" / "boundary-conformance-replay", work / "output"
    inputs.parent.mkdir(parents=True)
    shutil.copytree(source, inputs)
    before = {p.name: digest(p) for p in inputs.iterdir()}
    for p in inputs.iterdir():
        p.chmod(stat.S_IREAD)
    try:
        notebook = nbformat.read(root / "notebook/replay.ipynb", as_version=4)
        nbformat.validate(notebook)
        environment = os.environ.copy()
        environment["MLTOOLS_INPUT"] = str(inputs.parent)
        environment["MLTOOLS_OUTPUT"] = str(outputs)
        # Keep Jupyter runtime files inside the authorized workspace.
        environment["JUPYTER_RUNTIME_DIR"] = str(work / "jupyter-runtime")
        environment["IPYTHONDIR"] = str(work / "ipython")
        client = NotebookClient(
            notebook,
            timeout=240,
            kernel_name="python3",
            allow_errors=False,
            resources={"metadata": {"path": str(work)}},
        )
        executed = client.execute(env=environment)
        assert {p.name: digest(p) for p in inputs.iterdir()} == before
        config = json.loads((Path(__file__).parent / "config.json").read_bytes())
        release = publish.validate(root, config)
        verified = publish.verify_outputs(root, config, outputs)
        record = Path(record)
        record.mkdir(parents=True, exist_ok=False)
        nbformat.write(executed, record / "executed.ipynb")
        result = {
            "schema_version": 1,
            "status": "LOCAL_FRESH_KERNEL_VERIFIED",
            "python": sys.version.split()[0],
            "release_id": release["release_id"],
            "input_files_unchanged": before,
            "executed_code_cells": sum(c.cell_type == "code" for c in executed.cells),
            "replay_process_network": "DENIED_BY_AUDIT_HOOK",
            "input_file_mode": "read-only",
            "hosted_kaggle_run": "NOT_RUN_NO_CREDENTIALS_USED",
            "output_verification": verified,
        }
        (record / "verification.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(result, indent=2))
    finally:
        for p in inputs.iterdir():
            p.chmod(stat.S_IREAD | stat.S_IWRITE)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    a = p.parse_args()
    smoke(a.root, a.work, a.record)
