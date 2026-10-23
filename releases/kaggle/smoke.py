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


def smoke(root, work, record, expanded_input=False, platform_runtime_file=False):
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
    if expanded_input:
        from integrity import extract

        extract(inputs / "payload.zip", inputs / "payload")
        (inputs / "payload.zip").unlink()
    input_files = [p for p in inputs.rglob("*") if p.is_file()]
    if platform_runtime_file:
        outputs.mkdir()
        (outputs / "__notebook__.ipynb").write_bytes(b'{"fixture":"platform-managed notebook"}\n')
    before = {p.relative_to(inputs).as_posix(): digest(p) for p in input_files}
    for p in input_files:
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
        assert {p.relative_to(inputs).as_posix(): digest(p) for p in input_files} == before
        config = json.loads((Path(__file__).parent / "config.json").read_bytes())
        release = publish.validate(root, config)
        if platform_runtime_file:
            raw_platform_report = json.loads((outputs / "replay-report.json").read_bytes())
            assert "__notebook__.ipynb" not in raw_platform_report["files"]
            assert "__notebook__.ipynb" in raw_platform_report["platform_runtime_files_not_exported"]
            (outputs / "__notebook__.ipynb").rename(work / "platform-managed-notebook-fixture.ipynb")
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
            "input_representation": "expanded-directory" if expanded_input else "archive",
            "platform_runtime_fixture": platform_runtime_file,
            "hosted_kaggle_run": "NOT_RUN_NO_CREDENTIALS_USED",
            "output_verification": verified,
        }
        (record / "verification.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps(result, indent=2))
    finally:
        for p in input_files:
            p.chmod(stat.S_IREAD | stat.S_IWRITE)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--expanded-input", action="store_true")
    p.add_argument("--platform-runtime-file", action="store_true")
    a = p.parse_args()
    smoke(a.root, a.work, a.record, a.expanded_input, a.platform_runtime_file)
