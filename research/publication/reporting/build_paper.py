"""Compile an immutable manuscript snapshot with the verified local compiler."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "research/publication/paper"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", type=Path, required=True)
    parser.add_argument("--allow-downloads", action="store_true")
    args = parser.parse_args()
    compiler = json.loads((PAPER / "compiler-tool.json").read_text(encoding="utf-8"))
    if sha(args.tectonic) != compiler["executable_sha256"]:
        raise ValueError("Compiler does not match the recorded official executable")
    format_record = json.loads((PAPER / "venue-format.json").read_text(encoding="utf-8"))
    for row in format_record["files"]:
        if sha(PAPER / row["path"]) != row["sha256"]:
            raise ValueError("Official style bytes changed")
    output = PAPER / "output"
    output.mkdir(exist_ok=True)
    number = 1
    while (output / f"compile-{number:03}").exists():
        number += 1
    folder = output / f"compile-{number:03}"
    snapshot = ROOT / f".test-tmp/paper-source-{number:03}"
    snapshot.mkdir(parents=True, exist_ok=False)
    folder.mkdir(exist_ok=False)
    sources = {}
    for path in sorted(PAPER.rglob("*")):
        relative = path.relative_to(PAPER)
        if not path.is_file() or "output" in relative.parts:
            continue
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        sources[relative.as_posix()] = sha(destination)
    env = os.environ.copy()
    env["TECTONIC_CACHE_DIR"] = str(ROOT / ".test-tmp/tectonic-cache")
    env["SOURCE_DATE_EPOCH"] = subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", "7513755e9627098c3529c46b3ae9a056d76434a5"],
        cwd=ROOT,
        text=True,
    ).strip()
    command = [str(args.tectonic.resolve()), "--untrusted", "--keep-logs"]
    if not args.allow_downloads:
        command.append("--only-cached")
    command += ["--outdir", str(folder), str(snapshot / "main.tex")]
    start = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env, check=False)
    (folder / "compiler.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (folder / "compiler.stderr.txt").write_text(result.stderr, encoding="utf-8")
    stable = all(sha(snapshot / p) == h and sha(PAPER / p) == h for p, h in sources.items())
    record = {
        "schema_version": 1,
        "command": command,
        "returncode": result.returncode,
        "wall_seconds": time.perf_counter() - start,
        "source_date_epoch": env["SOURCE_DATE_EPOCH"],
        "source_file_sha256": sources,
        "source_stable_during_compilation": stable,
        "compiler_sha256": sha(args.tectonic),
        "builder_sha256": sha(Path(__file__)),
        "network_downloads_allowed": args.allow_downloads,
    }
    if result.returncode == 0 and stable:
        shutil.copyfile(folder / "main.pdf", output / "main.pdf")
        record["pdf_sha256"] = sha(output / "main.pdf")
    write(folder / "compile.json", record)
    if result.returncode or not stable:
        raise RuntimeError("Compile failed or manuscript changed; retained logs identify this attempt")
    write(
        output / "latest-build.json",
        {
            "compile_record": folder.relative_to(ROOT).as_posix() + "/compile.json",
            "compile_record_sha256": sha(folder / "compile.json"),
            "pdf_sha256": record["pdf_sha256"],
        },
    )
    print("Compiled stable manuscript snapshot to", output / "main.pdf")


if __name__ == "__main__":
    main()
