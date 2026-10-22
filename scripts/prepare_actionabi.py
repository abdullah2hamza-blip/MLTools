"""Explicitly download pinned sources and a verified portable Windows CPU toolchain."""

import argparse
import hashlib
import json
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True, help="New local setup directory")
    args = parser.parse_args()
    folder = args.destination.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    ledger = json.loads((ROOT / "integrations/actionabi/sources.json").read_bytes())
    for name, source in ledger["repositories"].items():
        path = folder / "sources" / name
        path.mkdir(parents=True)
        commands = [
            ["git", "init", "--quiet", str(path)],
            [
                "git",
                "-C",
                str(path),
                "remote",
                "add",
                "origin",
                f"https://github.com/{source['repository']}.git",
            ],
            ["git", "-C", str(path), "fetch", "--quiet", "--depth=1", "origin", source["revision"]],
            [
                "git",
                "-C",
                str(path),
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.eol=lf",
                "checkout",
                "--quiet",
                "--detach",
                "FETCH_HEAD",
            ],
        ]
        for command in commands:
            subprocess.run(command, check=True, timeout=180)
        print(f"Fetched pinned {source['repository']} source", flush=True)
    tool = ledger["toolchain"]
    archive = folder / tool["archive"]
    with urllib.request.urlopen(tool["url"], timeout=60) as response, archive.open("wb") as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk)
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != tool["sha256"] or archive.stat().st_size != tool["bytes"]:
        raise ValueError("Official toolchain archive identity mismatch; extraction withheld")
    with zipfile.ZipFile(archive) as bundle:
        for entry in bundle.infolist():
            if not (folder / entry.filename).resolve().is_relative_to(folder):
                raise ValueError("Unsafe toolchain archive entry; extraction withheld")
        bundle.extractall(folder)
    print("Verified and extracted portable toolchain; no global installation")


if __name__ == "__main__":
    main()
