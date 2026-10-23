"""Preserve raw downloads and verify report-bound outputs apart from CLI log sidecars."""

import argparse
import json
import tempfile
from pathlib import Path

import publish


def stage(root, config, raw, output):
    root, raw, output = Path(root).resolve(), Path(raw).resolve(), Path(output).resolve()
    publish.require(raw.is_dir(), "raw hosted download directory")
    publish.require(not output.exists(), "new verified output directory")
    publish.require(not output.is_relative_to(root), "outputs outside upload root")
    publish.require(
        not output.is_relative_to(raw) and not raw.is_relative_to(output), "separate raw/verified roots"
    )
    publish.validate(root, config)
    report = publish.parse(publish.read(raw / "replay-report.json"))
    reported = publish.hashes(report.get("files"), "hosted report files")
    selected = set(reported) | {"replay-report.json"}
    paths = list(raw.rglob("*"))
    publish.require(not any(p.is_symlink() for p in paths), "no raw output links")
    actual = {p.relative_to(raw).as_posix() for p in paths if p.is_file()}
    sidecars = actual - selected
    publish.require(selected <= actual, "all report-bound files downloaded")
    publish.require(sidecars <= {publish.NOTEBOOK_SLUG + ".log"}, "unknown provider sidecar requires review")
    snapshots = {name: publish.read(raw / publish.safe_name(name)) for name in sorted(actual)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".verify-hosted-", dir=output.parent) as temporary:
        candidate = Path(temporary) / "outputs"
        candidate.mkdir()
        for name in sorted(selected):
            path = candidate / publish.safe_name(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(snapshots[name])
        verification = publish.verify_outputs(root, config, candidate)
        publish.require(not output.exists(), "verified output destination still new")
        candidate.rename(output)
    publish.require(
        all(publish.digest(publish.read(raw / n)) == publish.digest(v) for n, v in snapshots.items()),
        "raw download unchanged",
    )
    return {
        "schema_version": 1,
        "status": "REPORT_BOUND_OUTPUTS_VERIFIED",
        "verification": verification,
        "raw_file_sha256": {name: publish.digest(value) for name, value in snapshots.items()},
        "provider_sidecars_excluded": {name: publish.digest(snapshots[name]) for name in sorted(sidecars)},
        "scope": (
            "Raw downloads preserved; only the CLI-generated notebook log is "
            "outside the report-bound output set"
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    result = stage(args.root, publish.parse(publish.read(args.config)), args.raw, args.output)
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_bytes(publish.encoded(result))
    print(json.dumps(result, indent=2))
