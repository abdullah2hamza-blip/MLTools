"""Export a self-contained failing fixture with its required episode context."""

from copy import deepcopy
from pathlib import Path

from .io import InputError, digest, load_bundle, local_path, write_json


def export_counterexample(bundle_path: str | Path, trace: dict, report: dict, destination: Path) -> None:
    witness = report.get("counterexample")
    if witness is None:
        raise InputError("No event-level divergence to export")
    bundle, root = load_bundle(bundle_path)
    for artifact in bundle["artifacts"]:
        if artifact["path"] in {"bundle.json", "trace.json", "reproducer.json"}:
            raise InputError("Artifact path conflicts with counterexample output filenames")
    # Refuse overwriting arbitrary user files. Existing output should be inspected or a new path selected.
    destination.mkdir(parents=True, exist_ok=False)
    reduced = deepcopy(bundle)
    reduced["scenario"] = witness["events"]
    reduced["contract"]["checks"] = [report["first_divergence"]["check"]]
    for artifact in reduced["artifacts"]:
        source = local_path(root, artifact["path"])
        target = local_path(destination, artifact["path"])
        if source.is_file():
            digest(source)  # Enforce the same size bound before copying.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    path = destination / "bundle.json"
    write_json(path, reduced)
    captured = deepcopy(trace)
    captured["events"] = trace["events"][witness["start_event"] : witness["original_event"] + 1]
    captured["bundle_sha256"] = digest(path)
    captured.setdefault("execution", {})["reduced_from_bundle"] = report["bundle_sha256"]
    write_json(destination / "trace.json", captured)
    write_json(
        destination / "reproducer.json",
        {
            "schema_version": 1,
            "original_bundle_sha256": report["bundle_sha256"],
            "original_event": witness["original_event"],
            "replay": "mltools check bundle.json --trace trace.json",
            "execute": "mltools check bundle.json --execute lerobot",
            "limit": "Input witness or episode prefix, not a globally minimal trajectory.",
        },
    )
