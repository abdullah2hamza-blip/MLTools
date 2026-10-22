"""Bounded explicit worker for unmodified original LeRobot fixture capture."""

import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stateful_fixtures import ROOT, require, sources  # noqa: E402

from mltools.io import digest  # noqa: E402


def main():
    raw = sys.stdin.buffer.read(2 * 1024**2 + 1)
    require(len(raw) <= 2 * 1024**2, "worker request size")
    request = json.loads(raw)
    require(1 <= len(request["bundles"]) <= 16, "bounded native batch")
    initial = sources()
    files = {}
    for name, expected in request["file_sha256"].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT) and digest(path) == expected, "worker input identity")
        files[name] = expected
    events = []

    def audit(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            events.append(event)
            raise RuntimeError("Native comparison worker forbids network connections")

    sys.addaudithook(audit)
    start = time.perf_counter()
    import torch

    from mltools.lerobot import capture, verify_installation

    torch.set_num_threads(1)
    upstream = verify_installation()
    captures, timings = {}, {}
    import_seconds = time.perf_counter() - start
    for name, relative in request["bundles"].items():
        require(relative in files, "capture bundle is in input snapshot")
        before = time.perf_counter()
        captures[name] = capture(ROOT / relative)
        timings[name] = time.perf_counter() - before
    require(upstream == verify_installation() and initial == sources(), "native source stability")
    require(
        all(digest(ROOT / name) == expected for name, expected in files.items()), "native input stability"
    )
    require(not events, "network attempts")
    result = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "source_file_sha256": initial,
        "input_file_sha256": files,
        "captures": captures,
        "acquisition_seconds": timings,
        "runtime": {
            "python": platform.python_version(),
            "imports_seconds": import_seconds,
            "wall_seconds": time.perf_counter() - start,
            "network_events": events,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
        "scope": "Unmodified mltools.lerobot.capture, genuine pinned processors and explicit fixture queues",
    }
    sys.stdout.buffer.write(json.dumps(result, allow_nan=False).encode())


if __name__ == "__main__":
    main()
