"""Opt-in genuine processor acquisition for research query designs, not public Suite replay."""

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Explicit trusted project code, also in the source snapshot. Python -I ignores
# ambient PYTHONPATH/user-site. The OpenPI CPU worker does not install MLTools.
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from query_designs import Design, fingerprint, require  # noqa: E402

from mltools._openpi_worker import _rss  # noqa: E402
from mltools.io import digest, local_path  # noqa: E402
from mltools.probe_cases import load  # noqa: E402

MAX_BYTES = 2 * 1024 * 1024


def sources():
    paths = list((ROOT / "src/mltools").glob("*.py")) + [Path(__file__), ROOT / "scripts/query_designs.py"]
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def openpi_acquisition(stats_text, configuration):
    from mltools._openpi_worker import verify_installation

    identity = verify_installation()
    import numpy as np
    from openpi import transforms
    from openpi.models.model import ModelType
    from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
    from openpi.shared.normalize import deserialize_json

    stats = deserialize_json(stats_text)
    mode = {"pi0": ModelType.PI0, "pi05": ModelType.PI05, "pi0_fast": ModelType.PI0_FAST}[
        configuration["model_type"]
    ]
    inputs, output = LiberoInputs(mode), LiberoOutputs()
    unnormalize = transforms.Unnormalize(
        {"actions": stats["actions"]}, use_quantiles=configuration["use_quantiles"]
    )
    absolute = transforms.AbsoluteActions(tuple(configuration["mask"]))

    def capture(design):
        result, rows = {}, []
        for query in design.probes:
            points = {(a.port, a.name): v for a, v in zip(design.axes, query.values, strict=True)}

            def vector(port, size, points=points):
                return np.asarray([points[(port, str(i))] for i in range(size)], dtype=np.float64)

            origin, current = vector("origin", 8), vector("current", 8)
            prediction = np.stack([vector("action", 7)] * 2)
            observed = inputs(
                {
                    "observation/state": origin.copy(),
                    "observation/image": np.full((4, 5, 3), 31, dtype=np.uint8),
                    "observation/wrist_image": np.full((4, 5, 3), 197, dtype=np.uint8),
                }
            )
            raw = unnormalize({"actions": np.pad(prediction, ((0, 0), (0, 25)))})["actions"]
            commands = output(absolute({"state": observed["state"].copy(), "actions": raw.copy()}))["actions"]
            if configuration["anchor"] == "current":
                commands[1:] = output(absolute({"state": current.copy(), "actions": raw[1:].copy()}))[
                    "actions"
                ]
            queue = deque(commands.copy())
            first, second = queue.popleft(), queue.popleft()
            result[query.id] = second.tolist()
            rows.append(
                {
                    "query": query.id,
                    "origin": origin.tolist(),
                    "current": current.tolist(),
                    "normalized_prediction": prediction[0].tolist(),
                    "first_action": first.tolist(),
                    "second_action": second.tolist(),
                    "queue_remaining": len(queue),
                }
            )
        return {
            "outputs": result,
            "identity": identity,
            "traces": rows,
            "stats_sha256": hashlib.sha256(stats_text.encode()).hexdigest(),
            "configuration": configuration,
            "scope": "Genuine OpenPI deployed-input/unnormalize/absolute/output transforms; fixture queue",
        }

    return capture, lambda: verify_installation() == identity


def execute(request, request_sha):
    initial_sources = sources()
    case_path = (ROOT / request["case_path"]).resolve()
    require(case_path.is_relative_to(ROOT), "case path outside research workspace")
    case_sha = digest(case_path)
    require(request["case_sha256"] == case_sha, "case input identity")
    case, root, original_suite, _reference = load(case_path)
    artifact = next(a for a in case["artifacts"] if a["id"] == case["adapter"]["stats_artifact"])
    stats_path = local_path(root, artifact["path"])
    stats_sha = digest(stats_path)
    require(request["stats_sha256"] == stats_sha == artifact["sha256"], "statistics snapshot identity")
    require(len(request["designs"]) <= 8 and request["designs"], "design batch size")
    designs = {name: Design.from_dict(value) for name, value in request["designs"].items()}
    expected_axes = original_suite.to_dict()["axes"]
    for design in designs.values():
        require(
            [vars(a) for a in design.axes] == expected_axes and design.outputs == original_suite.outputs,
            "research design changed original bounded ports",
        )
    network_events = []

    def audit(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            network_events.append(event)
            raise RuntimeError("Research processor execution forbids network connections")

    sys.addaudithook(audit)
    started = time.perf_counter()
    framework = case["adapter"]["framework"]
    config = case["adapter"]["configuration"]
    if framework == "lerobot":
        from mltools.lerobot import verify_installation
        from mltools.probe_capture import capture

        identity = verify_installation()
        import torch

        require(torch.get_num_threads() == 1, "CPU thread budget")
        for module in (
            "lerobot.configs",
            "lerobot.lerobot_types",
            "lerobot.processor.normalize_processor",
            "lerobot.processor.policy_robot_bridge",
            "lerobot.processor.relative_action_processor",
        ):
            importlib.import_module(module)
        if stats_path.suffix.lower() == ".safetensors":
            importlib.import_module("safetensors.torch")

        def acquire(design):
            # The internal capture consumes axes/probes/sha256; no public canonical
            # Suite is forged. All methods use this identical native acquisition.
            return capture(design, stats_path, config)

        def stable():
            return verify_installation() == identity
    else:
        require(stats_path.stat().st_size <= 65536, "OpenPI statistics size")
        acquire, stable = openpi_acquisition(stats_path.read_text(encoding="utf-8"), config)
    import_seconds = time.perf_counter() - started
    measurements = {}
    for name, design in designs.items():
        start = time.perf_counter()
        trace = acquire(design)
        trace["design_sha256"] = design.sha256
        trace["schedule_sha256"] = fingerprint(design.schedule)
        trace["case_sha256"] = case_sha
        trace["framework"] = framework
        trace["acquisition_seconds_after_imports"] = time.perf_counter() - start
        measurements[name] = trace
    peak = _rss()
    require(peak <= 2 * 1024**3, "observed peak working set exceeds development budget")
    require(stable(), "installed upstream sources changed during acquisition")
    require(initial_sources == sources(), "project acquisition source changed")
    require(digest(case_path) == case_sha and digest(stats_path) == stats_sha, "case/statistics changed")
    require(not network_events, "intercepted network attempt")
    return {
        "schema_version": 1,
        "format": "mltools-research-query-capture",
        "development_only": True,
        "final_evaluation": False,
        "request_sha256": request_sha,
        "source_file_sha256": initial_sources,
        "measurements": measurements,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "imports_and_setup_seconds": import_seconds,
            "wall_seconds": time.perf_counter() - started,
            "peak_working_set_bytes": peak,
            "network_events": network_events,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
        "scope": "New genuine-processor queries and two-action schedule; no inference or hardware",
    }


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        require(len(raw) <= MAX_BYTES, "request size")
        result = execute(json.loads(raw), hashlib.sha256(raw).hexdigest())
        output = json.dumps(result, allow_nan=False).encode("utf-8")
        require(len(output) <= MAX_BYTES, "capture size")
        sys.stdout.buffer.write(output)
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
