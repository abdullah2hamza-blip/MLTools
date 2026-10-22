"""Explicit opt-in LeRobot execution of a compiled distinguishing suite."""

import hashlib
import platform
import time
from collections import deque
from pathlib import Path

from .checker import validate_statistics
from .io import InputError, digest, parse_json, read_bytes
from .lerobot import verify_installation
from .obligations import Suite


def capture(suite: Suite, stats_path: Path, configuration: dict) -> dict:
    start = time.perf_counter()
    producer_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    size = len(configuration["action_names"])
    expected = {(p, str(i)) for p in ("action", "origin", "current") for i in range(size)}
    if {(a.port, a.name) for a in suite.axes} != expected or len(suite.outputs) != size:
        raise InputError("LeRobot probes require matching action/origin/current index axes and outputs")
    if set(suite.outputs) != {f"{name}.pos" for name in configuration["motor_names"]}:
        raise InputError("LeRobot output labels must match the named controller ports")
    stats_bytes = read_bytes(stats_path)
    stats_sha = hashlib.sha256(stats_bytes).hexdigest()
    identity = verify_installation()
    import torch
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.lerobot_types import TransitionKey as Key
    from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
    from lerobot.processor.policy_robot_bridge import PolicyActionToRobotActionProcessorStep
    from lerobot.processor.relative_action_processor import (
        AbsoluteActionsProcessorStep,
        RelativeActionsProcessorStep,
    )

    if stats_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load

        saved = load(stats_bytes)
        stats = {}
        for name in ("mean", "std"):
            value = saved.get(f"action.{name}")
            if value is None or tuple(value.shape) != (size,) or not bool(torch.isfinite(value).all()):
                raise InputError(
                    "Checkpoint action statistics require finite mean/std vectors of the port size"
                )
            stats[name] = value.to(torch.float64).tolist()
        if any(x <= 0 for x in stats["std"]):
            raise InputError("Standard deviations must be positive")
    else:
        stats = validate_statistics(parse_json(stats_bytes), size)

    unnormalizer = UnnormalizerProcessorStep(
        features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(size,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={"action": stats},
        device="cpu",
        dtype=torch.float64,
    )
    relative = RelativeActionsProcessorStep(
        enabled=configuration["relative"],
        action_names=configuration["action_names"],
        exclude_joints=configuration["exclude_joints"],
    )
    absolute = AbsoluteActionsProcessorStep(enabled=configuration["relative"], relative_step=relative)
    bridge = PolicyActionToRobotActionProcessorStep(motor_names=configuration["motor_names"])
    queue = deque()
    if configuration["bind_queue"]:
        relative.bind_action_queue(lambda: len(queue))
    results, traces = {}, []
    with torch.inference_mode():
        for probe in suite.probes:
            queue.clear()
            relative.reset()
            reset_cleared = relative.get_cached_state() is None
            points = {(a.port, a.name): v for a, v in zip(suite.axes, probe.values, strict=True)}
            origin = [points[("origin", str(i))] for i in range(size)]
            current = [points[("current", str(i))] for i in range(size)]
            prediction = [points[("action", str(i))] for i in range(size)]
            relative({Key.OBSERVATION: {"observation.state": torch.tensor([origin], dtype=torch.float64)}})
            queue.extend([prediction[:], prediction[:]])

            def dequeue():
                transformed = unnormalizer({Key.ACTION: torch.tensor([queue.popleft()], dtype=torch.float64)})
                commands = bridge.action(absolute(transformed)[Key.ACTION][0])
                return [float(commands[name].item()) for name in suite.outputs]

            first = dequeue()
            relative({Key.OBSERVATION: {"observation.state": torch.tensor([current], dtype=torch.float64)}})
            # Anchor must be observed while the second action is still queued.
            cached = relative.get_cached_state()[0].tolist()
            second = dequeue()
            results[probe.id] = second
            traces.append(
                {
                    "query": probe.id,
                    "origin": origin,
                    "current": current,
                    "normalized_prediction": prediction,
                    "first_action": first,
                    "second_action": second,
                    "cached_anchor_before_second": cached,
                    "queue_remaining": len(queue),
                    "reset_cleared_before_query": reset_cleared,
                }
            )
    if digest(stats_path) != stats_sha:
        raise InputError("Statistics changed during execution; discard this capture")
    return {
        "schema_version": 1,
        "suite_sha256": suite.sha256,
        "outputs": results,
        "identity": identity,
        "stats_sha256": stats_sha,
        "statistics_loader": "safetensors.torch.load"
        if stats_path.suffix.lower() == ".safetensors"
        else "bounded mean/std JSON",
        "configuration": configuration,
        "traces": traces,
        "producer": {"module": "mltools.probe_capture", "source_sha256": producer_sha},
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "wall_seconds": time.perf_counter() - start,
            "dtype": "float64",
            "device": "cpu",
        },
        "scope": "Genuine LeRobot processors; two-action numerical probes; no trained inference or hardware",
    }
