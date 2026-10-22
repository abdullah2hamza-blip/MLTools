"""Explicit opt-in execution of a pinned, installed LeRobot processor path."""

import hashlib
import importlib.metadata
import importlib.util
import json
from collections import deque
from pathlib import Path

from .checker import statistics
from .io import InputError, digest, load_bundle, local_path, read_json

REVISION = "5517a326eeb9b457cfe07d59b9bfdf291799c8d9"
BLOBS = {
    "processor/relative_action_processor.py": "3405402904cf15ca18227b3a3fe006d7936f9d53",
    "processor/normalize_processor.py": "b0eafcef75e187887c8ea642cfe138f9cf8e79a9",
    "processor/rename_processor.py": "d17f93dcd990861a2b5ade5914f4595e85eece16",
    "processor/policy_robot_bridge.py": "bd43c0e81a52c4b75c0c3029e110541484300c89",
    "processor/pipeline.py": "3cfdf680b27da74d75c9d1b4f8c7f69182c28b02",
    "configs/types.py": "18359ef0575a4f560a65f21e2d6ddf534edc770f",
    "robots/so_follower/so_follower.py": "be78bc9623b72155bb87660763d7e2c2ed481d5f",
    "robots/so_follower/config_so_follower.py": "af4dda782b562064c1b80902ab5948b07ddc43de",
}


def verify_installation() -> dict:
    try:
        distribution = importlib.metadata.distribution("lerobot")
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != REVISION:
            raise InputError("LeRobot must be installed from the documented pinned Git revision")
        spec = importlib.util.find_spec("lerobot")
        root = Path(next(iter(spec.submodule_search_locations)))
        verified = {}
        for relative, expected in BLOBS.items():
            raw = (root / relative).read_bytes().replace(b"\r\n", b"\n")
            actual = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
            if actual != expected:
                raise InputError(f"Installed LeRobot source identity mismatch: {relative}")
            verified[relative] = actual
        return {"revision": REVISION, "git_blobs": verified, "source_newlines": "canonical LF"}
    except (importlib.metadata.PackageNotFoundError, OSError, AttributeError, StopIteration) as exc:
        raise InputError("Install the optional pinned integration with uv sync --extra lerobot") from exc


def capture(bundle_path: str | Path) -> dict:
    """Run reviewed built-in wiring only. A bundle cannot select arbitrary modules/callbacks."""
    bundle, root = load_bundle(bundle_path)
    identity = verify_installation()
    try:
        import torch
        from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.lerobot_types import TransitionKey as Key
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        from lerobot.processor.policy_robot_bridge import PolicyActionToRobotActionProcessorStep
        from lerobot.processor.relative_action_processor import (
            AbsoluteActionsProcessorStep,
            RelativeActionsProcessorStep,
        )
        from lerobot.processor.rename_processor import RenameObservationsProcessorStep
    except ImportError as exc:
        raise InputError(f"LeRobot dependency import failed: {exc}") from exc

    adapter = bundle["adapter"]
    artifacts = {a["id"]: local_path(root, a["path"]) for a in bundle["artifacts"]}
    size = len(bundle["contract"]["joints"])
    try:
        runtime_stats = statistics(artifacts[adapter["stats_artifact"]], size)
        training_stats = statistics(artifacts[bundle["contract"]["normalization"]["artifact"]], size)
        training_config = read_json(artifacts["checkpoint_config"])
        training_order = training_config["state_order"]
        training_relative = RelativeActionsProcessorStep(
            enabled=training_config["relative"],
            exclude_joints=training_config["exclude_joints"],
            action_names=training_config["action_names"],
        )
    except (KeyError, TypeError) as exc:
        raise InputError(f"Missing integration artifact/configuration: {exc}") from exc

    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(size,))}

    def normalization(cls, stats):
        return cls(
            features=features,
            norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats={"action": stats},
            device="cpu",
            dtype=torch.float64,
        )

    normalizer = normalization(NormalizerProcessorStep, training_stats)
    unnormalizer = normalization(UnnormalizerProcessorStep, runtime_stats)
    relative = RelativeActionsProcessorStep(
        enabled=adapter["relative"],
        exclude_joints=adapter["exclude_joints"],
        action_names=adapter["action_names"],
    )
    absolute = AbsoluteActionsProcessorStep(enabled=adapter["relative"], relative_step=relative)
    rename = RenameObservationsProcessorStep(rename_map=adapter["camera_map"])
    training_rename = RenameObservationsProcessorStep(rename_map=training_config.get("camera_map", {}))
    bridge = PolicyActionToRobotActionProcessorStep(motor_names=adapter["motor_names"])
    queue = deque()
    if adapter["bind_queue"]:
        relative.bind_action_queue(lambda: len(queue))
    events = []
    with torch.inference_mode():
        for event in bundle["scenario"]:
            if event["kind"] == "reset":
                queue.clear()
                if adapter["reset"]:
                    relative.reset()
                training_relative.reset()
                events.append(
                    {
                        "kind": "reset",
                        "anchor_cleared": relative.get_cached_state() is None,
                        "queue_remaining": len(queue),
                    }
                )
                continue
            values = [event["state"][name] for name in adapter["state_order"]]
            training_values = [event["state"][name] for name in training_order]
            state = torch.tensor([values], dtype=torch.float64, device="cpu")
            relative({Key.OBSERVATION: {"observation.state": state}})
            cameras = rename.observation(event["cameras"])
            observed = {
                "kind": "tick",
                "cameras": cameras,
                "state_names": adapter["state_order"],
                "state_values": values,
                "training_cameras": training_rename.observation(event["cameras"]),
                "training_state_names": training_order,
                "training_state_values": training_values,
            }
            if event.get("prediction") is not None:
                queue.extend(event["prediction"])
                observed["prediction_horizon"] = len(event["prediction"])
                observed["training_horizon"] = training_config.get("horizon")
                degrees = training_config.get("use_degrees")
                observed["training_units"] = (
                    {
                        name: (
                            "range_0_100"
                            if name == "gripper"
                            else "degree"
                            if degrees is True
                            else "range_m100_100"
                        )
                        for name in training_order
                    }
                    if type(degrees) is bool
                    else None
                )
                if event.get("training_targets") is not None:
                    training_state = torch.tensor([training_values], dtype=torch.float64)
                    transition = {
                        Key.OBSERVATION: {"observation.state": training_state},
                        Key.ACTION: torch.tensor([event["training_targets"]], dtype=torch.float64),
                    }
                    transformed = normalizer(training_relative(transition))
                    observed["training_normalized"] = transformed[Key.ACTION][0].tolist()
            if queue:
                action = torch.tensor([queue.popleft()], dtype=torch.float64)
                transformed = unnormalizer({Key.ACTION: action})
                denormalized = transformed[Key.ACTION][0].tolist()
                commands = (
                    bridge.action(absolute(transformed)[Key.ACTION][0])
                    if adapter.get("emit_commands", True)
                    else None
                )
                observed["action"] = {
                    "denormalized": denormalized,
                    "commands": {k: float(v.item()) for k, v in commands.items()}
                    if commands is not None
                    else None,
                    "units": adapter.get("units"),
                }
            observed["queue_remaining"] = len(queue)
            events.append(observed)
    identity.update(
        {
            "adapter": "mltools.lerobot",
            "framework": "lerobot",
            "device": "cpu",
            "torch_version": str(torch.__version__),
            "dtype": "float64",
            "scope": "Real processors; fixture configuration, known actions; no model inference or hardware",
        }
    )
    return {
        "schema_version": 1,
        "bundle_sha256": digest(Path(bundle_path)),
        "execution": identity,
        "events": events,
    }
