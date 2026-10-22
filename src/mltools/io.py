"""Bounded data-only loading; no artifact code execution or deserialization."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

MAX_BYTES = 2 * 1024 * 1024
CHECKS = {"artifacts", "cameras", "joint_order", "normalization", "actions", "reset"}


class InputError(ValueError):
    """Invalid data or invocation, distinct from a conformance finding."""


def read_bytes(path: Path) -> bytes:
    """Read one bounded snapshot, including files that grow after the size check."""
    if path.stat().st_size > MAX_BYTES:
        raise InputError(f"File exceeds {MAX_BYTES} bytes: {path.name}")
    with path.open("rb") as source:
        data = source.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise InputError(f"File exceeds {MAX_BYTES} bytes: {path.name}")
    return data


def digest(path: Path) -> str:
    return hashlib.sha256(read_bytes(path)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise InputError(f"Non-finite JSON number: {value}")


def ensure_json(value) -> None:
    """Bound direct API values too; JSON float overflow is legal syntax but unusable evidence."""
    stack = [(value, 0)]
    count = 0
    while stack:
        item, depth = stack.pop()
        count += 1
        if count > 100_000 or depth > 40:
            raise InputError("JSON exceeds node/depth bounds")
        if type(item) in (int, float) and not number(item):
            raise InputError("JSON number exceeds supported finite range")
        if isinstance(item, dict):
            if not all(isinstance(k, str) for k in item):
                raise InputError("JSON object keys must be strings")
            stack.extend((x, depth + 1) for x in item.values())
        elif isinstance(item, list):
            stack.extend((x, depth + 1) for x in item)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise InputError("Only JSON data values are supported")


def parse_json(data: bytes) -> dict:
    """Parse exactly the bounded bytes whose identity a producer records."""
    try:
        if len(data) > MAX_BYTES:
            raise InputError(f"JSON exceeds {MAX_BYTES} bytes")
        result = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant)
        if not isinstance(result, dict):
            raise InputError("JSON document must be an object")
        ensure_json(result)
        return result
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InputError(f"Cannot parse JSON: {exc}") from exc


def read_json(path: Path) -> dict:
    try:
        return parse_json(read_bytes(path))
    except OSError as exc:
        raise InputError(f"Cannot read JSON {path.name}: {exc}") from exc


def local_path(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or ":" in value or "\\" in value:
        raise InputError("Artifact paths must be relative POSIX paths")
    path = (root / value).resolve()
    if Path(value).is_absolute() or not path.is_relative_to(root.resolve()):
        raise InputError("Artifact path escapes bundle directory")
    return path


def number(value: Any) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def vector(value: Any, size: int, label: str) -> None:
    if not isinstance(value, list) or len(value) != size or not all(number(x) for x in value):
        raise InputError(f"{label}: expected {size} finite numbers")


def names(value: Any, label: str) -> None:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 32
        or not all(isinstance(x, str) and x for x in value)
        or len(set(value)) != len(value)
    ):
        raise InputError(f"{label}: expected 1..32 unique nonempty names")


def load_bundle(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).resolve()
    bundle = read_json(path)
    try:
        if type(bundle["schema_version"]) is not int or bundle["schema_version"] != 1:
            raise InputError("Unsupported bundle schema_version")
        contract = bundle["contract"]
        joints = contract["joints"]
        names([j["name"] for j in joints], "contract.joints")
        for joint in joints:
            if joint.get("unit") is not None and not isinstance(joint["unit"], str):
                raise InputError("Joint unit must be a string or null")
            if joint.get("mode") not in ("relative", "absolute", None):
                raise InputError("Joint mode must be relative, absolute, or null")
        if type(contract["horizon"]) is not int or not 1 <= contract["horizon"] <= 64:
            raise InputError("horizon must be 1..64")
        names(contract["checks"], "contract.checks")
        if not isinstance(contract.get("normalization", {}), dict):
            raise InputError("contract.normalization must be an object")
        if contract.get("training_action_transform") not in (None, "identity", "joint_modes"):
            raise InputError("Unsupported training_action_transform")
        cameras = contract.get("cameras", [])
        if len(cameras) > 16:
            raise InputError("At most 16 cameras are supported")
        for camera in cameras:
            if camera.get("role") is not None and not isinstance(camera["role"], str):
                raise InputError("Camera role must be a string or null")
            if not all(isinstance(camera.get(k), str) and camera[k] for k in ("source", "target")):
                raise InputError("Cameras require source and target keys")
        if len({x["target"] for x in cameras}) != len(cameras):
            raise InputError("Contract camera targets must be unique")
        adapter = bundle["adapter"]
        if "emit_commands" in adapter and type(adapter["emit_commands"]) is not bool:
            raise InputError("adapter.emit_commands must be boolean")
        for key in ("state_order", "motor_names", "action_names"):
            names(adapter[key], f"adapter.{key}")
            if len(adapter[key]) != len(joints):
                raise InputError(f"adapter.{key} must match joint count")
        if not isinstance(adapter["camera_map"], dict):
            raise InputError("adapter.camera_map must be an object")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in adapter["camera_map"].items()):
            raise InputError("camera_map keys and values must be strings")
        names(adapter["exclude_joints"] or ["__none__"], "adapter.exclude_joints")
        for key in ("relative", "bind_queue", "reset"):
            if type(adapter[key]) is not bool:
                raise InputError(f"adapter.{key} must be boolean")
        artifacts = bundle["artifacts"]
        if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 32:
            raise InputError("artifacts must contain 1..32 entries")
        names([a["id"] for a in artifacts], "artifacts")
        for artifact in artifacts:
            local_path(path.parent, artifact["path"])
            sha = artifact.get("sha256")
            if sha is not None and (
                not isinstance(sha, str) or len(sha) != 64 or any(x not in "0123456789abcdef" for x in sha)
            ):
                raise InputError("sha256 must be lowercase hex or null")
        events = bundle["scenario"]
        if not isinstance(events, list) or len(events) > 500:
            raise InputError("scenario must contain at most 500 events")
        queue = 0
        for event in events:
            if event["kind"] == "reset":
                queue = 0
                continue
            if event["kind"] != "tick":
                raise InputError("Events must be reset or tick")
            state = event["state"]
            if not isinstance(state, dict) or not all(number(x) for x in state.values()):
                raise InputError("State must map names to finite numbers")
            if set(state) != {j["name"] for j in joints}:
                raise InputError("State must contain exactly the contract joints")
            if not isinstance(event["cameras"], dict):
                raise InputError("tick.cameras must be an object")
            if event.get("prediction") is not None:
                if queue:
                    raise InputError("Cannot replace an unconsumed chunk")
                chunk = event["prediction"]
                if not isinstance(chunk, list) or not 1 <= len(chunk) <= 64:
                    raise InputError("prediction must contain 1..64 actions")
                for row in chunk:
                    vector(row, len(joints), "prediction")
                targets = event.get("training_targets")
                if targets is not None:
                    if not isinstance(targets, list) or len(targets) != len(chunk):
                        raise InputError("training_targets must match prediction horizon")
                    for row in targets:
                        vector(row, len(joints), "training_targets")
                queue = len(chunk)
            queue = max(0, queue - 1)
        return bundle, path.parent
    except (KeyError, TypeError, AttributeError) as exc:
        raise InputError(f"Invalid bundle structure: {exc}") from exc


def write_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
