"""Independent scalar reference; deliberately imports no robot/ML framework."""

from collections import deque
from pathlib import Path

from .io import CHECKS, InputError, digest, ensure_json, load_bundle, local_path, number, read_json, vector
from .report import Finding, Status, make_report

ATOL = 1e-7
RTOL = 1e-7


def equivalent(expected, actual) -> bool:
    if type(expected) in (int, float) or type(actual) in (int, float):
        if not number(expected) or not number(actual):
            return False
    if number(expected) and number(actual):
        return abs(expected - actual) <= ATOL + RTOL * abs(expected)
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected.keys() == actual.keys() and all(equivalent(v, actual[k]) for k, v in expected.items())
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(
            equivalent(a, b) for a, b in zip(expected, actual, strict=True)
        )
    return type(expected) is type(actual) and expected == actual


def verify_artifacts(bundle: dict, root: Path) -> tuple[list[Finding], dict[str, Path]]:
    findings, verified = [], {}
    for artifact in bundle["artifacts"]:
        path = local_path(root, artifact["path"])
        expected, actual = artifact.get("sha256"), None
        if not path.is_file():
            status, message = Status.UNKNOWN, "Artifact unavailable; identity cannot be verified"
        else:
            actual = digest(path)
            if expected is None:
                status, message = Status.UNKNOWN, "Expected digest is missing"
            elif actual != expected:
                status, message = Status.FAIL, "Artifact differs from pinned identity"
            else:
                status, message = Status.PASS, "Artifact SHA-256 verified"
                verified[artifact["id"]] = path
        findings.append(
            Finding("artifacts", status, artifact["id"], message, expected=expected, actual=actual)
        )
    return findings, verified


def statistics(path: Path | None, size: int) -> dict | None:
    if path is None:
        return None
    stats = read_json(path)
    try:
        vector(stats["mean"], size, "stats.mean")
        vector(stats["std"], size, "stats.std")
        if any(x <= 0 for x in stats["std"]):
            raise InputError("Standard deviations must be positive")
        return stats
    except KeyError as exc:
        raise InputError("Statistics require mean and std") from exc


def check(bundle_path: str | Path, trace: dict | str | Path | None = None) -> dict:
    """Check a bounded JSON bundle against a captured trace. Does not execute adapters."""
    bundle, root = load_bundle(bundle_path)
    if isinstance(trace, (str, Path)):
        trace = read_json(Path(trace))
    findings, verified = verify_artifacts(bundle, root)
    contract = bundle["contract"]
    requested = contract["checks"]
    # Identity is always checked, even if callers omit it from their requested semantic checks.
    for name in requested:
        if name not in CHECKS:
            findings.append(
                Finding(name, Status.UNKNOWN, "coverage", "No executable reference for this check")
            )
    if trace is not None:
        ensure_json(trace)
        if not isinstance(trace, dict):
            raise InputError("Trace must be an object")
        if (
            type(trace.get("schema_version")) is not int
            or trace.get("schema_version") != 1
            or trace.get("bundle_sha256") != digest(Path(bundle_path))
        ):
            raise InputError("Trace schema or bundle digest mismatch; recapture against this exact bundle")
        if not isinstance(trace.get("events"), list) or len(trace["events"]) != len(bundle["scenario"]):
            raise InputError("Trace must contain one event for each scenario event")
        execution = trace.get("execution")
        if (
            not isinstance(execution, dict)
            or not isinstance(execution.get("scope"), str)
            or not execution["scope"]
        ):
            findings.append(
                Finding("execution", Status.UNKNOWN, "provenance", "Trace execution scope missing")
            )
    authority = contract.get("authority")
    reference_known = (
        isinstance(authority, dict) and isinstance(authority.get("source"), str) and bool(authority["source"])
    )
    joints = contract["joints"]
    names = [j["name"] for j in joints]
    scalar_units = {"degree", "radian", "meter", "range_0_100"}
    semantics_known = all(
        j.get("unit") in scalar_units and j.get("mode") in ("relative", "absolute") for j in joints
    )
    norm = contract.get("normalization", {})
    stats = (
        statistics(verified.get(norm.get("artifact")), len(joints))
        if norm.get("kind") == "mean_std"
        else None
    )
    queue = deque()
    anchor = None

    def compare(name, boundary, expected, actual, index, message):
        if name not in requested:
            return
        if not reference_known or trace is None or actual is None:
            findings.append(
                Finding(
                    name,
                    Status.UNKNOWN,
                    boundary,
                    "Independent reference or trace evidence missing",
                    index,
                    expected,
                    actual,
                )
            )
        else:
            status = Status.PASS if equivalent(expected, actual) else Status.FAIL
            findings.append(Finding(name, status, boundary, message, index, expected, actual))

    for i, event in enumerate(bundle["scenario"]):
        observed = trace["events"][i] if trace else {}
        if (
            isinstance(observed, dict)
            and observed.get("action") is not None
            and not isinstance(observed["action"], dict)
        ):
            raise InputError(f"Trace event {i} action must be an object")
        if not isinstance(observed, dict) or (observed and observed.get("kind") != event["kind"]):
            raise InputError(f"Trace event {i} kind does not match scenario")
        if event["kind"] == "reset":
            queue.clear()
            anchor = None
            compare(
                "reset",
                "episode.reset",
                True,
                observed.get("anchor_cleared"),
                i,
                "Reset clears cached anchor",
            )
            compare(
                "reset", "episode.queue", 0, observed.get("queue_remaining"), i, "Reset clears action queue"
            )
            continue
        state = [event["state"][name] for name in names]
        if not queue:
            anchor = state
        expected_cameras = {}
        cameras_available = bool(contract.get("cameras"))
        for camera in contract.get("cameras", []):
            if (
                camera["source"] not in event["cameras"]
                or not camera.get("role")
                or event["cameras"].get(camera["source"]) in (None, [], {})
            ):
                cameras_available = False
            else:
                expected_cameras[camera["target"]] = event["cameras"][camera["source"]]
        frames = list(expected_cameras.values())
        if any(equivalent(a, b) for k, a in enumerate(frames) for b in frames[k + 1 :]):
            cameras_available = False
        if not cameras_available:
            if "cameras" in requested:
                findings.append(
                    Finding("cameras", Status.UNKNOWN, "policy.images", "Camera reference missing", i)
                )
        else:
            compare(
                "cameras",
                "training.policy.images",
                expected_cameras,
                observed.get("training_cameras"),
                i,
                "Training camera routing must match independently labeled camera roles",
            )
            compare(
                "cameras",
                "policy.images",
                expected_cameras,
                observed.get("cameras"),
                i,
                "Camera content must reach the checkpoint's independently labeled input",
            )
        compare(
            "joint_order",
            "training.state_names",
            names,
            observed.get("training_state_names"),
            i,
            "Training input order must match the independent checkpoint contract",
        )
        compare(
            "joint_order",
            "training.state",
            state,
            observed.get("training_state_values"),
            i,
            "Training state values must preserve declared joint identity",
        )
        compare(
            "joint_order",
            "policy.state_names",
            names,
            observed.get("state_names"),
            i,
            "Checkpoint joint order must match executed input order",
        )
        compare(
            "joint_order",
            "policy.state",
            state,
            observed.get("state_values"),
            i,
            "Executed state vector must preserve named joint identity",
        )
        prediction = event.get("prediction")
        if prediction is not None:
            compare(
                "actions",
                "policy.horizon",
                contract["horizon"],
                observed.get("prediction_horizon"),
                i,
                "Predicted chunk must match declared horizon",
            )
            compare(
                "actions",
                "training.horizon",
                contract["horizon"],
                observed.get("training_horizon"),
                i,
                "Training chunk horizon must match contract",
            )
            if all(j.get("unit") in scalar_units for j in joints):
                compare(
                    "joint_order",
                    "training.units",
                    {j["name"]: j["unit"] for j in joints},
                    observed.get("training_units"),
                    i,
                    "Training convention must match declared units",
                )
            elif "joint_order" in requested:
                findings.append(
                    Finding(
                        "joint_order",
                        Status.UNKNOWN,
                        "training.units",
                        "Missing or unsupported scalar units",
                        i,
                    )
                )
            queue.extend(prediction)
            identity_training = contract.get("training_action_transform") == "identity"
            if stats and (semantics_known or identity_training) and event.get("training_targets") is not None:
                expected_training = []
                for target in event["training_targets"]:
                    vector(target, len(joints), "training_targets")
                    relative = [
                        x - anchor[k] if not identity_training and joints[k].get("mode") == "relative" else x
                        for k, x in enumerate(target)
                    ]
                    expected_training.append(
                        [(x - stats["mean"][k]) / (stats["std"][k] + 1e-8) for k, x in enumerate(relative)]
                    )
                compare(
                    "normalization",
                    "training.normalized_actions",
                    expected_training,
                    observed.get("training_normalized"),
                    i,
                    "Executed training transform matches scalar reference",
                )
            elif "normalization" in requested:
                findings.append(
                    Finding(
                        "normalization",
                        Status.UNKNOWN,
                        "training.normalized_actions",
                        "Training targets, statistics, or scalar semantics missing",
                        i,
                    )
                )
        if not queue:
            if observed.get("action") and "actions" in requested and reference_known:
                findings.append(
                    Finding(
                        "actions",
                        Status.FAIL,
                        "controller.commands",
                        "Unexpected command without a queued policy output",
                        i,
                        None,
                        observed["action"],
                    )
                )
            continue
        normalized = queue.popleft()
        action = observed.get("action") or {}
        if not isinstance(action, dict):
            raise InputError(f"Trace event {i} action must be an object")
        if not stats:
            if "normalization" in requested:
                findings.append(
                    Finding(
                        "normalization",
                        Status.UNKNOWN,
                        "action.unnormalize",
                        "Pinned training statistics or supported normalization missing",
                        i,
                    )
                )
            if "actions" in requested:
                findings.append(
                    Finding(
                        "actions",
                        Status.UNKNOWN,
                        "controller.commands",
                        "Cannot establish command semantics without training statistics",
                        i,
                    )
                )
            continue
        denormalized = [x * stats["std"][k] + stats["mean"][k] for k, x in enumerate(normalized)]
        vector(denormalized, len(joints), "computed reference action")
        compare(
            "normalization",
            "action.unnormalize",
            denormalized,
            action.get("denormalized"),
            i,
            "Deployment must interpret outputs using pinned training statistics",
        )
        if semantics_known:
            units = action.get("units")
            if units is not None and not isinstance(units, dict):
                raise InputError(f"Trace event {i} units must be an object")
            known_units = {
                name: unit
                for name, unit in (units or {}).items()
                if isinstance(unit, str) and unit in scalar_units and name in names
            }
            compare(
                "actions",
                "controller.units",
                {j["name"]: j["unit"] for j in joints if j["name"] in known_units},
                known_units or None,
                i,
                "Controller units must match the physical contract",
            )
            commands = {
                name + ".pos": denormalized[k] + (anchor[k] if joints[k]["mode"] == "relative" else 0)
                for k, name in enumerate(names)
            }
            vector(list(commands.values()), len(joints), "computed reference commands")
            if len(known_units) == len(joints):
                compare(
                    "actions",
                    "controller.commands",
                    commands,
                    action.get("commands"),
                    i,
                    "Named commands must preserve modes, gripper convention, and originating chunk anchor",
                )
            elif "actions" in requested:
                if known_units and isinstance(action.get("commands"), dict):
                    subset = {name + ".pos": commands[name + ".pos"] for name in known_units}
                    compare(
                        "actions",
                        "controller.commands.known_units",
                        subset,
                        {key: action["commands"].get(key) for key in subset},
                        i,
                        "Compare commands only where controller units are established",
                    )
                findings.append(
                    Finding(
                        "actions",
                        Status.UNKNOWN,
                        "controller.commands",
                        "Cannot interpret commands without controller units",
                        i,
                    )
                )
        elif "actions" in requested:
            findings.append(
                Finding("actions", Status.UNKNOWN, "controller.commands", "Joint units or modes missing", i)
            )
        compare(
            "actions",
            "policy.queue",
            len(queue),
            observed.get("queue_remaining"),
            i,
            "Queue consumption matches the declared one-action-per-tick lifecycle",
        )
    for name in requested:
        if not any(f.check == name for f in findings):
            findings.append(Finding(name, Status.UNKNOWN, "coverage", "No applicable fixture evidence"))
    report = make_report(findings, bundle, trace)
    report["bundle_sha256"] = digest(Path(bundle_path))
    report["tolerances"] = {"absolute": ATOL, "relative": RTOL}
    return report
