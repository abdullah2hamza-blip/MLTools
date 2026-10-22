"""Versioned affine research cases, explicit execution, replay and regression exports."""

import hashlib
import json
from pathlib import Path

from .io import InputError, digest, ensure_json, local_path, names, read_bytes, read_json, write_json
from .obligations import Reference, Suite, evaluate
from .report import aggregate

FORMAT = "mltools-affine-case"


def load(path):
    path = Path(path).resolve()
    case = read_json(path)
    try:
        if type(case["schema_version"]) is not int or case["schema_version"] != 1 or case["format"] != FORMAT:
            raise InputError("Unsupported affine case format/version")
        suite = Suite.from_dict(case["suite"])
        reference = Reference(**case["reference"])
        # Check reference dimensions and numbers even before evidence is supplied.
        evaluate(suite, reference, {"suite_sha256": suite.sha256, "outputs": {}})
        artifacts = case["artifacts"]
        if not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 32:
            raise InputError("Case requires 1..32 artifacts")
        names([a["id"] for a in artifacts], "artifacts")
        paths = []
        for artifact in artifacts:
            paths.append(local_path(path.parent, artifact["path"]))
            value = artifact.get("sha256")
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise InputError("Artifact hashes must be lowercase SHA-256 or unknown")
        if len(set(paths)) != len(paths) or path in paths:
            raise InputError("Artifacts must be distinct from each other and the case file")
        adapter = case["adapter"]
        if adapter["stats_artifact"] not in {a["id"] for a in artifacts}:
            raise InputError("Statistics artifact is not listed")
        config = adapter["configuration"]
        if adapter["framework"] == "openpi":
            if set(config) != {"model_type", "use_quantiles", "mask", "anchor"}:
                raise InputError("OpenPI configuration requires model_type/use_quantiles/mask/anchor")
            if (
                config["model_type"] not in ("pi0", "pi05", "pi0_fast")
                or type(config["use_quantiles"]) is not bool
            ):
                raise InputError("Invalid OpenPI normalization/model type")
            if (
                not isinstance(config["mask"], list)
                or len(config["mask"]) != 7
                or any(type(x) is not bool for x in config["mask"])
            ):
                raise InputError("OpenPI mask requires seven booleans")
            if config["anchor"] not in ("origin", "current"):
                raise InputError("Invalid OpenPI anchor")
        elif adapter["framework"] == "lerobot":
            if set(config) != {"relative", "bind_queue", "exclude_joints", "action_names", "motor_names"}:
                raise InputError("Invalid LeRobot probe configuration fields")
            if any(type(config[k]) is not bool for k in ("relative", "bind_queue")):
                raise InputError("LeRobot relative/bind_queue must be booleans")
            for field in ("action_names", "motor_names"):
                names(config[field], field)
            if len(config["action_names"]) != len(config["motor_names"]):
                raise InputError("Action and motor counts must match")
            names(config["exclude_joints"] or ["__none__"], "exclude_joints")
        else:
            raise InputError("Unsupported probe framework")
        if adapter["framework"] == "openpi":
            dimensions = (("action", 7), ("origin", 8), ("current", 8))
            outputs = 7
        else:
            if not isinstance(config["exclude_joints"], list):
                raise InputError("Excluded joints must be a list")
            outputs = len(config["action_names"])
            dimensions = tuple((port, outputs) for port in ("action", "origin", "current"))
            if set(suite.outputs) != {f"{name}.pos" for name in config["motor_names"]}:
                raise InputError("LeRobot output labels must match named motor ports")
        expected = {(port, str(i)) for port, n in dimensions for i in range(n)}
        if {(a.port, a.name) for a in suite.axes} != expected or len(suite.outputs) != outputs:
            raise InputError("Compiled suite does not match the reviewed framework port layout")
        if "checks" in case:
            names(case["checks"], "checks")
        return case, path.parent, suite, reference
    except (KeyError, TypeError, AttributeError) as exc:
        raise InputError(f"Invalid affine case: {exc}") from exc


def _sources(framework):
    if framework == "lerobot":
        from .lerobot import BLOBS, REVISION

        return REVISION, BLOBS, None
    from ._openpi_worker import BLOBS, REVISION

    return REVISION, BLOBS, "7a971b9d5f6b1495fd6cdea202ffa607d8b34bf0"


def capture(path, *, python=None):
    """Explicitly execute only a reviewed built-in integration; no artifact code loading."""
    source_sha = digest(Path(path))
    case, root, suite, _reference = load(path)
    if digest(Path(path)) != source_sha:
        raise InputError("Case changed while preparing execution")
    adapter = case["adapter"]
    artifact = next(a for a in case["artifacts"] if a["id"] == adapter["stats_artifact"])
    stats = local_path(root, artifact["path"])
    if adapter["framework"] == "openpi":
        from .openpi import capture as run

        if python is None:
            raise InputError("OpenPI execution requires an explicit worker interpreter")
        trace = run(suite, read_bytes(stats).decode("utf-8"), python=python, **adapter["configuration"])
    else:
        from .probe_capture import capture as run

        if python is not None:
            raise InputError("LeRobot executes in the current optional-integration environment")
        trace = run(suite, stats, adapter["configuration"])
    if digest(Path(path)) != source_sha or digest(stats) != trace["stats_sha256"]:
        raise InputError("Case/statistics changed during execution; discard this capture")
    trace["case_sha256"] = source_sha
    trace["framework"] = adapter["framework"]
    return trace


def check(path, trace=None):
    """Data-only replay; missing provenance stays UNKNOWN even if numbers agree."""
    case, root, suite, reference = load(path)
    if isinstance(trace, (str, Path)):
        trace = read_json(Path(trace))
    trace = {} if trace is None else trace
    ensure_json(trace)
    if not isinstance(trace, dict):
        raise InputError("Probe trace must be a JSON object")
    if trace.get("schema_version") is not None and (
        type(trace["schema_version"]) is not int or trace["schema_version"] != 1
    ):
        raise InputError("Unsupported probe trace version")
    guards = []

    def compare(check_name, field, expected, actual):
        status = "UNKNOWN" if expected is None or actual is None else "PASS" if expected == actual else "FAIL"
        guards.append(
            {"check": check_name, "boundary": field, "status": status, "expected": expected, "actual": actual}
        )

    case_sha = digest(Path(path))
    compare("binding", "schema_version", 1, trace.get("schema_version"))
    compare("binding", "case_sha256", case_sha, trace.get("case_sha256"))
    compare("binding", "suite_sha256", suite.sha256, trace.get("suite_sha256"))
    compare("binding", "framework", case["adapter"]["framework"], trace.get("framework"))
    compare("binding", "configuration", case["adapter"]["configuration"], trace.get("configuration"))
    revision, blobs, client_blob = _sources(case["adapter"]["framework"])
    identity = trace.get("identity") or {}
    if not isinstance(identity, dict):
        raise InputError("Trace identity must be an object")
    compare("producer", "revision", revision, identity.get("revision"))
    for name, expected in blobs.items():
        actual = identity.get("git_blobs") or {}
        if not isinstance(actual, dict):
            raise InputError("Source blobs must be an object")
        compare("producer", f"source:{name}", expected, actual.get(name))
    if client_blob:
        compare("producer", "client_image_tools_blob", client_blob, identity.get("client_image_tools_blob"))
    for artifact in case["artifacts"]:
        file = local_path(root, artifact["path"])
        actual = digest(file) if file.is_file() else None
        compare("artifacts", artifact["id"], artifact.get("sha256"), actual)
        if artifact["id"] == case["adapter"]["stats_artifact"]:
            compare("artifacts", "executed_stats_sha256", actual, trace.get("stats_sha256"))
    bound = trace.get("suite_sha256") == suite.sha256
    observations = trace if bound else {"suite_sha256": suite.sha256, "outputs": {}}
    numeric = evaluate(suite, reference, observations)
    checks = {
        name: aggregate(g["status"] for g in guards if g["check"] == name)
        for name in ("binding", "producer", "artifacts")
    }
    checks["obligations"] = numeric["status"]
    for name in case.get("checks", []):
        if name not in checks:
            checks[name] = "UNKNOWN"
            guards.append(
                {
                    "check": name,
                    "boundary": "unsupported_check",
                    "status": "UNKNOWN",
                    "expected": "independently supported check",
                    "actual": None,
                }
            )
    return {
        "schema_version": 1,
        "format": "mltools-affine-report",
        "status": aggregate(checks.values()),
        "case_sha256": case_sha,
        "suite_sha256": suite.sha256,
        "checks": checks,
        "provenance_findings": guards,
        "numeric_report": numeric,
        "trace_sha256": hashlib.sha256(
            json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "first_divergence": next((g for g in guards if g["status"] == "FAIL"), numeric["first_divergence"]),
        "limitations": numeric["limitations"]
        + [
            "Recorded producer/source identity is asserted metadata, not cryptographic proof of execution",
            "Only affine numerical obligations and explicit artifact/binding/source checks are evaluated",
            "Unbound numerical evidence is not reinterpreted as observations of the requested suite",
        ],
    }


def export(path, trace, report, destination):
    """Preserve original replay and provide a reduced case for fresh real execution."""
    if report["status"] != "FAIL":
        raise InputError("Only a failing case has a regression export")
    case, root, suite, _reference = load(path)
    if isinstance(trace, (str, Path)):
        trace = read_json(Path(trace))
    trace = {} if trace is None else trace
    if check(path, trace) != report:
        raise InputError("Export requires the current case/trace report")
    destination = Path(destination).resolve()
    inputs = [Path(path).resolve()] + [local_path(root, a["path"]) for a in case["artifacts"]]
    if destination.exists() or any(p.is_relative_to(destination) for p in inputs):
        raise InputError("Regression destination must be new and cannot contain case inputs")
    destination.mkdir(parents=True)
    copied_root = destination / "original"
    copied_root.mkdir()
    for artifact in case["artifacts"]:
        source = local_path(root, artifact["path"])
        target = local_path(copied_root, artifact["path"])
        if source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
    copied_case = copied_root / Path(path).name
    copied_case.write_bytes(Path(path).read_bytes())
    write_json(destination / "trace.json", trace)
    write_json(destination / "report.json", report)
    queries = {q["query"] for q in report["numeric_report"]["witness"]}
    child_path = None
    if queries:
        reduced = Suite(suite.axes, suite.outputs, tuple(p for p in suite.probes if p.id in queries))
        child = dict(case, suite=reduced.to_dict())
        reserved = {local_path(copied_root, a["path"]) for a in case["artifacts"]} | {copied_case}
        for index in range(len(reserved) + 2):
            candidate = copied_root / f"witness-{index}.case.json"
            if candidate not in reserved and not candidate.exists():
                child_path = candidate
                break
        write_json(child_path, child)
    write_json(
        destination / "export.json",
        {
            "schema_version": 1,
            "source_case_sha256": report["case_sha256"],
            "retained_queries": sorted(queries),
            "offline_case": copied_case.relative_to(destination).as_posix(),
            "offline_trace": "trace.json",
            "witness_case": child_path.relative_to(destination).as_posix() if child_path else None,
            "scope": "Original replay retained; reduced case requires fresh capture",
        },
    )
    (destination / "README.txt").write_text(
        f"Offline: mltools probe original/{Path(path).name} --trace trace.json\n"
        "Fresh execution: use the case or reduced witness with --execute and its required environment.\n"
        "A reduced witness proves a local violated relation, not a globally minimal causal trajectory.\n",
        encoding="utf-8",
        newline="\n",
    )
