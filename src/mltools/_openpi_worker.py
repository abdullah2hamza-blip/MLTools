"""Standalone Python 3.11 worker; invoked only by explicit OpenPI capture.

No MLTools import is needed in this incompatible dependency environment.
This is an executed deployment composition around genuine upstream transforms,
not a replacement for OpenPI model inference or its training runtime.
"""

import ctypes
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import sys
import time
from collections import deque
from pathlib import Path

REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
BLOBS = {
    "transforms.py": "272375ea95a5e43a42a6c0ff14cf89d34da76a45",
    "shared/normalize.py": "5a75faecfbf4267329824b103e4cdd091d729840",
    "policies/libero_policy.py": "fe5aab0add5795531913363d7c46d916c81d1f9b",
    "models/tokenizer.py": "8a4966d6298619e52c7ba53359ccbbb8ba0b8cf6",
    "models/model.py": "29618b49453742266fe6e4a5815ceee06d815f3b",
    "shared/array_typing.py": "569eafef1401d4a34b0289c0efb2125f71d21dfb",
    "models_pytorch/pi0_pytorch.py": "e68ddb7cc02c2fd256e9b48ac9d0fb9df966536a",
    "models_pytorch/gemma_pytorch.py": "ecc597a9a23ffb26db8c1907fd264c4af797d786",
}
MAX_BYTES = 2 * 1024 * 1024


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _json(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_reject_constant)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def verify_installation():
    direct_urls = {}
    for name in ("openpi", "openpi-client"):
        dist = importlib.metadata.distribution(name)
        direct = _json(dist.read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != REVISION:
            raise ValueError(f"{name} requires the documented immutable Git revision")
        direct_urls[name] = direct
    spec = importlib.util.find_spec("openpi")
    root = Path(next(iter(spec.submodule_search_locations)))
    for relative, expected in BLOBS.items():
        raw = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
        if blob != expected:
            raise ValueError(f"Installed OpenPI source identity mismatch: {relative}")
    client_spec = importlib.util.find_spec("openpi_client")
    client_root = Path(next(iter(client_spec.submodule_search_locations)))
    raw = (client_root / "image_tools.py").read_bytes().replace(b"\r\n", b"\n")
    client_blob = hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()
    if client_blob != "7a971b9d5f6b1495fd6cdea202ffa607d8b34bf0":
        raise ValueError("Installed OpenPI client image source identity mismatch")
    return {
        "revision": REVISION,
        "git_blobs": BLOBS,
        "source_newlines": "canonical LF",
        "direct_urls": direct_urls,
        "client_image_tools_blob": client_blob,
    }


def _rss():
    if os.name != "nt":
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == "darwin" else value * 1024)

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_ws",
                "ws",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    info = Counters()
    info.cb = ctypes.sizeof(info)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
        raise OSError("Cannot measure worker peak working set")
    return info.peak_ws


def execute(request):
    start = time.perf_counter()
    if (
        not isinstance(request, dict)
        or type(request.get("schema_version")) is not int
        or request["schema_version"] != 1
    ):
        raise ValueError("Unsupported worker request")
    suite = request["suite"]
    canonical = json.dumps(suite, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if _sha(canonical) != request["suite_sha256"]:
        raise ValueError("Compiled suite fingerprint mismatch")
    axes, probes = suite["axes"], suite["probes"]
    if (
        type(suite.get("schema_version")) is not int
        or suite["schema_version"] != 1
        or suite.get("method") != "affine-obligations-development"
    ):
        raise ValueError("Unsupported compiled suite")
    keys = [(a["port"], a["name"]) for a in axes]
    expected = {(p, str(i)) for p, n in (("action", 7), ("origin", 8), ("current", 8)) for i in range(n)}
    if len(keys) != 23 or set(keys) != expected or len(suite["outputs"]) != 7:
        raise ValueError("Unsupported LIBERO port layout")
    if not isinstance(probes, list) or not 1 <= len(probes) <= 193:
        raise ValueError("Invalid query count")
    centers = []
    for axis in axes:
        lower, upper = axis["lower"], axis["upper"]
        if not _finite(lower) or not _finite(upper) or not lower < upper:
            raise ValueError("Invalid probe bounds")
        center, radius = lower / 2 + upper / 2, upper / 2 - lower / 2
        if not _finite(center) or not _finite(radius) or radius <= 0:
            raise ValueError("Probe bounds exceed numeric range")
        if not lower < center < upper:
            raise ValueError("Probe bounds do not admit a distinct representable center")
        centers.append(center)
    canonical_probes = {"center": (centers, [0.0] * 23)}
    for j, axis in enumerate(axes):
        for sign, word in ((1.0, "plus"), (-1.0, "minus")):
            values, coords = centers[:], [0.0] * 23
            values[j], coords[j] = axis["upper"] if sign == 1 else axis["lower"], sign
            canonical_probes[f"{word}:{axis['port']}:{axis['name']}"] = values, coords
    identifiers = set()
    for probe in probes:
        if not isinstance(probe["id"], str) or probe["id"] in identifiers:
            raise ValueError("Query identifiers must be unique strings")
        identifiers.add(probe["id"])
        if len(probe["values"]) != 23 or not all(_finite(x) for x in probe["values"]):
            raise ValueError("Query inputs must contain 23 finite numbers")
        coordinates = probe["coordinates"]
        if len(coordinates) != 23 or not all(_finite(x) for x in coordinates):
            raise ValueError("Query coordinates must contain 23 finite numbers")
        if canonical_probes.get(probe["id"]) != (probe["values"], coordinates):
            raise ValueError("Query identity, coordinates and values do not match the bounded design")
        for value, axis in zip(probe["values"], axes, strict=True):
            if (
                not _finite(axis["lower"])
                or not _finite(axis["upper"])
                or not axis["lower"] <= value <= axis["upper"]
            ):
                raise ValueError("Probe exceeds declared bounds")
    mode, quantiles, mask, anchor = (request[k] for k in ("model_type", "use_quantiles", "mask", "anchor"))
    if mode not in ("pi0", "pi05", "pi0_fast") or type(quantiles) is not bool:
        raise ValueError("Unsupported model or normalization mode")
    if not isinstance(mask, list) or len(mask) != 7 or any(type(x) is not bool for x in mask):
        raise ValueError("Mask requires seven explicit booleans")
    if anchor not in ("origin", "current"):
        raise ValueError("Unsupported anchor")
    stats_text = request["stats_json"]
    if not isinstance(stats_text, str) or len(stats_text.encode("utf-8")) > 65536:
        raise ValueError("Statistics exceed size bound")
    stats_data = _json(stats_text)
    entries = stats_data["norm_stats"]
    if not isinstance(entries, dict) or set(entries) != {"actions", "state"}:
        raise ValueError("LIBERO statistics require exactly actions and state")
    for entry in entries.values():
        if set(entry) - {"mean", "std", "q01", "q99"} or not {"mean", "std"} <= set(entry):
            raise ValueError("Unsupported statistics fields")
        sizes = set()
        for values in entry.values():
            if values is not None:
                if (
                    not isinstance(values, list)
                    or not 1 <= len(values) <= 32
                    or not all(_finite(x) for x in values)
                ):
                    raise ValueError("Statistics require 1..32 finite numbers per vector")
                sizes.add(len(values))
        if len(sizes) != 1:
            raise ValueError("Statistics vectors have inconsistent sizes")
        if any(x < 0 for x in entry["std"]):
            raise ValueError("Standard deviations must be nonnegative")
        if quantiles and (entry.get("q01") is None or entry.get("q99") is None):
            raise ValueError("Quantile mode requires q01 and q99")
        if entry.get("q01") is not None and entry.get("q99") is not None:
            if any(a > b for a, b in zip(entry["q01"], entry["q99"], strict=True)):
                raise ValueError("Quantiles must be ordered")
    if len(entries["actions"]["mean"]) < 7 or len(entries["state"]["mean"]) < 8:
        raise ValueError("Statistics do not cover LIBERO action/state dimensions")
    identity = verify_installation()
    import numpy as np
    from openpi import transforms as transforms
    from openpi.models.model import ModelType
    from openpi.policies.libero_policy import LiberoInputs, LiberoOutputs
    from openpi.shared.normalize import deserialize_json

    stats = deserialize_json(stats_text)
    model_type = {"pi0": ModelType.PI0, "pi05": ModelType.PI05, "pi0_fast": ModelType.PI0_FAST}[mode]
    inputs, outputs = LiberoInputs(model_type), LiberoOutputs()
    normalizer = transforms.Normalize(stats, use_quantiles=quantiles, strict=True)
    unnormalizer = transforms.Unnormalize({"actions": stats["actions"]}, use_quantiles=quantiles)
    delta, absolute = transforms.DeltaActions(mask), transforms.AbsoluteActions(mask)
    base = np.full((4, 5, 3), 31, dtype=np.uint8)
    wrist = np.full((4, 5, 3), 197, dtype=np.uint8)
    image_hashes = {_sha(base.tobytes()): "base", _sha(wrist.tobytes()): "wrist"}
    # No normalized-state values are reused as the physical offset of a later action.
    result, traces = {}, []
    for probe in probes:
        coordinates = dict(zip(keys, probe["values"], strict=True))

        def vector(port, size, coordinates=coordinates):
            return np.array([coordinates[(port, str(i))] for i in range(size)], dtype=np.float64)

        origin, current = vector("origin", 8), vector("current", 8)
        prediction = np.stack([vector("action", 7)] * 2)
        deployed_input = inputs(
            {
                "observation/state": origin.copy(),
                "observation/image": base.copy(),
                "observation/wrist_image": wrist.copy(),
            }
        )
        # Reuse the numerical action probe as a synthetic training-side target.
        # It is not independently supplied physical training data or evidence of
        # parity with the checkpoint's actual training distribution/conventions.
        training_targets = prediction.copy()
        training = inputs(
            {
                "observation/state": origin.copy(),
                "observation/image": base.copy(),
                "observation/wrist_image": wrist.copy(),
                "actions": training_targets.copy(),
            }
        )
        relative = delta(training)
        training_relative = relative["actions"].copy()
        normalized = normalizer(relative)
        normalized_actions = normalized["actions"].copy()
        padded = transforms.PadStatesAndActions(32)(normalized)
        # Real inference postprocessing unnormalizes a full predicted chunk before
        # queuing. The current-anchor variant is an explicit composition fault.
        unnormalized = unnormalizer({"actions": np.pad(prediction, ((0, 0), (0, 25)))})["actions"]
        converted = absolute({"state": deployed_input["state"].copy(), "actions": unnormalized.copy()})
        commands = outputs(converted)["actions"]
        if anchor == "current":
            moved = absolute({"state": current.copy(), "actions": unnormalized[1:].copy()})
            commands[1:] = outputs(moved)["actions"]
        queue = deque(commands.copy())
        first, second = queue.popleft(), queue.popleft()
        result[probe["id"]] = second.tolist()
        traces.append(
            {
                "query": probe["id"],
                "origin": origin.tolist(),
                "current": current.tolist(),
                "normalized_prediction": prediction.tolist(),
                "unnormalized_chunk": unnormalized.tolist(),
                "first_action": first.tolist(),
                "second_action": second.tolist(),
                "queue_remaining": len(queue),
                "training_targets": training_targets.tolist(),
                "training_relative": training_relative.tolist(),
                "training_normalized": normalized_actions.tolist(),
                "training_padded_state_shape": list(padded["state"].shape),
                "training_padded_action_shape": list(padded["actions"].shape),
            }
        )
    camera_routes = {
        k: image_hashes.get(_sha(v.tobytes()), "padding_or_unknown")
        for k, v in deployed_input["image"].items()
    }
    peak = _rss()
    if peak > 2 * 1024**3:
        raise ValueError("Worker exceeded development peak working-set budget")
    return {
        "schema_version": 1,
        "suite_sha256": request["suite_sha256"],
        "outputs": result,
        "identity": identity,
        "stats_sha256": _sha(stats_text.encode("utf-8")),
        "training_evidence_scope": "Reused synthetic probes; not independent physical training data",
        "configuration": {"model_type": mode, "use_quantiles": quantiles, "mask": mask, "anchor": anchor},
        "camera_routes": camera_routes,
        "camera_masks": {k: bool(v) for k, v in deployed_input["image_mask"].items()},
        "traces": traces,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "wall_seconds": time.perf_counter() - start,
            "peak_working_set_bytes": peak,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
        "scope": "Development processor execution; no policy inference or hardware command",
    }


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError("Worker request exceeds size bound")
        response = execute(_json(raw))
        encoded = json.dumps(response, allow_nan=False)
        if len(encoded.encode()) > MAX_BYTES:
            raise ValueError("Worker result exceeds size bound")
        print(encoded)
        return 0
    except Exception as exc:
        print(json.dumps({"schema_version": 1, "error": f"{type(exc).__name__}: {exc}"}))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
