"""Genuine historical OpenPI data factories; unused model preprocessing is isolated."""

import argparse
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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "integrations/openpi-history"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(f"Historical OpenPI capture: {message}")


def installation(phase, ledger):
    identity = ledger["phases"][phase]
    direct_urls = {}
    for distribution, module, key in (
        ("openpi", "openpi", "module_git_blobs"),
        ("openpi-client", "openpi_client", "client_module_git_blobs"),
    ):
        dist = importlib.metadata.distribution(distribution)
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        require(direct.get("vcs_info", {}).get("commit_id") == identity["revision"], distribution)
        direct_urls[distribution] = direct
        spec = importlib.util.find_spec(module)
        base = Path(next(iter(spec.submodule_search_locations)))
        actual_paths = {p.relative_to(base).as_posix() for p in base.rglob("*.py")}
        require(actual_paths == set(identity[key]), f"{module} installed source inventory")
        for name, expected in identity[key].items():
            raw = (base / name).read_bytes().replace(b"\r\n", b"\n")
            blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
            require(blob == expected, f"{module}/{name} source identity")
    return {**identity, "direct_urls": direct_urls, "source_newlines": "canonical LF"}


def peak_rss():
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    initial = {
        p: p.read_bytes()
        for p in (Path(__file__), args.request, PROFILE / "fixture.json", PROFILE / "sources.json")
    }
    initial_hashes = {p: hashlib.sha256(raw).hexdigest() for p, raw in initial.items()}
    require(len(initial[args.request]) <= 128 * 1024, "request size")
    request = json.loads(initial[args.request])
    fixture = json.loads(initial[PROFILE / "fixture.json"])
    ledger = json.loads(initial[PROFILE / "sources.json"])
    require(request["fixture_sha256"] == initial_hashes[PROFILE / "fixture.json"], "fixture identity")
    suite = request["suite"]
    require(1 <= len(suite["probes"]) <= 31, "query count")
    keys = [(a["port"], a["name"]) for a in suite["axes"]]
    expected_keys = [(p, str(i)) for p, n in (("action", 7), ("origin", 8)) for i in range(n)]
    require(keys == expected_keys, "query port layout")
    canonical = json.dumps(suite, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(hashlib.sha256(canonical).hexdigest() == request["suite_sha256"], "suite identity")
    seen = set()
    centers = fixture["legacy_targets"][0] + fixture["origin"]
    radius = fixture["probe_radius"]
    for query in suite["probes"]:
        require(query["id"] not in seen, "duplicate query")
        seen.add(query["id"])
        coordinates = [0.0] * 15
        values = centers[:]
        if query["id"] != "center":
            sign, port, name = query["id"].split(":")
            require(sign in ("plus", "minus") and (port, name) in keys, "query identity")
            index = keys.index((port, name))
            coordinates[index] = 1.0 if sign == "plus" else -1.0
            values[index] += radius * coordinates[index]
        require(query["coordinates"] == coordinates, "query coordinates")
        require(
            len(query["values"]) == 15
            and all(math.isfinite(v) for v in query["values"])
            and all(abs(a - b) <= 1e-15 for a, b in zip(query["values"], values, strict=True)),
            "query values",
        )
    identity = installation(args.phase, ledger)
    blocked_network_events = []

    def audit(event, _arguments):
        if event in ("socket.connect", "socket.getaddrinfo"):
            blocked_network_events.append(event)
            raise RuntimeError("Historical processor capture forbids network connections")

    sys.addaudithook(audit)
    started = time.perf_counter()
    import dataclasses

    import jax
    import numpy as np
    from openpi import transforms
    from openpi.training import config

    args.assets.mkdir(parents=True, exist_ok=False)
    train = config.get_config("pi0_libero")
    require(type(train.data) is config.LeRobotLiberoDataConfig, "named factory type")
    calls = []
    original = config.ModelTransformFactory.__call__

    def unused_model_preprocessing(_self, model_config):
        calls.append(type(model_config).__name__)
        return transforms.Group()

    config.ModelTransformFactory.__call__ = unused_model_preprocessing
    try:
        # Historical urllib URL parsing misclassifies a Windows C: absolute path.
        # A relative local path takes its genuine local-file branch unchanged.
        assets_path = Path(os.path.relpath(args.assets, Path.cwd())).as_posix()
        factory = dataclasses.replace(train.data, assets=config.AssetsConfig(assets_dir=assets_path))
        data = factory.create(args.assets, train.model)
    finally:
        config.ModelTransformFactory.__call__ = original
    require(len(calls) == 1, "model-preprocessing isolation call count")
    require(data.norm_stats is None, "empty local assets unexpectedly yielded statistics")
    require(not data.model_transforms.inputs and not data.model_transforms.outputs, "isolated scope")
    require(train.model.action_dim == 32, "historical PI0 padding dimension")

    def names(items):
        return [type(item).__name__ for item in items]

    inputs = names(data.data_transforms.inputs)
    outputs = names(data.data_transforms.outputs)
    require(
        inputs == (["LiberoInputs"] if args.phase == "before" else ["LiberoInputs", "DeltaActions"])
        and outputs
        == (["LiberoOutputs"] if args.phase == "before" else ["AbsoluteActions", "LiberoOutputs"]),
        "genuine factory transform order",
    )

    def incoming(state, actions=None, repack=False):
        image = np.full((4, 5, 3), 17, dtype=np.uint8)
        wrist = np.full((4, 5, 3), 91, dtype=np.uint8)
        state = np.asarray(state, dtype=np.float64).copy()
        if repack:
            value = {"image": image, "wrist_image": wrist, "state": state, "prompt": "fixture"}
        else:
            value = {
                "observation/image": image,
                "observation/wrist_image": wrist,
                "observation/state": state,
                "prompt": "fixture",
            }
        if actions is not None:
            value["actions"] = np.asarray(actions, dtype=np.float64).copy()
        if repack:
            value = transforms.compose(data.repack_transforms.inputs)(value)
        return transforms.compose(data.data_transforms.inputs)(value)

    def outgoing(state, actions):
        padded = np.zeros((len(actions), 32), dtype=np.float64)
        padded[:, :7] = actions
        state = incoming(state)["state"]
        return transforms.compose(data.data_transforms.outputs)({"state": state, "actions": padded})[
            "actions"
        ]

    origin = fixture["origin"]
    trained = incoming(origin, fixture["dataset_actions"], repack=True)
    deployed = outgoing(origin, fixture["legacy_targets"])
    roundtrip = outgoing(origin, trained["actions"][:, :7])
    queries = {}
    for query in suite["probes"]:
        values = query["values"]
        # Two independently supplied legacy model outputs, both sharing one origin.
        action = values[:7]
        observed = outgoing(values[7:], [action, action])
        require(np.array_equal(observed[0], observed[1]), "chunk rows use different origins")
        queries[query["id"]] = observed[1].tolist()
    require(not blocked_network_events, "a network attempt was intercepted")
    capture = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "classification": "genuine_historical_data_factory_with_isolated_unused_model_preprocessing",
        "phase": args.phase,
        "identity": identity,
        "fixture_sha256": initial_hashes[PROFILE / "fixture.json"],
        "request_sha256": initial_hashes[args.request],
        "worker_sha256": initial_hashes[Path(__file__)],
        "source_ledger_sha256": initial_hashes[PROFILE / "sources.json"],
        "suite_sha256": request["suite_sha256"],
        "outputs": queries,
        "factory": {
            "name": "pi0_libero",
            "repo_id": data.repo_id,
            "repack_inputs": names(data.repack_transforms.inputs),
            "model_inputs": names(data.model_transforms.inputs),
            "model_outputs": names(data.model_transforms.outputs),
            "data_inputs": inputs,
            "data_outputs": outputs,
            "normalization_statistics": None,
            "model_preprocessing_substitution": "Only ModelTransformFactory.__call__: genuine empty Group",
            "substitution_calls": calls,
            "original_method_restored": config.ModelTransformFactory.__call__ is original,
            "model_action_dim": train.model.action_dim,
        },
        "boundary": {
            "training_targets": trained["actions"][:, :7].tolist(),
            "training_shape": list(trained["actions"].shape),
            "training_padding_zero": bool(np.all(trained["actions"][:, 7:] == 0)),
            "state_padding_zero": bool(np.all(trained["state"][8:] == 0)),
            "legacy_output_actions": deployed.tolist(),
            "native_roundtrip": roundtrip.tolist(),
            "image_pixels": {key: int(value[0, 0, 0]) for key, value in trained["image"].items()},
            "image_masks": {key: bool(value) for key, value in trained["image_mask"].items()},
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": str(jax.__version__),
            "numpy": str(np.__version__),
            "seconds_including_imports": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss(),
            "dependencies": {
                dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
            },
            "network_guard": "Audit hook rejects socket.connect/getaddrinfo; not OS isolation",
            "blocked_network_events": blocked_network_events,
        },
        "limits": [
            "Unmodified installed Python sources; only unused model preprocessing isolated in memory",
            "CPU processor subset, not original full declared training dependency environment",
            "Synthetic values and source-backed configured legacy convention; no checkpoint outputs",
            "No normalization, tokenizer/model, controller, simulator, reset or policy queue",
            "No physical-unit calibration, checkpoint-training attestation or task-success claim",
        ],
    }
    require(config.ModelTransformFactory.__call__ is original, "method restoration")
    require(installation(args.phase, ledger) == identity, "installed sources changed during capture")
    require(all(sha(p) == digest for p, digest in initial_hashes.items()), "capture inputs changed")
    args.output.write_text(
        json.dumps(capture, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{args.phase}: genuine data factory executed ({len(queries)} queries)", flush=True)


if __name__ == "__main__":
    main()
