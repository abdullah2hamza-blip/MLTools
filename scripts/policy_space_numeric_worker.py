"""Explicit offline native Policy Space custom-contract replay worker."""

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from action_study_common import fingerprint  # noqa: E402
from normalization_profiles import require  # noqa: E402
from policy_space_numeric import NEGATIVE_CONTROLS, metadata, scalar_descriptor, selection  # noqa: E402
from validate_native_baselines import identities  # noqa: E402

from mltools._openpi_worker import _rss  # noqa: E402


def execute(plan, decoder, contract_dir):
    """Reusable native engine; the public entry point enforces development scope."""
    require(
        all(importlib.util.find_spec(m) is None for m in ("torch", "lerobot", "openpi")),
        "isolated no-ML Policy Space environment",
    )
    start = time.perf_counter()
    identity = identities()
    import numpy as np
    from policy_space import PolicyModel
    from policy_space.runtime.server import (
        PolicySpaceServer,
        _pack_array,
        _safe_packb,
        _safe_unpackb,
        _unpack_array,
    )

    class ReplayModel(PolicyModel):
        def __init__(self):
            super().__init__({})
            self.calls = dict.fromkeys(("initialize", "ingest", "infer_replay", "finalize"), 0)
            self.actions, self.initialized_with, self.ingested = None, None, None

        def initialize_episode(self, episode_info=None):
            self.calls["initialize"] += 1
            self.initialized_with = episode_info

        def ingest_observation(self, obs):
            self.calls["ingest"] += 1
            self.ingested = obs["state"]["normalized_action"].copy()

        def infer_actions(self):
            self.calls["infer_replay"] += 1
            return self.actions.copy()

        def finalize_episode(self, episode_info=None):
            self.calls["finalize"] += 1

    def dispatch(server, endpoint, payload):
        raw = _safe_packb(payload, default=_pack_array)
        decoded = _safe_unpackb(raw, object_hook=_unpack_array)
        response = server._dispatch(endpoint, decoded)
        return _safe_unpackb(_safe_packb(response, default=_pack_array), object_hook=_unpack_array)

    def describe(array):
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": [scalar_descriptor(v) for v in array.reshape(-1).tolist()],
        }

    def infer(server, model, observation, step, episode_id):
        model.actions = np.asarray([observation["output"]], dtype=np.float64)
        payload = {
            "episode_id": episode_id,
            "step": step,
            "state": {"normalized_action": np.asarray(observation["input"], dtype=np.float64)},
        }
        return dispatch(server, "infer_actions", payload)

    previous = os.environ.get("POLICY_SPACE_CONTRACTS_DIR")
    os.environ["POLICY_SPACE_CONTRACTS_DIR"] = str(contract_dir)
    setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    rows, controls, accepted, rejected, registry = {}, {}, 0, 0, None
    try:
        for cell_id, cell in plan["cells"].items():
            rows[cell_id] = {}
            for condition in cell["conditions"]:
                meta = metadata(cell, condition)
                model = ReplayModel()
                server = PolicySpaceServer(model, metadata=meta)
                loaded = dataclasses.asdict(server._contract_registry)
                require(registry is None or registry == loaded, "stable native contract registry")
                registry = loaded
                selected = selection(cell_id, condition, meta)
                initialized = dispatch(server, "initialize_episode", selected)
                selected_pair = server._active_schema_pair.to_mapping()
                designs, step = {}, 0
                for design_id, observations in decoder["cells"][cell_id][condition]["designs"].items():
                    designs[design_id] = []
                    for observation in observations:
                        result = infer(server, model, observation, step, selected["episode_id"])
                        designs[design_id].append(
                            {
                                "decision": "ACCEPT",
                                "query": observation["query"],
                                "input": observation["input"],
                                "ingested_input": model.ingested.tolist(),
                                "replayed_output": model.actions[0].tolist(),
                                "input_dtype": str(model.ingested.dtype),
                                "wire_output": result["actions"][0].tolist(),
                                "wire_shape": list(result["actions"].shape),
                                "wire_dtype": str(result["actions"].dtype),
                            }
                        )
                        step += 1
                        accepted += 1
                finalized = dispatch(server, "finalize_episode", {"episode_id": selected["episode_id"]})
                rows[cell_id][condition] = {
                    "metadata": meta,
                    "initialized_with": model.initialized_with,
                    "initialize_response": initialized,
                    "selected_schema_pair": selected_pair,
                    "finalize_response": finalized,
                    "designs": designs,
                    "callback_counts": model.calls,
                }
            controls[cell_id] = {}
            base = decoder["cells"][cell_id]["numeric_control"]["designs"]["canonical"][0]
            for name in NEGATIVE_CONTROLS:
                model = ReplayModel()
                meta = metadata(cell, "numeric_control")
                server = PolicySpaceServer(model, metadata=meta)
                selected = selection(cell_id, "numeric_control", meta, "-" + name)
                dispatch(server, "initialize_episode", selected)
                model.actions = np.asarray([base["output"]], dtype=np.float64)
                value = np.asarray(base["input"], dtype=np.float64)
                if name == "wrong_output_width":
                    model.actions = model.actions[:, :-1]
                elif name == "nonfinite_output":
                    model.actions[0, 0] = np.nan
                elif name == "wrong_input_width":
                    value = value[:-1]
                else:
                    value[0] = np.nan
                payload = {
                    "episode_id": selected["episode_id"],
                    "step": 0,
                    "state": {"normalized_action": value},
                }
                decoded = _safe_unpackb(_safe_packb(payload, default=_pack_array), object_hook=_unpack_array)
                actual_arrays = {
                    "request_input": describe(decoded["state"]["normalized_action"]),
                    "model_output": describe(model.actions),
                }
                try:
                    server._dispatch("infer_actions", decoded)
                except ValueError as exc:
                    control = {"decision": "REJECT", "native_error": str(exc)}
                    rejected += 1
                else:
                    control = {"decision": "ACCEPT", "native_error": ""}
                dispatch(server, "finalize_episode", {"episode_id": selected["episode_id"]})
                controls[cell_id][name] = {
                    **control,
                    "base_query": base,
                    "actual_arrays": actual_arrays,
                    "callback_counts": model.calls,
                }
    finally:
        if previous is None:
            os.environ.pop("POLICY_SPACE_CONTRACTS_DIR", None)
        else:
            os.environ["POLICY_SPACE_CONTRACTS_DIR"] = previous
    require(identities() == identity, "native source identities before/after execution")
    return {
        "engine": "policy_space_numeric",
        "upstream_identity": identity,
        "registry": registry,
        "cells": rows,
        "negative_controls": controls,
        "runtime": {
            "setup_seconds": setup_seconds,
            "dispatch_seconds": time.perf_counter() - start,
            "accepted_replay_queries": accepted,
            "rejected_structural_controls": rejected,
        },
    }


def main():
    from run_policy_space_numeric import inputs, request, sources, verify_contracts

    raw = sys.stdin.buffer.read(1024**2 + 1)
    require(len(raw) <= 1024**2, "bounded explicit native request")
    req = json.loads(raw)
    require(type(req["schema_version"]) is int and req["schema_version"] == 1, "request version")
    matched = (ROOT / req["matched_path"]).resolve()
    contract_dir = (ROOT / req["contracts_path"]).resolve()
    require(matched.is_relative_to(ROOT) and contract_dir.is_relative_to(ROOT), "contained native inputs")
    plan, decoder = inputs(matched)
    require(fingerprint(req) == fingerprint(request(matched, contract_dir, plan)), "exact native request")
    verify_contracts(contract_dir, plan)
    before, events = sources(), []

    def audit(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.bind"):
            events.append(event)
            raise RuntimeError("Numeric transport replay forbids network connections/listeners")

    sys.addaudithook(audit)
    start = time.perf_counter()
    result = execute(plan, decoder, contract_dir)
    require(before == sources() and inputs(matched) == (plan, decoder), "stable source/decoder inputs")
    verify_contracts(contract_dir, plan)
    require(not events and _rss() <= 2 * 1024**3, "observed network/resource controls")
    result.update(
        schema_version=1,
        development_only=True,
        final_evaluation=False,
        source_file_sha256=before,
        request_sha256=hashlib.sha256(raw).hexdigest(),
    )
    result["runtime"].update(
        python=platform.python_version(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        total_wall_seconds=time.perf_counter() - start,
        peak_rss_bytes=_rss(),
        network_events=events,
    )
    sys.stdout.buffer.write(json.dumps(result, allow_nan=False).encode())


if __name__ == "__main__":
    main()
