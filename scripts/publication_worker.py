"""Opt-in native phases behind the complete committed publication protocol."""

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from action_designs import ActionDesign  # noqa: E402
from action_study_common import fingerprint  # noqa: E402
from action_study_worker import factory  # noqa: E402
from matched_study_worker import execute_decoder, execute_robotruth  # noqa: E402
from normalization_profiles import require  # noqa: E402
from policy_space_numeric_worker import execute as execute_transport  # noqa: E402
from publication_network import activate  # noqa: E402
from publication_protocol import verify  # noqa: E402

from mltools._openpi_worker import _rss  # noqa: E402


def execute_witness(plan, folder, framework, selected):
    """Fresh process, fresh actual artifact load, only the retained native calls."""
    start = time.perf_counter()
    import torch
    from safetensors.torch import load

    torch.set_num_threads(1)
    if framework == "lerobot":
        from mltools.lerobot import verify_installation
    else:
        from mltools._openpi_worker import verify_installation
    identity = verify_installation()
    operators, cells = {}, {k: v for k, v in plan["cells"].items() if v["framework"] == framework}
    rows = {k: {} for k in cells}
    for cell_id, conditions in selected.items():
        cell = cells[cell_id]
        operators[cell_id] = {}
        for condition in conditions:
            entry = cell["conditions"][condition]
            raw = (folder / entry["directory"] / entry["post_state_file"]).read_bytes()
            require(
                hashlib.sha256(raw).hexdigest() == entry["file_sha256"][entry["post_state_file"]],
                "fresh actual artifact",
            )
            tensors = load(raw)
            statistics = {
                k: tensors["action." + k].to(dtype=torch.float64).tolist()
                for k in entry["specification"]["statistics"]
            }
            require(
                statistics == entry["specification"]["statistics"],
                "fresh independently decoded/native arrays",
            )
            operator, config = factory({**entry["specification"], "statistics": statistics})
            operators[cell_id][condition] = operator
            rows[cell_id][condition] = {
                "loaded_artifact": {
                    "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                    "statistics": statistics,
                    "tensor_dtypes": {k: str(tensors["action." + k].dtype) for k in statistics},
                },
                "configuration": config,
                "designs": {"canonical": []},
            }
    setup = time.perf_counter() - start
    start, count = time.perf_counter(), 0
    for cell_id, conditions in selected.items():
        design = ActionDesign.from_dict(cells[cell_id]["designs"]["canonical"])
        for condition, ids in conditions.items():
            for query in design.probes:
                if query.id not in ids:
                    continue
                output = operators[cell_id][condition](list(query.values), True)
                rows[cell_id][condition]["designs"]["canonical"].append(
                    {"query": query.id, "input": list(query.values), "output": output}
                )
                count += 1
    require(verify_installation() == identity, "fresh native source stability")
    return {
        "engine": "decoder_witness",
        "framework": framework,
        "upstream_identity": identity,
        "cells": rows,
        "runtime": {
            "setup_seconds": setup,
            "main_seconds": time.perf_counter() - start,
            "primary_decoder_calls": count,
            "auxiliary_native_calls": 0,
        },
    }


def main():
    from publication_analysis import select_witnesses
    from run_policy_space_numeric import verify_contracts
    from run_publication_study import load_main, request, verify_attempt

    raw = sys.stdin.buffer.read(1024**2 + 1)
    require(len(raw) <= 1024**2, "bounded native phase request")
    req = json.loads(raw)
    require(type(req["schema_version"]) is int and req["schema_version"] == 1, "typed phase request")
    protocol_path = (ROOT / req["protocol_path"]).resolve()
    protocol, plan = verify(protocol_path, req["freeze_commit"])
    folder = (ROOT / req["attempt_path"]).resolve()
    require(folder.is_relative_to(ROOT), "contained native attempt")
    verify_attempt(folder, protocol, req["freeze_commit"], req["phase"])
    engine = req["phase"].removesuffix("_witness")
    environment = json.loads((folder / "attempt-start.json").read_bytes())["environments"][engine]
    installed = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    require(
        installed == environment["packages"]
        and platform.python_version() == environment["python"]
        and str(Path(sys.prefix).resolve()) == environment["prefix"],
        "same preflighted isolated native environment before execution claim",
    )
    require(
        fingerprint(req)
        == fingerprint(request(folder, protocol_path, protocol, req["freeze_commit"], req["phase"])),
        "exact gated phase inputs",
    )
    claim = {
        "schema_version": 1,
        "phase": req["phase"],
        "status": "STARTED",
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "freeze_commit": req["freeze_commit"],
        "protocol_sha256": req["protocol_sha256"],
    }
    with (folder / (req["phase"] + ".claim.json")).open("xb") as handle:
        handle.write((json.dumps(claim, indent=2) + "\n").encode())
    before = fingerprint(protocol)
    events, denied_probes = activate()
    start, cpu_start = time.perf_counter(), time.process_time()
    assets = ROOT / protocol["assets_path"]
    phase = req["phase"]
    if phase == "robotruth":
        result = execute_robotruth(plan, assets)
    elif phase in ("lerobot", "openpi"):
        result = execute_decoder(plan, assets, phase)
    else:
        decoder = load_main(folder, protocol_path, protocol, plan, req["freeze_commit"])
        if phase.endswith("_witness"):
            framework = phase.removesuffix("_witness")
            selections = select_witnesses(plan, decoder)
            result = execute_witness(plan, assets, framework, selections[framework])
        else:
            require(phase == "policy_space", "known native phase")
            verify_contracts(folder / "contracts", plan)
            result = execute_transport(plan, decoder, folder / "contracts")
    after, _ = verify(protocol_path, req["freeze_commit"])
    require(fingerprint(after) == before, "stable frozen sources/inputs")
    require(not events, f"native phase attempted forbidden network operations: {events}")
    rss = _rss()
    require(
        rss <= protocol["specification"]["resources"]["observed_peak_rss_limit_bytes"],
        "observed native RSS budget",
    )
    result.update(
        schema_version=1,
        format="mltools-publication-native-phase",
        role=protocol["role"],
        phase=phase,
        freeze_commit=req["freeze_commit"],
        protocol_sha256=req["protocol_sha256"],
        source_file_sha256=protocol["source_file_sha256"],
        request_sha256=hashlib.sha256(raw).hexdigest(),
    )
    result["runtime"].update(
        python=platform.python_version(),
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        total_wall_seconds=time.perf_counter() - start,
        process_cpu_seconds=time.process_time() - cpu_start,
        peak_rss_bytes=rss,
        network_events=events,
        denied_dependency_capability_probes=denied_probes,
        environment_prefix=str(Path(sys.prefix).resolve()),
    )
    require(result["runtime"]["packages"] == installed, "stable installed distribution versions")
    sys.stdout.buffer.write(json.dumps(result, allow_nan=False).encode())


if __name__ == "__main__":
    main()
