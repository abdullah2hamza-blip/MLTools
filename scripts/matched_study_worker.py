"""Stock baseline and decoder engines over the same immutable leaf artifacts."""

import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from action_designs import ActionDesign  # noqa: E402
from action_study_worker import factory  # noqa: E402
from normalization_profiles import require  # noqa: E402
from study_assets import verify  # noqa: E402
from validate_robotruth import identities as robotruth_identity  # noqa: E402

from mltools._openpi_worker import _rss  # noqa: E402


def execute_robotruth(plan, folder):
    require(
        all(importlib.util.find_spec(m) is None for m in ("torch", "lerobot", "openpi")),
        "separate no-ML robotruth environment",
    )
    identity = robotruth_identity()
    from robotruth.contract import check_contract, extract_spec

    start = time.perf_counter()
    rows, extracts, checks = {}, 0, 0
    for cell_id, cell in plan["cells"].items():
        base = folder / cell["conditions"]["numeric_control"]["directory"]
        evaluated = extract_spec(base, family="lerobot", overlay={}, role="evaluated")
        extracts += 1
        rows[cell_id] = {}
        for condition, entry in cell["conditions"].items():
            deployed = extract_spec(
                folder / entry["directory"], family="lerobot", overlay={}, role="deployed"
            )
            extracts += 1
            result = check_contract(evaluated, deployed, allow_unknown=False, ignore=set())
            checks += 1
            rows[cell_id][condition] = {
                "evaluated": evaluated.model_dump(mode="json"),
                "deployed": deployed.model_dump(mode="json"),
                "native_passed": result.passed,
                "native_summary": result.summary(),
                "findings": dataclasses.asdict(result)["findings"],
                "allow_unknown": False,
                "ignore": [],
                "overlays": {"evaluated": {}, "deployed": {}},
                "extractor_family": "lerobot",
                "target_decoder_framework": cell["framework"],
            }
    require(robotruth_identity() == identity, "complete native robotruth source stability")
    return {
        "engine": "robotruth",
        "upstream_identity": identity,
        "cells": rows,
        "runtime": {
            "wall_seconds": time.perf_counter() - start,
            "extract_calls": extracts,
            "check_calls": checks,
        },
    }


def execute_decoder(plan, folder, framework):
    setup_start = time.perf_counter()
    import torch

    torch.set_num_threads(1)
    if framework == "lerobot":
        from mltools.lerobot import verify_installation
    else:
        from mltools._openpi_worker import verify_installation
    identity = verify_installation()
    from safetensors.torch import load

    cells = {k: v for k, v in plan["cells"].items() if v["framework"] == framework}
    require(cells, "applicable decoder cells")
    operators, loaded = {}, {}
    for cell_id, cell in cells.items():
        operators[cell_id], loaded[cell_id] = {}, {}
        for condition, entry in cell["conditions"].items():
            raw = (folder / entry["directory"] / entry["post_state_file"]).read_bytes()
            require(
                hashlib.sha256(raw).hexdigest() == entry["file_sha256"][entry["post_state_file"]],
                "actual deployed artifact identity",
            )
            tensors = load(raw)
            spec = entry["specification"]
            native = {k: tensors["action." + k].to(dtype=torch.float64).tolist() for k in spec["statistics"]}
            require(native == spec["statistics"], "independent versus native Safetensors decoding")
            operators[cell_id][condition] = factory({**spec, "statistics": native})
            loaded[cell_id][condition] = {
                "artifact_sha256": hashlib.sha256(raw).hexdigest(),
                "statistics": native,
                "tensor_dtypes": {k: str(tensors["action." + k].dtype) for k in native},
            }
    setup_seconds = time.perf_counter() - setup_start
    rows, main_calls, auxiliary = {}, 0, 0
    start = time.perf_counter()
    for cell_id, cell in cells.items():
        rows[cell_id] = {}
        for condition in cell["conditions"]:
            call, configuration = operators[cell_id][condition]
            values = {}
            for name, raw_design in cell["designs"].items():
                d = ActionDesign.from_dict(raw_design)
                values[name] = [
                    {"query": q.id, "input": list(q.values), "output": call(list(q.values), True)}
                    for q in d.probes
                ]
                main_calls += len(d.probes)
            rows[cell_id][condition] = {
                "loaded_artifact": loaded[cell_id][condition],
                "configuration": configuration,
                "designs": values,
            }
    main_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for cell_id, cell in cells.items():
        intended, _ = operators[cell_id]["numeric_control"]
        for condition in cell["conditions"]:
            candidate, _ = operators[cell_id][condition]
            results = []
            for row in rows[cell_id][condition]["designs"]["canonical"]:
                x = intended(row["input"], True)
                z = intended(x, False)
                results.append(
                    {
                        "query": row["query"],
                        "self_encoded": candidate(row["output"], False),
                        "intended_decoded": x,
                        "cross_encoded": z,
                        "cross_decoded": candidate(z, True),
                    }
                )
                auxiliary += 4
            rows[cell_id][condition]["roundtrips"] = results
    require(verify_installation() == identity, "native decoder source stability")
    return {
        "engine": "decoder",
        "framework": framework,
        "upstream_identity": identity,
        "cells": rows,
        "runtime": {
            "setup_seconds": setup_seconds,
            "main_seconds": main_seconds,
            "roundtrip_seconds": time.perf_counter() - start,
            "primary_decoder_calls": main_calls,
            "auxiliary_native_calls": auxiliary,
        },
    }


def main():
    from run_matched_study import sources

    raw = sys.stdin.buffer.read(1024**2 + 1)
    require(len(raw) <= 1024**2, "bounded engine request")
    req = json.loads(raw)
    require(
        type(req["schema_version"]) is int
        and req["schema_version"] == 1
        and req["engine"] in ("robotruth", "lerobot", "openpi")
        and req["source_file_sha256"] == sources(),
        "native producer binding",
    )
    folder = (ROOT / req["assets_path"]).resolve()
    require(folder.is_relative_to(ROOT), "workspace-contained matched assets")
    require(
        hashlib.sha256((folder / "plan.json").read_bytes()).hexdigest() == req["plan_sha256"],
        "asset plan identity",
    )
    plan = verify(folder)
    require(plan["role"] == "development", "no prospective engine execution before committed protocol freeze")
    before = sources()
    events = []

    def audit(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            events.append(event)
            raise RuntimeError("Matched native workflows forbid network connections")

    sys.addaudithook(audit)
    started = time.perf_counter()
    result = (
        execute_robotruth(plan, folder)
        if req["engine"] == "robotruth"
        else execute_decoder(plan, folder, req["engine"])
    )
    require(sources() == before and verify(folder) == plan and not events, "stable engine inputs/sources")
    require(_rss() <= 2 * 1024**3, "measured working-set bound")
    result.update(
        schema_version=1,
        development_only=True,
        final_evaluation=False,
        plan_sha256=req["plan_sha256"],
        source_file_sha256=before,
        request_sha256=hashlib.sha256(raw).hexdigest(),
    )
    result["runtime"].update(
        python=platform.python_version(),
        peak_rss_bytes=_rss(),
        network_events=events,
        total_wall_seconds=time.perf_counter() - started,
        packages={d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    )
    sys.stdout.buffer.write(json.dumps(result, allow_nan=False).encode())


if __name__ == "__main__":
    main()
