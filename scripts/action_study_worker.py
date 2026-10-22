"""Opt-in native development acquisition with separately counted baseline work."""

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from action_designs import ActionDesign  # noqa: E402
from action_study_common import development_inputs, fingerprint, sources  # noqa: E402
from normalization_profiles import finite, require, validate  # noqa: E402

from mltools._openpi_worker import _rss  # noqa: E402


def factory(spec):
    n = validate(spec)
    profile = spec["profile"]
    if profile.startswith("lerobot"):
        import torch
        from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.lerobot_types import TransitionKey as Key
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

        mode = NormalizationMode.MEAN_STD if "mean_std" in profile else NormalizationMode.QUANTILES

        def build(cls):
            return cls(
                features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(n,))},
                norm_map={FeatureType.ACTION: mode},
                stats={"action": spec["statistics"]},
                device="cpu",
                dtype=torch.float64,
                eps=1e-8,
            )

        encoder, decoder = build(NormalizerProcessorStep), build(UnnormalizerProcessorStep)

        def call(row, inverse):
            with torch.inference_mode():
                return (decoder if inverse else encoder)(
                    {Key.ACTION: torch.tensor(row, dtype=torch.float64)}
                )[Key.ACTION].tolist()

        configuration = {
            "encoder": encoder.get_config(),
            "decoder": decoder.get_config(),
            "device": "cpu",
            "dtype": "float64",
        }
    else:
        import numpy as np
        from openpi import transforms
        from openpi.shared.normalize import NormStats

        stats = {
            "actions": NormStats(
                mean=np.zeros(n),
                std=np.ones(n),
                q01=np.asarray(spec["statistics"]["q01"], dtype=np.float64),
                q99=np.asarray(spec["statistics"]["q99"], dtype=np.float64),
            )
        }
        encoder = transforms.Normalize(stats, use_quantiles=True)
        decoder = transforms.Unnormalize(stats, use_quantiles=True)

        def call(row, inverse):
            return (decoder if inverse else encoder)({"actions": np.asarray(row, dtype=np.float64)})[
                "actions"
            ].tolist()

        configuration = {
            "encoder": "openpi.transforms.Normalize",
            "decoder": "openpi.transforms.Unnormalize",
            "use_quantiles": True,
            "device": "cpu",
            "dtype": "float64",
        }
    # Round-trip through JSON so tuples in genuine native configuration are
    # represented identically in the raw stdout and parsed capture.
    return call, json.loads(json.dumps(configuration))


def execute(request):
    initial = sources()
    require(request["source_file_sha256"] == initial, "producer source snapshot")
    require(request["development_only"] is True and request["final_evaluation"] is False, "development gate")
    framework = request["framework"]
    require(framework in ("lerobot", "openpi"), "native framework")
    expected = {k: v for k, v in development_inputs().items() if v["framework"] == framework}
    require(request["cells"] == expected, "fixed development-only cells; no prospective execution")
    designs = {
        cell: {name: ActionDesign.from_dict(value) for name, value in entries.items()}
        for cell, entries in request["designs"].items()
    }
    require(set(designs) == set(expected), "complete development design inventory")
    for cell, entries in designs.items():
        n = validate(expected[cell]["intended"])
        require(1 <= len(entries) <= 8, "design budget")
        require(all(len(d.axes) == len(d.outputs) == n for d in entries.values()), "native port dimensions")
    selections = request["selected_queries"]
    require(request["phase"] in ("main", "witness"), "explicit acquisition phase")
    require((selections is None) == (request["phase"] == "main"), "fresh subset selection phase")
    if selections is not None:
        require(isinstance(selections, dict) and set(selections) <= set(expected), "witness cell inventory")
        for cell, conditions in selections.items():
            require(set(conditions) <= set(expected[cell]["conditions"]), "witness condition inventory")
            for ids in conditions.values():
                valid = {p.id for p in designs[cell]["canonical"].probes}
                require(
                    isinstance(ids, list) and ids and len(set(ids)) == len(ids) and set(ids) <= valid,
                    "witness IDs",
                )
    events = []

    def audit(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            events.append(event)
            raise RuntimeError("Action-only worker forbids network connections")

    sys.addaudithook(audit)
    started = time.perf_counter()
    if framework == "lerobot":
        import torch

        from mltools.lerobot import verify_installation

        torch.set_num_threads(1)
    else:
        from mltools._openpi_worker import verify_installation
    identity = verify_installation()
    operators = {}
    for cell_id, cell in expected.items():
        artifact = cell["source_artifact"]
        if artifact is not None:
            raw = (ROOT / artifact["path"]).read_bytes()
            require(hashlib.sha256(raw).hexdigest() == artifact["sha256"], "source statistics bytes")
            from safetensors.torch import load

            native = load(raw)
            require(
                all(native["action." + k].tolist() == v for k, v in cell["intended"]["statistics"].items()),
                "genuine Safetensors decoding matches separately decoded source statistics",
            )
        operators[cell_id] = {
            "intended": factory(cell["intended"]),
            "candidates": {
                name: factory(value["specification"]) for name, value in cell["conditions"].items()
            },
        }
    measurements, baseline_measurements = {}, {}
    setup_seconds = time.perf_counter() - started
    main_calls, auxiliary_calls = 0, 0
    main_start = time.perf_counter()
    for cell_id, cell in expected.items():
        measurements[cell_id] = {}
        for condition, intervention in cell["conditions"].items():
            selected = None if selections is None else selections.get(cell_id, {}).get(condition)
            if selections is not None and selected is None:
                continue
            candidate, config = operators[cell_id]["candidates"][condition]
            rows = {}
            for name, design in designs[cell_id].items():
                if selections is not None and name != "canonical":
                    continue
                captured = []
                for query in design.probes:
                    if selected is not None and query.id not in selected:
                        continue
                    result = candidate(list(query.values), True)
                    main_calls += 1
                    require(
                        len(result) == len(design.outputs) and all(map(finite, result)),
                        "native finite output",
                    )
                    captured.append({"query": query.id, "input": list(query.values), "output": result})
                rows[name] = {
                    "design_sha256": design.sha256,
                    "rows": captured,
                    "decoder_calls": len(captured),
                }
            measurements[cell_id][condition] = {
                "deployment_sha256": fingerprint(intervention["specification"]),
                "configuration": config,
                "designs": rows,
            }
    main_seconds = time.perf_counter() - main_start
    baseline_start = time.perf_counter()
    if request["phase"] == "main":
        # Baseline acquisition is a separate phase. The primary design performs
        # exactly one decoder invocation/query; four auxiliary calls/query below
        # are retained and charged separately to the round-trip workflows.
        for cell_id, cell in expected.items():
            baseline_measurements[cell_id] = {}
            intended, _ = operators[cell_id]["intended"]
            for condition in cell["conditions"]:
                candidate, _ = operators[cell_id]["candidates"][condition]
                rows = []
                # Canonical observations only: no redundant auxiliary baseline
                # queries are invented for random geometry comparisons.
                primary = measurements[cell_id][condition]["designs"]["canonical"]["rows"]
                for row in primary:
                    self_encoded = candidate(row["output"], False)
                    intended_decoded = intended(row["input"], True)
                    cross_encoded = intended(intended_decoded, False)
                    cross_decoded = candidate(cross_encoded, True)
                    auxiliary_calls += 4
                    rows.append(
                        {
                            "query": row["query"],
                            "self_encoded": self_encoded,
                            "intended_decoded": intended_decoded,
                            "cross_encoded": cross_encoded,
                            "cross_decoded": cross_decoded,
                        }
                    )
                baseline_measurements[cell_id][condition] = rows
    baseline_seconds = time.perf_counter() - baseline_start
    require(verify_installation() == identity and sources() == initial, "stable native/producer sources")
    require(not events and _rss() <= 2 * 1024**3, "network/observed peak-RSS controls")
    return {
        "schema_version": 1,
        "format": "mltools-action-normalization-native-development",
        "development_only": True,
        "final_evaluation": False,
        "framework": framework,
        "phase": request["phase"],
        "source_file_sha256": initial,
        "upstream_identity": identity,
        "measurements": measurements,
        "roundtrips": baseline_measurements,
        "runtime": {
            "python": platform.python_version(),
            "setup_seconds": setup_seconds,
            "main_seconds": main_seconds,
            "roundtrip_seconds": baseline_seconds,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": _rss(),
            "main_decoder_calls": main_calls,
            "auxiliary_native_calls": auxiliary_calls,
            "network_events": events,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
    }


def main():
    raw = sys.stdin.buffer.read(4 * 1024**2 + 1)
    require(len(raw) <= 4 * 1024**2, "request byte bound")
    result = execute(json.loads(raw))
    result["request_sha256"] = hashlib.sha256(raw).hexdigest()
    sys.stdout.buffer.write(json.dumps(result, allow_nan=False).encode())


if __name__ == "__main__":
    main()
