"""Actual isolated native decoders for designed certificate parity controls."""

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from certificate_cases import ROOT, sha, sources  # noqa: E402
from normalization_certificate import require, validate  # noqa: E402


def native_lerobot(request):
    import torch
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.lerobot_types import TransitionKey as Key
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    n = len(request["coordinates"])

    def step(cls, data):
        return cls(
            features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(n,))},
            norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
            stats={"action": {k: data[k] for k in ("mean", "std")}},
            device="cpu",
            dtype=torch.float64,
        )

    values = torch.tensor(request["calibration"]["values"], dtype=torch.float64)
    with torch.inference_mode():
        z = (
            step(NormalizerProcessorStep, request["intended"])({Key.ACTION: values})[Key.ACTION]
            if request["calibration"]["space"] == "physical"
            else values
        )
        correct = step(UnnormalizerProcessorStep, request["intended"])({Key.ACTION: z})[Key.ACTION]
        candidate = step(UnnormalizerProcessorStep, request["candidate"])({Key.ACTION: z})[Key.ACTION]
    return {
        "normalized_actions": z.tolist(),
        "intended_decoded": correct.tolist(),
        "candidate_decoded": candidate.tolist(),
    }


def native_openpi(request):
    import numpy as np
    from openpi import transforms
    from openpi.shared.normalize import NormStats

    n = len(request["coordinates"])

    def stats(data):
        return {
            "actions": NormStats(
                mean=np.zeros(n),
                std=np.ones(n),
                q01=np.asarray(data["q01"], dtype=np.float64),
                q99=np.asarray(data["q99"], dtype=np.float64),
            )
        }

    values = np.asarray(request["calibration"]["values"], dtype=np.float64)
    z = (
        transforms.Normalize(stats(request["intended"]), use_quantiles=True)({"actions": values})["actions"]
        if request["calibration"]["space"] == "physical"
        else values
    )
    correct = transforms.Unnormalize(stats(request["intended"]), use_quantiles=True)({"actions": z})[
        "actions"
    ]
    candidate = transforms.Unnormalize(stats(request["candidate"]), use_quantiles=True)({"actions": z})[
        "actions"
    ]
    return {
        "normalized_actions": z.tolist(),
        "intended_decoded": correct.tolist(),
        "candidate_decoded": candidate.tolist(),
    }


def main():
    raw = sys.stdin.buffer.read(2 * 1024**2 + 1)
    require(len(raw) <= 2 * 1024**2, "native request bound")
    request = json.loads(raw)
    framework = request["framework"]
    require(framework in ("openpi", "lerobot"), "native framework")
    require(1 <= len(request["cases"]) <= 8, "native batch bound")
    initial = sources()
    require(request["source_file_sha256"] == initial, "native producer snapshot")
    for case in request["cases"].values():
        validate(case)
        require(
            case["profile"]
            == framework + ("_quantile_extension" if framework == "openpi" else "_mean_std_extension"),
            "native profile applicability",
        )
    events = []

    def audit(event, _args):
        if event in ("socket.connect", "socket.getaddrinfo"):
            events.append(event)
            raise RuntimeError("Certificate worker denies network connections")

    sys.addaudithook(audit)
    spec = importlib.util.spec_from_file_location(
        "certificate_openpi_identity", ROOT / "src/mltools/_openpi_worker.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    start = time.perf_counter()
    if framework == "openpi":
        verify = helper.verify_installation
        native = native_openpi
    else:
        import torch

        import mltools
        from mltools.lerobot import verify_installation

        torch.set_num_threads(1)
        installed = Path(mltools.__file__).parent
        require(
            all(sha(installed / p.name) == sha(p) for p in (ROOT / "src/mltools").glob("*.py")),
            "installed core identity",
        )
        verify = verify_installation
        native = native_lerobot
    identity = verify()
    captures = {name: native(case) for name, case in request["cases"].items()}
    require(identity == verify() and sources() == initial, "native source stability")
    require(events == [], "native network attempts")
    result = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "source_file_sha256": initial,
        "framework": framework,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "upstream_identity": identity,
        "captures": captures,
        "runtime": {
            "python": platform.python_version(),
            "wall_seconds": time.perf_counter() - start,
            "peak_rss_bytes": helper._rss(),
            "network_events": events,
            "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        },
    }
    sys.stdout.buffer.write(json.dumps(result, allow_nan=False).encode())


if __name__ == "__main__":
    main()
