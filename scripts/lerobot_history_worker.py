"""Execute genuine before/fix LeRobot processors with explicit synthetic queue inputs."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
from collections import deque
from pathlib import Path

SOURCES = {
    "before": (
        "240ea44cf9f887df944019cf46e18a0d496e1a56",
        {
            "processor/relative_action_processor.py": "4c1af387b21e203bf4f364c7f98be3bf7c12f757",
            "policies/pretrained.py": "dbada7b0ad4d0dec63e3f3eb8007dde1f5f8d435",
            "processor/__init__.py": "f8c14a6b7376490e3924764f4fe1cdaf51e4f7fe",
            "rollout/context.py": "c25cf05188c018afe9dabd6212ceb1c8bbff9899",
            "rollout/inference/sync.py": "8b905df6192d28232140418dad93c01c87d556c5",
            "scripts/lerobot_eval.py": "6dcc5de189f47b0e4217893d36b284e4b0382e15",
        },
    ),
    "after": (
        "7c98c1b2ce309f94e542bd4b9ee3fe176301edbf",
        {
            "processor/relative_action_processor.py": "346acd1b4c4906087467985cd89a483de2635de2",
            "policies/pretrained.py": "1bbed0378a43310113fc661241e7995fc2e0c548",
            "processor/__init__.py": "455c90d672970fdd5da36f3965bd32d9e278b930",
            "rollout/context.py": "7fc3c0d7e402af745a1a6dfcf64da0aab1b0162a",
            "rollout/inference/sync.py": "30631b7939cdd68595c7140f86924746f649226a",
            "scripts/lerobot_eval.py": "6aa586ed032a775cc17470a5256d61a116f3c73d",
        },
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=SOURCES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Historical capture output must be new")
    revision, blobs = SOURCES[args.phase]
    dist = importlib.metadata.distribution("lerobot")
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    if direct.get("vcs_info", {}).get("commit_id") != revision:
        raise ValueError("Historical LeRobot installation has the wrong Git revision")
    spec = importlib.util.find_spec("lerobot")
    root = Path(next(iter(spec.submodule_search_locations)))
    for name, expected in blobs.items():
        raw = (root / name).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if actual != expected:
            raise ValueError(f"Historical source mismatch: {name}")
    import torch
    from lerobot.lerobot_types import TransitionKey as Key
    from lerobot.processor.relative_action_processor import (
        AbsoluteActionsProcessorStep,
        RelativeActionsProcessorStep,
    )

    relative = RelativeActionsProcessorStep(enabled=True, action_names=["scalar"])
    absolute = AbsoluteActionsProcessorStep(enabled=True, relative_step=relative)
    queue = deque()
    if args.phase == "after":
        relative.bind_action_queue(lambda: len(queue))
    rows = []
    # Independent expected controller values come from pre-existing chunk-origin
    # semantics, not the observed processor state or inverse of its own output.
    scenario = [
        (1.0, [0.1, 0.2, 0.3], 1.1),
        (5.0, None, 1.2),
        (9.0, None, 1.3),
        (20.0, [0.4], 20.4),
        (30.0, [0.5, 0.6], 30.5),
    ]
    with torch.inference_mode():
        for state, prediction, expected in scenario:
            relative({Key.OBSERVATION: {"observation.state": torch.tensor([[state]], dtype=torch.float64)}})
            if prediction is not None:
                if queue:
                    raise AssertionError("Fixture cannot replace a pending chunk")
                queue.extend(prediction)
            offset = queue.popleft()
            action = absolute({Key.ACTION: torch.tensor([[offset]], dtype=torch.float64)})[Key.ACTION].item()
            rows.append(
                {
                    "state": state,
                    "offset": offset,
                    "expected": expected,
                    "actual": action,
                    "anchor": relative.get_cached_state().item(),
                    "queue_remaining": len(queue),
                }
            )
        pending_before_reset = len(queue)
        queue.clear()
        relative.reset()
        reset_cleared = relative.get_cached_state() is None
        relative({Key.OBSERVATION: {"observation.state": torch.tensor([[40.0]], dtype=torch.float64)}})
        final = absolute({Key.ACTION: torch.tensor([[0.7]], dtype=torch.float64)})[Key.ACTION].item()
    match = all(abs(row["actual"] - row["expected"]) <= 1e-12 for row in rows)
    if match != (args.phase == "after") or not reset_cleared or pending_before_reset != 1:
        raise AssertionError("Historical boundary outcome did not match the independently audited source")
    if abs(final - 40.7) > 1e-12:
        raise AssertionError("Post-reset independent chunk did not adopt its new origin")
    result = {
        "schema_version": 1,
        "development_only": True,
        "final_evaluation": False,
        "phase": args.phase,
        "status": "PASS" if match else "FAIL",
        "classification": "historical_processor_source_reproduction_with_synthetic_boundary_inputs",
        "identity": {"revision": revision, "git_blobs": blobs, "direct_url": direct},
        "events": rows,
        "reset": {
            "pending_before_reset": pending_before_reset,
            "anchor_cleared": reset_cleared,
            "new_chunk_actual": final,
            "new_chunk_expected": 40.7,
        },
        "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "dependency_scope": "shared current locked CPU dependencies; not original full training runtime",
        },
        "limits": [
            "Real complete installed historical source; no extracted code fragments",
            "Fixture queue supplies known outputs; no policy inference or checkpoint weights",
            "Public evaluation-style processor sequence, not full evaluation environment",
            "Parent synchronous rollout guard and RTC exclusions remain as documented",
            "No ACT-relative integration, hardware units, robot safety or task-success claim",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"{args.phase}: {result['status']}; verified six source blobs and reset/refill", flush=True)


if __name__ == "__main__":
    main()
