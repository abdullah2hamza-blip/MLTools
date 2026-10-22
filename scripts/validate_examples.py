"""Execute every public fixture against pinned real LeRobot processors."""

import platform
from pathlib import Path

from mltools.checker import check
from mltools.io import write_json
from mltools.lerobot import capture

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "good": "PASS",
    "camera_swap": "FAIL",
    "joint_swap": "FAIL",
    "state_swap": "FAIL",
    "stale_stats": "FAIL",
    "moving_anchor": "FAIL",
    "relative_gripper": "FAIL",
    "no_reset": "FAIL",
    "absolute_body": "FAIL",
    "missing_identity": "UNKNOWN",
    "unsupported_frames": "UNKNOWN",
}
WITNESSES = {
    "camera_swap": ("cameras", "policy.images"),
    "joint_swap": ("actions", "controller.commands"),
    "state_swap": ("joint_order", "policy.state_names"),
    "stale_stats": ("normalization", "action.unnormalize"),
    "moving_anchor": ("actions", "controller.commands"),
    "relative_gripper": ("actions", "controller.commands"),
    "no_reset": ("reset", "episode.reset"),
    "absolute_body": ("actions", "controller.commands"),
    "missing_identity": ("artifacts", "training_stats"),
    "unsupported_frames": ("frames", "coverage"),
}


def run():
    cases = {}
    for name, expected in EXPECTED.items():
        bundle = ROOT / "examples" / "so101" / f"{name}.json"
        trace = capture(bundle)
        report = check(bundle, trace)
        actual = report["status"]
        print(f"{name}: {actual}")
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, got {actual}")
        if name in WITNESSES:
            wanted_check, wanted_boundary = WITNESSES[name]
            if not any(
                f["check"] == wanted_check and f["boundary"] == wanted_boundary and f["status"] == expected
                for f in report["findings"]
            ):
                raise RuntimeError(f"{name}: missing intended {wanted_check}/{wanted_boundary} witness")
        cases[name] = {
            "expected": expected,
            "actual": actual,
            "checks": report["checks"],
            "bundle_sha256": report["bundle_sha256"],
            "first_divergence": report["first_divergence"],
        }
        if name in ("good", "moving_anchor"):
            write_json(bundle.with_suffix(".trace.json"), trace)
            write_json(bundle.with_suffix(".report.json"), report)
    result = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.system(),
        "execution": trace["execution"],
        "cases": cases,
        "limits": "Injected faults and software fixtures; no historical full-policy/hardware reproduction.",
    }
    (ROOT / "validation").mkdir(exist_ok=True)
    write_json(ROOT / "validation" / "integration.json", result)
    published = {}
    for name, expected in (("metadata", "UNKNOWN"), ("wrong_stats", "FAIL")):
        bundle = ROOT / "examples" / "published_so101" / f"{name}.json"
        trace = capture(bundle)
        report = check(bundle, trace)
        if report["status"] != expected or report["checks"]["normalization"] != (
            "PASS" if name == "metadata" else "FAIL"
        ):
            raise RuntimeError(f"Published checkpoint {name}: incorrect partial-evidence verdict")
        write_json(bundle.with_suffix(".trace.json"), trace)
        published[name] = {
            "status": report["status"],
            "checks": report["checks"],
            "first_divergence": report["first_divergence"],
            "bundle_sha256": report["bundle_sha256"],
        }
        print(f"published/{name}: {report['status']}")
    write_json(
        ROOT / "validation" / "published_checkpoint.json",
        {
            "schema_version": 1,
            "execution": trace["execution"],
            "cases": published,
            "source_manifest": "examples/published_so101/source_manifest.json",
            "scope": "Published processor configs/statistics; no weights, tokenizer, inference or hardware",
        },
    )
    return result


if __name__ == "__main__":
    run()
