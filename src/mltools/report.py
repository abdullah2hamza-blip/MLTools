from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Status(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


@dataclass
class Finding:
    check: str
    status: Status
    boundary: str
    message: str
    event: int | None = None
    expected: Any = None
    actual: Any = None


def aggregate(statuses) -> Status:
    statuses = list(statuses)
    if Status.FAIL in statuses:
        return Status.FAIL
    if not statuses or Status.UNKNOWN in statuses:
        return Status.UNKNOWN
    return Status.PASS


def make_report(findings: list[Finding], bundle: dict, trace: dict | None) -> dict:
    first = next((f for f in findings if f.status == Status.FAIL), None)
    counterexample = None
    if first and first.event is not None:
        context_end = first.event if bundle["scenario"][first.event]["kind"] == "reset" else first.event + 1
        start = max(
            (i for i, e in enumerate(bundle["scenario"][:context_end]) if e["kind"] == "reset"),
            default=0,
        )
        if first.check in ("cameras", "joint_order"):
            start = first.event
        counterexample = {
            "original_event": first.event,
            "start_event": start,
            "events": bundle["scenario"][start : first.event + 1],
            "scope": "Single event for input checks; episode prefix preserves stateful failure context.",
        }
    return {
        "schema_version": 1,
        "status": aggregate(f.status for f in findings),
        "embodiment": bundle["contract"].get("embodiment"),
        "checks": {
            c: aggregate(f.status for f in findings if f.check == c)
            for c in dict.fromkeys(bundle["contract"]["checks"] + [f.check for f in findings])
        },
        "findings": [asdict(f) for f in findings],
        "first_divergence": asdict(first) if first else None,
        "counterexample": counterexample,
        "execution": trace.get("execution") if trace else None,
        "limitations": [
            "Finite fixtures, independent declared reference, and listed boundaries only.",
            "No trained-weight inference, task-success, calibration, or robot-safety claim.",
            "Integrators must establish the authority and physical identity of reference labels.",
            "Recorded traces are producer assertions, not authenticated execution attestations.",
        ],
    }
