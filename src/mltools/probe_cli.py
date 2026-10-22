"""CLI dispatch for versioned research probe cases."""

import json

from . import probe_cases
from .io import InputError, local_path, write_json


def run(args):
    case, root, _suite, _reference = probe_cases.load(args.case)
    protected = {args.case.resolve()} | {local_path(root, a["path"]) for a in case["artifacts"]}
    if args.trace:
        protected.add(args.trace.resolve())
    outputs = [p.resolve() for p in (args.save_trace, args.json_path) if p]
    if any(p in protected for p in outputs) or len(set(outputs)) != len(outputs):
        raise InputError("Output paths must be distinct and cannot overwrite case inputs")
    if args.counterexample:
        destination = args.counterexample.resolve()
        if any(p.is_relative_to(destination) or destination.is_relative_to(p) for p in outputs):
            raise InputError("Separate output paths cannot overlap the regression export directory")
    if args.save_trace and not args.execute:
        raise InputError("--save-trace requires explicit --execute")
    if args.python and args.execute != "openpi":
        raise InputError("--python is only used for explicit OpenPI execution")
    trace = args.trace
    if args.execute:
        if args.execute != case["adapter"]["framework"]:
            raise InputError("Execution framework must match the case adapter")
        trace = probe_cases.capture(args.case, python=args.python)
        if args.save_trace:
            write_json(args.save_trace, trace)
    report = probe_cases.check(args.case, trace)
    if args.counterexample:
        probe_cases.export(args.case, trace, report, args.counterexample)
    if args.json_path:
        write_json(args.json_path, report)
    if args.format == "json":
        print(json.dumps(report, indent=2, allow_nan=False))
    else:
        print(f"{report['status']} | affine research obligations | {case['adapter']['framework']}")
        for name, status in report["checks"].items():
            print(f"  {status:7} {name}")
        if report["first_divergence"]:
            first = report["first_divergence"]
            print(f"  first={first.get('boundary', first.get('obligation'))}")
            print(f"  expected={first['expected']} actual={first['actual']}")
        print(
            "Scope: declared affine fragment and trusted evidence; no task-success or hardware-safety claim."
        )
    return {"PASS": 0, "FAIL": 1, "UNKNOWN": 2}[report["status"]]
