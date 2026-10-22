import argparse
import json
import sys
from pathlib import Path

from .checker import check
from .io import InputError, load_bundle, local_path, read_json, write_json


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(3, f"ERROR: {message}\n")


def main(argv=None) -> int:
    parser = Parser(description="Evidence-scoped robot-policy semantic conformance")
    parser.add_argument("--version", action="version", version="mltools 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", help="Compare a bundle with an executed or recorded trace")
    check_parser.add_argument("bundle", type=Path)
    source = check_parser.add_mutually_exclusive_group()
    source.add_argument(
        "--execute", choices=["lerobot"], help="Explicitly execute the pinned installed integration"
    )
    source.add_argument("--trace", type=Path, help="Data-only replay; does not import LeRobot")
    check_parser.add_argument("--save-trace", type=Path, help="Save execution evidence for offline replay")
    check_parser.add_argument("--json", dest="json_path", type=Path, help="Write the versioned report")
    check_parser.add_argument(
        "--counterexample", type=Path, help="Export a failing regression into a new directory"
    )
    check_parser.add_argument("--format", choices=["text", "json"], default="text")
    probe = sub.add_parser("probe", help="Execute or replay versioned research probe cases")
    probe.add_argument("case", type=Path)
    source = probe.add_mutually_exclusive_group()
    source.add_argument("--execute", choices=["lerobot", "openpi"])
    source.add_argument("--trace", type=Path)
    probe.add_argument("--python", type=Path, help="Explicit isolated OpenPI worker interpreter")
    probe.add_argument("--save-trace", type=Path)
    probe.add_argument("--json", dest="json_path", type=Path)
    probe.add_argument("--counterexample", type=Path)
    probe.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            from .probe_cli import run

            return run(args)
        if args.save_trace and not args.execute:
            raise InputError("--save-trace requires --execute")
        bundle, root = load_bundle(args.bundle)
        protected = {args.bundle.resolve()} | {local_path(root, a["path"]) for a in bundle["artifacts"]}
        if args.trace:
            protected.add(args.trace.resolve())
        outputs = [p.resolve() for p in (args.save_trace, args.json_path) if p]
        if any(p in protected for p in outputs) or len(set(outputs)) != len(outputs):
            raise InputError(
                "Output files must be distinct and cannot overwrite bundle, artifacts or input trace"
            )
        trace = args.trace
        if args.execute:
            from .lerobot import capture

            trace = capture(args.bundle)
            if args.save_trace:
                write_json(args.save_trace, trace)
        report = check(args.bundle, trace)
        if args.counterexample:
            from .reproduce import export_counterexample

            captured = read_json(trace) if isinstance(trace, Path) else trace
            export_counterexample(args.bundle, captured, report, args.counterexample)
        if args.json_path:
            write_json(args.json_path, report)
        if args.format == "json":
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            print(f"{report['status']} | {report['embodiment']} | finite fixture conformance")
            for name, status in report["checks"].items():
                print(f"  {status:7} {name}")
            for finding in report["findings"]:
                if finding["status"] != "PASS":
                    print(
                        f"  {finding['status']} event={finding['event']} {finding['boundary']}: "
                        f"{finding['message']}"
                    )
            first = report["first_divergence"]
            if first:
                print(f"  expected={first['expected']}\n  actual={first['actual']}")
            print("Scope: declared references and fixtures only; no robot-safety or model-performance claim.")
        return {"PASS": 0, "FAIL": 1, "UNKNOWN": 2}[report["status"]]
    except (InputError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
