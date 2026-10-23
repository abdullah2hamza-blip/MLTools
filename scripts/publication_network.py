"""Deny audited network operations; classify one denied dependency probe."""

import hashlib
import importlib.metadata
import sys
from pathlib import Path

from action_study_common import fingerprint
from normalization_profiles import require

PROBE = {
    "event": "socket.bind",
    "classification": "denied_import_time_ipv6_capability_probe",
    "distribution": "urllib3",
    "version": "2.8.0",
    "source": "urllib3/util/connection.py",
    "source_sha256_lf": "2633bbdb69731e5ccb5cf4e4afd65605d86c7979cc5633126f50c92d5ad74a74",
    "function": "_has_ipv6",
    "address": ["::1", 0],
    "operation_executed": False,
}


def activate():
    unexpected, denied_probes = [], []
    try:
        distribution = importlib.metadata.distribution("urllib3")
    except importlib.metadata.PackageNotFoundError:
        distribution, source = None, None
    else:
        source = Path(distribution.locate_file(PROBE["source"])).resolve()

    def audit(event, args):
        if event not in ("socket.connect", "socket.getaddrinfo", "socket.bind", "socket.listen"):
            return
        unexpected.append(event)
        caller = sys._getframe(1)
        known = (
            event == "socket.bind"
            and len(args) == 2
            and isinstance(args[1], tuple)
            and list(args[1]) == PROBE["address"]
            and caller.f_code.co_name == PROBE["function"]
            and source is not None
            and Path(caller.f_code.co_filename).resolve() == source
            and distribution.version == PROBE["version"]
            and hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            == PROBE["source_sha256_lf"]
        )
        if known:
            unexpected.pop()
            denied_probes.append(dict(PROBE))
        raise RuntimeError("Publication audit denies all network operations")

    sys.addaudithook(audit)
    return unexpected, denied_probes


def validate(runtime):
    require(runtime["network_events"] == [], "no unclassified network attempts")
    probes = runtime["denied_dependency_capability_probes"]
    require(
        isinstance(probes, list)
        and len(probes) <= 1
        and all(fingerprint(p) == fingerprint(PROBE) for p in probes),
        "exact source-identified denied dependency capability probe",
    )
