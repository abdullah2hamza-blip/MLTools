"""Opt-in, isolated execution of OpenPI's reviewed LIBERO processor boundary.

The worker uses its own Python 3.11 environment. No policy inference, tokenizer
construction, network download or robot interface is requested by this adapter.
"""

import json
import os
import subprocess
from pathlib import Path

from .io import MAX_BYTES, InputError, ensure_json
from .obligations import Suite


def capture(
    suite: Suite,
    stats_json: str,
    *,
    python: str | Path,
    model_type: str = "pi0",
    use_quantiles: bool = False,
    mask: tuple[bool, ...] = (True,) * 6 + (False,),
    anchor: str = "origin",
    timeout: float = 180,
) -> dict:
    """Execute a fresh two-action chunk for every compiled intervention.

    Axes must cover action indices 0..6 and origin/current state indices 0..7.
    The returned output is the second dequeued action. ``anchor='current'`` is
    an explicit injected deployment variant, not an upstream OpenPI defect.
    References and physical interpretation are supplied independently to evaluate().
    """
    expected = {(p, str(i)) for p, n in (("action", 7), ("origin", 8), ("current", 8)) for i in range(n)}
    if {(a.port, a.name) for a in suite.axes} != expected or len(suite.outputs) != 7:
        raise InputError("OpenPI LIBERO probes require 7 action and 8 origin/current axes, and 7 outputs")
    if model_type not in ("pi0", "pi05", "pi0_fast") or type(use_quantiles) is not bool:
        raise InputError("Unsupported OpenPI model type or normalization mode")
    if len(mask) != 7 or any(type(x) is not bool for x in mask):
        raise InputError("OpenPI mask must contain seven explicit booleans")
    if anchor not in ("origin", "current"):
        raise InputError("Anchor must be origin or current")
    if not isinstance(stats_json, str) or len(stats_json.encode("utf-8")) > 65536:
        raise InputError("OpenPI statistics must be bounded JSON text")
    if type(timeout) not in (int, float) or not 0 < timeout <= 180:
        raise InputError("Worker timeout must be positive and at most 180 seconds")
    request = {
        "schema_version": 1,
        "suite": suite.to_dict(),
        "suite_sha256": suite.sha256,
        "stats_json": stats_json,
        "model_type": model_type,
        "use_quantiles": use_quantiles,
        "mask": list(mask),
        "anchor": anchor,
    }
    raw = json.dumps(request, allow_nan=False).encode("utf-8")
    if len(raw) > MAX_BYTES:
        raise InputError("Worker request exceeds input size bound")
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        JAX_PLATFORMS="cpu",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONUTF8="1",
        PYTHONNOUSERSITE="1",
    )
    try:
        result = subprocess.run(
            [str(python), "-I", str(Path(__file__).with_name("_openpi_worker.py"))],
            input=raw,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InputError(f"OpenPI worker could not complete: {exc}") from exc
    if len(result.stdout) > MAX_BYTES or len(result.stderr) > MAX_BYTES:
        raise InputError("OpenPI worker exceeded output bound")
    try:
        data = json.loads(result.stdout)
        ensure_json(data)
    except (ValueError, UnicodeError) as exc:
        raise InputError(f"Invalid OpenPI worker response (exit {result.returncode})") from exc
    if result.returncode or not isinstance(data, dict) or "error" in data:
        error = data.get("error", "invalid response") if isinstance(data, dict) else "invalid response"
        raise InputError(f"OpenPI worker failed: {error}")
    if data.get("suite_sha256") != suite.sha256:
        raise InputError("OpenPI worker response has a different probe-suite identity")
    return data
