"""Development integration tests; synthetic fixtures, not benchmark incidents."""

import hashlib
import json
import os
from dataclasses import replace

import pytest

from mltools._openpi_worker import execute
from mltools.io import InputError
from mltools.obligations import Axis, Reference, Suite, compile_suite, evaluate
from mltools.openpi import capture

PYTHON = os.environ.get("MLTOOLS_OPENPI_PYTHON")
live = pytest.mark.skipif(
    not PYTHON, reason="Opt in with MLTOOLS_OPENPI_PYTHON pointing to pinned worker env"
)


def fixture():
    axes = tuple(
        Axis(port, str(i), "synthetic_normalized" if port == "action" else "synthetic_state", -radius, radius)
        for port, size, radius in (("action", 7, 1), ("origin", 8, 0.25), ("current", 8, 0.25))
        for i in range(size)
    )
    suite = compile_suite(axes, tuple(f"command.{i}" for i in range(7)))
    stats = {
        "norm_stats": {
            name: {
                "mean": [0.3 * (i + 1) for i in range(n)],
                "std": [0.5 + i * 0.1 for i in range(n)],
                "q01": [-2 - i * 0.2 for i in range(n)],
                "q99": [3 + i * 0.3 for i in range(n)],
            }
            for name, n in (("actions", 7), ("state", 8))
        }
    }
    return suite, json.dumps(stats)


def independent_fixture_reference(suite, *, quantiles):
    # These fixture expectations are authored independently of the adapter and
    # its returned metadata. They do not attest any published checkpoint.
    scales = [((5 + i * 0.5 + 1e-6) / 2 if quantiles else 0.5 + i * 0.1 + 1e-6) for i in range(7)]
    offsets = [((0.5 + i * 0.05 + 0.5e-6) if quantiles else 0.3 * (i + 1)) for i in range(7)]
    coefficients = tuple(
        tuple(
            scales[i]
            if a.port == "action" and a.name == str(i)
            else 0.25
            if a.port == "origin" and a.name == str(i) and i < 6
            else 0.0
            for a in suite.axes
        )
        for i in range(7)
    )
    return Reference(
        coefficients,
        tuple(offsets),
        ("synthetic_command",) * 7,
        "independently authored development fixture; not a published checkpoint",
        True,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mask": (True,)},
        {"use_quantiles": 1},
        {"anchor": "auto"},
        {"model_type": "unknown"},
        {"timeout": float("nan")},
    ],
)
def test_invalid_configuration_is_rejected_before_worker_launch(kwargs):
    suite, stats = fixture()
    with pytest.raises(InputError):
        capture(suite, stats, python="nonexistent-worker", **kwargs)


def test_raw_worker_rejects_collapsed_inputs_even_with_recomputed_fingerprint():
    suite, stats = fixture()
    plan = suite.to_dict()
    for probe in plan["probes"]:
        probe["values"] = [0.0] * 23
    digest = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ValueError, match="bounded design"):
        execute({"schema_version": 1, "suite": plan, "suite_sha256": digest, "stats_json": stats})


@live
@pytest.mark.parametrize(("model", "quantiles"), [("pi0", False), ("pi05", True), ("pi0_fast", True)])
def test_actual_openpi_boundaries_and_model_specific_camera_masks(model, quantiles):
    suite, stats = fixture()
    trace = capture(suite, stats, python=PYTHON, model_type=model, use_quantiles=quantiles)
    report = evaluate(suite, independent_fixture_reference(suite, quantiles=quantiles), trace)
    assert report["status"] == "PASS"
    assert trace["identity"]["revision"] == "215abfb217dbac7d5f1273282331b9b1866c0479"
    assert trace["camera_routes"] == {
        "base_0_rgb": "base",
        "left_wrist_0_rgb": "wrist",
        "right_wrist_0_rgb": "padding_or_unknown",
    }
    assert trace["camera_masks"]["right_wrist_0_rgb"] is (model == "pi0_fast")
    assert all(t["training_padded_action_shape"] == [2, 32] for t in trace["traces"])
    assert all(t["training_padded_state_shape"] == [32] for t in trace["traces"])
    assert all(t["queue_remaining"] == 0 for t in trace["traces"])
    # Nontrivial training-side values distinguish delta-before-normalization
    # from normalization-before-delta without using the worker as its own oracle.
    row = next(t for t in trace["traces"] if t["query"] == "plus:origin:0")
    expected = (-0.25 - 0.3) / (0.5 + 1e-6)
    if quantiles:
        expected = 2 * (-0.25 + 2) / (5 + 1e-6) - 1
    assert row["training_relative"][0][0] == -0.25
    assert row["training_normalized"][0][0] == pytest.approx(expected)


@live
def test_real_two_query_witness_replays_in_fresh_worker_with_partial_reference():
    suite, stats = fixture()
    ref = independent_fixture_reference(suite, quantiles=False)
    ref = replace(ref, offset=(None,) * 7)
    evidence = capture(suite, stats, python=PYTHON, anchor="current")
    report = evaluate(suite, ref, evidence)
    assert report["status"] == "FAIL"
    assert report["first_divergence"]["obligation"] == "effect:origin:0"
    ids = {row["query"] for row in report["witness"]}
    assert len(ids) == 2
    reduced = Suite(suite.axes, suite.outputs, tuple(p for p in suite.probes if p.id in ids))
    replay = capture(reduced, stats, python=PYTHON, anchor="current")
    rerun = evaluate(reduced, ref, replay)
    assert rerun["status"] == "FAIL"
    assert rerun["first_divergence"]["obligation"] == "effect:origin:0"
    assert rerun["coverage"]["planned_queries"] == 2
    # The injected fault changes the later action; the first action still uses
    # its origin, preserving the temporal context of the experiment.
    plus = next(t for t in evidence["traces"] if t["query"] == "plus:origin:0")
    assert plus["first_action"][0] - plus["second_action"][0] == pytest.approx(0.25)
