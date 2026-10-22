"""Action-only schedule separation, canonical binding and shared-oracle controls."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import action_designs as ad  # noqa: E402
import query_designs as qd  # noqa: E402

from mltools.obligations import (  # noqa: E402
    Axis,
    Reference,  # noqa: E402
    compile_suite,
    evaluate,
)


def fixture():
    return compile_suite(
        (Axis("action", "0", "normalized", -0.8, 0.8), Axis("action", "1", "normalized", -0.8, 0.8)),
        ("raw0", "raw1"),
    )


def ref():
    return {
        "coefficients": [[1.6, 0], [0, 2.4]],
        "offset": [1.0, -2.0],
        "output_units": ["raw_index_unit", "raw_index_unit"],
        "authority": "Independent designed numerical fixture",
        "affine_fragment_verified": True,
    }


@pytest.mark.parametrize("name", list(ad.generate_all(fixture())))
def test_equal_budget_roundtrip_and_independent_oracle(name):
    design = ad.generate_all(fixture())[name]
    assert len(design.probes) == 5
    assert ad.ActionDesign.from_dict(json.loads(json.dumps(design.to_dict()))) == design
    assert design.to_dict()["schedule"] != qd.SCHEDULE
    assert design.to_dict()["schedule"]["native_decoder_calls_per_query"] == 1
    observed = {
        "design_sha256": design.sha256,
        "outputs": {p.id: [1 + 2 * p.values[0], -2 + 3 * p.values[1]] for p in design.probes},
    }
    report = qd.analyze(design, ref(), observed)
    assert report["full_affine_request_status"] == "PASS"
    # A changed slope is compared against the independent intended contract.
    observed["outputs"] = {p.id: [1 + 4 * p.values[0], -2 + 3 * p.values[1]] for p in design.probes}
    assert qd.analyze(design, ref(), observed)["full_affine_request_status"] == "FAIL"


def test_canonical_queries_bind_the_real_core_suite():
    suite = fixture()
    design = ad.generate(suite, "canonical")
    assert [(p.id, p.coordinates, p.values) for p in design.probes] == [
        (p.id, p.coordinates, p.values) for p in suite.probes
    ]
    observed = {p.id: [1 + 2 * p.values[0], -2 + 3 * p.values[1]] for p in design.probes}
    assert (
        evaluate(suite, Reference(**ref()), {"suite_sha256": suite.sha256, "outputs": observed})["status"]
        == "PASS"
    )


@pytest.mark.parametrize(
    "tamper",
    [
        "temporal_schedule",
        "state_axis",
        "input",
        "id",
        "seed",
        "budget",
        "extra_field",
        "boolean",
        "call_type",
    ],
)
def test_reject_false_execution_and_generator_claims(tamper):
    value = copy.deepcopy(ad.generate(fixture(), "canonical").to_dict())
    if tamper == "temporal_schedule":
        value["schedule"] = qd.SCHEDULE.copy()
    elif tamper == "state_axis":
        value["axes"][0]["port"] = "origin"
    elif tamper == "input":
        value["probes"][1]["values"][0] /= 2
    elif tamper == "id":
        value["probes"][1]["id"] = "unexecuted-query"
    elif tamper == "seed":
        value["seed"] = True
    elif tamper == "budget":
        value["probes"].pop()
    elif tamper == "boolean":
        value["probes"][1]["coordinates"][0] = True
    elif tamper == "call_type":
        value["schedule"]["native_decoder_calls_per_query"] = True
    else:
        value["temporal_trace"] = True
    with pytest.raises(ValueError):
        ad.ActionDesign.from_dict(value)


def test_hash_ownership_seed_binding_and_stateless_scope():
    first = ad.generate(fixture(), "random_dense", 7)
    second = ad.generate(fixture(), "random_dense", 23)
    assert first.sha256 != second.sha256
    altered = first.to_dict()
    altered["schedule"]["queue_operations"] = True
    assert first.to_dict()["schedule"]["queue_operations"] is False
    with pytest.raises(ValueError):
        ad.ActionDesign.from_dict(altered)
    with pytest.raises(ValueError):
        qd.Design.from_dict(first.to_dict())
    state_suite = compile_suite((Axis("current", "0", "raw", -1, 1),), ("out",))
    with pytest.raises(ValueError):
        ad.generate(state_suite, "canonical")
