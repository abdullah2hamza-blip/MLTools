"""Authored raw-vector transport contracts; no inferred robot embodiment."""

import math
import struct

from action_study_common import fingerprint
from normalization_profiles import require


def contracts(plan):
    """JSON is a YAML subset accepted by the pinned native registry."""
    result = {"protocol/v2.yaml": {"protocol_version": 2}}
    for cell in plan["cells"].values():
        n = len(cell["intended"]["statistics"][next(iter(cell["intended"]["statistics"]))])
        observation, action = schema_ids(n)
        result[f"observations/{observation}.yaml"] = {
            "id": observation,
            "description": (
                "Authored research input profile: normalized raw-index vector; no robot state claim"
            ),
            "fields": {"state": {"normalized_action": {"shape": [n], "dtype": "float64"}}},
        }
        result[f"actions/{action}.yaml"] = {
            "id": action,
            "description": (
                "Authored research output profile: raw-index vector; no physical units or embodiment"
            ),
            "response_key": "actions",
            "action_dim": n,
            "dtype": "float32",
            "representation": "raw_coordinate_vector",
        }
    return result


def schema_ids(n):
    return f"mltools_normalized_vector_{n}@1", f"mltools_raw_vector_{n}@1"


def metadata(cell, condition):
    spec = cell["conditions"][condition]["specification"]
    n = len(next(iter(spec["statistics"].values())))
    observation, action = schema_ids(n)
    return {
        "supported_schema_pairs": [{"observation_schema": observation, "action_schema": action}],
        "action_horizon": 1,
        "study_evidence": {
            "profile_authority": "Authored raw-index transport contract; not an official embodiment schema",
            "source_checkpoint": {
                k: cell[k] for k in ("model", "revision", "artifact_manifest", "artifact_manifest_sha256")
            },
            "framework": cell["framework"],
            "intended": cell["intended"],
            "deployment": spec,
            "intended_file_sha256": cell["conditions"]["numeric_control"]["file_sha256"],
            "deployment_file_sha256": cell["conditions"][condition]["file_sha256"],
            "checkpoint_physical_authority_status": "UNKNOWN",
            "execution": "Replay of actual captured decoder outputs; no learned policy inference",
        },
    }


def selection(cell_id, condition, meta, suffix=""):
    return {
        "episode_id": f"mltools-{cell_id}-{condition}{suffix}",
        "study_evidence": meta["study_evidence"],
    }


def float32(values):
    require(all(type(v) in (float, int) and math.isfinite(v) for v in values), "finite numeric wire values")
    result = [struct.unpack("<f", struct.pack("<f", v))[0] for v in values]
    require(all(math.isfinite(v) for v in result), "finite float32 wire values")
    return result


def registry_documents(plan):
    docs = contracts(plan)
    return {
        "protocol": docs["protocol/v2.yaml"],
        "observations": {v["id"]: v for k, v in docs.items() if k.startswith("observations/")},
        "actions": {v["id"]: v for k, v in docs.items() if k.startswith("actions/")},
        "embodiments": {},
    }


NEGATIVE_CONTROLS = ("wrong_output_width", "nonfinite_output", "wrong_input_width", "nonfinite_input")


def callback_counts(n):
    return {"initialize": 1, "ingest": n, "infer_replay": n, "finalize": 1}


def negative_error(name, n):
    if name == "wrong_output_width":
        return f"actions must have shape [T,{n}] with T >= 1; got (1, {n - 1})"
    if name == "nonfinite_output":
        return "actions contains NaN or infinity"
    if name == "wrong_input_width":
        return f"state field 'normalized_action' must have shape ({n},); got ({n - 1},)"
    return "state field 'normalized_action' contains NaN or infinity"


def scalar_descriptor(value):
    if math.isnan(value):
        return {"nonfinite": "NaN"}
    if math.isinf(value):
        return {"nonfinite": "+infinity" if value > 0 else "-infinity"}
    return value


def negative_arrays(base, name):
    """Independent expected fixed intervention, including the exact rejected arrays."""
    inputs, outputs = list(base["input"]), list(base["output"])
    if name == "wrong_output_width":
        outputs.pop()
    elif name == "nonfinite_output":
        outputs[0] = {"nonfinite": "NaN"}
    elif name == "wrong_input_width":
        inputs.pop()
    else:
        inputs[0] = {"nonfinite": "NaN"}
    return {
        "request_input": {"dtype": "float64", "shape": [len(inputs)], "values": inputs},
        "model_output": {"dtype": "float64", "shape": [1, len(outputs)], "values": outputs},
    }


def validate_result(data, plan, decoder):
    """Data-only replay binds all outcomes to the actual source observations."""
    require(
        fingerprint(data["registry"]) == fingerprint(registry_documents(plan)),
        "actual custom registry documents without invented embodiment",
    )
    require(set(data["cells"]) == set(plan["cells"]), "complete transport cell inventory")
    count, negatives = 0, 0
    for cell_id, cell in plan["cells"].items():
        cells = data["cells"][cell_id]
        require(set(cells) == set(cell["conditions"]), "complete transport condition inventory")
        for condition in cell["conditions"]:
            row = cells[condition]
            meta = metadata(cell, condition)
            selected = selection(cell_id, condition, meta)
            require(
                fingerprint(row["metadata"]) == fingerprint(meta)
                and fingerprint(row["initialized_with"]) == fingerprint(selected),
                "equal available evidence supplied to native service and replay model",
            )
            require(
                row["initialize_response"] == {"ok": True, "resumed": False}
                and row["finalize_response"] == {"ok": True}
                and row["selected_schema_pair"]
                == {**meta["supported_schema_pairs"][0], "embodiment_constraints": []},
                "actual lifecycle and native single-pair selection without embodiment negotiation",
            )
            wanted = decoder["cells"][cell_id][condition]["designs"]
            require(set(row["designs"]) == set(wanted), "all eight shared query geometries")
            total = 0
            for design_id, captured in wanted.items():
                observations = row["designs"][design_id]
                require(len(observations) == len(captured), "exact query denominator")
                for actual, native in zip(observations, captured, strict=True):
                    require(
                        actual["decision"] == "ACCEPT"
                        and actual["query"] == native["query"]
                        and fingerprint(actual["input"]) == fingerprint(native["input"])
                        and fingerprint(actual["ingested_input"]) == fingerprint(native["input"])
                        and fingerprint(actual["replayed_output"]) == fingerprint(native["output"])
                        and actual["input_dtype"] == "float64"
                        and actual["wire_dtype"] == "float32"
                        and actual["wire_shape"] == [1, len(native["output"])]
                        and fingerprint(actual["wire_output"]) == fingerprint(float32(native["output"])),
                        "actual decoded inputs, replay outputs and native float32 conversion",
                    )
                total += len(observations)
            require(row["callback_counts"] == callback_counts(total), "actual replay/lifecycle counts")
            count += total
        controls = data["negative_controls"][cell_id]
        require(set(controls) == set(NEGATIVE_CONTROLS), "complete structural negative controls")
        for name in NEGATIVE_CONTROLS:
            row = controls[name]
            output_fault = name in NEGATIVE_CONTROLS[:2]
            n = len(next(iter(cell["intended"]["statistics"].values())))
            base = decoder["cells"][cell_id]["numeric_control"]["designs"]["canonical"][0]
            require(
                row["decision"] == "REJECT"
                and row["native_error"] == negative_error(name, n)
                and row["callback_counts"] == callback_counts(int(output_fault))
                and fingerprint(row["base_query"]) == fingerprint(base)
                and fingerprint(row["actual_arrays"]) == fingerprint(negative_arrays(base, name)),
                "exact rejected mutation, native structural refusal and callback accounting",
            )
            negatives += 1
    require(set(data["negative_controls"]) == set(plan["cells"]), "negative-control cell inventory")
    require(
        data["runtime"]["accepted_replay_queries"] == count
        and data["runtime"]["rejected_structural_controls"] == negatives,
        "actual native query/control totals",
    )
    return count, negatives
