# Policy Space custom numeric transport

The pinned Policy Space registry supports caller-defined contracts through `POLICY_SPACE_CONTRACTS_DIR`. `scripts/run_policy_space_numeric.py` exercises that supported path on the actual outputs in `validation/matched-workflows-development-001/`. It uses the complete isolated installation, verifies the pinned distribution revisions and relevant source/schema blobs before and after execution, and calls the stock dispatcher with native MessagePack encoding/decoding.

The authored contract declares a normalized raw-index input vector and one raw-index output vector. It supplies neither physical units nor a robot embodiment, camera assignment or gripper convention. These are research contracts, not official Policy Space embodiment schemas. The server selects its single declared observation/action pair automatically; the capture records that actual selection. An exploratory attempt with partial request selectors was rejected by the native API, which requires an embodiment when using explicit negotiation. Automatic single-pair selection is a supported path and avoids inventing that evidence.

The adapter model replays the actual captured float64 decoder outputs. It receives the captured normalized inputs, while both the native service metadata and initialization callback receive the exact intended/deployment statistics and file identities. No trained policy, checkpoint adapter, WebSocket connection, robot or simulator runs. Native transport converts the responses to float32; all resulting arrays and conversion error are retained. Conversion error has raw-coordinate units and no fitted safety threshold.

Joined numerical verdicts describe the original float64 decoder boundary, before the transport conversion. They are not numerical checks on the float32 wire response. The final protocol must name its numerical boundary explicitly; this adapter establishes the applicability of the structural workflow and records the additional conversion.

## Actual development evidence

`validation/policy-space-numeric-development-001/` records 520 accepted query replays: five conditions of one reused ACT move-cube artifact lineage, eight geometries and 13 queries per geometry. The pristine and unchanged-value F64 serialization controls pass all numerical methods; the three explicit statistic interventions fail them. Policy Space accepts their valid raw-vector transport in every condition. This is a difference in contract scope, not a scored semantic miss or evidence of MLTools superiority. Robotruth statistics identity and withheld certification remain separate columns in the joined analysis.

Four additional structural controls use the actual pristine first query as their base. Wrong output width and nonfinite output are rejected after the model callback. Wrong input width and nonfinite input are rejected before ingestion/inference. These controls invoke four separate initialization/finalization pairs and two additional replay callbacks. They are implementation applicability controls, not four independent research samples. Typed descriptors retain the actual codec-decoded inputs and actual model outputs, including explicit nonfinite tags; replay checks their exact intervention recipes and native error messages.

The capture contains nine initialization/finalization pairs in total, 522 ingestion/replay callbacks, and no observed socket connection, DNS or bind event. The parent has a 180-second subprocess timeout; the worker records its measured RSS against a 2 GiB post-measurement bound. These controls are not an OS resource sandbox or cryptographic execution attestation. Raw stdout/stderr, the exact request, authored schema files, native outputs and regenerated analysis are hash-bound. Data-only verification imports neither NumPy nor Policy Space nor an ML framework.

## Reproduce or replay

Use the environment setup in `integrations/baselines/README.md` and choose a new output directory:

```powershell
.venv-smoke/Scripts/python.exe scripts/run_policy_space_numeric.py --matched validation/matched-workflows-development-001 --python .venv-baselines/Scripts/python.exe --output validation/policy-space-numeric-development-NEW
.venv-smoke/Scripts/python.exe -O scripts/run_policy_space_numeric.py --verify validation/policy-space-numeric-development-001
```

The native entry point rejects prospective prepared plans before importing the native tool. This development validation resolves the custom-transport applicability gate. A complete committed protocol, selected-case execution, fresh witnesses on those exact artifacts, clean main-result reproduction and the paper/PDF still remain. Stock physical-schema mapping remains ungrounded for the selected raw-coordinate metadata.
