# Native robotruth contract workflow

This optional development environment installs the complete base robotruth package at `8eb1704cf446239e3f03687c2ebd707cbab68e98`, with its declared runtime dependencies and no ML extras. The pinned root license is Apache-2.0. Only stock `extract_spec` and `check_contract` are executed; statistical/outcome evaluation, policyci, VLMs, inference and hardware are outside this comparison path.

From the repository root:

```powershell
uv venv --python 3.13 .venv-robotruth
uv pip sync --python .venv-robotruth/Scripts/python.exe --require-hashes integrations/robotruth/requirements.lock
uv pip install --python .venv-robotruth/Scripts/python.exe --no-deps --build-constraints integrations/robotruth/build.lock -r integrations/robotruth/frameworks.txt
.venv-robotruth/Scripts/python.exe -I scripts/validate_robotruth.py --output validation/robotruth-native-NEW
```

Installation needs network. Acquisition has a Python socket/DNS audit tripwire and checks the complete installed Python inventory before/after execution. `sources.json` pins all 43 package Python blobs. Windows Git line-ending conversions are explicitly recorded: each file must either match its exact Git blob or match after CRLF-to-LF conversion. Both the upstream blob and raw installed SHA-256 are saved. This is not a claim that every wheel byte equals the original Git file.

The dependency and build locks contain hashes. `pip sync` removes packages outside its lock, so install the pinned Git package afterward. Dependency installation/download cost is separate from the metadata-only experiment. The runner requires an environment without Torch, LeRobot or OpenPI and imports no policy framework.

## Controls and evidence contract

Eight controls use independently authored synthetic LeRobot-shaped metadata and a tiny valid Safetensors file containing two synthetic identity values. Those values are not trained weights. Explicit synthetic overlays state gripper convention, units and embodiment. Native warnings are preserved. Interventions separately change identity bytes, statistics, chunk length, camera name and declared units, or remove required gripper/weight evidence. Related variants are not independent research units.

Three additional controls copy original, hash-verified metadata from the already inspected move-cube SmolVLA/ACT development lineages. No weights or missing physical fields are supplied. One ACT copy has an explicitly injected +1 change to the first F32 `action.mean` in its postprocessor statistics. This is a metadata/statistics intervention; no new policy or processor execution follows from it. Unobserved candidate lineages are excluded.

The native APIs receive `allow_unknown=False` and no ignored fields. Extraction, full manifests, explicit overlays and all native findings are saved. A native FAIL caused by unknown fatal fields means certification was withheld; it is not a detected incompatibility or false alarm. Known unequal statistics hashes are separately creditable findings even when other required fields remain unknown. Native PASS with warnings does not establish all physical semantics.

The extractor defaults some LeRobot action semantics from configuration heuristics and may use a dataset repository ID for the embodiment. Save those stock outputs without treating them as independent physical authority. There are no hidden `robotruth.spec.json` overlays, fabricated published weights, invented calibration or invented publisher metadata. The full serialized manifests are retained because the native fingerprint omits runtime fields.

## Replay and comparison limits

Saved evidence can be verified using only stdlib Python:

```powershell
.venv-smoke/Scripts/python.exe -O scripts/validate_robotruth.py --verify validation/robotruth-native-001
```

This path checks identities, expected control facts, finding/manifest binding, native aggregation and the development-only scope. It does not rerun upstream algorithms. A new native run uses the isolated environment and a new output directory. Producer assertions and file hashes are not cryptographic execution attestation.

These controls exercise genuine extraction plus checking and credit stock capabilities. They are not a final MLTools comparison, a blind checkpoint evaluation or a timing study. Final adapters must supply the same available artifact/authority evidence to every method, preserve each native decision vocabulary, distinguish missing certification evidence from known mismatches, and declare temporal/physical applicability before evaluation. Robotruth's broader product capabilities are not judged by this one module.

`docs/matched-statistics-study.md` now documents an additional exact-shared-file development adapter. Stock extraction/checking and genuine LeRobot decoding consume the same pinned intended/deployed leaf artifacts, with original processor-style filenames. Its unchanged-value F64 serialization control separates numerical equality from native strict identity findings. Full certification remains withheld for missing evidence. The native union statistics hash produces both action/observation findings; the adapter preserves both and counts one underlying artifact difference. Final prospective execution still requires its committed protocol gate.
