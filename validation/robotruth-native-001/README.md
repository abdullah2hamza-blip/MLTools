# Stock robotruth development workflow controls

The complete robotruth base package at `8eb1704cf446239e3f03687c2ebd707cbab68e98` was installed with hash-locked dependencies in a separate Python 3.13.7 environment. Native `extract_spec` and `check_contract` ran on all 11 cases with default fail-closed handling, explicit saved overlays and no ignored fields. All 43 package Python sources were checked before and after execution. Four match their original Git blob bytes; 39 match after explicitly recorded Windows CRLF-to-LF conversion. Every raw installed source hash is preserved.

| Control | Native aggregate | Interpretation |
| --- | --- | --- |
| Synthetic complete declaration | PASS, 2 warnings | Calibration hashes are absent; this is not full physical conformance |
| Changed synthetic identity bytes | FAIL | Known weight-identity mismatch |
| Changed synthetic statistics | FAIL | Known action/observation statistics-hash mismatches |
| Changed synthetic chunk length | FAIL | Known chunk mismatch |
| Changed synthetic camera name | FAIL | Known camera-set/preprocessing mismatches |
| Changed synthetic declared units | PASS, 3 warnings | Units mismatch has native warning severity |
| Missing synthetic gripper convention | FAIL | Certification withheld on unknown required evidence |
| Missing synthetic identity file | FAIL | Certification withheld on unknown required evidence |
| Published SmolVLA metadata | FAIL | Missing weights, control rate and gripper convention |
| Published ACT metadata | FAIL | Missing weights, control rate and gripper convention |
| Published ACT metadata with injected statistics change | FAIL | Known statistics mismatches plus the same missing certification evidence |

`inputs/` preserves all extraction inputs. Synthetic tensor values are explicitly authored identity fixtures, not trained weights. Public inputs copy only the original, hash-verified six metadata/statistics files from each already inspected development bundle. The ACT intervention adds 1 to the first F32 `action.mean` in the copied postprocessor file; original publisher files are unchanged. No missing published physical field is filled in. Newly selected candidate lineages are excluded.

`cases/` contains both complete extracted manifests, explicit overlays and full native findings for every condition. The extractor's stock action-mode/space defaults and dataset-name embodiment heuristic are not authenticated physical semantics. Unknown fatal fields withhold certification; they are not detections or false alarms. Multiple changed findings from one statistics artifact are one intervention, and the public ACT/SmolVLA move-cube lineage is shared development evidence.

`summary.json` binds all 143 input/report files, original source identities, runner/lock bytes, upstream inventory, packages, native outcomes and runtime. Acquisition observed no socket/DNS audit events or ML framework imports. The audit tripwire is not an OS sandbox. The single 2.16-second workflow run is not a timing estimate or a speed comparison; installation is excluded. There was no trained inference, temporal processor execution, statistical/outcome evaluation, simulator, hardware or network service.

Data-only verification, including Python optimized mode:

```powershell
.venv-smoke/Scripts/python.exe -O scripts/validate_robotruth.py --verify validation/robotruth-native-001
```

This verifies recorded identities, control facts and finding/manifest binding; it does not rerun the native checker. For a new native acquisition use the pinned environment and a new destination as documented in `integrations/robotruth/README.md`. A final fair comparison still needs frozen equal-evidence adapters and applicability rules. No superiority, final evaluation or publication-readiness claim follows from this development record.
