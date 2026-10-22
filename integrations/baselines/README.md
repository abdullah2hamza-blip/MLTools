# Native baseline applicability environment

These are development workflow controls for two complete installed packages. They are not final comparative experiments. The environment has no Torch, model weights, simulators or hardware support. No network service is opened by the runner.

From the repository root, using uv and Python 3.13:

```powershell
uv venv --python 3.13 .venv-baselines
uv pip sync --python .venv-baselines/Scripts/python.exe --require-hashes integrations/baselines/requirements.lock
uv pip install --python .venv-baselines/Scripts/python.exe --no-deps --build-constraints integrations/baselines/build.lock -r integrations/baselines/frameworks.txt
.venv-baselines/Scripts/python.exe -I scripts/validate_native_baselines.py --output validation/native-baselines-NEW
```

`pip sync` removes packages outside the dependency lock, so run the pinned Git installation afterward. `frameworks.txt` fixes both upstream commits; `sources.json` checks installed Git metadata and nine Policy Space source/schema blobs plus the preflight implementation before import. Build constraints pin setuptools and wheel. Dependency and build locks contain hashes. Installation needs network; subsequent runner execution performs no downloads.

The runner invokes the actual preflight CLI on four explicit dataset/policy/robot contracts. Correct declarations pass, declared shape and unit mismatches fail, and absent unit metadata returns the native input-error exit code. It preserves this error as an input error instead of counting it as detection.

Policy Space runs eight controls through its actual server dispatcher with native MessagePack encoding/decoding, built-in schema selection, observation/action validation and episode lifecycle. Five incompatible structural inputs are rejected. A valid control and two schema-valid relabelings are accepted. This uses a synthetic fixture model; the runner does not execute a policy adapter, a WebSocket client/server, a trained checkpoint, or robot commands. Lifecycle callbacks establish that the server invokes them, not that arbitrary models clear their private state correctly.

The native single-arm contract has **eight action coordinates**: seven arm joints and one gripper. The current OpenPI LIBERO processor captures have seven end-effector/gripper coordinates. Feeding one directly to the other would manufacture a schema mismatch. The final comparison must supply an appropriate, independently grounded adapter or record inapplicability.

Recorded development output is `validation/native-baselines-002/summary.json`. `research/publication/baseline-audit.md` documents the wider comparison requirements. Tether, XPolicyLab, robotruth and ActionABI are not executed by this runner. No full-product superiority claim follows.

The separate `integrations/robotruth/` profile now executes stock metadata extraction and checking without ML dependencies. Its recorded controls and decision interpretation are documented there; this two-package runner remains scoped as above.

The separate `integrations/actionabi/` profile builds and exercises the native CPU scoring, equivalence/refusal and separating-probe library APIs. Its trajectory assumptions and synthetic development controls are documented there.

The pinned Policy Space registry also supports `POLICY_SPACE_CONTRACTS_DIR` for caller-defined contracts. Lack of a grounded stock embodiment mapping therefore does not make every raw numeric transport workflow inapplicable. `research/publication/method-applicability.json` keeps a clearly authored custom raw-coordinate dispatcher/codec path pending before the main freeze, without inventing physical joint/gripper semantics or learned-policy execution.
