# Offline Panda software-controller profile

This optional Windows x64 profile installs the complete released `robosuite==1.4.1` wheel with MuJoCo 2.3.7 and all 14 resolved packages from `requirements.lock`. It creates a real, renderer-free `Lift` environment with a Panda robot and `OSC_POSE` controller. It exercises software operators, two full controller goal setters and populated gripper reset behavior. It performs zero environment steps, torque-controller calls or renders. No policy weights, learned-policy execution, task-success result or physical robot are involved.

## Setup and execution

Use Python 3.11.13 for the captured Windows environment. Dependency setup includes the approximately 185 MiB robosuite wheel and is separate from benchmark metadata. Commands run from the repository root; choose new environment and output names:

```powershell
uv venv --python 3.11.13 .venv-panda-NEW
uv pip sync --python .venv-panda-NEW/Scripts/python.exe --require-hashes integrations/panda/requirements.lock
uv --cache-dir .test-tmp/uv-panda-cache pip check --python .venv-panda-NEW/Scripts/python.exe
.venv-smoke/Scripts/python.exe scripts/validate_panda.py --python .venv-panda-NEW/Scripts/python.exe --output validation/panda-software-NEW
```

The parent interpreter needs the installed MLTools core; it does not need ML or simulation dependencies. The worker uses the explicitly selected interpreter in isolated mode, one thread per numerical library and a 180-second subprocess timeout. A Python audit hook rejects socket connection/address-resolution events. Peak RSS is measured and checked against 2 GiB after execution; this is not an OS memory limit or network sandbox.

Data-only replay requires no NumPy, robosuite, MuJoCo, Torch, LeRobot or OpenPI imports:

```powershell
.venv-smoke/Scripts/python.exe -O scripts/validate_panda.py --verify validation/panda-software-001
.venv-smoke/Scripts/python.exe -O scripts/validate_panda.py --verify validation/panda-software-reproduction-001
.venv-smoke/Scripts/python.exe -O scripts/compare_panda_reproduction.py validation/panda-software-001 validation/panda-software-reproduction-001 --output .test-tmp/panda-comparison-NEW.json
```

A new `.venv-panda-clean` installed the same hash-locked packages from the local cache. Its actual native execution is retained in `validation/panda-software-reproduction-001/`. Nine semantic capture fields match the original; inputs, requests and analysis match byte for byte. Runtime measurements and environment-specific stderr paths differ. This is a development workflow reproduction, not a new experimental unit or reproduction of the unfinished main study.

## Explicit Windows import compatibility

The released wheel attempts to load an absent `robosuite/utils/mujoco.dll`. During import only, the worker redirects that exact `ctypes.WinDLL` path to the installed, separately pinned `mujoco/mujoco.dll`. At the same import point it sets the native `MUJOCO_GPU_RENDERING` runtime macro to `False` so that `MUJOCO_GL=disable` is respected. It immediately restores the ctypes loader. Native source files are never edited, and no `macros_private.py` is permitted.

`distribution.json` pins the release wheel hash, complete installed RECORD fingerprint, bootstrap source files and DLL hash. All 1,027 hashed robosuite files (613,108,465 bytes) are checked before and after execution. Eight semantic source files additionally match Git blobs from `ARISE-Initiative/robosuite@fbee5844ff5632f5b5698e204ec5357ca50be0df`. The complete release wheel differs from that Git checkout, especially in its loader/macros; the wheel is not described as a full installation of that commit. RECORD and producer hashes establish recorded integrity, not cryptographic execution attestation.

The package's MIT notice, including its MuJoCo attribution, is copied in `LICENSE.robosuite.txt`. Dependencies retain their installed licenses. See `docs/panda-software.md` for contracts, results and limitations.
