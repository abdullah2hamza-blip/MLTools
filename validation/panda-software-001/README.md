# Panda software development capture

Actual native robosuite 1.4.1/MuJoCo 2.3.7 execution in `.venv-panda` on Windows x64, Python 3.11.13. The complete pinned wheel and all locked dependency versions were checked. No learned policy, robot hardware, rendering or physics stepping was used.

`summary.json` binds exactly six evidence files: `inputs.json`, `request.json`, `native.stdout.txt`, `native.stderr.txt`, `capture.json` and `analysis.json`. Raw output bytes are preserved. The summary additionally binds core/producer/environment source files. This README is documentation outside that closed evidence set.

The capture includes 114 affine operator queries, two full native goal setters, four orientation calls and 238 gripper formatting calls with populated-state native/omitted reset controls. Rotation/gripper cases retain affine UNKNOWN. `docs/panda-software.md` explains authority, injection labels, numerical scope, outcomes and limits. This is one repeatedly measured development software cluster, not final evaluation or checkpoint execution.

Offline verification: `.venv-smoke/Scripts/python.exe -O scripts/validate_panda.py --verify validation/panda-software-001`. Native rerun setup and clean reproduction are documented in `integrations/panda/README.md`.
