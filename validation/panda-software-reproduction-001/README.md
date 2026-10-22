# Clean-environment Panda development reproduction

Actual native run in the new `.venv-panda-clean` environment, installed from the hash-locked local package cache. Python 3.11.13 and all 14 resolved package versions match the original. `uv pip check` passed. This is an environment reproduction, not another independent research sample.

The closed manifest binds the same six evidence filenames as the original capture. Raw stdout/stderr retain exact bytes and environment-specific paths. Nine semantic capture fields reproduce exactly; `inputs.json`, `request.json` and `analysis.json` match byte for byte. Runtime/peak RSS differ and are recorded, not treated as deterministic results.

`validation/panda-reproduction-verification.json` records the actual comparison. Offline replay: `.venv-smoke/Scripts/python.exe -O scripts/validate_panda.py --verify validation/panda-software-reproduction-001`. See `integrations/panda/README.md` and `docs/panda-software.md` for commands, Windows compatibility, source authority and scientific limits. This README is outside the six-file evidence manifest.
