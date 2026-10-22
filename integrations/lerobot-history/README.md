# Historical LeRobot processor reproduction

This executes complete installed parent/fix source revisions from PR4057, with explicit synthetic queue boundary inputs. It uses the current locked CPU dependencies as a compatibility environment, not the historical full training runtime. No model, policy inference, simulator or live robot is constructed.

Prepare the main dependency environment first, then install each genuine historical package before adding the shared dependency path:

```powershell
uv sync --locked --extra lerobot --group dev
uv venv .venv-lerobot-history --python .venv/Scripts/python.exe
uv pip install --python .venv-lerobot-history/Scripts/python.exe --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@240ea44cf9f887df944019cf46e18a0d496e1a56"
uv venv .venv-lerobot-history-after --python .venv/Scripts/python.exe
uv pip install --python .venv-lerobot-history-after/Scripts/python.exe --no-deps "lerobot @ git+https://github.com/huggingface/lerobot.git@7c98c1b2ce309f94e542bd4b9ee3fe176301edbf"
$taskDependencySite = (Resolve-Path .venv/Lib/site-packages).Path
Set-Content .venv-lerobot-history/Lib/site-packages/mltools_shared_deps.pth $taskDependencySite -Encoding ascii
Set-Content .venv-lerobot-history-after/Lib/site-packages/mltools_shared_deps.pth $taskDependencySite -Encoding ascii
uv run --no-sync python scripts/reproduce_lerobot_history.py --before-python .venv-lerobot-history/Scripts/python.exe --after-python .venv-lerobot-history-after/Scripts/python.exe --output .test-tmp/history-NEW
```

The `.pth` files contain only a local directory path; they expose the existing locked dependencies after each environment's own package directory. They do not patch source or import extracted functions. Historical workers use isolated Python mode and verify their installed Git direct URL and six canonical-LF source blobs before ML imports. The fixed and parent sources can have the same distribution version; the immutable Git identities are required.

The runner executes one ML worker at a time with a 180-second timeout. It records observed values, reset/refill behavior, source identities, project worker hash and dependency-lock hash. Failed source checks or unexpected outcomes stop execution. To verify the committed development evidence without ML imports:

```powershell
uv run --no-sync python scripts/reproduce_lerobot_history.py --verify validation/lerobot-history-001
```

The independent semantic reference is the pre-existing action-representation guide and chunk-start training statistics computation. The public evaluation-style processor sequence is the reproduced boundary. The parent's normal synchronous rollout already rejected relative actions, and RTC is excluded. This does not establish ACT-relative integration or trained-policy task success. See `research/publication/lerobot-history.md` for exact authority and scope.
