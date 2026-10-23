# Main-result reproduction

The supported acquisition platform is Windows AMD64, Python 3.13.7 for the core/LeRobot/stock tools and Python 3.11.13 for the OpenPI processor subset. `machine.json` records generic local hardware. This is processor and offline software-controller research; it loads no policy weights and operates no robot.

`../runs/prospective-001` is the original single attempt. `../runs/clean-reproduction-001` is the successful single clean reproduction. Their semantic analysis, witness report, joined external workflows and CSV are byte-identical. Metrics agree with measured phase costs excluded. Four separately installed environments provide distinct prefixes for all six native phases. The same machine and package caches are allowed; this is not independent hardware replication or another research sample.

## Replay saved evidence

In a fresh clone, first install uv and the requested Python versions. Use the checked-in lock without changing it:

```powershell
uv sync --locked --group dev --python 3.13.7
uv run --no-sync python -O scripts/run_publication_study.py --verify research/publication/runs/prospective-001
uv run --no-sync python -O scripts/run_publication_study.py --verify research/publication/runs/clean-reproduction-001
```

These commands do not import either ML framework or reacquire native outcomes. They require the complete repository history containing freeze `7513755e9627098c3529c46b3ae9a056d76434a5`, and exact protected file bytes. Do not apply root-wide formatting or line-ending normalization to evidence. `../protocol-freeze-001.json` and its full Git commit remain immutable; snapshot flags in earlier selection documents do not represent later live state.

## Install new exact profiles

Build the core wheel and prepare a bare OpenPI-version interpreter. The core wheel is checked against frozen Python source bytes by the setup script:

```powershell
uv build --wheel
uv venv .venv-openpi --python 3.11.13
uv run --no-sync python research/publication/reproduction/prepare_environments.py --uv C:/absolute/path/to/uv.exe --freeze-commit 7513755e9627098c3529c46b3ae9a056d76434a5 --environment-tag replica-a --output research/publication/reproduction/replica-a-setup
```

All setup output paths and environment names must be new. The script selects exact registry versions/hashes from existing locks and installs fixed Git revisions without dependency re-resolution. It verifies the full frozen Python/distribution inventory. Its source, compiler-independent wheel identity, commands, outputs, timing and profiles are saved. `replica-a` produces `.venv-replica-a-lerobot-clean`, `-openpi-clean`, `-robotruth-clean` and `-policy-space-clean` environments. Base `.venv` and `.venv-openpi` are used only to locate the requested Python runtimes.

The actual clean setup used the default `study` tag. `environment-build-001` retains the Windows long-filename checkout failure. `environment-build-002` retains the exact-profile rejection caused by an extra `hf-xet` distribution. `environment-build-003` succeeded after using process-local Git long-path configuration and selecting the frozen subset from the broader hashed lock. It preserves the earlier setup/source hashes. These are dependency setup iterations before the clean experiment, not native evaluation retries. No required package version, processor source, input, method or threshold changed.

## Reacquire in an isolated checkout

Committed result directories are immutable and single-use. A new native reproduction uses a separate checkout of the freeze, where those output directories do not yet exist. From the full clone:

```powershell
$studyRepo = (Get-Location).Path
$replicaRoot = Join-Path $studyRepo '.test-tmp/replica-workspace'
git worktree add --detach $replicaRoot 7513755e9627098c3529c46b3ae9a056d76434a5
$primaryOutput = Join-Path $replicaRoot 'research/publication/runs/prospective-001'
$frozenProtocol = Join-Path $replicaRoot 'research/publication/protocol-freeze-001.json'
& "$studyRepo/.venv/Scripts/python.exe" "$replicaRoot/scripts/run_publication_study.py" --protocol $frozenProtocol --freeze-commit 7513755e9627098c3529c46b3ae9a056d76434a5 --output $primaryOutput --lerobot-python "$studyRepo/.venv-replica-a-lerobot-clean/Scripts/python.exe" --openpi-python "$studyRepo/.venv-replica-a-openpi-clean/Scripts/python.exe" --robotruth-python "$studyRepo/.venv-replica-a-robotruth-clean/Scripts/python.exe" --policy-space-python "$studyRepo/.venv-replica-a-policy-space-clean/Scripts/python.exe"
```

Prepare a second set with `--environment-tag replica-b` and another new setup evidence directory. Run the same isolated runner with `--output "$replicaRoot/research/publication/runs/clean-reproduction-001"`, `--reproduce $primaryOutput` and all four `replica-b` interpreters. This executes the protocol's one original plus one clean attempt in that new checkout. The runner enforces source bytes, environment profiles, attempt order, time/resource bounds and the exact original-summary binding. Retain failures; do not overwrite outputs or silently retune/retry a prospective attempt. Additional executions are replication evidence, not additional independent samples in the paper.

Installation may use network. Acquisition denies audited network operations, uses offline CPU settings and one native worker at a time. Parent wall time and native interval CPU have different scope; OS/file caches and machine load are uncontrolled. Producer hashes and traces are not execution attestation.
