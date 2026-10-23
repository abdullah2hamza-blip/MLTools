"""Install fresh native environments from the committed study's exact profiles."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import tomllib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from normalization_profiles import require  # noqa: E402
from publication_protocol import FREEZE  # noqa: E402
from publication_protocol import verify as verify_protocol  # noqa: E402
from run_publication_study import environment  # noqa: E402
from run_publication_study import verify as verify_run  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lerobot_dependencies(protocol):
    """Select existing hash-locked rows; never resolve newer package versions."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    normalize = lambda name: name.lower().replace("_", "-")  # noqa: E731
    rows = {(normalize(p["name"]), p.get("version")): p for p in lock["package"]}
    lines = ["# Frozen Windows CPU profile; excludes separately installed local/Git packages."]
    for name, version in sorted(protocol["native_environments"]["lerobot"]["packages"].items()):
        if normalize(name) in {"lerobot", "mltools"}:
            continue
        row = rows[(normalize(name), version)]
        require("registry" in row["source"], "hash-locked registry dependency")
        artifacts = [*row.get("wheels", []), *([row["sdist"]] if "sdist" in row else [])]
        hashes = sorted({a["hash"] for a in artifacts})
        require(hashes and all(h.startswith("sha256:") for h in hashes), "dependency hashes")
        lines.append(name + "==" + version + " " + " ".join("--hash=" + h for h in hashes))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--freeze-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-setup", type=Path)
    args = parser.parse_args()
    folder = args.output.resolve()
    require(folder.is_relative_to(ROOT), "workspace-contained installation evidence")
    require(args.uv.is_file(), "explicit uv executable")
    protocol, _ = verify_protocol(ROOT / FREEZE, args.freeze_commit)
    original = ROOT / protocol["specification"]["attempt_policy"]["primary_output"]
    verify_run(original)
    wheel = ROOT / "dist/mltools-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel) as archive:
        require(
            all(
                archive.read("mltools/" + p.name) == p.read_bytes()
                for p in (ROOT / "src/mltools").glob("*.py")
            ),
            "installed wheel contains the current frozen core sources",
        )
    prefixes = {
        name: ROOT / (".venv-study-" + name.replace("_", "-") + "-clean")
        for name in protocol["native_environments"]
    }
    previous = None
    if args.resume_setup is not None:
        previous_path = args.resume_setup.resolve()
        require(previous_path.is_relative_to(ROOT), "contained earlier setup evidence")
        previous = json.loads((previous_path / "setup.json").read_text(encoding="utf-8"))
        require(previous["status"] == "FAILED", "resume installation only after retained setup failure")
        require(previous["freeze_commit"] == args.freeze_commit, "same frozen installation target")
        require(
            digest(previous_path / "prepare_environments.py") == previous["setup_source_sha256"],
            "earlier exact setup source retained",
        )
    else:
        require(all(not p.exists() for p in prefixes.values()), "new separately installed environments")
    folder.mkdir(parents=True, exist_ok=False)
    dependencies = folder / "lerobot-requirements.lock"
    dependencies.write_text(lerobot_dependencies(protocol), encoding="utf-8")
    commands = []
    profiles = {}
    record = {
        "schema_version": 1,
        "status": "STARTED",
        "freeze_commit": args.freeze_commit,
        "original_summary_sha256": digest(original / "summary.json"),
        "setup_source_sha256": digest(Path(__file__)),
        "uv_sha256": digest(args.uv),
        "core_wheel_sha256": digest(wheel),
        "generated_lock_sha256": digest(dependencies),
        "resumes_setup": None
        if previous is None
        else {
            "path": previous_path.relative_to(ROOT).as_posix(),
            "setup_sha256": digest(previous_path / "setup.json"),
            "reason": (
                "Git subprocess long-path configuration repairs dependency checkout only; "
                "no native evaluation has run in these environments."
            ),
        },
        "commands": commands,
        "environments": profiles,
        "scope": (
            "Separate fresh installations on the same Windows machine; package caches may be reused. "
            "No native benchmark calls occur during setup. Not an independent machine or research sample."
        ),
    }
    cache = ROOT / ".test-tmp/uv-cache"
    process_env = os.environ.copy()
    config_index = int(process_env.get("GIT_CONFIG_COUNT", "0"))
    git_overrides = {
        "GIT_CONFIG_COUNT": str(config_index + 1),
        f"GIT_CONFIG_KEY_{config_index}": "core.longpaths",
        f"GIT_CONFIG_VALUE_{config_index}": "true",
    }
    process_env.update(git_overrides)
    record["git_subprocess_environment_overrides"] = git_overrides

    def run(arguments):
        command = [str(args.uv.resolve()), *map(str, arguments), "--cache-dir", str(cache)]
        start = time.perf_counter()
        result = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, check=False, env=process_env
        )
        commands.append(
            {
                "command": command,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "wall_seconds": time.perf_counter() - start,
            }
        )
        write(folder / "setup.json", record)
        print(arguments[0], arguments[1], "returncode", result.returncode, flush=True)
        require(result.returncode == 0, "installation command succeeded; see retained setup.json")

    try:
        for name, prefix in prefixes.items():
            python = prefix / "Scripts/python.exe"
            if previous is not None and name in previous["environments"]:
                actual = environment(python)
                require(actual == previous["environments"][name], "unchanged completed fresh installation")
                require(
                    {k: actual[k] for k in ("python", "packages")} == protocol["native_environments"][name],
                    "completed installation still matches freeze",
                )
                profiles[name] = actual
                continue
            base = ROOT / (
                ".venv-openpi/Scripts/python.exe" if name == "openpi" else ".venv/Scripts/python.exe"
            )
            if not prefix.exists():
                run(["venv", prefix, "--python", base])
            lock = {
                "lerobot": dependencies,
                "openpi": ROOT / "integrations/openpi/requirements-cpu.lock",
                "robotruth": ROOT / "integrations/robotruth/requirements.lock",
                "policy_space": ROOT / "integrations/baselines/requirements.lock",
            }[name]
            extra = ["--torch-backend", "cpu"] if name == "openpi" else []
            run(["pip", "sync", "--python", python, "--require-hashes", lock, *extra])
            if name == "lerobot":
                run(
                    [
                        "pip",
                        "install",
                        "--python",
                        python,
                        "--no-deps",
                        "--build-constraints",
                        ROOT / "integrations/baselines/build.lock",
                        "lerobot @ git+https://github.com/huggingface/lerobot.git@5517a326eeb9b457cfe07d59b9bfdf291799c8d9",
                    ]
                )
                run(["pip", "install", "--python", python, "--no-deps", wheel])
            else:
                directory = "baselines" if name == "policy_space" else name
                run(
                    [
                        "pip",
                        "install",
                        "--python",
                        python,
                        "--no-deps",
                        "--build-constraints",
                        ROOT / "integrations" / directory / "build.lock",
                        "-r",
                        ROOT / "integrations" / directory / "frameworks.txt",
                    ]
                )
            actual = environment(python)
            require(
                {k: actual[k] for k in ("python", "packages")} == protocol["native_environments"][name],
                "exact frozen native Python/package profile",
            )
            profiles[name] = actual
            write(folder / "setup.json", record)
        verify_protocol(ROOT / FREEZE, args.freeze_commit)
        record["status"] = "SUCCEEDED"
    except BaseException as exc:
        record.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write(folder / "setup.json", record)
    print("Installed four exact frozen profiles in distinct new environments")


if __name__ == "__main__":
    main()
