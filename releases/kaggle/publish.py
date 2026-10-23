"""Plan a private Kaggle release; external operations require explicit execution.

This module uses only the standard library. It never imports Kaggle. The pinned
CLI is imported in a child process only after --execute and authorization.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

CLI_VERSION = "2.2.5"
CLI_REVISION = "8b88a6aeffed5acae0b1e43137d946afd5c43351"
OWNER = "abdullah2hamza"
SLUG = "boundary-conformance-replay"
IDENTITY = OWNER + "/" + SLUG
TITLE = "Boundary Conformance Replay"
DATASET_TITLE = "ML Boundary Conformance: Saved Trace Replay"
DATASET_FILES = {
    "dataset-metadata.json",
    "payload.zip",
    "conditions.csv",
    "README.md",
    "release-manifest.json",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
}
NOTEBOOK_FILES = {"kernel-metadata.json", "replay.ipynb"}
SEMANTIC_OUTPUTS = {
    "conditions.csv",
    "analysis.json",
    "metrics.json",
    "witnesses.json",
    "workflow-analysis.json",
}
REQUIRED_OUTPUTS = SEMANTIC_OUTPUTS | {"replay-summary.json"}
REMOTE_STEPS = {"dataset-create", "dataset-status", "notebook-push", "notebook-status", "output-pull"}
MUTATIONS = {"dataset-create", "notebook-push"}
STEPS = REMOTE_STEPS | {"plan", "public-plan", "output-verify", "reconcile"}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_FILE = 64 * 1024 * 1024
MAX_PAYLOAD = 512 * 1024 * 1024
MAX_MEMBERS = 10000
PRIVATE_PATTERNS = (
    rb"(?i)[a-z]:[\\/]+Users[\\/]+[^\\/\s\"']+",
    rb"(?i)/(?:home|Users)/abdul(?:[\\/]|[\s\"']|$)",
    rb"(?i)(?:git@github\.|ssh://|https?://[^/\s:@]+:[^/\s@]+@)",
    rb"(?i)(?:KAGGLE_API_TOKEN|KAGGLE_KEY|KAGGLE_USERNAME|HF_TOKEN|github_pat_|ghp_)",
    rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
)
FORBIDDEN_PARTS = {".git", ".kaggle", ".aws", ".ssh", ".codex", ".agents", "__pycache__"}
FORBIDDEN_NAMES = {".env", "kaggle.json", "credentials", "credentials.json", "id_rsa", "id_ed25519"}
FORBIDDEN_SUFFIXES = {".exe", ".dll", ".so", ".dylib", ".whl", ".pkl", ".pickle", ".pt", ".pth", ".ckpt"}
FORBIDDEN_IMPORTS = {
    "kaggle",
    "kagglehub",
    "kaggle_secrets",
    "requests",
    "httpx",
    "urllib",
    "socket",
    "huggingface_hub",
    "torch",
    "lerobot",
    "openpi",
    "transformers",
}


def require(condition, message):
    if not condition:
        raise ValueError("Kaggle publication: " + message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def parse(raw):
    def bad(value):
        raise ValueError("Kaggle publication: nonfinite JSON constant " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def read(path, limit=MAX_FILE):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "regular file required: " + path.name)
    require(path.stat().st_size <= limit, "bounded file: " + path.name)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "bounded file snapshot")
    return raw


def privacy(raw, label):
    require(not any(re.search(pattern, raw) for pattern in PRIVATE_PATTERNS), "privacy pattern in " + label)


def safe_name(name):
    require(isinstance(name, str) and name and "\\" not in name, "relative POSIX file name")
    parts = PurePosixPath(name).parts
    require(
        not name.startswith("/")
        and ":" not in name
        and "\x00" not in name
        and all(p not in ("", ".", "..") for p in name.split("/")),
        "safe relative file name",
    )
    lowered = [p.lower() for p in parts]
    require(
        not any(p in FORBIDDEN_PARTS or p in FORBIDDEN_NAMES or p.startswith(".venv") for p in lowered)
        and PurePosixPath(name).suffix.lower() not in FORBIDDEN_SUFFIXES,
        "excluded private/runtime file: " + name,
    )
    return name


def hashes(value, label):
    require(isinstance(value, dict) and value, "nonempty " + label)
    for name, sha in value.items():
        safe_name(name)
        require(isinstance(sha, str) and SHA256.fullmatch(sha), "SHA256 in " + label)
    return value


def boolean(value, expected):
    return type(value) is bool and value is expected or value == str(expected).lower()


def closed(folder, expected):
    require(folder.is_dir() and not folder.is_symlink(), "regular directory: " + folder.name)
    require({p.name for p in folder.iterdir()} == expected, "closed inventory: " + folder.name)
    require(all(p.is_file() and not p.is_symlink() for p in folder.iterdir()), "flat regular files")


def check_payload(path):
    """Inspect names/bytes without extracting or executing archive content."""
    require(path.stat().st_size <= MAX_PAYLOAD, "bounded payload archive")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(0 < len(entries) <= MAX_MEMBERS, "bounded payload member count")
        require(len({item.filename for item in entries}) == len(entries), "unique payload member names")
        require(sum(item.file_size for item in entries) <= MAX_PAYLOAD, "bounded expanded payload")
        for item in entries:
            safe_name(item.filename)
            require(not item.is_dir(), "payload contains files only")
            kind = stat.S_IFMT(item.external_attr >> 16)
            require(kind in (0, stat.S_IFREG), "no archive links or special files")
            require(not item.flag_bits & 1 and item.file_size <= MAX_FILE, "unencrypted bounded member")
            raw = archive.read(item)
            privacy(raw, item.filename)
            if item.filename.endswith(".safetensors"):
                require(len(raw) <= 128 * 1024 and len(raw) >= 10, "small statistics, not model weights")
                header_size = int.from_bytes(raw[:8], "little")
                require(2 <= header_size <= len(raw) - 8, "bounded Safetensors header")
                header = parse(raw[8 : 8 + header_size])
                tensors = [v for k, v in header.items() if k != "__metadata__"]
                require(
                    tensors
                    and all(isinstance(v, dict) and v.get("dtype") in ("F32", "F64") for v in tensors),
                    "numeric statistics tensors",
                )
    return len(entries)


def check_notebook(raw, payload_sha):
    notebook = parse(raw)
    require(type(notebook.get("nbformat")) is int and notebook["nbformat"] == 4, "notebook format 4")
    require(isinstance(notebook.get("cells"), list) and notebook["cells"], "notebook cells")
    sources = []
    for cell in notebook["cells"]:
        require(isinstance(cell, dict) and cell.get("cell_type") in ("code", "markdown"), "notebook cell")
        source = cell.get("source")
        require(
            isinstance(source, str) or isinstance(source, list) and all(isinstance(s, str) for s in source),
            "text notebook source",
        )
        source = source if isinstance(source, str) else "".join(source)
        sources.append(source)
        require(not cell.get("attachments"), "no hidden notebook attachments")
        if cell["cell_type"] == "code":
            require(
                cell.get("execution_count") is None and cell.get("outputs") == [], "clean notebook outputs"
            )
            tree = ast.parse(source)
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = (
                        [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                    )
                    require(
                        not any(n.split(".")[0] in FORBIDDEN_IMPORTS for n in names),
                        "no credential/network/native notebook imports",
                    )
                if isinstance(node, ast.Attribute):
                    require(
                        node.attr not in ("getenv", "get_secret", "authenticate"),
                        "no credential/environment notebook access",
                    )
                    if node.attr == "environ":
                        method = parents.get(node)
                        call = parents.get(method)
                        allowed = {"MLTOOLS_INPUT": "/kaggle/input", "MLTOOLS_OUTPUT": "/kaggle/working"}
                        require(
                            isinstance(node.value, ast.Name)
                            and node.value.id == "os"
                            and isinstance(method, ast.Attribute)
                            and method.attr == "get"
                            and isinstance(call, ast.Call)
                            and call.func is method
                            and not call.keywords
                            and len(call.args) == 2
                            and all(isinstance(a, ast.Constant) for a in call.args)
                            and isinstance(call.args[0].value, str)
                            and call.args[0].value in allowed
                            and call.args[1].value == allowed[call.args[0].value],
                            "only two exact local input/output environment overrides",
                        )
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    require(
                        not re.search(r"(?i)(?:https?://|\b(?:curl|wget|pip install|uv sync)\b)", node.value),
                        "no network/install notebook commands",
                    )
    require(payload_sha in "\n".join(sources), "notebook pins exact payload SHA256")


def validate(root, config):
    """Pure read-only validation. No CLI discovery, imports, auth, or writes."""
    require(isinstance(config, dict), "configuration object")
    required_config = {
        "schema_version",
        "owner",
        "dataset_slug",
        "notebook_slug",
        "license",
        "license_approved",
    }
    require(
        required_config <= set(config) <= required_config | {"dataset_title", "notebook_title"},
        "exact publication configuration keys",
    )
    require(
        config.get("dataset_title", DATASET_TITLE) == DATASET_TITLE
        and config.get("notebook_title", TITLE) == TITLE,
        "fixed configured dataset/notebook titles",
    )
    require(type(config["schema_version"]) is int and config["schema_version"] == 1, "configuration version")
    require(
        config["owner"] == OWNER and config["dataset_slug"] == config["notebook_slug"] == SLUG,
        "approved owner and slugs",
    )
    require(type(config["license_approved"]) is bool, "explicit license approval")
    require(
        config["license"] is None or isinstance(config["license"], str) and bool(config["license"]),
        "license string or pending null",
    )
    require(not config["license_approved"] or config["license"] is not None, "approval names a license")
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "release root directory")
    require({p.name for p in root.iterdir()} == {"dataset", "notebook"}, "closed release root")
    closed(root / "dataset", DATASET_FILES)
    closed(root / "notebook", NOTEBOOK_FILES)
    snapshots = {}
    for folder, names in (("dataset", DATASET_FILES), ("notebook", NOTEBOOK_FILES)):
        for name in sorted(names):
            raw = read(root / folder / name, MAX_PAYLOAD if name == "payload.zip" else MAX_FILE)
            if name != "payload.zip":
                privacy(raw, folder + "/" + name)
            snapshots[folder + "/" + name] = raw
    manifest = parse(snapshots["dataset/release-manifest.json"])
    require(
        type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1, "manifest version"
    )
    require(
        isinstance(manifest.get("release_id"), str)
        and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", manifest["release_id"]),
        "release identity",
    )
    expected = hashes(manifest.get("files"), "dataset file hashes")
    require(set(expected) == DATASET_FILES - {"release-manifest.json"}, "manifest covers every dataset file")
    require(
        all(digest(snapshots["dataset/" + n]) == sha for n, sha in expected.items()), "dataset byte hashes"
    )
    payload_sha = digest(snapshots["dataset/payload.zip"])
    require(manifest.get("payload_sha256") == payload_sha, "manifest payload SHA256")
    replay = hashes(manifest.get("expected_replay_outputs"), "expected replay outputs")
    require(set(replay) == SEMANTIC_OUTPUTS, "five expected semantic output hashes")
    require(replay["conditions.csv"] == expected["conditions.csv"], "expected conditions are published CSV")
    metadata = parse(snapshots["dataset/dataset-metadata.json"])
    require(
        metadata.get("id") == IDENTITY and "id_no" not in metadata, "dataset identity cannot be overridden"
    )
    require(
        set(metadata)
        <= {
            "id",
            "title",
            "subtitle",
            "description",
            "licenses",
            "resources",
            "keywords",
            "userSpecifiedSources",
            "expectedUpdateFrequency",
        },
        "known dataset metadata fields",
    )
    require(isinstance(metadata.get("title"), str) and 6 <= len(metadata["title"]) <= 50, "dataset title")
    if "dataset_title" in config:
        require(metadata["title"] == config["dataset_title"], "configured dataset title")
    licenses = metadata.get("licenses")
    require(
        isinstance(licenses, list)
        and len(licenses) == 1
        and isinstance(licenses[0], dict)
        and set(licenses[0]) == {"name"},
        "one explicit dataset license",
    )
    if config["license"] is not None:
        require(licenses[0]["name"] == config["license"], "dataset license matches approved configuration")
    kernel = parse(snapshots["notebook/kernel-metadata.json"])
    require(
        set(kernel)
        <= {
            "id",
            "title",
            "code_file",
            "language",
            "kernel_type",
            "is_private",
            "enable_gpu",
            "enable_internet",
            "machine_shape",
            "dataset_sources",
            "competition_sources",
            "kernel_sources",
            "model_sources",
        },
        "known kernel fields; no numeric ID override",
    )
    require(kernel.get("id") == IDENTITY and kernel.get("title") == TITLE, "notebook identity/title")
    require(
        kernel.get("code_file") == "replay.ipynb"
        and kernel.get("language") == "python"
        and kernel.get("kernel_type") == "notebook",
        "notebook code path/type",
    )
    require(
        boolean(kernel.get("is_private"), True)
        and boolean(kernel.get("enable_gpu"), False)
        and boolean(kernel.get("enable_internet"), False),
        "private CPU offline notebook",
    )
    require(kernel.get("dataset_sources") == [IDENTITY], "exact dataset attachment")
    require(
        all(kernel.get(k, []) == [] for k in ("competition_sources", "kernel_sources", "model_sources"))
        and kernel.get("machine_shape", "") == "",
        "no extra input sources or accelerators",
    )
    check_notebook(snapshots["notebook/replay.ipynb"], payload_sha)
    members = check_payload(root / "dataset/payload.zip")
    pending = config["license"] is None or not config["license_approved"]
    identity = {n: digest(raw) for n, raw in sorted(snapshots.items())}
    return {
        "schema_version": 1,
        "release_id": manifest["release_id"],
        "payload_sha256": payload_sha,
        "build_sha256": digest(encoded(identity)),
        "file_sha256": identity,
        "payload_members": members,
        "ready": not pending,
        "blockers": ["License approval is pending"] if pending else [],
        "expected_replay_outputs": replay,
    }


def versioned(version):
    require(type(version) is int and version > 0, "explicit positive notebook version")
    return IDENTITY + "/" + str(version)


def command(step, root, output=None, notebook_version=None):
    root = Path(root).resolve()
    base = [sys.executable, "-I", "-c", "from kaggle.cli import main; main()"]
    if step == "dataset-create":
        tail = ["datasets", "create", "--path", str(root / "dataset"), "--keep-tabular", "--dir-mode", "skip"]
    elif step == "dataset-status":
        tail = ["datasets", "status", IDENTITY]
    elif step == "notebook-push":
        tail = ["kernels", "push", "--path", str(root / "notebook")]
    elif step == "notebook-status":
        tail = ["kernels", "status", versioned(notebook_version)]
    elif step == "output-pull":
        require(output is not None, "new output directory required")
        tail = ["kernels", "output", versioned(notebook_version), "--path", str(Path(output).resolve())]
    else:
        raise ValueError("Kaggle publication: no external command for " + step)
    return base + tail


def plan(root, config, step="plan", output=None, notebook_version=None):
    require(step in STEPS, "known explicit publication step")
    result = validate(root, config)
    result.update(
        {
            "step": step,
            "cli_version": CLI_VERSION,
            "cli_revision": CLI_REVISION,
            "executes": False,
            "privacy": "private",
            "credentials_accessed": False,
        }
    )
    steps = (
        ("dataset-create", "dataset-status", "notebook-push") if step in ("plan", "public-plan") else (step,)
    )
    result["commands"] = [command(s, root, output, notebook_version) for s in steps if s in REMOTE_STEPS]
    if step == "public-plan":
        require(output is not None, "verified hosted outputs required before a public review plan")
        result["hosted_output_verification"] = verify_outputs(root, config, output)
        result["commands"] = []
        result["next_action"] = (
            "Review hosted outputs, then obtain separate approval and change visibility manually in Kaggle UI"
        )
    return result


def verify_outputs(root, config, output):
    release = validate(root, config)
    folder = Path(output)
    require(folder.is_dir() and not folder.is_symlink(), "hosted output directory")
    report = parse(read(folder / "replay-report.json"))
    require(
        type(report.get("schema_version")) is int and report["schema_version"] == 1, "hosted report version"
    )
    require(
        report.get("status") == "VERIFIED_SAVED_TRACE_REPLAY"
        and report.get("release_id") == release["release_id"]
        and report.get("payload_sha256") == release["payload_sha256"],
        "hosted report release/status binding",
    )
    files = hashes(report.get("files"), "hosted output files")
    require(REQUIRED_OUTPUTS <= set(files) and "replay-report.json" not in files, "required hosted outputs")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    require(actual == {"replay-report.json", *files}, "closed hosted output inventory")
    require(not any(p.is_symlink() for p in folder.rglob("*")), "no hosted output links")
    for name, sha in files.items():
        raw = read(folder / safe_name(name))
        privacy(raw, name)
        require(digest(raw) == sha, "hosted output hash: " + name)
    require(
        all(files[n] == sha for n, sha in release["expected_replay_outputs"].items()),
        "hosted semantic bytes match predeclared release outputs",
    )
    return {
        "schema_version": 1,
        "status": "VERIFIED_SAVED_TRACE_REPLAY",
        "release_id": release["release_id"],
        "payload_sha256": release["payload_sha256"],
        "report_sha256": digest(read(folder / "replay-report.json")),
        "scope": "Saved-trace replay outputs; no fresh native acquisition or execution attestation",
    }


def exclusive(path, value):
    raw = encoded(value)
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def previous(state, step):
    base = state / step
    if not base.exists():
        return []
    require(base.is_dir() and not base.is_symlink(), "regular attempt-state directory")
    entries = sorted(base.iterdir())
    require(
        all(re.fullmatch(r"[0-9]{4}", p.name) and p.is_dir() and not p.is_symlink() for p in entries),
        "closed attempt-state inventory",
    )
    require(
        [p.name for p in entries] == [f"{i:04d}" for i in range(1, len(entries) + 1)],
        "contiguous preserved attempts",
    )
    return entries


def authorize(text):
    require(isinstance(text, str) and len(text.strip()) >= 12, "explicit authorization text required")
    return digest(text.encode())


def installed_version():
    return importlib.metadata.version("kaggle")


def run_cli(argv):
    """Only called from the explicit execution path; no shell or package import here."""
    return subprocess.run(argv, capture_output=True, timeout=600, check=False)


def execute(
    root,
    config,
    step,
    state,
    authorization,
    *,
    output=None,
    notebook_version=None,
    runner=None,
    version_getter=None,
):
    require(step in REMOTE_STEPS, "only an explicit private remote step can execute")
    approved = authorize(authorization)
    release = validate(root, config)
    require(release["ready"], "license approval required before external operations")
    root, state = Path(root).resolve(), Path(state).resolve()
    require(not state.is_relative_to(root), "private attempt state must stay outside upload root")
    if output is not None:
        output = Path(output).resolve()
        require(
            not output.is_relative_to(root) and not output.is_relative_to(state),
            "output separate from upload/state",
        )
    if step == "output-pull":
        require(output is not None and not output.exists(), "output download needs a fresh directory")
    # Checking installed metadata does not import/authenticate Kaggle. The actual
    # import occurs only in the authorized child command, using this interpreter.
    actual_version = (version_getter or installed_version)()
    require(actual_version == CLI_VERSION, "exact pinned Kaggle CLI " + CLI_VERSION)
    old = previous(state, step)
    if step in MUTATIONS and old:
        resolution = old[-1] / "reconciliation.json"
        decision = parse(read(resolution)) if resolution.is_file() else {}
        claim_path = old[-1] / "claim.json"
        require(
            decision.get("outcome") == "CONFIRMED_NO_REMOTE_MUTATION"
            and type(decision.get("schema_version")) is int
            and decision["schema_version"] == 1
            and decision.get("step") == step
            and claim_path.is_file()
            and decision.get("claim_sha256") == digest(read(claim_path))
            and isinstance(decision.get("evidence_sha256"), str)
            and SHA256.fullmatch(decision["evidence_sha256"])
            and isinstance(decision.get("authorization_sha256"), str)
            and SHA256.fullmatch(decision["authorization_sha256"]),
            "prior mutation cannot be retried without explicit no-mutation reconciliation",
        )
    base = state / step
    base.mkdir(parents=True, exist_ok=True)
    attempt = base / f"{len(old) + 1:04d}"
    attempt.mkdir(exist_ok=False)
    staged = attempt / "staged"
    argv = command(step, staged if step in MUTATIONS else root, output, notebook_version)
    exclusive(
        attempt / "claim.json",
        {
            "schema_version": 1,
            "status": "STARTED",
            "step": step,
            "release_id": release["release_id"],
            "payload_sha256": release["payload_sha256"],
            "build_sha256": release["build_sha256"],
            "authorization_sha256": approved,
            "cli_version": actual_version,
            "cli_revision": CLI_REVISION,
            "command": argv,
            "started_utc": datetime.now(UTC).isoformat(),
        },
    )
    try:
        if step in MUTATIONS:
            for relative, sha in release["file_sha256"].items():
                raw = read(root / relative, MAX_PAYLOAD if relative.endswith("payload.zip") else MAX_FILE)
                require(digest(raw) == sha, "stable source before staging")
                path = staged / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as stream:
                    stream.write(raw)
            require(
                validate(staged, config)["build_sha256"] == release["build_sha256"],
                "validated upload snapshot",
            )
        result = (runner or run_cli)(argv)
        stdout = result.stdout.encode() if isinstance(result.stdout, str) else result.stdout or b""
        stderr = result.stderr.encode() if isinstance(result.stderr, str) else result.stderr or b""
        (attempt / "stdout.txt").write_bytes(stdout)
        (attempt / "stderr.txt").write_bytes(stderr)
        status = (
            "SUBMITTED"
            if result.returncode == 0 and step in MUTATIONS
            else "COMPLETED"
            if result.returncode == 0
            else "UNKNOWN"
        )
        receipt = {
            "schema_version": 1,
            "status": status,
            "step": step,
            "returncode": result.returncode,
            "stdout_sha256": digest(stdout),
            "stderr_sha256": digest(stderr),
            "scope": "CLI result only; inspect native status and verify hosted outputs separately",
        }
        exclusive(attempt / "result.json", receipt)
        return receipt
    except BaseException as error:
        if isinstance(error, subprocess.TimeoutExpired):
            for name, data in (("stdout.txt", error.stdout), ("stderr.txt", error.stderr)):
                (attempt / name).write_bytes(data.encode() if isinstance(data, str) else data or b"")
        if not (attempt / "result.json").exists():
            exclusive(
                attempt / "result.json",
                {"schema_version": 1, "status": "UNKNOWN", "step": step, "error_type": type(error).__name__},
            )
        raise


def reconcile(root, config, state, target, outcome, evidence, authorization):
    release = validate(root, config)
    approved = authorize(authorization)
    require(
        target in MUTATIONS and outcome in ("CONFIRMED_REMOTE_MUTATION", "CONFIRMED_NO_REMOTE_MUTATION"),
        "explicit mutation reconciliation",
    )
    state = Path(state).resolve()
    require(not state.is_relative_to(Path(root).resolve()), "state outside release")
    old = previous(state, target)
    require(old, "existing mutation attempt required")
    claim = parse(read(old[-1] / "claim.json"))
    require(claim["build_sha256"] == release["build_sha256"], "reconcile the exact attempted build")
    evidence = Path(evidence)
    safe_name(evidence.name)
    raw = read(evidence, 1024 * 1024)
    require(len(raw.strip()) >= 20, "substantive externally checked reconciliation evidence")
    record = {
        "schema_version": 1,
        "outcome": outcome,
        "step": target,
        "claim_sha256": digest(read(old[-1] / "claim.json")),
        "evidence_sha256": digest(raw),
        "authorization_sha256": approved,
        "scope": "Human-supplied reconciliation; a nonzero exit alone never proves no remote mutation",
    }
    exclusive(old[-1] / "reconciliation.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--step", choices=sorted(STEPS), default="plan")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization")
    parser.add_argument("--state", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notebook-version", type=int)
    parser.add_argument("--target", choices=sorted(MUTATIONS))
    parser.add_argument("--outcome", choices=("CONFIRMED_REMOTE_MUTATION", "CONFIRMED_NO_REMOTE_MUTATION"))
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    safe_name(args.config.name)
    config = parse(read(args.config))
    if args.step == "output-verify":
        require(not args.execute and args.output is not None, "output verification is local/read-only")
        result = verify_outputs(args.root, config, args.output)
    elif args.step == "reconcile" and args.execute:
        require(
            args.state is not None and args.evidence is not None, "reconciliation state/evidence required"
        )
        result = reconcile(
            args.root, config, args.state, args.target, args.outcome, args.evidence, args.authorization
        )
    elif args.execute:
        require(args.state is not None, "private attempt state directory required")
        result = execute(
            args.root,
            config,
            args.step,
            args.state,
            args.authorization,
            output=args.output,
            notebook_version=args.notebook_version,
        )
    else:
        result = plan(args.root, config, args.step, args.output, args.notebook_version)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
