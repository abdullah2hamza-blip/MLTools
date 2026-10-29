"""Package a clean tagged research revision beside an already validated replay."""

import argparse
import importlib.util
import io
import json
import re
import subprocess
import zipfile
from pathlib import Path

from publication_guard import (
    FREEZE_COMMIT,
    PROFILE,
    check_values,
    encoded,
    parse,
    require,
    safe_name,
    sha,
    validate,
    verify_frozen_closure,
)

ROOT = Path(__file__).resolve().parents[2]
PAPER = "research/publication/paper/"
STYLES = {PAPER + x for x in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty")}
ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    ".python-version",
    "README.md",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "pyproject.toml",
    "uv.lock",
    "CONTRIBUTING.md",
    "SECURITY.md",
}
PREFIXES = (
    ".github/",
    "docs/",
    "examples/",
    "integrations/",
    "research/",
    "scripts/",
    "src/",
    "tests/",
    "validation/",
    "releases/",
)
PUBLIC_REPOSITORY = "https://github.com/abdullah2hamza-blip/MLTools"


def git(root, *args, input_data=None):
    return subprocess.check_output(["git", *args], cwd=root, input=input_data)


def clean_tag(root, ref):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", ref) is not None, "explicit safe tag name")
    commit = git(root, "rev-parse", "--verify", "refs/tags/" + ref + "^{commit}").decode().strip()
    require(commit == git(root, "rev-parse", "HEAD").decode().strip(), "check out the tagged revision")
    require(not git(root, "status", "--porcelain", "--untracked-files=no").strip(), "clean tracked checkout")
    subprocess.run(["git", "merge-base", "--is-ancestor", FREEZE_COMMIT, commit], cwd=root, check=True)
    return commit


def revision_files(root, ref):
    entries = {}
    for row in git(root, "ls-tree", "-rz", "--full-tree", ref).split(b"\0"):
        if not row:
            continue
        header, name = row.split(b"\t", 1)
        mode, kind, blob = header.decode().split()
        require(mode in ("100644", "100755") and kind == "blob", "regular tracked files only")
        path = name.decode("utf-8")
        safe_name(path)
        entries[path] = blob
    ids = sorted(set(entries.values()))
    stream = io.BytesIO(git(root, "cat-file", "--batch", input_data=("\n".join(ids) + "\n").encode()))
    blobs = {}
    for expected in ids:
        blob, kind, count = stream.readline().decode().split()
        require(blob == expected and kind == "blob", "complete Git blob identity")
        blobs[blob] = stream.read(int(count))
        require(stream.read(1) == b"\n", "Git batch framing")
    return {name: blobs[blob] for name, blob in sorted(entries.items())}


def omission(path):
    if path in STYLES:
        return "External unmodified TeX style; exact pinned acquisition instructions supplied"
    if path.startswith("releases/kaggle/artifacts/") or path.startswith("releases/publication/artifacts/"):
        return "Historical release binaries/archives omitted to avoid recursive copies"
    if path.startswith(PAPER + "output/") and path != PAPER + "output/main.pdf":
        return "Intermediate/historical paper build output"
    if path.startswith("releases/kaggle/validation/") and path.endswith(".ipynb"):
        return "Historical executed release notebook; bounded proofs retained"
    return None


def source_snapshot(all_files):
    selected, omitted = {}, {}
    for name, data in all_files.items():
        require(name in ROOT_FILES or name.startswith(PREFIXES), "review an unknown source root: " + name)
        reason = omission(name)
        if reason:
            omitted[name] = reason
            continue
        check_values(data, name)
        selected[name] = data
    verify_frozen_closure(selected)
    return selected, omitted


def archive(files):
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as result:
        for name, raw in sorted(files.items()):
            safe_name(name)
            check_values(raw, name)
            item = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            item.compress_type = zipfile.ZIP_DEFLATED
            result.writestr(item, raw)
    return data.getvalue()


def source_notes(ref, commit, format_record):
    rows = [
        "MLTools public research source snapshot",
        "Repository: " + PUBLIC_REPOSITORY,
        "Tag: " + ref,
        "Source commit: " + commit,
        "Original frozen protocol: " + FREEZE_COMMIT,
        "",
        "The snapshot retains every protected file in the 406-file frozen closure.",
        "It omits Git history; original strict verification requires a full clone:",
        "git clone " + PUBLIC_REPOSITORY + ".git",
        "cd MLTools",
        "git checkout --detach " + commit,
        "Follow research/publication/reproduction/README.md from that checkout.",
        "The Kaggle notebook reanalyzes saved traces, not fresh framework acquisition.",
        "",
        "Unmodified TeX styles are omitted from both source ZIPs. Obtain these exact",
        "files in the manuscript directory before compiling, preserving their notices:",
    ]
    for item in format_record["files"]:
        if item["path"] in ("tmlr.sty", "tmlr.bst", "fancyhdr.sty"):
            rows += [
                "https://raw.githubusercontent.com/JmlrOrg/tmlr-style-file/"
                + format_record["template_commit"]
                + "/"
                + item["upstream_path"],
                "SHA256 " + item["sha256"],
            ]
    rows += [
        "",
        "Original authored code/results/manuscript/figures: Apache-2.0; root LICENSE/NOTICE.",
        "Pinned publisher metadata retains its declared Apache-2.0 cards/licenses.",
        "TMLR style: Apache-2.0. fancyhdr: LPPL1.3 or later. tmlr.bst: LPPL1 or later.",
        "The LPPL1.3c text is supplied; style headers and complete upstream work terms prevail.",
        "https://www.latex-project.org/lppl/lppl-1-3c/",
        "The included tiny fixture_weight tensors are authored identity controls, not learned weights.",
        "No learned weights, recorded robot episodes, external dependency packages or compiler binaries.",
        "Omitted files and exact ZIP member checksums are listed in public-release-manifest.json.",
        "Kaggle may expand the ZIPs; verify the same member inventories and hashes in that form.",
    ]
    return ("\n".join(rows) + "\n").encode()


def supplements(all_files, ref, commit, replay):
    selected, omitted = source_snapshot(all_files)
    format_record = parse(all_files[PAPER + "venue-format.json"])
    notes = source_notes(ref, commit, format_record)
    license_path = "releases/publication/LICENSE.LPPL-1.3c.txt"
    require(license_path in all_files, "tag includes the official LPPL license copy")
    selected["PUBLIC-SOURCE-NOTES.txt"] = notes
    paper = {
        name[len(PAPER) :]: raw
        for name, raw in selected.items()
        if name.startswith(PAPER) and not name.startswith(PAPER + "output/")
    }
    paper["PUBLIC-SOURCE-NOTES.txt"] = notes
    paper["LICENSE.LPPL-1.3c.txt"] = all_files[license_path]
    for name in ("LICENSE", "NOTICE"):
        paper[name] = all_files[name]
    pdf = all_files[PAPER + "output/main.pdf"]
    latest = parse(all_files[PAPER + "output/latest-build.json"])
    require(sha(pdf) == latest["pdf_sha256"], "latest compiled PDF bytes")
    compile_raw = all_files[latest["compile_record"]]
    require(sha(compile_raw) == latest["compile_record_sha256"], "compile record bytes")
    compiled = parse(compile_raw)
    require(
        compiled["returncode"] == 0 and compiled["source_stable_during_compilation"],
        "successful stable PDF build",
    )
    for name, digest in compiled["source_file_sha256"].items():
        require(sha(all_files[PAPER + name]) == digest, "manuscript changed since compile: " + name)
    files = {
        "manuscript.pdf": pdf,
        "manuscript-source.zip": archive(paper),
        "research-source.zip": archive(selected),
    }
    manifest = {
        "schema_version": 1,
        "profile": PROFILE,
        "source_tag": ref,
        "source_commit": commit,
        "freeze_commit": FREEZE_COMMIT,
        "original_research_commit": "86e35c0fb4cc7c7f0e363d1e250a281f673d72ce",
        "repository": PUBLIC_REPOSITORY,
        "files": {name: sha(raw) for name, raw in files.items()},
        "zip_members": {
            name: {p: sha(raw) for p, raw in sorted(items.items())}
            for name, items in (("manuscript-source.zip", paper), ("research-source.zip", selected))
        },
        "omitted_git_files": omitted,
        "replay": {
            name: replay[name] for name in ("release_id", "payload_sha256", "expected_replay_outputs")
        },
        "frozen_protected_file_count": 406,
        "verification_scope": (
            "Source/package byte verification; full original Git gate requires public clone/ancestry. "
            "Notebook is saved-trace replay."
        ),
        "rights": (
            "Original Apache-2.0 content; retained publisher licenses. TeX styles use exact acquisition "
            "and distinct LPPL terms; see PUBLIC-SOURCE-NOTES.txt."
        ),
    }
    files["public-release-manifest.json"] = encoded(manifest)
    return files


def publisher(root):
    path = root / "releases/kaggle/publish.py"
    spec = importlib.util.spec_from_file_location("mltools_publication_publisher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(root, ref, replay_root, output, config):
    root, replay_root, output = Path(root).resolve(), Path(replay_root).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(replay_root), "new separate output root")
    commit = clean_tag(root, ref)
    publish = publisher(root)
    review = publish.validate(replay_root, config)
    require("public_supplement" not in review, "start from a focused replay, not a recursive supplement")
    all_files = revision_files(root, commit)
    additions = supplements(all_files, ref, commit, review)
    snapshots = {name: (replay_root / name).read_bytes() for name in review["file_sha256"]}
    for name, raw in additions.items():
        snapshots["dataset/" + name] = raw
    extra_text = (
        "\n\n## Complete research supplement\n\n"
        "The accompanying manuscript.pdf is an unreviewed research preprint. "
        "manuscript-source.zip contains its editable sources; research-source.zip contains the tagged "
        "source, frozen inputs, saved results and reproduction tools. public-release-manifest.json "
        "lists all files, exclusions, licenses, source commit and ZIP member hashes. "
        "Original Git-ancestry verification requires a full public clone. "
        "Repository: " + PUBLIC_REPOSITORY + "/tree/" + commit + ".\n"
    )
    card = snapshots["dataset/README.md"].decode() + extra_text
    snapshots["dataset/README.md"] = card.encode()
    metadata = parse(snapshots["dataset/dataset-metadata.json"])
    metadata["description"] = card
    snapshots["dataset/dataset-metadata.json"] = encoded(metadata)
    release = parse(snapshots["dataset/release-manifest.json"])
    release["distribution_profile"] = PROFILE
    release["files"] = {
        name.removeprefix("dataset/"): sha(raw)
        for name, raw in sorted(snapshots.items())
        if name.startswith("dataset/") and name != "dataset/release-manifest.json"
    }
    snapshots["dataset/release-manifest.json"] = encoded(release)
    validate(snapshots, release)
    for name, raw in snapshots.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    result = publish.validate(output, config)
    require(clean_tag(root, ref) == commit, "tagged commit and inputs stayed unchanged")
    return {
        "schema_version": 1,
        "status": "COMPLETE_RESEARCH_PACKAGE_BUILT",
        "source_tag": ref,
        "source_commit": commit,
        "release_id": review["release_id"],
        "payload_sha256": review["payload_sha256"],
        "files": result["file_sha256"],
        "supplement": result["public_supplement"],
        "credentials_used": False,
        "remote_mutations": False,
        "hosted_validation": "NOT_RUN_BY_PACKAGING",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Existing checked-out clean tag")
    parser.add_argument("--replay-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--record", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "releases/kaggle/config.json")
    args = parser.parse_args()
    require(not args.record.resolve().is_relative_to(args.output.resolve()), "lock outside uploaded root")
    require(not args.record.exists(), "new lock record")
    result = build(ROOT, args.ref, args.replay_root, args.output, parse(args.config.read_bytes()))
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_bytes(encoded(result))
    print(json.dumps(result, indent=2))
