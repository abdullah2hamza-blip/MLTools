"""Publisher controls use a stub runner exclusively; no Kaggle import/call."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "mltools_kaggle_publish", Path(__file__).with_name("publish.py")
)
publish = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publish)


class PublisherTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[2] / ".test-tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="kaggle-publish-", dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "release"
        self.dataset = self.root / "dataset"
        self.notebook = self.root / "notebook"
        self.dataset.mkdir(parents=True)
        self.notebook.mkdir()
        self.config = {
            "schema_version": 1,
            "owner": publish.OWNER,
            "dataset_slug": publish.SLUG,
            "notebook_slug": publish.SLUG,
            "license": "apache-2.0",
            "license_approved": True,
        }
        self.outputs = {
            name: ("case,status\ncontrol,PASS\n" if name == "conditions.csv" else "{}\n").encode()
            for name in publish.SEMANTIC_OUTPUTS
        }
        (self.dataset / "conditions.csv").write_bytes(self.outputs["conditions.csv"])
        (self.dataset / "README.md").write_text("Saved-trace replay. No native framework acquisition.\n")
        (self.dataset / "LICENSE").write_text("Apache License, Version 2.0\n")
        (self.dataset / "THIRD_PARTY_NOTICES.md").write_text(
            "Source metadata retains Apache-2.0 attribution.\n"
        )
        self.write_json(
            self.dataset / "dataset-metadata.json",
            {
                "id": publish.IDENTITY,
                "title": publish.TITLE,
                "licenses": [{"name": "apache-2.0"}],
            },
        )
        with zipfile.ZipFile(self.dataset / "payload.zip", "w") as archive:
            archive.writestr("release/source.py", "print('saved data only')\n")
        self.kernel = {
            "id": publish.IDENTITY,
            "title": publish.TITLE,
            "code_file": "replay.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_internet": False,
            "dataset_sources": [publish.IDENTITY],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
            "machine_shape": "",
        }
        self.write_json(self.notebook / "kernel-metadata.json", self.kernel)
        self.manifest()

    @staticmethod
    def write_json(path, value):
        path.write_bytes(publish.encoded(value))

    def manifest(self, source=None):
        payload = publish.digest((self.dataset / "payload.zip").read_bytes())
        self.write_json(
            self.notebook / "replay.ipynb",
            {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {},
                "cells": [
                    {
                        "cell_type": "markdown",
                        "metadata": {},
                        "source": "Saved-trace payload SHA256: " + payload,
                    },
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": source or "print('offline replay')",
                        "execution_count": None,
                        "outputs": [],
                    },
                ],
            },
        )
        self.write_json(
            self.dataset / "release-manifest.json",
            {
                "schema_version": 1,
                "release_id": "test-release-001",
                "payload_sha256": payload,
                "files": {
                    name: publish.digest((self.dataset / name).read_bytes())
                    for name in publish.DATASET_FILES - {"release-manifest.json"}
                },
                "expected_replay_outputs": {name: publish.digest(raw) for name, raw in self.outputs.items()},
            },
        )

    def stub(self, argv):
        self.calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=b"submitted\n", stderr=b"")

    def execute(self, step="dataset-create", runner=None, **kwargs):
        self.calls = []
        return publish.execute(
            self.root,
            self.config,
            step,
            self.base / "state",
            "Human approved private upload",
            runner=runner or self.stub,
            version_getter=lambda: publish.CLI_VERSION,
            **kwargs,
        )

    def hosted(self):
        output = self.base / "hosted"
        output.mkdir()
        for name, raw in {**self.outputs, "replay-summary.json": b"{}\n"}.items():
            (output / name).write_bytes(raw)
        self.write_json(
            output / "replay-report.json",
            {
                "schema_version": 1,
                "release_id": "test-release-001",
                "payload_sha256": publish.digest((self.dataset / "payload.zip").read_bytes()),
                "status": "VERIFIED_SAVED_TRACE_REPLAY",
                "files": {p.name: publish.digest(p.read_bytes()) for p in output.iterdir()},
            },
        )
        return output

    def test_default_plan_never_discovers_cli_or_writes(self):
        before = {p.relative_to(self.base): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        with (
            patch.object(publish, "installed_version", side_effect=AssertionError("no CLI metadata read")),
            patch.object(publish, "run_cli", side_effect=AssertionError("no external execution")),
        ):
            result = publish.plan(self.root, self.config)
        self.assertTrue(result["ready"])
        self.assertFalse(result["executes"])
        self.assertEqual(
            before, {p.relative_to(self.base): p.read_bytes() for p in self.base.rglob("*") if p.is_file()}
        )
        self.assertIn("--keep-tabular", result["commands"][0])
        self.assertNotIn("--public", str(result["commands"]))

    def test_pending_license_blocks_execution(self):
        self.config.update(license=None, license_approved=False)
        self.assertFalse(publish.plan(self.root, self.config)["ready"])
        with self.assertRaisesRegex(ValueError, "license approval"):
            self.execute()
        self.assertFalse((self.base / "state").exists())

    def test_license_not_approved(self):
        self.config["license_approved"] = False
        self.assertFalse(publish.validate(self.root, self.config)["ready"])

    def test_tampered_dataset_hash(self):
        (self.dataset / "conditions.csv").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "dataset byte hashes"):
            publish.plan(self.root, self.config)

    def test_closed_upload_inventory(self):
        (self.dataset / "extra.txt").write_text("not allowlisted")
        with self.assertRaisesRegex(ValueError, "closed inventory"):
            publish.plan(self.root, self.config)

    def test_rehashed_private_path_rejected(self):
        (self.dataset / "README.md").write_text(r"C:\Users\PrivatePerson\repo\trace")
        self.manifest()
        with self.assertRaisesRegex(ValueError, "privacy pattern"):
            publish.plan(self.root, self.config)

    def test_archive_privacy_and_unsafe_names_rejected(self):
        for name, body in (
            ("release/trace.json", r'{"prefix":"C:\\Users\\PrivatePerson\\repo"}'),
            ("../outside.py", "pass"),
            ("release/.kaggle/kaggle.json", "{}"),
        ):
            with self.subTest(name=name):
                with zipfile.ZipFile(self.dataset / "payload.zip", "w") as archive:
                    archive.writestr(name, body)
                self.manifest()
                with self.assertRaises(ValueError):
                    publish.plan(self.root, self.config)

    def test_rehashed_notebook_settings_fail_closed(self):
        for key, value in (
            ("is_private", False),
            ("enable_internet", True),
            ("enable_gpu", True),
            ("id_no", 123),
            ("dataset_sources", ["other/data"]),
            ("machine_shape", "NvidiaL4"),
        ):
            with self.subTest(key=key):
                bad = {**self.kernel, key: value}
                self.write_json(self.notebook / "kernel-metadata.json", bad)
                with self.assertRaises(ValueError):
                    publish.plan(self.root, self.config)
        self.write_json(self.notebook / "kernel-metadata.json", self.kernel)

    def test_no_credential_or_native_notebook_imports(self):
        for source in (
            "import kaggle",
            "from kaggle_secrets import UserSecretsClient",
            "import torch",
            "import os\nprint(os.environ)",
            "import os\nprint(os.getenv('SOMETHING'))",
        ):
            with self.subTest(source=source):
                self.manifest(source)
                with self.assertRaises(ValueError):
                    publish.plan(self.root, self.config)

    def test_only_exact_two_path_environment_overrides_allowed(self):
        self.manifest(
            "import os\n"
            "a = os.environ.get('MLTOOLS_INPUT', '/kaggle/input')\n"
            "b = os.environ.get('MLTOOLS_OUTPUT', '/kaggle/working')\n"
        )
        self.assertTrue(publish.validate(self.root, self.config)["ready"])
        for source in (
            "import os\na = os.environ.get('OTHER', '/kaggle/input')",
            "import os\na = os.environ.get('MLTOOLS_INPUT', '/tmp')",
            "import os\na = os.environ.get('MLTOOLS_INPUT')",
            "import os\na = os.environ['MLTOOLS_INPUT']",
            "import os\na = os.environ.copy()",
            "import os\na = os.environ.get('MLTOOLS_INPUT', default='/kaggle/input')",
        ):
            with self.subTest(source=source):
                self.manifest(source)
                with self.assertRaisesRegex(ValueError, "two exact"):
                    publish.validate(self.root, self.config)

    def test_public_upstream_home_path_is_preserved_but_researcher_path_rejected(self):
        for path, allowed in (("/home/upstream-publisher/training", True), ("/home/abdul/workspace", False)):
            with self.subTest(path=path):
                with zipfile.ZipFile(self.dataset / "payload.zip", "w") as archive:
                    archive.writestr("metadata/train_config.json", json.dumps({"root": path}))
                self.manifest()
                if allowed:
                    self.assertTrue(publish.validate(self.root, self.config)["ready"])
                else:
                    with self.assertRaisesRegex(ValueError, "privacy pattern"):
                        publish.validate(self.root, self.config)

    def test_optional_fixed_titles(self):
        config = {**self.config, "dataset_title": publish.DATASET_TITLE, "notebook_title": publish.TITLE}
        path = self.dataset / "dataset-metadata.json"
        metadata = json.loads(path.read_bytes())
        metadata["title"] = publish.DATASET_TITLE
        self.write_json(path, metadata)
        self.manifest()
        self.assertTrue(publish.validate(self.root, config)["ready"])
        with self.assertRaisesRegex(ValueError, "fixed configured"):
            publish.validate(self.root, {**config, "dataset_title": "different"})

    def test_payload_hash_must_be_in_notebook(self):
        path = self.notebook / "replay.ipynb"
        notebook = json.loads(path.read_bytes())
        notebook["cells"][0]["source"] = "unbound notebook"
        self.write_json(path, notebook)
        with self.assertRaisesRegex(ValueError, "pins exact payload"):
            publish.plan(self.root, self.config)

    def test_mutation_claim_exists_before_stub_and_upload_is_staged(self):
        def runner(argv):
            claim = self.base / "state/dataset-create/0001/claim.json"
            self.assertTrue(claim.is_file())
            self.assertEqual(json.loads(claim.read_bytes())["status"], "STARTED")
            staged = Path(argv[argv.index("--path") + 1])
            self.assertEqual(staged.name, "dataset")
            self.assertIn("staged", staged.parts)
            self.assertEqual(
                (staged / "payload.zip").read_bytes(), (self.dataset / "payload.zip").read_bytes()
            )
            return SimpleNamespace(returncode=0, stdout=b"submitted", stderr=b"")

        self.assertEqual(self.execute(runner=runner)["status"], "SUBMITTED")
        with self.assertRaisesRegex(ValueError, "prior mutation"):
            self.execute()

    def test_unknown_nonzero_exit_blocks_retry(self):
        result = self.execute(
            runner=lambda _: SimpleNamespace(returncode=1, stdout=b"uncertain", stderr=b"error")
        )
        self.assertEqual(result["status"], "UNKNOWN")
        with self.assertRaisesRegex(ValueError, "prior mutation"):
            self.execute()

    def test_timeout_preserves_partial_streams_and_blocks_retry(self):
        def runner(argv):
            raise subprocess.TimeoutExpired(argv, 600, output=b"partial", stderr=b"partial error")

        with self.assertRaises(subprocess.TimeoutExpired):
            self.execute(runner=runner)
        attempt = self.base / "state/dataset-create/0001"
        self.assertEqual((attempt / "stdout.txt").read_bytes(), b"partial")
        self.assertEqual(json.loads((attempt / "result.json").read_bytes())["status"], "UNKNOWN")
        with self.assertRaisesRegex(ValueError, "prior mutation"):
            self.execute()

    def test_incomplete_attempt_blocks_retry(self):
        attempt = self.base / "state/dataset-create/0001"
        attempt.mkdir(parents=True)
        (attempt / "claim.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "prior mutation"):
            self.execute()

    def test_explicit_reconciliation_preserves_attempt_and_allows_one_retry(self):
        self.execute(runner=lambda _: SimpleNamespace(returncode=1, stdout=b"", stderr=b"unknown"))
        evidence = self.base / "manual-evidence.txt"
        evidence.write_text("Human inspected the remote account and confirmed the dataset was not created.")
        publish.reconcile(
            self.root,
            self.config,
            self.base / "state",
            "dataset-create",
            "CONFIRMED_NO_REMOTE_MUTATION",
            evidence,
            "Human authorizes reconciliation",
        )
        self.execute()
        self.assertTrue((self.base / "state/dataset-create/0001/result.json").is_file())
        self.assertTrue((self.base / "state/dataset-create/0002/claim.json").is_file())
        with self.assertRaises(ValueError):
            self.execute()

    def test_confirmed_remote_mutation_never_allows_duplicate(self):
        self.execute()
        evidence = self.base / "manual-evidence.txt"
        evidence.write_text("Human verified that the exact dataset exists remotely and inspected its files.")
        publish.reconcile(
            self.root,
            self.config,
            self.base / "state",
            "dataset-create",
            "CONFIRMED_REMOTE_MUTATION",
            evidence,
            "Human authorizes reconciliation",
        )
        with self.assertRaisesRegex(ValueError, "prior mutation"):
            self.execute()

    def test_incomplete_reconciliation_cannot_unlock_retry(self):
        self.execute(runner=lambda _: SimpleNamespace(returncode=1, stdout=b"", stderr=b"unknown"))
        self.write_json(
            self.base / "state/dataset-create/0001/reconciliation.json",
            {"outcome": "CONFIRMED_NO_REMOTE_MUTATION"},
        )
        with self.assertRaisesRegex(ValueError, "prior mutation"):
            self.execute()

    def test_tampered_reconciled_claim_cannot_unlock_retry(self):
        self.execute(runner=lambda _: SimpleNamespace(returncode=1, stdout=b"", stderr=b"unknown"))
        evidence = self.base / "manual-evidence.txt"
        evidence.write_text("Human verified remote absence through independent account and file inspection.")
        publish.reconcile(
            self.root,
            self.config,
            self.base / "state",
            "dataset-create",
            "CONFIRMED_NO_REMOTE_MUTATION",
            evidence,
            "Human authorizes reconciliation",
        )
        path = self.base / "state/dataset-create/0001/claim.json"
        claim = json.loads(path.read_bytes())
        claim["build_sha256"] = "0" * 64
        self.write_json(path, claim)
        with self.assertRaisesRegex(ValueError, "prior mutation"):
            self.execute()

    def test_authorization_version_and_state_location_gates(self):
        with self.assertRaisesRegex(ValueError, "authorization"):
            publish.execute(self.root, self.config, "dataset-create", self.base / "state", "")
        with self.assertRaisesRegex(ValueError, "exact pinned"):
            publish.execute(
                self.root,
                self.config,
                "dataset-create",
                self.base / "state",
                "approved private operation",
                runner=self.stub,
                version_getter=lambda: "2.2.4",
            )
        with self.assertRaisesRegex(ValueError, "outside upload"):
            publish.execute(
                self.root, self.config, "dataset-create", self.root / "state", "approved private operation"
            )

    def test_status_commands_are_version_bound_and_never_public(self):
        self.assertEqual(publish.command("dataset-status", self.root)[-1], publish.IDENTITY)
        self.assertEqual(
            publish.command("notebook-status", self.root, notebook_version=4)[-1], publish.IDENTITY + "/4"
        )
        with self.assertRaisesRegex(ValueError, "positive notebook version"):
            publish.command("output-pull", self.root, self.base / "output")
        self.assertNotIn("--public", publish.command("notebook-push", self.root))

    def test_hosted_output_exact_replay(self):
        output = self.hosted()
        self.assertEqual(
            publish.verify_outputs(self.root, self.config, output)["status"], "VERIFIED_SAVED_TRACE_REPLAY"
        )
        public = publish.plan(self.root, self.config, "public-plan", output)
        self.assertEqual(public["commands"], [])
        with self.assertRaisesRegex(ValueError, "explicit private remote step"):
            self.execute("public-plan")

    def test_rehashed_wrong_hosted_output_is_rejected(self):
        output = self.hosted()
        (output / "analysis.json").write_text('{"fabricated":"different"}')
        report = json.loads((output / "replay-report.json").read_bytes())
        report["files"]["analysis.json"] = publish.digest((output / "analysis.json").read_bytes())
        self.write_json(output / "replay-report.json", report)
        with self.assertRaisesRegex(ValueError, "predeclared release"):
            publish.verify_outputs(self.root, self.config, output)

    def test_hosted_closed_inventory_and_report_binding(self):
        output = self.hosted()
        (output / "unreported.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "closed hosted"):
            publish.verify_outputs(self.root, self.config, output)
        (output / "unreported.txt").unlink()
        report = json.loads((output / "replay-report.json").read_bytes())
        report["payload_sha256"] = "0" * 64
        self.write_json(output / "replay-report.json", report)
        with self.assertRaisesRegex(ValueError, "release/status binding"):
            publish.verify_outputs(self.root, self.config, output)

    def test_config_and_metadata_owner_cannot_drift(self):
        for config in ({**self.config, "owner": "someoneelse"}, {**self.config, "extra": "surprise"}):
            with self.assertRaises(ValueError):
                publish.validate(self.root, config)
        path = self.dataset / "dataset-metadata.json"
        data = json.loads(path.read_bytes())
        data["id_no"] = 123
        self.write_json(path, data)
        self.manifest()
        with self.assertRaisesRegex(ValueError, "cannot be overridden"):
            publish.validate(self.root, self.config)

    def test_json_duplicate_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            publish.parse(b'{"id":1,"id":2}')


if __name__ == "__main__":
    unittest.main()
