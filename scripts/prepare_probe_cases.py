"""Create development-only versioned probe cases, including published ACT statistics."""

from pathlib import Path

from prepare_act_candidates import FILES

from mltools.io import digest, read_json, write_json
from mltools.obligations import Axis, Reference, compile_suite

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples/probes"


def case(suite, reference, adapter, artifacts, description, checks=None):
    return {
        "schema_version": 1,
        "format": "mltools-affine-case",
        "development_only": True,
        "suite": suite.to_dict(),
        "reference": reference.to_dict(),
        "adapter": adapter,
        "artifacts": artifacts,
        "description": description,
        **({"checks": checks} if checks else {}),
    }


def prepare_lerobot():
    folder = OUT / "lerobot"
    folder.mkdir(parents=True, exist_ok=True)
    stats = folder / "stats.json"
    write_json(stats, {"mean": [1, 2, 3, 4, 5, 15], "std": [2, 3, 4, 5, 6, 4]})
    labels = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    axes = tuple(
        Axis(
            port,
            str(i),
            "normalized" if port == "action" else "range_0_100" if i == 5 else "degree",
            -1 if port == "action" else 0 if i == 5 else -10,
            1 if port == "action" else 100 if i == 5 else 10,
        )
        for port in ("action", "origin", "current")
        for i in range(6)
    )
    suite = compile_suite(axes, tuple(f"{name}.pos" for name in labels))
    scales = [2, 3, 4, 5, 6, 4]
    ref = Reference(
        tuple(
            tuple(
                scales[i]
                if a.port == "action" and a.name == str(i)
                else 10
                if a.port == "origin" and a.name == str(i) and i < 5
                else 0
                for a in axes
            )
            for i in range(6)
        ),
        (1, 2, 3, 4, 5, 15),
        ("degree",) * 5 + ("range_0_100",),
        "Independently authored SO101 calibration fixture, not a trained checkpoint",
        True,
    )
    for name, bind in (("good", True), ("moving_anchor", False)):
        adapter = {
            "framework": "lerobot",
            "stats_artifact": "stats",
            "configuration": {
                "relative": True,
                "bind_queue": bind,
                "exclude_joints": ["gripper"],
                "action_names": labels,
                "motor_names": labels,
            },
        }
        write_json(
            folder / f"{name}.json",
            case(
                suite,
                ref,
                adapter,
                [{"id": "stats", "path": "stats.json", "sha256": digest(stats)}],
                "Synthetic calibration; explicit queue-binding intervention",
            ),
        )


def prepare_openpi():
    folder = OUT / "openpi"
    folder.mkdir(parents=True, exist_ok=True)
    prior = ROOT / "research/publication/development/openpi-001"
    from mltools.obligations import Suite

    for name, original in (("good", "pi0"), ("moving_anchor", "moving_anchor")):
        source = read_json(prior / f"{original}.case.json")
        stats = folder / "stats.json"
        stats.write_text(source["stats_json"], encoding="utf-8", newline="")
        adapter = {
            "framework": "openpi",
            "stats_artifact": "stats",
            "configuration": {
                "model_type": "pi0",
                "use_quantiles": False,
                "mask": [True] * 6 + [False],
                "anchor": "origin" if name == "good" else "current",
            },
        }
        write_json(
            folder / f"{name}.json",
            case(
                Suite.from_dict(source["suite"]),
                Reference(**source["reference"]),
                adapter,
                [{"id": "stats", "path": "stats.json", "sha256": digest(stats)}],
                "Existing synthetic OpenPI development lineage; no model weights or physical training data",
            ),
        )


def prepare_act():
    # Move-cube shares the previously evaluated SmolVLA development lineage.
    # Boccia remains an unexecuted selection candidate, not development tuning data.
    source = ROOT / "research/publication/selection/act/act-so101-move-cube"
    manifest = read_json(source / "manifest.json")
    folder = OUT / "act"
    folder.mkdir(parents=True, exist_ok=True)
    files = [f for f in manifest["files"] if f["path"] in FILES]
    artifacts = []
    for index, item in enumerate(files):
        src, dest = source / item["path"], folder / item["path"]
        if digest(src) != item["sha256"]:
            raise ValueError("Pinned ACT selection metadata changed")
        dest.write_bytes(src.read_bytes())
        artifacts.append({"id": f"checkpoint_{index}", "path": item["path"], "sha256": item["sha256"]})
    (folder / "LICENSE-2.0.txt").write_bytes((source / "LICENSE-2.0.txt").read_bytes())
    artifacts.append(
        {"id": "checkpoint_license", "path": "LICENSE-2.0.txt", "sha256": digest(folder / "LICENSE-2.0.txt")}
    )
    processor = read_json(source / "policy_postprocessor.json")
    if [step["registry_name"] for step in processor["steps"]] != [
        "unnormalizer_processor",
        "device_processor",
    ]:
        raise ValueError("Unreviewed ACT postprocessing composition")
    first = processor["steps"][0]
    if first["config"]["norm_map"]["ACTION"] != "MEAN_STD" or first["config"]["eps"] != 1e-8:
        raise ValueError("Unreviewed normalization convention")
    tensors = read_json(source / "decoded_statistics.json")["tensors"]
    mean, std = (tensors[f"action.{name}"]["values"] for name in ("mean", "std"))
    axes = tuple(
        Axis(
            port,
            str(i),
            "normalized_model_output" if port == "action" else "uninterpreted_state_coordinate",
            -1,
            1,
        )
        for port in ("action", "origin", "current")
        for i in range(6)
    )
    labels = [f"action_coordinate_{i}" for i in range(6)]
    suite = compile_suite(axes, tuple(f"{name}.pos" for name in labels))
    ref = Reference(
        tuple(
            tuple(std[i] if a.port == "action" and a.name == str(i) else 0 for a in axes) for i in range(6)
        ),
        tuple(mean),
        ("checkpoint_raw_action_coordinate",) * 6,
        f"Independent preprocessor statistics decoding; {manifest['model']}@{manifest['revision']}",
        True,
    )
    stats_id = next(a["id"] for a in artifacts if a["path"] == first["state_file"])
    adapter = {
        "framework": "lerobot",
        "stats_artifact": stats_id,
        "configuration": {
            "relative": False,
            "bind_queue": True,
            "exclude_joints": [],
            "action_names": labels,
            "motor_names": labels,
        },
    }
    description = (
        "Published ACT normalization in raw index coordinates; no ACT inference, native ACT queue, "
        "physical joint mapping, camera preprocessing, or attested training dataset snapshot"
    )
    write_json(
        folder / "metadata.json",
        case(suite, ref, adapter, artifacts, description, checks=["physical_semantics"]),
    )
    wrong = folder / "wrong_stats.json"
    write_json(wrong, {"mean": [x + 5 for x in mean], "std": std})
    faulty = dict(adapter, stats_artifact="injected_stats")
    write_json(
        folder / "wrong_stats.case.json",
        case(
            suite,
            ref,
            faulty,
            artifacts + [{"id": "injected_stats", "path": wrong.name, "sha256": digest(wrong)}],
            description + "; separately labeled injected normalization offset",
            checks=["physical_semantics"],
        ),
    )
    write_json(
        folder / "source_manifest.json",
        {
            "schema_version": 1,
            "model": manifest["model"],
            "revision": manifest["revision"],
            "license": manifest["license"],
            "split": "development_shared_move_cube_lineage",
            "files": files,
            "reference_loader": "bounded independent stdlib F32/F64 decoder",
            "runtime_loader": "native safetensors.torch for the published postprocessor file",
            "physical_semantics": "UNKNOWN",
            "weights_downloaded": False,
        },
    )


if __name__ == "__main__":
    prepare_lerobot()
    prepare_openpi()
    prepare_act()
    print("Prepared development cases; unexecuted ACT boccia selection candidate left untouched")
