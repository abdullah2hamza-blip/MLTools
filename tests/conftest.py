import shutil
from pathlib import Path

import pytest

from mltools.io import read_json, write_json

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "so101"


@pytest.fixture
def sample(tmp_path):
    shutil.copytree(EXAMPLES / "artifacts", tmp_path / "artifacts")
    bundle = read_json(EXAMPLES / "good.json")
    path = tmp_path / "bundle.json"
    write_json(path, bundle)
    trace = read_json(EXAMPLES / "good.trace.json")
    return path, bundle, trace
