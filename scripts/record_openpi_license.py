"""Record attribution fields omitted by the connector's comment view; no model downloads."""

import hashlib
import json
import urllib.request
from pathlib import Path

from mltools.io import write_json

URL = "https://api.github.com/repos/Physical-Intelligence/openpi/issues/comments/2638064959"
ROOT = Path(__file__).resolve().parents[1]


def main():
    request = urllib.request.Request(URL, headers={"Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read(65537)
    if len(raw) > 65536:
        raise ValueError("Comment metadata exceeded size bound")
    value = json.loads(raw)
    if value["id"] != 2638064959 or value["user"]["login"] != "uzhilinsky":
        raise ValueError("Unexpected license evidence identity")
    destination = ROOT / "research/publication/selection/openpi-license.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        destination,
        {
            "schema_version": 1,
            "source_api": URL,
            "source_response_sha256": hashlib.sha256(raw).hexdigest(),
            "comment_url": value["html_url"],
            "comment_id": value["id"],
            "body": value["body"],
            "created_at": value["created_at"],
            "updated_at": value["updated_at"],
            "author": value["user"]["login"],
            "author_association": value["author_association"],
            "repository_license": "Apache-2.0 at pinned OpenPI code revision",
            "claim": "Contributor statement about project licensing, answering a weight-license question",
            "limits": [
                "Not an independently verified OWNER/MEMBER authorization",
                "Statement predates the selected May/September 2025 LIBERO object generations",
                "No object-specific license notice found for the selected GCS statistics",
                "Do not silently promote this statement to generation-specific publisher attestation",
            ],
            "asset_redistribution_status": "not resolved by this evidence record",
        },
    )
    print(f"Recorded primary API attribution: {value['author_association']}, {value['created_at']}")


if __name__ == "__main__":
    main()
