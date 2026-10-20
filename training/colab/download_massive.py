"""Download the official MASSIVE 1.1 archive and copy only en-US and license files."""

import argparse
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

from sllm_research.io import sha256_file, write_json

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"


def download(output: Path) -> None:
    manifest = output / "download.json"
    if manifest.exists():
        for filename, checksum in json.loads(manifest.read_text())["files_sha256"].items():
            if sha256_file(output / filename) != checksum:
                raise ValueError(f"Previously downloaded file changed: {filename}")
        print(f"Verified existing download: {output}")
        return
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "amazon-massive-dataset-1.1.tar.gz"
    if not archive.exists():
        partial = archive.with_suffix(".gz.part")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        partial.replace(archive)
    selected = {}
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = Path(member.name)
            if member.isfile() and (
                name.name == "en-US.jsonl"
                or name.name.upper()
                in {"LICENSE", "LICENSE.TXT", "NOTICE", "NOTICE.MD", "THIRD-PARTY.MD"}
            ):
                destination = output / name.name
                with source.extractfile(member) as stream, destination.open("wb") as target:
                    shutil.copyfileobj(stream, target)
                selected[name.name] = sha256_file(destination)
    if "en-US.jsonl" not in selected:
        raise ValueError("Official archive did not contain en-US.jsonl")
    selected[archive.name] = sha256_file(archive)
    write_json(
        manifest, {"dataset": "MASSIVE", "release": "1.1", "url": URL, "files_sha256": selected}
    )
    print(f"Saved en-US, license files, archive, and checksums to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/massive-1.1"))
    download(parser.parse_args().output)
