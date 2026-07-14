"""Compile the pinned TrickTrack source harness and regenerate its JSON fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile


REFERENCE_DIRECTORY = Path(__file__).resolve().parent
ARCHIVE = REFERENCE_DIRECTORY / "TrickTrack-b164fad.tar.gz"
HARNESS = REFERENCE_DIRECTORY / "source_harness.cpp"
FIXTURE = (
    REFERENCE_DIRECTORY.parents[5]
    / "tests/fixtures/reconnect/cellular_automaton/tricktrack_source.json"
)
MANIFEST = FIXTURE.with_name("manifest.json")
EXPECTED_ARCHIVE_SHA256 = (
    "144e836fe1bc64fc97bf09b47a5b6dd60e31899f720d3b4d559931d36780fbe1"
)


def regenerate_tricktrack_fixture() -> None:
    """Build the C++14 source harness and atomically write canonical JSON."""
    observed_hash = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    if observed_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError("pinned TrickTrack archive checksum mismatch")

    with tempfile.TemporaryDirectory(prefix="tricktrack-a06-") as temporary:
        temporary_path = Path(temporary)
        with tarfile.open(ARCHIVE, "r:gz") as archive:
            archive.extractall(temporary_path, filter="data")
        source_root = temporary_path / "TrickTrack-b164fad"
        executable = temporary_path / "tricktrack-source-harness"
        subprocess.run(
            [
                "c++",
                "-std=c++14",
                f"-I{source_root / 'include'}",
                str(HARNESS),
                "-o",
                str(executable),
            ],
            check=True,
        )
        completed = subprocess.run(
            [str(executable)], check=True, capture_output=True, text=True
        )
    parsed = json.loads(completed.stdout)
    rendered = json.dumps(parsed, separators=(",", ":"), sort_keys=True) + "\n"
    fixture_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    temporary_fixture = FIXTURE.with_suffix(".json.tmp")
    temporary_fixture.write_text(rendered, encoding="utf-8")
    temporary_fixture.replace(FIXTURE)
    manifest = {
        "archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "fixture": FIXTURE.name,
        "fixture_sha256": fixture_hash,
        "generator": str(Path(__file__).relative_to(REFERENCE_DIRECTORY.parents[5])),
        "harness_sha256": hashlib.sha256(HARNESS.read_bytes()).hexdigest(),
        "source_commit": "b164fad1361505ff8dbf328107b645753ce331ac",
    }
    rendered_manifest = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    temporary_manifest = MANIFEST.with_suffix(".json.tmp")
    temporary_manifest.write_text(rendered_manifest, encoding="utf-8")
    temporary_manifest.replace(MANIFEST)


if __name__ == "__main__":
    regenerate_tricktrack_fixture()
