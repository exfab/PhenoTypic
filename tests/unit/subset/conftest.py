"""Fixtures for the subset selector tests.

The candidate set deliberately spans **two parent-relative subdirectories**:
``scan_directory_structure`` treats one level of subdirectories as separate
datasets, so a bare filename cannot disambiguate two datasets that both contain
``plate_001.tif`` (§10.2). Every fixture here therefore carries parent-relative
paths, never bare names.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def parent_dir(tmp_path):
    """A parent image directory with ``plateA/`` (7) and ``plateB/`` (5)."""
    parent = tmp_path / "plates"
    for dataset, count in (("plateA", 7), ("plateB", 5)):
        (parent / dataset).mkdir(parents=True)
        for index in range(1, count + 1):
            (parent / dataset / f"{dataset}_{index:02d}.tif").write_bytes(
                f"{dataset}{index}".encode()
            )
    return parent


@pytest.fixture
def image_refs(parent_dir):
    """Twelve ``ImageRef``s across the two datasets, in sorted order."""
    from phenotypic.subset import ImageRef

    return [
        ImageRef(
            path=path,
            relative_path=path.relative_to(parent_dir).as_posix(),
        )
        for path in sorted(parent_dir.rglob("*.tif"))
    ]


@pytest.fixture
def batches_csv(tmp_path, image_refs):
    """A grouping CSV keyed by parent-relative path.

    The first two images are the ``rare`` batch and the remaining ten are
    ``common``, so ``equal`` and ``proportional`` allocation give visibly
    different answers.
    """
    csv = tmp_path / "batches.csv"
    rows = "\n".join(
        f"{ref.relative_path},{'rare' if index < 2 else 'common'}"
        for index, ref in enumerate(image_refs)
    )
    csv.write_text(f"image,Metadata_Batch\n{rows}\n")
    return csv


@pytest.fixture
def species_csv(tmp_path, image_refs):
    """A filtering CSV: ``plateA`` is ``A_nidulans``, ``plateB`` is ``A_niger``."""
    csv = tmp_path / "species.csv"
    rows = "\n".join(
        f"{ref.relative_path},"
        f"{'A_nidulans' if ref.dataset == 'plateA' else 'A_niger'}"
        for ref in image_refs
    )
    csv.write_text(f"image,Metadata_Species\n{rows}\n")
    return csv
