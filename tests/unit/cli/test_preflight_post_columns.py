"""The run preflight's post-column check (spec 2026-09-24-cli-preflight §8).

At 81d19ec a post op naming an absent column was swallowed at finalization
and discarded every post op's output (claim-verification report §7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from phenotypic import Image, ImagePipeline
from phenotypic._cli._cli_preflight import check_post_columns, intrinsic_metadata_headers
from phenotypic.abc_ import ObjectDetector
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.post import AppendString, ExpandMetadata, JoinMetadata
from phenotypic.sdk_ import is_metadata_header
from tests.unit.cli._preflight_support import make_context, make_datasets


def _tiff(path: Path) -> Path:
    image = np.full((32, 32, 3), 20, dtype=np.uint8)
    image[6:12, 6:12] = 220
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, image)
    return path


def _pipeline(**post) -> ImagePipeline:
    return ImagePipeline(ops={"d": OtsuDetector()}, meas={"s": MeasureSize()}, post=post)


def test_the_intrinsic_headers_match_a_real_read_and_measure(tmp_path: Path) -> None:
    """The in-memory derivation must equal what a real run inserts."""
    image = Image.imread(_tiff(tmp_path / "img001.tiff"))
    frame = _pipeline().apply_and_measure(image)
    # Schema ownership, never a string prefix (CLAUDE.md, Gotchas).
    real = {str(c) for c in frame.columns if is_metadata_header(str(c))}

    assert set(intrinsic_metadata_headers("Image")) == real


def test_a_missing_metadata_column_is_an_error_when_the_set_is_complete(tmp_path: Path) -> None:
    datasets = make_datasets(_tiff(tmp_path / "plate1" / "img001.tiff"))

    (finding,) = check_post_columns(
        make_context(_pipeline(tag=AppendString(column="DoesNotExist", value="_x")), datasets=datasets)
    )

    assert finding.code == "PF-POST-COLUMN" and finding.severity == "error"
    assert "Metadata_DoesNotExist" in finding.message


def test_columns_from_every_known_source_satisfy_the_check(tmp_path: Path) -> None:
    datasets = make_datasets(_tiff(tmp_path / "plate1" / "img001.tiff"))
    csv = tmp_path / "meta.csv"
    csv.write_text("ImageName,Strain\nimg001,WT_A\n", encoding="utf-8")
    pipeline = _pipeline(
        from_csv=AppendString(column="Strain", value="_x"),
        intrinsic=AppendString(column="ImageName", value="_x"),
        dataset=AppendString(column="Dataset", value="_x"),
        split=ExpandMetadata(column="Strain", labels=["Genotype", "Allele"]),
        from_split=AppendString(column="Genotype", value="_x"),
    )

    assert check_post_columns(make_context(pipeline, datasets=datasets, metadata_csv=csv)) == []


def test_an_input_carrying_phenotypic_metadata_softens_to_a_warning(tmp_path: Path) -> None:
    store = Image(np.zeros((16, 16, 3), dtype=np.uint8), name="img001").save2zarr(
        tmp_path / "plate1" / "img001.ome.zarr"
    )

    (finding,) = check_post_columns(
        make_context(_pipeline(tag=AppendString(column="DoesNotExist", value="_x")),
                     datasets=make_datasets(store))
    )

    assert finding.severity == "warning"


class _CustomDetector(ObjectDetector):
    """A detector from outside the phenotypic package; it may set metadata."""

    def _operate(self, image):
        return image


_CustomDetector.__module__ = "my_lab.detectors"


def test_a_custom_operation_softens_to_a_warning(tmp_path: Path) -> None:
    datasets = make_datasets(_tiff(tmp_path / "plate1" / "img001.tiff"))
    pipeline = ImagePipeline(
        ops={"d": _CustomDetector()}, post={"tag": AppendString(column="DoesNotExist", value="_x")}
    )

    (finding,) = check_post_columns(make_context(pipeline, datasets=datasets))

    assert finding.severity == "warning"


def test_join_metadata_on_an_unknown_measurement_header_warns(tmp_path: Path) -> None:
    datasets = make_datasets(_tiff(tmp_path / "plate1" / "img001.tiff"))
    table = tmp_path / "layout.csv"
    table.write_text("Grid_RowNum,Strain\n0,WT\n", encoding="utf-8")

    (finding,) = check_post_columns(
        make_context(_pipeline(layout=JoinMetadata(metadata=table, on=["Grid_RowNum"])),
                     datasets=datasets)
    )

    assert finding.severity == "warning"


def test_process_mode_never_runs_post(tmp_path: Path) -> None:
    datasets = make_datasets(_tiff(tmp_path / "plate1" / "img001.tiff"))

    assert check_post_columns(
        make_context(_pipeline(tag=AppendString(column="DoesNotExist", value="_x")), "process", datasets)
    ) == []


def _phenotypic_export(path: Path) -> Path:
    """A PNG or TIFF written by PhenoTypic, carrying its metadata on read."""
    image = Image(np.full((16, 16, 3), 40, dtype=np.uint8), name=path.stem)
    image.metadata["Strain"] = "WT"
    path.parent.mkdir(parents=True, exist_ok=True)
    image.rgb.imsave(path)
    return path


@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_a_file_carrying_phenotypic_metadata_softens_to_a_warning(tmp_path: Path, suffix: str) -> None:
    """Review D6 (M22, M23): the PNG text chunk and the TIFF description count too."""
    export = _phenotypic_export(tmp_path / "plate1" / f"img001{suffix}")

    (finding,) = check_post_columns(
        make_context(_pipeline(tag=AppendString(column="DoesNotExist", value="_x")),
                     datasets=make_datasets(export))
    )

    assert finding.severity == "warning"
