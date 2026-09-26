"""Input headers and the checks built on them (spec 2026-09-24-cli-preflight §7).

Every channel and bit-depth rule mirrors a probe of ``Image.imread`` recorded
in ``docs/superpowers/reports/2026-09-24-cli-preflight/header-behavior.md``;
``test_the_header_prediction_matches_imread`` re-checks that agreement on the
same files, so the table cannot drift from the reader it predicts.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image as PILImage

from phenotypic import Image, ImagePipeline
from phenotypic._cli._cli_input_headers import read_input_header, read_input_headers
from phenotypic._cli._cli_preflight import (
    check_bit_depth,
    check_detect_mode_on_gray,
    check_input_channels,
    check_input_headers_readable,
    check_raw_needs_rawpy,
    check_rgb_ops_on_gray,
    check_stem_collisions,
)
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureColor, MeasureSize
from tests.unit.cli._preflight_support import make_context, make_datasets

RNG = np.random.default_rng(0)


def _write(path: Path, kind: str) -> Path:
    """Write one probe file of *kind* at *path*."""
    if kind == "rgb_png":
        PILImage.fromarray(RNG.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(path)
    elif kind == "gray_png":
        PILImage.fromarray(RNG.integers(0, 255, (16, 16), dtype=np.uint8)).save(path)
    elif kind == "gray16_png":
        PILImage.fromarray(RNG.integers(0, 60000, (16, 16), dtype=np.uint16)).save(path)
    elif kind == "rgba_png":
        PILImage.fromarray(RNG.integers(0, 255, (16, 16, 4), dtype=np.uint8), "RGBA").save(path)
    elif kind == "palette_png":
        PILImage.fromarray(RNG.integers(0, 255, (16, 16, 3), dtype=np.uint8)).convert("P").save(path)
    elif kind == "la_png":
        PILImage.fromarray(RNG.integers(0, 255, (16, 16, 2), dtype=np.uint8), "LA").save(path)
    elif kind == "rgb_jpg":
        PILImage.fromarray(RNG.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save(path)
    elif kind == "rgb16_tiff":
        tifffile.imwrite(path, RNG.integers(0, 65535, (16, 16, 3), dtype=np.uint16))
    elif kind == "gray_tiff":
        tifffile.imwrite(path, RNG.integers(0, 255, (16, 16), dtype=np.uint8))
    elif kind == "two_ch_tiff":
        tifffile.imwrite(path, RNG.integers(0, 255, (16, 16, 2), dtype=np.uint8),
                         photometric="minisblack", planarconfig="contig")
    elif kind == "five_ch_tiff":
        tifffile.imwrite(path, RNG.integers(0, 255, (16, 16, 5), dtype=np.uint8),
                         photometric="minisblack", planarconfig="contig")
    elif kind == "multipage_gray_tiff":
        with tifffile.TiffWriter(path) as writer:
            for _ in range(3):
                writer.write(RNG.integers(0, 255, (16, 16), dtype=np.uint8),
                             photometric="minisblack")
    elif kind == "fiji_composite":
        tifffile.imwrite(path, RNG.integers(0, 255, (3, 16, 16), dtype=np.uint8),
                         imagej=True, metadata={"axes": "CYX"})
    elif kind == "ome_cyx":
        tifffile.imwrite(path, RNG.integers(0, 255, (3, 16, 16), dtype=np.uint8),
                         ome=True, metadata={"axes": "CYX"})
    elif kind == "stack3_tiff":
        tifffile.imwrite(path, RNG.integers(0, 255, (3, 16, 16), dtype=np.uint8),
                         photometric="minisblack")
    elif kind == "stack4_tiff":
        tifffile.imwrite(path, RNG.integers(0, 255, (4, 16, 16), dtype=np.uint8),
                         photometric="minisblack")
    elif kind == "stack2_tiff":
        tifffile.imwrite(path, RNG.integers(0, 255, (2, 16, 24), dtype=np.uint8),
                         photometric="minisblack")
    elif kind == "rgba_tiff":
        tifffile.imwrite(path, RNG.integers(0, 255, (16, 16, 4), dtype=np.uint8),
                         photometric="rgb", extrasamples=["unassalpha"])
    elif kind == "empty":
        path.write_bytes(b"")
    else:  # pragma: no cover - test authoring error
        raise ValueError(kind)
    return path


CASES = {
    # kind: (suffix, expected channels, expected raw_channels, expected bits)
    "rgb_png": (".png", 3, None, 8),
    "gray_png": (".png", 1, None, 8),
    "gray16_png": (".png", 1, None, 16),
    "rgba_png": (".png", 3, None, 8),
    "palette_png": (".png", 3, None, 8),
    "la_png": (".png", None, 2, None),
    "rgb_jpg": (".jpg", 3, None, 8),
    "rgb16_tiff": (".tiff", 3, None, 16),
    "gray_tiff": (".tiff", 1, None, 8),
    "two_ch_tiff": (".tiff", None, 2, 8),
    "five_ch_tiff": (".tiff", None, 5, 8),
    "multipage_gray_tiff": (".tiff", 1, None, 8),
    # One series of single-sample pages: skimage moves a leading axis of 3 or 4
    # to the end, so these decode to RGB (review D1).
    "fiji_composite": (".tif", 3, None, 8),
    "ome_cyx": (".ome.tif", 3, None, 8),
    "stack3_tiff": (".tiff", 3, None, 8),
    "stack4_tiff": (".tiff", 3, None, 8),
    # A 2-plane stack is not moved; imread then reads the width as a channel count.
    "stack2_tiff": (".tiff", None, 24, 8),
    "rgba_tiff": (".tiff", 3, None, 8),
}


@pytest.mark.parametrize("kind", sorted(CASES))
def test_headers_report_the_decoded_channel_count(kind: str, tmp_path: Path) -> None:
    suffix, channels, raw, bits = CASES[kind]
    header = read_input_header(_write(tmp_path / f"f{suffix}", kind))

    assert header.error is None
    assert (header.channels, header.raw_channels, header.bits) == (channels, raw, bits)


@pytest.mark.parametrize("kind", sorted(CASES))
def test_the_header_prediction_matches_imread(kind: str, tmp_path: Path) -> None:
    """The decoded-channel table must agree with the reader it predicts."""
    suffix, channels, raw, _ = CASES[kind]
    path = _write(tmp_path / f"f{suffix}", kind)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if raw is not None:
            with pytest.raises(ValueError, match="channels"):
                Image.imread(path)
            return
        image = Image.imread(path)
    assert (3 if not image.rgb.isempty() else 1) == channels


def test_an_empty_file_is_an_unreadable_header(tmp_path: Path) -> None:
    header = read_input_header(_write(tmp_path / "e.png", "empty"))

    # startswith, not "in": tmp_path's own name contains "empty" (review D6).
    assert header.error and header.error.startswith("the file is empty")


def test_a_garbage_file_is_an_unreadable_header(tmp_path: Path) -> None:
    path = tmp_path / "junk.png"
    path.write_bytes(b"this is not a png")

    assert read_input_header(path).error


def test_a_store_header_is_read_from_json_only(tmp_path: Path, monkeypatch) -> None:
    rgb_store = Image(RNG.integers(0, 255, (16, 16, 3), dtype=np.uint8)).save2zarr(tmp_path / "rgb.ome.zarr")
    gray_store = Image(RNG.integers(0, 255, (16, 16), dtype=np.uint8)).save2zarr(tmp_path / "gray.ome.zarr")
    import zarr

    monkeypatch.setattr(zarr, "open_array", lambda *a, **k: pytest.fail("opened an array"))

    rgb, gray = read_input_headers([rgb_store, gray_store])

    assert (rgb.channels, gray.channels) == (3, 1)
    assert rgb.carries_phenotypic_metadata and gray.carries_phenotypic_metadata


def test_a_raw_file_is_rgb_without_a_header_read(tmp_path: Path) -> None:
    path = tmp_path / "plate.nef"
    path.write_bytes(b"not parsed")

    header = read_input_header(path)

    assert (header.channels, header.bits, header.error) == (3, 16, None)


# --- the checks ---------------------------------------------------------------


def _tree(tmp_path: Path, *kinds: str) -> tuple:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, kind in enumerate(kinds):
        suffix = CASES.get(kind, (".png",))[0]
        paths.append(_write(tmp_path / f"img{index:03d}{suffix}", kind))
    return make_datasets(*paths)


def _detector() -> ImagePipeline:
    return ImagePipeline(ops={"d": OtsuDetector()}, meas={"s": MeasureSize()})


def test_detect_mode_on_all_gray_inputs_is_an_error(tmp_path: Path) -> None:
    datasets = _tree(tmp_path, "gray_png", "gray_tiff")

    (finding,) = check_detect_mode_on_gray(make_context(_detector(), datasets=datasets, detect_mode="red"))

    assert finding.code == "PF-DETECT-MODE-GRAY" and finding.severity == "error"


def test_detect_mode_on_some_gray_inputs_warns_and_lists_them(tmp_path: Path) -> None:
    datasets = _tree(tmp_path, "rgb_png", "gray_png")

    (finding,) = check_detect_mode_on_gray(make_context(_detector(), datasets=datasets, detect_mode="red"))

    assert finding.severity == "warning"
    assert finding.subjects == (str(datasets[0].images[1]),)


def test_detect_mode_gray_and_measure_mode_are_out_of_scope(tmp_path: Path) -> None:
    datasets = _tree(tmp_path, "gray_png")

    assert check_detect_mode_on_gray(make_context(_detector(), datasets=datasets)) == []
    assert check_detect_mode_on_gray(
        make_context(_detector(), "measure", datasets, detect_mode="red")
    ) == []


def test_an_rgb_measurer_on_gray_inputs(tmp_path: Path) -> None:
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, meas={"c": MeasureColor()})
    all_gray = _tree(tmp_path / "a", "gray_png", "gray_tiff")
    mixed = _tree(tmp_path / "b", "rgb_png", "gray_png")

    (error,) = check_rgb_ops_on_gray(make_context(pipeline, datasets=all_gray))
    (warning,) = check_rgb_ops_on_gray(make_context(pipeline, datasets=mixed))

    assert (error.severity, warning.severity) == ("error", "warning")
    assert "meas:c" in error.message


def test_an_rgb_measurer_is_out_of_scope_in_process_mode(tmp_path: Path) -> None:
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, meas={"c": MeasureColor()})
    datasets = _tree(tmp_path, "gray_png")

    assert check_rgb_ops_on_gray(make_context(pipeline, "process", datasets)) == []


def test_unreadable_headers_warn_for_a_subset(tmp_path: Path) -> None:
    datasets = _tree(tmp_path, "rgb_png", "empty")

    (finding,) = check_input_headers_readable(make_context(_detector(), datasets=datasets))

    assert finding.code == "PF-HEADER-UNREADABLE" and finding.severity == "warning"


def test_refused_channel_counts(tmp_path: Path) -> None:
    datasets = _tree(tmp_path, "la_png", "five_ch_tiff")

    (finding,) = check_input_channels(make_context(_detector(), datasets=datasets))

    assert finding.code == "PF-CHANNELS" and finding.severity == "error"
    assert len(finding.subjects) == 2


def test_a_bit_depth_that_contradicts_the_data(tmp_path: Path) -> None:
    datasets = _tree(tmp_path, "rgb16_tiff", "rgb16_tiff")

    (finding,) = check_bit_depth(make_context(_detector(), datasets=datasets, bit_depth=8))

    assert finding.code == "PF-BIT-DEPTH" and finding.severity == "error"
    assert check_bit_depth(make_context(_detector(), datasets=datasets, bit_depth=16)) == []
    assert check_bit_depth(make_context(_detector(), datasets=datasets)) == []


def test_raw_inputs_without_rawpy(monkeypatch, tmp_path: Path) -> None:
    import importlib.util

    raw = tmp_path / "plate.nef"
    raw.write_bytes(b"x")
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: None if name == "rawpy" else real(name, *a, **k))

    (finding,) = check_raw_needs_rawpy(make_context(_detector(), datasets=make_datasets(raw)))

    assert finding.code == "PF-RAW-NO-RAWPY" and finding.severity == "error"


def test_same_stem_inputs_collide(tmp_path: Path) -> None:
    datasets = make_datasets(tmp_path / "a.png", tmp_path / "a.tif", tmp_path / "b.png")

    (finding,) = check_stem_collisions(make_context(_detector(), datasets=datasets))

    assert finding.code == "PF-STEM-COLLISION" and finding.severity == "error"
    assert "a" in finding.message
    assert check_stem_collisions(
        make_context(_detector(), datasets=make_datasets(tmp_path / "a.png", tmp_path / "b.png"))
    ) == []


def test_sample_mode_still_checks_every_input(tmp_path: Path) -> None:
    """Review R26: a sample is a trial of the full run."""
    datasets = _tree(tmp_path, "gray_png", "gray_png", "gray_png")

    (finding,) = check_detect_mode_on_gray(
        make_context(_detector(), datasets=datasets, detect_mode="red", sample=1)
    )

    assert finding.severity == "error"


def test_a_z_stack_is_left_unknown(tmp_path: Path) -> None:
    """More than three dimensions: imread refuses, with no channel count to name."""
    path = tmp_path / "z.tiff"
    tifffile.imwrite(path, RNG.integers(0, 255, (5, 16, 16, 3), dtype=np.uint8), photometric="rgb")

    header = read_input_header(path)

    assert (header.channels, header.raw_channels, header.error) == (None, None, None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            Image.imread(path)


def test_a_zarr_v2_store_names_its_format(tmp_path: Path) -> None:
    """Review D9: the header error says why, not only that a file is missing."""
    store = tmp_path / "legacy.ome.zarr"
    store.mkdir()
    (store / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")

    header = read_input_header(store)

    assert header.error and "Zarr v2" in header.error


def test_an_rgb_measurer_accepts_a_fiji_composite(tmp_path: Path) -> None:
    """Review D1, end to end through the check: no false PF-RGB-OP-GRAY."""
    pipeline = ImagePipeline(ops={"d": OtsuDetector()}, meas={"c": MeasureColor()})
    datasets = _tree(tmp_path, "fiji_composite", "stack3_tiff")

    assert check_rgb_ops_on_gray(make_context(pipeline, datasets=datasets)) == []
    assert check_detect_mode_on_gray(
        make_context(_detector(), datasets=datasets, detect_mode="red")
    ) == []


def test_the_same_stem_in_two_datasets_is_not_a_collision(tmp_path: Path) -> None:
    """Review D6 (M14): ``plate1/a.png`` beside ``plate2/a.png`` is the ordinary layout."""
    from phenotypic._cli._cli_types import Dataset

    datasets = tuple(
        Dataset(name=name, images=[tmp_path / name / "a.png"], input_dir=tmp_path / name,
                output_dir=Path("out") / name)
        for name in ("plate1", "plate2")
    )

    assert check_stem_collisions(make_context(_detector(), datasets=datasets)) == []


def test_bit_depth_is_not_checked_for_jpeg(tmp_path: Path) -> None:
    """Review D3: imread records 8 bits for a JPEG whatever --bit-depth says."""
    datasets = _tree(tmp_path, "rgb_jpg", "rgb_jpg")

    assert check_bit_depth(make_context(_detector(), datasets=datasets, bit_depth=16)) == []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = Image.imread(datasets[0].images[0], bit_depth=16)
    assert image.bit_depth == 8
