"""The format default is layer-dependent, and two layers have no zarr form."""

from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from phenotypic._cli._cli_process_only import resolve_process_format

# `main`, not `process_single`: that name does not exist. The command object
# is `main`, declared `@click.command()` at _cli_process_single.py:420.
from phenotypic._cli._cli_process_single import main as process_single_worker


@pytest.mark.parametrize("layer", ["rgb", "gray"])
def test_primary_series_layers_default_to_zarr(layer: str) -> None:
    assert resolve_process_format(layer, None) == "zarr"


def test_detect_mat_defaults_to_tiff() -> None:
    """It has no store form at all; the default cannot be zarr."""
    assert resolve_process_format("detect_mat", None) == "tiff"


def test_objmap_defaults_to_tiff() -> None:
    """A bare `--mode process --layer objmap` must keep working."""
    assert resolve_process_format("objmap", None) == "tiff"


@pytest.mark.parametrize("layer", ["rgb", "gray", "detect_mat", "objmap"])
def test_an_explicit_tiff_request_is_always_honoured(layer: str) -> None:
    assert resolve_process_format(layer, "tiff") == "tiff"


@pytest.mark.parametrize("layer", ["rgb", "gray"])
def test_explicit_zarr_is_honoured_for_a_primary_series(layer: str) -> None:
    assert resolve_process_format(layer, "zarr") == "zarr"


def test_explicit_zarr_for_objmap_is_refused() -> None:
    with pytest.raises(click.UsageError, match="no single-series OME-Zarr form"):
        resolve_process_format("objmap", "zarr")


def test_explicit_zarr_for_detect_mat_is_refused() -> None:
    with pytest.raises(click.UsageError, match="no single-series OME-Zarr form"):
        resolve_process_format("detect_mat", "zarr")


def test_the_two_refusals_give_different_reasons() -> None:
    """One is an NGFF rule; the other is ours. A user deserves to know which.

    objmap is refused because NGFF 0.5 2.6 says a labels group is nested
    inside an image group and is not itself an image -- a format rule, and
    unfixable here. detect_mat is refused because PhenoTypic's own writer
    requires a primary series (`ngff_.primary_series`, ngff_.py:459-474) -- our
    rule, and changeable in its own design. Collapsing the two into one message
    would tell a detect_mat user that NGFF forbids something it does not.
    """
    with pytest.raises(click.UsageError) as objmap:
        resolve_process_format("objmap", "zarr")
    with pytest.raises(click.UsageError) as detect_mat:
        resolve_process_format("detect_mat", "zarr")

    assert "labels group" in str(objmap.value)
    assert "labels group" not in str(detect_mat.value)
    assert "primary series" in str(detect_mat.value)
    assert "primary series" not in str(objmap.value)
    # Each names the layer it is about, and neither claims to be about the
    # other -- the messages must not be interchangeable.
    assert "objmap" in str(objmap.value)
    assert "detect_mat" in str(detect_mat.value)
    # Each names a remedy the user can actually type.
    assert "--process-format tiff" in str(objmap.value)
    assert "--process-format tiff" in str(detect_mat.value)


def test_the_worker_advertises_the_option() -> None:
    result = CliRunner().invoke(process_single_worker, ["--help"])
    assert result.exit_code == 0
    assert "--process-format" in result.output
    assert "zarr for rgb/gray" in result.output  # the default is stated
