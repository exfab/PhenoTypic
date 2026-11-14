"""One current pipeline config per folder (spec A3, A4).

The finalize writer publishes ``pipeline.pht-pipe.json`` and retires an
old-typed ``pipeline.json.pht-pipe`` sibling; the plain V1 ``pipeline.json``
is left for V1 readers. The start-of-run seed writes nothing when a config
already exists under any of the three names.
"""

from __future__ import annotations

from pathlib import Path

from phenotypic._cli._cli_output_manager import _persist_pipeline_to_output_dir
from phenotypic.phenotypicCLI import _seed_pipeline_config


def test_finalize_publishes_new_name_and_removes_old(tmp_path, tiny_pipeline):
    old = tmp_path / "deliverables" / "pipeline.json.pht-pipe"  # legacy name, read path
    old.parent.mkdir()
    old.write_text(tiny_pipeline.to_json())
    written = _persist_pipeline_to_output_dir(tmp_path, tiny_pipeline)
    new = tmp_path / "deliverables" / "pipeline.pht-pipe.json"
    assert written == new
    assert new.is_file()
    assert not old.exists()


def test_finalize_keeps_plain_legacy_json(tmp_path, tiny_pipeline):
    plain = tmp_path / "deliverables" / "pipeline.json"  # legacy name, read path
    plain.parent.mkdir()
    plain.write_text(tiny_pipeline.to_json())
    _persist_pipeline_to_output_dir(tmp_path, tiny_pipeline)
    assert plain.is_file()
    assert (tmp_path / "deliverables" / "pipeline.pht-pipe.json").is_file()


def test_seed_skips_when_any_name_exists(tmp_path, tiny_pipeline_file):
    old = tmp_path / "deliverables" / "pipeline.json.pht-pipe"  # legacy name, read path
    old.parent.mkdir()
    old.write_text("{}")
    assert _seed_pipeline_config(tiny_pipeline_file, tmp_path) is None
    assert not (tmp_path / "deliverables" / "pipeline.pht-pipe.json").exists()
    assert old.read_text() == "{}"


def test_seed_writes_new_name_byte_for_byte(tmp_path, tiny_pipeline_file):
    seeded = _seed_pipeline_config(tiny_pipeline_file, tmp_path)
    assert seeded == tmp_path / "deliverables" / "pipeline.pht-pipe.json"
    assert seeded.read_bytes() == Path(tiny_pipeline_file).read_bytes()


def test_old_seeded_config_is_restart_scaffolding(tmp_path):
    """A tree seeded under the old typed name, with no image run, reads as fresh."""
    from phenotypic.phenotypicCLI import _prior_scientific_artifact_names

    old = tmp_path / "deliverables" / "pipeline.json.pht-pipe"  # legacy name, read path
    old.parent.mkdir()
    old.write_text("{}")
    assert _prior_scientific_artifact_names(tmp_path) == []


def test_process_copy_publishes_new_name_and_removes_old(tmp_path, tiny_pipeline_file):
    """The process-mode ``.phenotypic`` copy keeps one current file (spec A4)."""
    from phenotypic.phenotypicCLI import _publish_process_pipeline_copy

    cache = tmp_path / ".phenotypic"
    cache.mkdir()
    old = cache / "pipeline.json.pht-pipe"  # legacy name, read path
    old.write_text("{}")
    plain = cache / "pipeline.json"  # legacy name, read path
    plain.write_text("{}")
    written = _publish_process_pipeline_copy(tiny_pipeline_file, tmp_path)
    assert written == cache / "pipeline.pht-pipe.json"
    assert written.read_bytes() == Path(tiny_pipeline_file).read_bytes()
    assert not old.exists()
    assert plain.is_file()


def test_qc_recipe_write_publishes_new_name_and_removes_old(tmp_path, tiny_pipeline):
    """A QC write beside an old-typed sibling leaves one current file."""
    from phenotypic.sdk_._qc_recipe import QcRecipe

    old = tmp_path / "deliverables" / "pipeline.json.pht-pipe"  # legacy name, read path
    old.parent.mkdir()
    old.write_text(tiny_pipeline.to_json())
    recipe = QcRecipe.load(tmp_path)
    assert recipe._write_qc_array()
    assert (tmp_path / "deliverables" / "pipeline.pht-pipe.json").is_file()
    assert not old.exists()


def test_qc_recipe_write_keeps_plain_legacy_json(tmp_path, tiny_pipeline):
    from phenotypic.sdk_._qc_recipe import QcRecipe

    plain = tmp_path / "deliverables" / "pipeline.json"  # legacy name, read path
    plain.parent.mkdir()
    plain.write_text(tiny_pipeline.to_json())
    recipe = QcRecipe.load(tmp_path)
    assert recipe._write_qc_array()
    assert plain.is_file()
    assert (tmp_path / "deliverables" / "pipeline.pht-pipe.json").is_file()
