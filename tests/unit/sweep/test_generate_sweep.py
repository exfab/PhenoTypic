"""Tests for phenotypic.sweep manifest generation and loading."""

import json

import pytest

from phenotypic.sweep import Sweep, generate_sweep_manifest, load_sweep_manifest
from phenotypic import ImagePipeline
from phenotypic.enhance import GaussianBlur
from phenotypic.detect import OtsuDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_config():
    """A single config with one swept param per operation."""
    return [
        Sweep(GaussianBlur, sigma=(1.0, 2.0), truncate=4.0),
        Sweep(OtsuDetector, ignore_zeros=(True, False)),
    ]


@pytest.fixture
def named_configs():
    """Two named configs for multi-config tests."""
    return {
        "ConfigA": [
            Sweep(GaussianBlur, sigma=(1.0, 2.0)),
            Sweep(OtsuDetector),
        ],
        "ConfigB": [
            Sweep(GaussianBlur, sigma=(3.0,)),
            Sweep(OtsuDetector, ignore_zeros=(True, False)),
        ],
    }


# ---------------------------------------------------------------------------
# generate_sweep_manifest tests
# ---------------------------------------------------------------------------


class TestGenerateSweepManifest:

    def test_single_config_pipeline_count(self, simple_config):
        manifest = generate_sweep_manifest(simple_config)
        # 2 sigma values × 2 ignore_zeros values = 4 pipelines
        assert manifest["total_pipelines"] == 4

    def test_single_config_auto_named(self, simple_config):
        manifest = generate_sweep_manifest(simple_config)
        assert "Pipeline" in manifest["configs"]

    def test_cartesian_product_count(self):
        config = [
            Sweep(GaussianBlur, sigma=(1.0, 2.0, 3.0)),
            Sweep(OtsuDetector, ignore_zeros=(True, False)),
        ]
        manifest = generate_sweep_manifest(config)
        # 3 × 2 = 6
        assert manifest["total_pipelines"] == 6

    def test_fixed_params_preserved(self, simple_config):
        manifest = generate_sweep_manifest(simple_config)
        pipes = manifest["configs"]["Pipeline"]["pipelines"]
        for pipe_dict in pipes.values():
            gb_cfg = pipe_dict["pipe_cfgs"]["GaussianBlur"]
            assert gb_cfg["params"]["truncate"] == 4.0

    def test_multiple_named_configs(self, named_configs):
        manifest = generate_sweep_manifest(named_configs)
        assert "ConfigA" in manifest["configs"]
        assert "ConfigB" in manifest["configs"]

    def test_total_equals_sum_of_n_combinations(self, named_configs):
        manifest = generate_sweep_manifest(named_configs)
        total = sum(
            cfg["n_combinations"] for cfg in manifest["configs"].values()
        )
        assert manifest["total_pipelines"] == total

    def test_pipeline_names_sequential(self, simple_config):
        manifest = generate_sweep_manifest(simple_config)
        pipes = manifest["configs"]["Pipeline"]["pipelines"]
        names = sorted(pipes.keys())
        expected = [f"Pipeline_{i}" for i in range(len(names))]
        assert names == expected

    def test_each_pipeline_loadable(self, simple_config):
        manifest = generate_sweep_manifest(simple_config)
        pipes = manifest["configs"]["Pipeline"]["pipelines"]
        for pipe_name, pipe_dict in pipes.items():
            pipe = ImagePipeline.from_json(json.dumps(pipe_dict))
            assert pipe.name == pipe_name
            op_names = list(pipe._ops.keys())
            assert "GaussianBlur" in op_names
            assert "OtsuDetector" in op_names

    def test_loaded_pipeline_has_correct_params(self, simple_config):
        manifest = generate_sweep_manifest(simple_config)
        pipe_dict = manifest["configs"]["Pipeline"]["pipelines"]["Pipeline_0"]
        pipe = ImagePipeline.from_json(json.dumps(pipe_dict))
        gb = list(pipe._ops.values())[0]
        assert isinstance(gb, GaussianBlur)
        assert gb.sigma in [1.0, 2.0]
        assert gb.truncate == 4.0

    def test_meas_kwarg_attaches_measurements(self):
        from phenotypic.measure import MeasureShape

        config = [Sweep(OtsuDetector)]
        manifest = generate_sweep_manifest(config, meas=[MeasureShape()])
        pipe_dict = manifest["configs"]["Pipeline"]["pipelines"]["Pipeline_0"]
        assert "MeasureShape" in pipe_dict["meas"]

    def test_filepath_writes_json(self, simple_config, tmp_path):
        out = tmp_path / "sweep.json"
        manifest = generate_sweep_manifest(simple_config, filepath=out)
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded == manifest

    def test_no_file_when_filepath_none(self, simple_config, tmp_path):
        generate_sweep_manifest(simple_config)
        # No file should appear in tmp_path
        assert list(tmp_path.iterdir()) == []

    def test_empty_sweep_params_single_pipeline(self):
        config = [Sweep(GaussianBlur)]
        manifest = generate_sweep_manifest(config)
        assert manifest["total_pipelines"] == 1

    def test_version_in_manifest(self):
        import phenotypic

        config = [Sweep(GaussianBlur)]
        manifest = generate_sweep_manifest(config)
        assert manifest["version"] == phenotypic.__version__

    def test_description_stored(self, simple_config):
        manifest = generate_sweep_manifest(simple_config, desc="Test run")
        assert manifest["description"] == "Test run"

    def test_invalid_configs_type_raises(self):
        with pytest.raises(TypeError, match="list or dict"):
            generate_sweep_manifest("bad input")

    def test_non_sweep_in_list_raises(self):
        with pytest.raises(TypeError, match="Sweep instance"):
            generate_sweep_manifest([GaussianBlur()])


# ---------------------------------------------------------------------------
# load_sweep_manifest tests
# ---------------------------------------------------------------------------


class TestLoadSweepManifest:

    def test_round_trip(self, simple_config, tmp_path):
        out = tmp_path / "sweep.json"
        generate_sweep_manifest(simple_config, filepath=out)
        result = load_sweep_manifest(out)
        assert isinstance(result, dict)
        assert "Pipeline" in result
        for pipe_name, pipe in result["Pipeline"].items():
            assert isinstance(pipe, ImagePipeline)

    def test_returns_correct_structure(self, named_configs, tmp_path):
        out = tmp_path / "multi.json"
        generate_sweep_manifest(named_configs, filepath=out)
        result = load_sweep_manifest(out)
        assert "ConfigA" in result
        assert "ConfigB" in result
        for cfg_name, pipes in result.items():
            for pipe_name, pipe in pipes.items():
                assert isinstance(pipe, ImagePipeline)

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sweep_manifest(tmp_path / "does_not_exist.json")

    def test_loaded_ops_match_originals(self, simple_config, tmp_path):
        out = tmp_path / "sweep.json"
        generate_sweep_manifest(simple_config, filepath=out)
        result = load_sweep_manifest(out)
        for pipe_name, pipe in result["Pipeline"].items():
            ops = list(pipe._ops.values())
            assert isinstance(ops[0], GaussianBlur)
            assert isinstance(ops[1], OtsuDetector)
