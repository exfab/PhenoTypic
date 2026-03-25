"""Tests for phenotypic.sweep manifest generation and loading."""

import json

import pytest

from phenotypic.sweep import (
    Presence,
    Sweep,
    generate_sweep_manifest,
    load_pipeline_names_from_manifest,
    load_single_pipeline_from_manifest,
    load_sweep_manifest,
)
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


# ---------------------------------------------------------------------------
# Presence sweep tests
# ---------------------------------------------------------------------------


class TestPresenceSweep:

    def test_presence_adds_absent_variant(self):
        """Presence(GB, sigma=(1,2)) + Sweep(Otsu) → 3 pipelines."""
        config = [
            Presence(GaussianBlur, sigma=(1.0, 2.0)),
            Sweep(OtsuDetector),
        ]
        manifest = generate_sweep_manifest(config)
        # 2 sigma + 1 absent = 3
        assert manifest["total_pipelines"] == 3

    def test_presence_no_sweep_params(self):
        """Presence(GB) + Sweep(Otsu) → 2 pipelines (present + absent)."""
        config = [
            Presence(GaussianBlur),
            Sweep(OtsuDetector),
        ]
        manifest = generate_sweep_manifest(config)
        assert manifest["total_pipelines"] == 2

    def test_absent_pipeline_lacks_operation(self):
        """Absent variant should not contain the Presence op."""
        config = [
            Presence(GaussianBlur, sigma=(1.0,)),
            Sweep(OtsuDetector),
        ]
        manifest = generate_sweep_manifest(config)
        pipes = manifest["configs"]["Pipeline"]["pipelines"]
        op_sets = []
        for pipe_dict in pipes.values():
            op_names = list(pipe_dict["pipe_cfgs"].keys())
            op_sets.append(set(op_names))

        # One pipeline has GB, one does not
        has_gb = [s for s in op_sets if "GaussianBlur" in s]
        no_gb = [s for s in op_sets if "GaussianBlur" not in s]
        assert len(has_gb) == 1
        assert len(no_gb) == 1
        # All pipelines should have OtsuDetector
        for s in op_sets:
            assert "OtsuDetector" in s

    def test_present_pipeline_has_correct_params(self):
        """Present variant should have the correct op and params."""
        config = [
            Presence(GaussianBlur, sigma=(1.5,), truncate=3.0),
            Sweep(OtsuDetector),
        ]
        manifest = generate_sweep_manifest(config)
        pipes = manifest["configs"]["Pipeline"]["pipelines"]
        for pipe_dict in pipes.values():
            if "GaussianBlur" in pipe_dict["pipe_cfgs"]:
                gb_cfg = pipe_dict["pipe_cfgs"]["GaussianBlur"]
                assert gb_cfg["params"]["sigma"] == 1.5
                assert gb_cfg["params"]["truncate"] == 3.0

    def test_multiple_presence(self):
        """Two Presence ops → combos with neither, one, or both."""
        from phenotypic.enhance import MedianFilter

        config = [
            Presence(GaussianBlur, sigma=(1.0,)),
            Presence(MedianFilter, width=(3,)),
        ]
        manifest = generate_sweep_manifest(config)
        # Each Presence: 1 param combo + 1 absent = 2
        # 2 × 2 = 4
        assert manifest["total_pipelines"] == 4

        pipes = manifest["configs"]["Pipeline"]["pipelines"]
        combos = []
        for pipe_dict in pipes.values():
            op_names = set(pipe_dict["pipe_cfgs"].keys())
            combos.append(op_names)

        # Should include: both, GB only, MF only, neither
        assert {"GaussianBlur", "MedianFilter"} in combos
        assert {"GaussianBlur"} in combos
        assert {"MedianFilter"} in combos
        assert set() in combos

    def test_presence_with_fixed_params(self):
        """Fixed params present when op is included."""
        config = [
            Presence(GaussianBlur, sigma=(1.0,), truncate=4.0),
        ]
        manifest = generate_sweep_manifest(config)
        pipes = manifest["configs"]["Pipeline"]["pipelines"]
        for pipe_dict in pipes.values():
            if "GaussianBlur" in pipe_dict["pipe_cfgs"]:
                gb_cfg = pipe_dict["pipe_cfgs"]["GaussianBlur"]
                assert gb_cfg["params"]["truncate"] == 4.0

    def test_presence_pipeline_loadable(self, tmp_path):
        """Round-trip through JSON works for both present and absent."""
        config = [
            Presence(GaussianBlur, sigma=(1.0,)),
            Sweep(OtsuDetector),
        ]
        out = tmp_path / "opt_sweep.json"
        generate_sweep_manifest(config, filepath=out)
        result = load_sweep_manifest(out)
        pipes = result["Pipeline"]
        assert len(pipes) == 2
        for pipe in pipes.values():
            assert isinstance(pipe, ImagePipeline)

    def test_presence_repr(self):
        p = Presence(GaussianBlur, sigma=(1.0, 2.0), truncate=4.0)
        r = repr(p)
        assert r.startswith("Presence(")
        assert "GaussianBlur" in r

    def test_presence_is_sweep_subclass(self):
        p = Presence(GaussianBlur)
        assert isinstance(p, Sweep)

    def test_presence_wraps_sweep_instance(self):
        """Presence(Sweep(...)) copies operation_class and params."""
        inner = Sweep(GaussianBlur, sigma=(1.0, 2.0), truncate=4.0)
        p = Presence(inner)
        assert p.operation_class is GaussianBlur
        assert p.sweep_params == {"sigma": [1.0, 2.0]}
        assert p.fixed_params == {"truncate": 4.0}
        assert isinstance(p, Presence)

    def test_presence_wraps_sweep_manifest_count(self):
        """Presence(Sweep(GB, sigma=(1,2))) + Sweep(Otsu) → 3."""
        config = [
            Presence(Sweep(GaussianBlur, sigma=(1.0, 2.0))),
            Sweep(OtsuDetector),
        ]
        manifest = generate_sweep_manifest(config)
        assert manifest["total_pipelines"] == 3

    def test_presence_wraps_sweep_repr(self):
        """repr of wrapped Sweep still starts with 'Presence('."""
        p = Presence(Sweep(GaussianBlur, sigma=(1.0, 2.0)))
        assert repr(p).startswith("Presence(")
        assert "GaussianBlur" in repr(p)

    def test_presence_wraps_sweep_rejects_extra_params(self):
        """Passing **params alongside a Sweep instance is an error."""
        with pytest.raises(TypeError, match="Cannot pass \\*\\*params"):
            Presence(Sweep(GaussianBlur), sigma=1.0)


# ---------------------------------------------------------------------------
# Lazy loading tests
# ---------------------------------------------------------------------------


class TestLoadSinglePipelineFromManifest:

    def test_round_trips_correctly(self, simple_config, tmp_path):
        """Extracted JSON is valid and produces correct pipeline."""
        out = tmp_path / "sweep.json"
        generate_sweep_manifest(simple_config, filepath=out)

        json_str = load_single_pipeline_from_manifest(out, "Pipeline_0")
        pipe = ImagePipeline.from_json(json_str)
        assert pipe.name == "Pipeline_0"
        op_names = list(pipe._ops.keys())
        assert "GaussianBlur" in op_names
        assert "OtsuDetector" in op_names

    def test_matches_full_load(self, simple_config, tmp_path):
        """Single-pipeline load matches the corresponding full load."""
        out = tmp_path / "sweep.json"
        generate_sweep_manifest(simple_config, filepath=out)

        # Full load
        full = load_sweep_manifest(out)
        full_pipe = full["Pipeline"]["Pipeline_1"]

        # Lazy load
        json_str = load_single_pipeline_from_manifest(out, "Pipeline_1")
        lazy_pipe = ImagePipeline.from_json(json_str)

        assert full_pipe.to_json() == lazy_pipe.to_json()

    def test_not_found_raises_key_error(self, simple_config, tmp_path):
        out = tmp_path / "sweep.json"
        generate_sweep_manifest(simple_config, filepath=out)

        with pytest.raises(KeyError, match="not found in manifest"):
            load_single_pipeline_from_manifest(out, "NonexistentPipeline")

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_single_pipeline_from_manifest(
                tmp_path / "missing.json", "Pipeline_0",
            )

    def test_named_configs(self, named_configs, tmp_path):
        """Works with multi-config manifests."""
        out = tmp_path / "multi.json"
        generate_sweep_manifest(named_configs, filepath=out)

        json_str = load_single_pipeline_from_manifest(out, "ConfigA_0")
        pipe = ImagePipeline.from_json(json_str)
        assert pipe.name == "ConfigA_0"


class TestLoadPipelineNamesFromManifest:

    def test_returns_correct_names(self, simple_config, tmp_path):
        out = tmp_path / "sweep.json"
        manifest = generate_sweep_manifest(simple_config, filepath=out)

        names = load_pipeline_names_from_manifest(out)
        expected = list(manifest["configs"]["Pipeline"]["pipelines"].keys())
        assert names == expected

    def test_correct_count(self, simple_config, tmp_path):
        out = tmp_path / "sweep.json"
        generate_sweep_manifest(simple_config, filepath=out)

        names = load_pipeline_names_from_manifest(out)
        assert len(names) == 4  # 2 sigma × 2 ignore_zeros

    def test_named_configs_count(self, named_configs, tmp_path):
        out = tmp_path / "multi.json"
        manifest = generate_sweep_manifest(named_configs, filepath=out)

        names = load_pipeline_names_from_manifest(out)
        assert len(names) == manifest["total_pipelines"]

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline_names_from_manifest(tmp_path / "missing.json")
