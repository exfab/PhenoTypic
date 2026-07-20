"""Inject each A10 FilFinder adapter mutant and run its killing probe."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
from types import ModuleType
import warnings

import numpy as np

from phenotypic import Image


ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "src/phenotypic/detect/_filfinder_detector.py"


def _load_module(path: Path, name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"mutation site count is {source.count(old)}, expected 1: {old!r}"
        )
    start = source.index(old)
    mutated = source[:start] + new + source[start + len(old) :]
    if mutated[:start] != source[:start]:
        raise AssertionError("mutation changed text before its site")
    if mutated[start + len(new) :] != source[start + len(old) :]:
        raise AssertionError("mutation changed text after its site")
    return mutated


class _Quantity:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Quantity) and self.value == other.value


class _PixelUnit:
    def __rmul__(self, value: float) -> _Quantity:
        return _Quantity(value)


class _Pool:
    def __init__(self) -> None:
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(
        self, wait: bool = True, *, cancel_futures: bool = False
    ) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class _Source:
    instances: list[_Source] = []

    def __init__(
        self,
        image: np.ndarray,
        *,
        beamwidth: object,
        mask: np.ndarray,
        pool: object,
    ) -> None:
        self.image = image
        self.beamwidth = beamwidth
        self.input_mask = mask
        self.pool = pool
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.mask = mask.copy()
        self.skeleton = np.zeros_like(mask)
        self.skeleton[0, 0] = True
        self.skeleton[1, 1] = True
        self.skeleton[-1, -1] = True
        self.skeleton_longpath = np.zeros_like(mask)
        self.skeleton_longpath[0, 0] = True
        self.skeleton_longpath[1, 1] = True
        type(self).instances.append(self)

    def create_mask(self, *, use_existing_mask: bool) -> None:
        self.calls.append(
            ("create_mask", {"use_existing_mask": use_existing_mask})
        )

    def medskel(self, *, rng: int) -> None:
        self.calls.append(("medskel", {"rng": rng}))

    def analyze_skeletons(self, **kwargs: object) -> None:
        self.calls.append(("analyze_skeletons", kwargs))


def _image() -> Image:
    image = Image(np.zeros((5, 5, 3), dtype=np.uint8))
    values = np.zeros((5, 5), dtype=np.float32)
    values[0, 0] = 0.5
    values[1, 1] = 0.75
    values[-1, -1] = 1.0
    image._data.detect_mat = values
    return image


def _install_fake_runtime(module: ModuleType) -> list[_Pool]:
    _Source.instances = []
    pools: list[_Pool] = []

    def create_pool() -> _Pool:
        pool = _Pool()
        pools.append(pool)
        return pool

    module._load_filfinder_runtime = lambda: (
        _Source,
        types.SimpleNamespace(pix=_PixelUnit()),
    )
    module._create_warning_forwarding_pool = create_pool
    return pools


def _apply(
    module: ModuleType, **kwargs: object
) -> tuple[Image, _Source, list[_Pool]]:
    pools = _install_fake_runtime(module)
    result = module.FilFinderDetector(**kwargs).apply(_image())
    return result, _Source.instances[-1], pools


def _probe_float32(module: ModuleType) -> None:
    value = np.nextafter(np.float64(0.5), np.float64(0.0))
    copied = module._copy_float32_source(np.array([[value]]))
    if copied[0, 0] != 0.5:
        raise AssertionError("float32 seam was skipped")


def _probe_threshold(module: ModuleType) -> None:
    result, _, _ = _apply(module, output="mask", threshold=0.5)
    if result.objmap[:][0, 0] == 0:
        raise AssertionError("threshold equality became background")


def _probe_layer(module: ModuleType) -> None:
    _, source, _ = _apply(module, output="mask")
    np.testing.assert_array_equal(source.image, _image().detect_mat[:])


def _probe_create_mask(module: ModuleType) -> None:
    _, source, _ = _apply(module, output="mask")
    if source.calls != [("create_mask", {"use_existing_mask": True})]:
        raise AssertionError("existing-mask stage changed")


def _probe_beam(module: ModuleType) -> None:
    _, source, _ = _apply(module, output="mask", beamwidth_px=2.25)
    if source.beamwidth != _Quantity(2.25):
        raise AssertionError("beam width lost its pixel quantity")


def _analysis(
    module: ModuleType,
) -> tuple[dict[str, object], _Source, list[_Pool]]:
    _, source, pools = _apply(
        module,
        output="longest_path",
        branch_threshold_px=4.5,
        prune_criteria="intensity",
        relative_intensity_threshold=0.35,
        max_prune_iterations=13,
        rng_seed=23,
    )
    return source.calls[-1][1], source, pools


def _probe_branch(module: ModuleType) -> None:
    arguments, _, _ = _analysis(module)
    if arguments["branch_thresh"] != _Quantity(4.5):
        raise AssertionError("branch threshold lost its pixel quantity")


def _probe_skel(module: ModuleType) -> None:
    arguments, _, _ = _analysis(module)
    if arguments["skel_thresh"] != _Quantity(1.0):
        raise AssertionError("upstream one-pixel defect was changed")


def _probe_prune(module: ModuleType) -> None:
    arguments, _, _ = _analysis(module)
    if arguments["relintens_thresh"] != 0.35:
        raise AssertionError("prune fields were swapped")


def _probe_rng(module: ModuleType) -> None:
    _, source, _ = _analysis(module)
    if source.calls[1] != ("medskel", {"rng": 23}):
        raise AssertionError("RNG seed was not forwarded")


def _probe_fresh(module: ModuleType) -> None:
    pools = _install_fake_runtime(module)
    detector = module.FilFinderDetector(output="skeleton")
    detector.apply(_image())
    detector.apply(_image())
    if len(_Source.instances) != 2 or len(pools) != 2:
        raise AssertionError("source object or pool was reused")


def _probe_longest(module: ModuleType) -> None:
    result, _, _ = _apply(module, output="longest_path")
    if result.objmap[:][-1, -1] != 0:
        raise AssertionError("wrong post-analysis raster was selected")


def _probe_connectivity(module: ModuleType) -> None:
    structures: list[np.ndarray] = []
    original_label = module.ndimage.label

    def recording_label(
        values: np.ndarray, structure: np.ndarray | None = None
    ) -> tuple[np.ndarray, int]:
        structures.append(np.asarray(structure).copy())
        return original_label(values, structure=structure)

    module.ndimage.label = recording_label
    try:
        result, _, _ = _apply(module, output="skeleton")
    finally:
        module.ndimage.label = original_label
    if not structures:
        raise AssertionError("labeling was skipped")
    for structure in structures:
        np.testing.assert_array_equal(
            structure, np.ones((3, 3), dtype=np.uint8)
        )
    expected = np.zeros((5, 5), dtype=np.int32)
    expected[0, 0] = 1
    expected[1, 1] = 1
    expected[-1, -1] = 2
    np.testing.assert_array_equal(result.objmap[:], expected)


def _probe_objmask(module: ModuleType) -> None:
    result, _, _ = _apply(module, output="longest_path")
    np.testing.assert_array_equal(result.objmask[:], result.objmap[:] > 0)
    expected = np.zeros((5, 5), dtype=bool)
    expected[0, 0] = True
    expected[1, 1] = True
    np.testing.assert_array_equal(result.objmask[:], expected)


def _probe_layers(module: ModuleType) -> None:
    pools = _install_fake_runtime(module)
    image = _image()
    before = image.detect_mat[:].copy()
    module.FilFinderDetector(output="mask").apply(image, inplace=True)
    np.testing.assert_array_equal(image.detect_mat[:], before)
    if pools[-1].shutdown_calls != [(True, False)]:
        raise AssertionError("pool was not shut down")


def _probe_dependency(module: ModuleType) -> None:
    old_filfinder = sys.modules.get("fil_finder", ...)
    old_astropy = sys.modules.get("astropy", ...)
    sys.modules["fil_finder"] = None
    sys.modules["astropy"] = None
    try:
        try:
            module.FilFinderDetector(output="mask").apply(_image())
        except ImportError as error:
            if "topology" not in str(error):
                raise AssertionError(
                    "dependency error is not actionable"
                ) from error
        else:
            raise AssertionError("missing dependency was swallowed")
    finally:
        for name, old in (
            ("fil_finder", old_filfinder),
            ("astropy", old_astropy),
        ):
            if old is ...:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _probe_shutdown(module: ModuleType) -> None:
    _, _, pools = _apply(module, output="mask")
    if pools[-1].shutdown_calls != [(True, False)]:
        raise AssertionError("pool shutdown was skipped")


def _probe_stages(module: ModuleType) -> None:
    _, source, _ = _apply(module, output="mask")
    if [name for name, _ in source.calls] != ["create_mask"]:
        raise AssertionError("downstream stage ran for mask output")


def _probe_warning_scope(module: ModuleType) -> None:
    class WarningSource(_Source):
        def create_mask(self, *, use_existing_mask: bool) -> None:
            warnings.warn(module.EXPECTED_SUPPLIED_MASK_WARNING, UserWarning)
            warnings.warn(
                module.EXPECTED_SUPPLIED_MASK_WARNING, RuntimeWarning
            )
            warnings.warn("visible control", UserWarning)

    module._load_filfinder_runtime = lambda: (
        WarningSource,
        types.SimpleNamespace(pix=_PixelUnit()),
    )
    pools: list[_Pool] = []
    module._create_warning_forwarding_pool = (
        lambda: pools.append(_Pool()) or pools[-1]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module.FilFinderDetector(output="mask").apply(_image())
    if [str(item.message) for item in caught] != [
        module.EXPECTED_SUPPLIED_MASK_WARNING,
        "visible control",
    ]:
        raise AssertionError("warning filter was not exact")
    if [item.category for item in caught] != [RuntimeWarning, UserWarning]:
        raise AssertionError("warning filter was not exact")


def _probe_process_count(module: ModuleType) -> None:
    observed: list[int] = []

    class Pool:
        def __init__(self, *, max_workers: int) -> None:
            observed.append(max_workers)

    module._WarningForwardingProcessPool = Pool
    module._create_warning_forwarding_pool()
    if observed != [1]:
        raise AssertionError("pool did not use exactly one worker")


def _probe_keyed_warning(module: ModuleType) -> None:
    underlying: Future[
        tuple[int, str, int, list[tuple[str, type[Warning], str, int]]]
    ] = Future()
    underlying.set_result(
        (0, "worker", 7, [("warning", UserWarning, __file__, 1)])
    )
    sink: dict[int, dict[str, object]] = {}
    future = module._WarningForwardingFuture(underlying, 0, sink)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if future.result() != 7:
            raise AssertionError("worker result changed")
    if sink[0]["function"] != "worker":
        raise AssertionError("warning record lost its task key")


Probe = Callable[[ModuleType], None]
Mutation = tuple[str, str, str, str, Probe, bool]


MUTATIONS: tuple[Mutation, ...] = (
    (
        "FF-M01",
        "test_private_source_copy_kills_direct_float64_threshold_mutant",
        "quantized = np.asarray(detect_mat, dtype=np.float32)",
        "quantized = np.asarray(detect_mat, dtype=np.float64)",
        _probe_float32,
        False,
    ),
    (
        "FF-M02",
        "test_inclusive_threshold_and_nan_polarity",
        "threshold_mask = source_image >= self.threshold",
        "threshold_mask = source_image > self.threshold",
        _probe_threshold,
        False,
    ),
    (
        "FF-M03",
        "test_source_arguments_are_copied_float64_and_unit_bearing",
        "_copy_float32_source(image.detect_mat[:])",
        "_copy_float32_source(image.gray[:])",
        _probe_layer,
        False,
    ),
    (
        "FF-M04",
        "test_exact_stage_graph_and_pool_shutdown",
        "filfinder.create_mask(use_existing_mask=True)",
        "filfinder.create_mask(use_existing_mask=False)",
        _probe_create_mask,
        False,
    ),
    (
        "FF-M05",
        "test_source_arguments_are_copied_float64_and_unit_bearing",
        "beamwidth=self.beamwidth_px * units.pix",
        "beamwidth=self.beamwidth_px",
        _probe_beam,
        False,
    ),
    (
        "FF-M06",
        "test_source_arguments_are_copied_float64_and_unit_bearing",
        "else self.branch_threshold_px * units.pix",
        "else self.branch_threshold_px",
        _probe_branch,
        False,
    ),
    (
        "FF-M07",
        "test_source_arguments_are_copied_float64_and_unit_bearing",
        "skel_thresh=1.0 * units.pix",
        "skel_thresh=2.0 * units.pix",
        _probe_skel,
        False,
    ),
    (
        "FF-M08",
        "test_source_arguments_are_copied_float64_and_unit_bearing",
        "relintens_thresh=self.relative_intensity_threshold",
        "relintens_thresh=self.prune_criteria",
        _probe_prune,
        False,
    ),
    (
        "FF-M09",
        "test_source_arguments_are_copied_float64_and_unit_bearing",
        "filfinder.medskel(rng=self.rng_seed)",
        "filfinder.medskel(rng=0)",
        _probe_rng,
        False,
    ),
    (
        "FF-M10",
        "test_fresh_source_object_and_pool_per_apply",
        "pool = _create_warning_forwarding_pool()",
        "pool = getattr(self, '_mutation_pool', None)\n        if pool is None:\n            pool = _create_warning_forwarding_pool()\n            object.__setattr__(self, '_mutation_pool', pool)",
        _probe_fresh,
        False,
    ),
    (
        "FF-M11",
        "test_real_filfinder_matches_all_24_selected_oracle_outputs",
        "filfinder.skeleton_longpath,",
        "filfinder.skeleton,",
        _probe_longest,
        False,
    ),
    (
        "FF-M12",
        "test_selected_rasters_use_eight_connectivity_and_row_major_labels",
        "structure=np.ones((3, 3), dtype=np.uint8)",
        "structure=ndimage.generate_binary_structure(2, 1)",
        _probe_connectivity,
        False,
    ),
    (
        "FF-M13",
        "test_selected_rasters_use_eight_connectivity_and_row_major_labels",
        "image.objmask[:] = objmap > 0",
        "image.objmask[:] = threshold_mask",
        _probe_objmask,
        False,
    ),
    (
        "FF-M14",
        "test_source_arguments_are_copied_float64_and_unit_bearing",
        "image.objmap[:] = objmap",
        "image.detect_mat[:] = 0.0\n        image.objmap[:] = objmap",
        _probe_layers,
        False,
    ),
    (
        "FF-M15",
        "test_module_import_is_optional_dependency_free",
        "from scipy import ndimage",
        "from scipy import ndimage\nfrom fil_finder import FilFinder2D as _EAGER_FILFINDER",
        _probe_stages,
        True,
    ),
    (
        "FF-M16",
        "test_nonempty_mask_reports_missing_topology_extra",
        "raise ImportError(_TOPOLOGY_IMPORT_ERROR) from error",
        "return None, None",
        _probe_dependency,
        False,
    ),
    (
        "FF-M17",
        "test_pool_shutdown_is_guaranteed_after_failure",
        "finally:\n            pool.shutdown(wait=True)",
        "finally:\n            pass",
        _probe_shutdown,
        False,
    ),
    (
        "FF-M18",
        "test_exact_stage_graph_and_pool_shutdown",
        'if self.output == "mask":',
        "if False:",
        _probe_stages,
        False,
    ),
    (
        "FF-M19",
        "test_only_exact_supplied_mask_warning_is_suppressed",
        "category=UserWarning,",
        "category=Warning,",
        _probe_warning_scope,
        False,
    ),
    (
        "FF-M20",
        "test_real_process_pool_forwards_keyed_child_warning_to_parent",
        "self._warning_sink[task_index] = {",
        "self._warning_sink[-1] = {",
        _probe_keyed_warning,
        False,
    ),
    (
        "FF-M21",
        "test_real_process_pool_forwards_keyed_child_warning_to_parent",
        "return _WarningForwardingProcessPool(max_workers=1)",
        "return _WarningForwardingProcessPool(max_workers=2)",
        _probe_process_count,
        False,
    ),
)


def execute_mutations() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    digest = hashlib.sha256(source.encode()).hexdigest()
    results: list[tuple[str, str, str]] = []
    with tempfile.TemporaryDirectory(
        prefix="phenotypic-filfinder-mutants-"
    ) as temp:
        directory = Path(temp)
        baseline_path = directory / "baseline.py"
        baseline_path.write_text(source, encoding="utf-8")
        for index, (_, _, _, _, probe, block_import) in enumerate(MUTATIONS):
            if not block_import:
                baseline = _load_module(
                    baseline_path, f"filfinder_mutation_baseline_{index}"
                )
                probe(baseline)

        for mutant_id, test_name, old, new, probe, block_import in MUTATIONS:
            path = directory / f"{mutant_id.lower()}.py"
            path.write_text(_replace_once(source, old, new), encoding="utf-8")
            try:
                if block_import:
                    old_module = sys.modules.get("fil_finder", ...)
                    sys.modules["fil_finder"] = None
                    try:
                        _load_module(path, f"filfinder_{mutant_id.lower()}")
                    finally:
                        if old_module is ...:
                            sys.modules.pop("fil_finder", None)
                        else:
                            sys.modules["fil_finder"] = old_module
                    raise AssertionError("eager optional import mutant loaded")
                mutant = _load_module(path, f"filfinder_{mutant_id.lower()}")
                probe(mutant)
            except Exception as error:
                results.append((mutant_id, test_name, type(error).__name__))
            else:
                raise AssertionError(f"{mutant_id} survived {test_name}")

    restored = SOURCE.read_text(encoding="utf-8")
    if (
        restored != source
        or hashlib.sha256(restored.encode()).hexdigest() != digest
    ):
        raise AssertionError("production source changed during mutation run")
    if len(results) != len(MUTATIONS):
        raise AssertionError("not every mutant produced a result")
    for mutant_id, test_name, reason in results:
        print(f"{mutant_id}: KILLED by {test_name} ({reason})")
    print(f"Baseline restored: sha256={digest}")


if __name__ == "__main__":
    execute_mutations()
