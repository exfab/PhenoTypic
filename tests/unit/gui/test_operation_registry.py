"""Unit tests for OperationRegistry and ParamInfo."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import List, Optional

import pytest

from phenotypic.abc_ import ObjectDetector
from phenotypic.enhance import BlurGauss
from phenotypic._gui import _operation_registry
from phenotypic._gui._operation_registry import (
    OperationRegistry,
    ParamInfo,
    OperationInfo,
    get_registry,
)


class TestParamInfo:
    """Test ParamInfo dataclass."""

    def test_param_info_creation(self):
        """Test creating ParamInfo."""
        info = ParamInfo(
                name="sigma",
                type_hint=float,
                default=1.0,
                has_default=True,
                is_operation=False,
                is_pipeline=False,
                is_optional=False,
        )
        assert info.name == "sigma"
        assert info.type_hint is float
        assert info.default == 1.0
        assert info.has_default is True


class TestOperationInfo:
    """Test OperationInfo dataclass."""

    def test_operation_info_creation(self):
        """Test creating OperationInfo."""
        info = OperationInfo(
                cls=BlurGauss,
                name="BlurGauss",
                category="Enhancer",
                module="phenotypic.enhance",
                docstring="Test docstring",
                parameters={},
        )
        assert info.cls is BlurGauss
        assert info.name == "BlurGauss"
        assert info.category == "Enhancer"


class TestOperationRegistry:
    """Test OperationRegistry functionality."""

    @pytest.fixture
    def registry(self):
        """Create fresh registry for each test."""
        reg = OperationRegistry()
        reg.discover()
        return reg

    def test_discover_operations(self, registry):
        """Test that operations are discovered."""
        assert len(registry._operations) > 0
        assert "BlurGauss" in registry._operations
        assert "OtsuDetector" in registry._operations

    def test_get_categories(self, registry):
        """Test getting operation categories."""
        categories = registry.get_categories()
        assert "Enhancer" in categories
        assert "Detector" in categories
        assert isinstance(categories, list)
        # Should be sorted
        assert categories == sorted(categories)

    def test_get_by_category(self, registry):
        """Test getting operations by category."""
        enhancers = registry.get_by_category("Enhancer")
        assert len(enhancers) > 0
        assert all(info.category == "Enhancer" for info in enhancers)

        # Check specific operations
        enhancer_names = [info.name for info in enhancers]
        assert "BlurGauss" in enhancer_names
        assert "EnhanceLocalContrast" in enhancer_names

    def test_legacy_bm3d_alias_not_registered(self, registry):
        """Deserializer aliases must not appear as duplicate palette entries."""
        assert registry.get("EnhanceBlockMatch") is not None
        assert registry.get("BM3DDenoiser") is None

    def test_get_operation(self, registry):
        """Test getting specific operation by name."""
        info = registry.get("BlurGauss")
        assert info is not None
        assert info.cls is BlurGauss
        assert info.name == "BlurGauss"
        assert info.category == "Enhancer"

    def test_get_nonexistent_operation(self, registry):
        """Test getting operation that doesn't exist."""
        info = registry.get("NonexistentOperation")
        assert info is None

    def test_get_all_operations(self, registry):
        """Test getting all operations."""
        all_ops = registry.get_all()
        assert isinstance(all_ops, dict)
        assert len(all_ops) > 0
        assert "BlurGauss" in all_ops

    def test_create_instance(self, registry):
        """Test creating operation instance."""
        blur = registry.create_instance("BlurGauss")
        assert isinstance(blur, BlurGauss)

        # With parameters
        blur2 = registry.create_instance("BlurGauss", sigma=2.5)
        assert blur2.sigma == 2.5

    def test_create_instance_nonexistent(self, registry):
        """Test creating instance of nonexistent operation."""
        with pytest.raises(KeyError):
            registry.create_instance("NonexistentOperation")

    def test_extract_parameters(self, registry):
        """Test parameter extraction."""
        info = registry.get("BlurGauss")
        assert "sigma" in info.parameters

        param_info = info.parameters["sigma"]
        assert param_info.name == "sigma"
        assert param_info.has_default is True

    def test_extract_parameters_with_operations(self, registry):
        """Test parameter extraction for operations with nested operation params."""
        # FilamentousFungiDetector has operation parameters
        info = registry.get("FilamentousFungiDetector")
        if info:  # Only test if this detector exists
            assert "inoculum_detector" in info.parameters
            param = info.parameters["inoculum_detector"]
            assert param.is_operation or param.is_pipeline
            assert param.is_optional  # It's Union[..., None]

    def test_global_registry(self):
        """Test global registry singleton."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2  # Should be same instance

    def test_operation_has_docstring(self, registry):
        """Test that operations include docstrings."""
        info = registry.get("BlurGauss")
        assert info.docstring is not None
        assert len(info.docstring) > 0

    def test_operation_module_path(self, registry):
        """Test that module paths are captured."""
        info = registry.get("BlurGauss")
        assert "phenotypic" in info.module
        assert "enhance" in info.module


class TestPointPickerMarker:
    """Operations that mix in ``PointPickerMixin`` are flagged in the registry.

    The Dash builder uses these flags to swap a free-form text input for an
    interactive point picker on the matching parameter.
    """

    @pytest.fixture
    def registry(self):
        reg = OperationRegistry()
        reg.discover()
        return reg

    def test_point_picker_marker_propagates(self, registry):
        """ManualPointDetector and ManualRefine advertise the mixin marker."""
        det = registry.get("ManualPointDetector")
        assert det is not None
        assert det.is_point_pickable is True
        assert det.point_picker_param == "centers"

        sel = registry.get("ManualRefine")
        assert sel is not None
        assert sel.is_point_pickable is True
        assert sel.point_picker_param == "centers"

    def test_non_pickable_ops_have_falsy_marker(self, registry):
        """Operations without the mixin do not gain a stray pickable flag."""
        otsu = registry.get("OtsuDetector")
        assert otsu is not None
        assert otsu.is_point_pickable is False
        assert otsu.point_picker_param is None

        blur = registry.get("BlurGauss")
        assert blur is not None
        assert blur.is_point_pickable is False
        assert blur.point_picker_param is None

    def test_threshold_based_manual_detector_is_not_pickable(self, registry):
        """UserThreshold takes a scalar threshold, not points — not flagged."""
        man = registry.get("UserThreshold")
        assert man is not None
        assert man.is_point_pickable is False
        assert man.point_picker_param is None


class TestColumnRefDetection:
    """`_extract_parameters` populates `ParamInfo.column_ref` from `Annotated`."""

    @pytest.fixture(scope="class")
    def registry(self):
        reg = OperationRegistry()
        reg.discover()
        return reg

    @pytest.mark.parametrize(
            "cls_name,param_name,expected_multi",
            [
                ("EdgeCorrector", "on", False),
                ("EdgeCorrector", "groupby", True),
                ("EdgeCorrector", "time_label", False),
                ("TukeyOutlierRemover", "on", False),
                ("TukeyOutlierRemover", "groupby", True),
                ("LogGrowthModel", "on", False),
                ("LogGrowthModel", "groupby", True),
                ("LogGrowthModel", "time_label", False),
                ("LinearLagModel", "on", False),
                ("LinearLagModel", "groupby", True),
                ("LinearLagModel", "time_label", False),
                ("LinearCapAndLagModel", "on", False),
                ("LinearCapAndLagModel", "groupby", True),
                ("LinearCapAndLagModel", "time_label", False),
            ],
    )
    def test_column_ref_populated(
            self, registry, cls_name, param_name, expected_multi
    ):
        info = registry.get(cls_name)
        assert info is not None, f"{cls_name} not registered"
        p = info.parameters.get(param_name)
        assert p is not None, f"{cls_name}.{param_name} missing"
        assert p.column_ref is not None, (
            f"{cls_name}.{param_name} has no column_ref"
        )
        assert p.column_ref.source == "measurements"
        assert p.column_ref.multi is expected_multi
        assert p.column_ref.with_alt is False

    def test_kmax_label_is_column_ref_with_alt(self, registry):
        """`Kmax_label: ColumnRef | None` — the alt branch flips with_alt."""
        info = registry.get("LogGrowthModel")
        assert info is not None
        p = info.parameters.get("Kmax_label")
        assert p is not None
        assert p.column_ref is not None
        assert p.column_ref.source == "measurements"
        assert p.column_ref.multi is False
        assert p.column_ref.with_alt is True

    def test_non_column_params_have_no_column_ref(self, registry):
        info = registry.get("EdgeCorrector")
        assert info is not None
        for name in ("nrows", "ncols", "top_n", "pvalue"):
            p = info.parameters.get(name)
            assert p is not None
            assert p.column_ref is None, f"{name} should not be a column ref"

    def test_non_analyzer_op_has_no_column_ref(self, registry):
        """Builder-side operations don't carry the marker."""
        blur = registry.get("BlurGauss")
        assert blur is not None
        for p in blur.parameters.values():
            assert p.column_ref is None


class TestIsListDetection:
    """`_extract_parameters` populates ``ParamInfo.is_list`` for list-typed params.

    The flag distinguishes list-typed aux ports (e.g.
    ``CompositeDetector.ops: List[Union[ObjectDetector, ImagePipeline]]``)
    from scalar variants (e.g.
    ``FilamentousFungiDetector.inoculum_detector: Union[ObjectDetector,
    ImagePipeline, None]``) so the GUI builder can render multi-port
    ``+``/``×`` controls only on list slots.
    """

    @pytest.fixture(scope="class")
    def registry(self):
        reg = OperationRegistry()
        reg.discover()
        return reg

    def test_composite_detector_detectors_is_list(self, registry):
        """``CompositeDetector.ops: List[Union[Op, Pipeline]]``."""
        info = registry.get("CompositeDetector")
        assert info is not None
        p = info.parameters.get("ops")
        assert p is not None
        assert p.is_list is True
        assert p.is_operation is True
        assert p.is_pipeline is True
        assert p.is_optional is False

    def test_filamentous_inoculum_is_scalar_optional(self, registry):
        """``inoculum_detector: Union[Op, Pipeline, None]`` is scalar+optional."""
        info = registry.get("FilamentousFungiDetector")
        assert info is not None
        p = info.parameters.get("inoculum_detector")
        assert p is not None
        assert p.is_list is False
        assert p.is_operation is True
        assert p.is_pipeline is True
        assert p.is_optional is True

    def test_optional_list_of_operations(self):
        """``Optional[List[ObjectDetector]]`` peels both wrappers."""

        class _SyntheticOptionalListOp:
            def __init__(self, param: Optional[List[ObjectDetector]] = None):
                self.param = param

        reg = OperationRegistry()
        params = reg._extract_parameters(_SyntheticOptionalListOp)
        p = params["param"]
        assert p.is_list is True
        assert p.is_operation is True
        assert p.is_optional is True

    def test_bare_list_no_args(self):
        """Bare ``list`` annotation flags ``is_list`` without op/pipeline."""

        class _SyntheticBareListOp:
            def __init__(self, param: list = []):
                self.param = param

        reg = OperationRegistry()
        params = reg._extract_parameters(_SyntheticBareListOp)
        p = params["param"]
        assert p.is_list is True
        assert p.is_operation is False
        assert p.is_pipeline is False

    def test_scalar_param_is_not_list(self, registry):
        """Scalar params keep ``is_list=False`` (regression guard)."""
        blur = registry.get("BlurGauss")
        assert blur is not None
        for p in blur.parameters.values():
            assert p.is_list is False


class TestQualityCheckCategory:
    """`_discover_analyzers` routes QualityCheck subclasses to a dedicated category.

    Per spec §1278–1283, ``QualityCheck`` subclasses (e.g.
    ``ExpectedVsDetectedCount``, ``ReplicateAgreement``) must land under
    the ``"quality_check"`` category so the QC tab's add-check dropdown
    has something to render. Without the explicit branch, the
    fall-through default would mis-route them into ``"Filter"``.

    Per spec §1419–1428, ``OperationRegistry._extract_parameters`` skips
    the inherited ``agg_func`` parameter on QC subclasses that opt out
    via ``_exposes_agg_func: ClassVar[bool] = False`` (the default for
    every v1 check) so the param form doesn't surface an unused dropdown.
    Backward-compat: analyzers without the attribute (``EdgeCorrector``,
    ``LogGrowthModel``) keep their ``agg_func`` parameter exposed.
    """

    @pytest.fixture(scope="class")
    def registry(self):
        reg = OperationRegistry()
        reg.discover()
        return reg

    def test_quality_check_subclasses_get_quality_check_category(self, registry):
        """``ExpectedVsDetectedCount`` and ``ReplicateAgreement`` register here."""
        qc_ops = registry.get_by_category("quality_check")
        qc_names = {info.name for info in qc_ops}
        assert "ExpectedVsDetectedCount" in qc_names
        assert "ReplicateAgreement" in qc_names
        # Each registered op must carry the matching category attribute.
        for info in qc_ops:
            assert info.category == "quality_check"

    def test_quality_check_classes_excluded_from_filter_or_model_categories(
            self, registry
    ):
        """QC classes must not leak into ``"Filter"`` or ``"Model"`` buckets."""
        filter_names = {info.name for info in registry.get_by_category("Filter")}
        model_names = {info.name for info in registry.get_by_category("Model")}
        assert "ExpectedVsDetectedCount" not in filter_names
        assert "ExpectedVsDetectedCount" not in model_names
        assert "ReplicateAgreement" not in filter_names
        assert "ReplicateAgreement" not in model_names

    def test_quality_check_base_class_itself_not_registered(self, registry):
        """The abstract ``QualityCheck`` ABC is excluded from every category."""
        all_ops = registry.get_all()
        assert "QualityCheck" not in all_ops
        for category in registry.get_categories():
            names = {info.name for info in registry.get_by_category(category)}
            assert "QualityCheck" not in names

    def test_quality_check_params_omit_agg_func_when_exposes_agg_func_is_false(
            self, registry
    ):
        """``_exposes_agg_func=False`` filters ``agg_func`` out of params."""
        info = registry.get("ExpectedVsDetectedCount")
        assert info is not None
        assert "agg_func" not in info.parameters

    def test_non_quality_check_analyzers_still_expose_agg_func(self, registry):
        """Backward-compat: analyzers without the flag keep ``agg_func``."""
        info = registry.get("EdgeCorrector")
        assert info is not None
        assert "agg_func" in info.parameters


class TestEdgeCorrectionCategory:
    def test_edge_corrector_is_edge_category(self):
        reg = OperationRegistry()
        reg.discover()
        info = reg.get("EdgeCorrector")
        assert info is not None
        assert info.category == "Edge Correction"
        filter_names = {i.name for i in reg.get_by_category("Filter")}
        model_names = {i.name for i in reg.get_by_category("Model")}
        assert "EdgeCorrector" not in filter_names
        assert "EdgeCorrector" not in model_names
        assert "EdgeCorrector" in {i.name for i in reg.get_by_category("Edge Correction")}


class TestGetRegistryConcurrentFirstCall:
    """``get_registry()`` under two simultaneous first callers.

    The builder and analysis sub-apps both build lazily on their first request,
    so two Flask worker threads can reach :func:`get_registry` at the same
    time, and ``discover()`` imports the whole operation library, so the
    construction window is ~1s wide. Three properties have to hold together,
    and the tests below are written so that each one fails on its own:

    1. **Always complete** -- nobody ever receives a registry with zero
       operations. Publishing the singleton before ``discover()`` populates it
       hands the second caller *the same object*, empty, and it renders its
       layout from that: empty dropdowns, no exception, no log line.
    2. **Exactly one instance** -- ``id(get_registry())`` never changes.
       ``_resolve_dag_accepts_for_class_port`` (``builder/_layout.py``) is an
       ``lru_cache`` keyed on ``id(registry)`` that falls back to an uncached
       per-port walk on every render when the ids diverge, so a fix that
       publishes without a first-writer-wins check trades a visible bug for a
       silent permanent slowdown.
    3. **No lock held across imports** -- ``discover()`` runs outside the lock,
       so a second caller is never blocked behind another thread's imports.

    A fix that satisfies (1) but not (2) is the one most likely to be written
    by accident, which is why identity is asserted and not merely assumed.
    """

    @pytest.fixture
    def isolated_singleton(self):
        """Clear the process-wide singleton for one test, then restore it."""
        saved = _operation_registry._REGISTRY
        _operation_registry._REGISTRY = None
        try:
            yield
        finally:
            _operation_registry._REGISTRY = saved

    @pytest.fixture
    def discover_window(self, monkeypatch):
        """Hold the *first* ``discover()`` call open until the test releases it.

        **Only the first call is parked, and that is load-bearing -- do not
        "simplify" this to park every call.** Doing so makes the correct
        implementation and the broken one indistinguishable, *in the wrong
        direction*: because ``get_registry()`` builds outside the lock, a
        second first-time caller runs its own ``discover()``, so parking every
        call parks the second caller too. The tests below would then observe it
        blocked and conclude it had been made to wait -- failing the correct
        implementation while passing a publish-without-first-writer-wins one,
        which never blocks anybody.

        Parking only the first call is what makes the interleaving
        deterministic in both directions: the first caller is pinned
        mid-construction and cannot have published anything, while a second
        caller that builds its own registry runs to completion inside that
        window instead of deadlocking against it. The identity assertion in
        particular is racy under the park-everything design, because both
        threads resume at the same instant and their ``_REGISTRY`` reads
        interleave arbitrarily.
        """
        entered = threading.Event()
        release = threading.Event()
        calls: List[OperationRegistry] = []
        counter_lock = threading.Lock()
        real_discover = OperationRegistry.discover

        def parked_discover(registry: OperationRegistry) -> None:
            with counter_lock:
                calls.append(registry)
                is_first = len(calls) == 1
            if is_first:
                entered.set()
                assert release.wait(timeout=60.0), (
                    "the test never released the first discover() call"
                )
            real_discover(registry)

        monkeypatch.setattr(OperationRegistry, "discover", parked_discover)
        try:
            yield SimpleNamespace(entered=entered, release=release, calls=calls)
        finally:
            # Never leave a worker parked if the test failed mid-way.
            release.set()

    @staticmethod
    def _race_a_second_caller_into_the_construction_window(window) -> dict:
        """Run two first-time callers with the second landing mid-construction.

        Ordering is forced with events, never with sleeps: the second caller is
        started only after the first has signalled from inside ``discover()``,
        and the first cannot leave ``discover()`` until this function releases
        it. Everything the second caller does therefore happens while the
        first has published nothing.
        """
        observed: dict = {}
        second_calling = threading.Event()
        second_returned = threading.Event()

        def first_caller() -> None:
            observed["first"] = get_registry()

        def second_caller() -> None:
            second_calling.set()
            registry = get_registry()
            observed["second"] = registry
            # Snapshot the size at the instant of return. Reading it later,
            # from the main thread, is how this kind of test goes vacuous.
            observed["second_size"] = len(registry.get_all())
            second_returned.set()

        one = threading.Thread(target=first_caller, name="registry-first")
        two = threading.Thread(target=second_caller, name="registry-second")

        one.start()
        assert window.entered.wait(timeout=60.0), (
            "the first caller never entered discover()"
        )

        two.start()
        assert second_calling.wait(timeout=60.0), (
            "the second caller never reached get_registry()"
        )
        observed["second_returned_inside_window"] = second_returned.wait(timeout=60.0)

        window.release.set()
        one.join(timeout=60.0)
        two.join(timeout=60.0)
        assert not one.is_alive(), "the first caller never finished"
        assert not two.is_alive(), "the second caller never finished"
        return observed

    def test_a_second_caller_never_observes_an_unpopulated_registry(
            self, isolated_singleton, discover_window
    ):
        """Property 1: whatever the second caller gets, it is fully discovered."""
        observed = self._race_a_second_caller_into_the_construction_window(
                discover_window
        )

        assert observed.get("second_size", 0) > 0, (
            "the second caller observed a registry carrying 0 operations: "
            "get_registry() published _REGISTRY before discover() populated it"
        )
        assert observed["second_size"] == len(observed["second"].get_all()), (
            "the registry grew after the second caller had already received it"
        )

    def test_a_concurrent_first_call_does_not_swap_the_published_instance(
            self, isolated_singleton, discover_window
    ):
        """Property 2: first writer wins, so ``id(get_registry())`` is stable.

        Both callers build their own registry here -- the first is parked, so
        it cannot have published anything the second could reuse. Exactly one
        of those two objects may reach ``_REGISTRY``, and both callers must
        come away holding it. A fix that publishes unconditionally fails this
        while passing the completeness test above.

        **Why the failure is deterministic rather than a coin flip**, which is
        the whole reason the fixture parks only the first ``discover()``: the
        second caller finds ``_REGISTRY is None`` because the first is parked
        and has published nothing, so it builds its own and publishes it via
        the *unparked* second ``discover()`` call. Only then does the release
        let the first caller finish and overwrite ``_REGISTRY`` with its own
        object. The two objects therefore differ **by construction**, and the
        write order is fixed by the release rather than by scheduling luck.
        Park every ``discover()`` instead and both threads resume at the same
        instant, their ``_REGISTRY`` reads interleave arbitrarily, and this
        assertion starts passing or failing at random.
        """
        observed = self._race_a_second_caller_into_the_construction_window(
                discover_window
        )

        assert observed["first"] is observed["second"], (
            "the two callers came away with different registry instances: "
            "id(get_registry()) is not stable, which silently demotes the "
            "id(registry)-keyed lru_cache in builder/_layout.py to a "
            "per-render walk"
        )
        assert _operation_registry._REGISTRY is observed["first"], (
            "the published singleton is neither of the objects handed out"
        )
        assert len(discover_window.calls) <= 2, (
            f"discover() ran {len(discover_window.calls)} times; at most one "
            "duplicate build is expected from a two-caller race"
        )

    def test_the_second_caller_is_not_blocked_behind_the_first_ones_imports(
            self, isolated_singleton, discover_window
    ):
        """Property 3: ``discover()`` runs outside the lock.

        The first caller is parked inside ``discover()`` for the whole window,
        so a second caller that completes within it cannot have been waiting on
        a lock held across those imports.

        This test deliberately **forbids** wrapping ``discover()`` in
        ``with _REGISTRY_LOCK:``. That form is otherwise correct and is the
        obvious simplification, which is exactly why it needs a test: building
        outside the lock is a decision with a stated reason (see property 3 in
        ``get_registry``'s docstring -- the lock's safety would otherwise
        depend on eight operation packages never importing ``phenotypic._gui``,
        with nothing watching), not an accident to be tidied away.
        """
        observed = self._race_a_second_caller_into_the_construction_window(
                discover_window
        )

        assert observed["second_returned_inside_window"], (
            "the second caller was still blocked while the first sat inside "
            "discover(): the registry lock is being held across imports"
        )

    def test_repeated_calls_return_one_stable_instance(self, isolated_singleton):
        """The uncontended path: same object, populated, and recorded globally."""
        first = get_registry()
        second = get_registry()
        assert first is second
        assert _operation_registry._REGISTRY is first
        assert first.get_all(), "the published singleton must be populated"
