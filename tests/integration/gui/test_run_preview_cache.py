"""End-to-end exercise of the builder preview-cache contract.

Drives :func:`phenotypic.gui.builder._callbacks._bake_preview_cache` against
a small ``BuilderState`` (``GaussianBlur → OtsuDetector → MeasureSize``) and
asserts:

- Cache slots for ops nodes hold raw PNG :class:`bytes`.
- Cache slots for measurement / post nodes hold a :class:`pandas.DataFrame`.
- No :class:`Image` / :class:`GridImage` instance is referenced from the
  cache afterward (verified via ``gc.get_referents`` walk).
"""

from __future__ import annotations

import gc
import weakref
from pathlib import Path

import pandas as pd
import pytest

from phenotypic.data._synthetic_data import load_synth_yeast_plate
from phenotypic.gui.builder._callbacks import _bake_preview_cache
from phenotypic.gui.builder._session import IntermediatesCache, PreviewRenderError

# The public ``BuilderScope`` / ``BuilderState`` names are permanent aliases
# for the DAG schema.  The legacy-path tests below exercise
# ``_bake_preview_cache_legacy``, which still walks the linear-list model, so
# they bind the ``_Legacy*`` types directly (same pattern as
# ``test_doc_section.py`` / ``test_state_dataclasses.py``).  ``to_pipeline``
# already operates on the legacy scope.
from phenotypic.gui.builder._state import (
    _LegacyBuilderScope as BuilderScope,
    _LegacyBuilderState as BuilderState,
    _LegacyStepNode as StepNode,
    to_pipeline,
)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _seed_pipeline_state() -> BuilderState:
    """A minimal three-step state spanning ops + meas."""
    return BuilderState(
        root=BuilderScope(
            nodes=[
                StepNode(node_id="aaa", class_name="GaussianBlur",
                         params={"sigma": 1.0}, label="GaussianBlur"),
                StepNode(node_id="bbb", class_name="OtsuDetector",
                         label="OtsuDetector"),
                StepNode(node_id="ccc", class_name="MeasureSize",
                         label="MeasureSize"),
            ],
            name="bake-test",
        ),
    )


def test_bake_preview_cache_populates_bytes_for_ops():
    state = _seed_pipeline_state()
    pipeline = to_pipeline(state.root)
    image = load_synth_yeast_plate()
    result = pipeline.apply_with_intermediates(image)

    cache = IntermediatesCache()
    _bake_preview_cache(state, pipeline, result, "sess-1", cache)

    enhancer_blob = cache.get_intermediate("sess-1", "aaa")
    detector_blob = cache.get_intermediate("sess-1", "bbb")
    assert isinstance(enhancer_blob, bytes)
    assert isinstance(detector_blob, bytes)
    assert enhancer_blob[:8] == PNG_MAGIC
    assert detector_blob[:8] == PNG_MAGIC


def test_bake_preview_cache_stores_dataframe_for_measurements():
    state = _seed_pipeline_state()
    pipeline = to_pipeline(state.root)
    result = pipeline.apply_with_intermediates(load_synth_yeast_plate())

    cache = IntermediatesCache()
    _bake_preview_cache(state, pipeline, result, "sess-1", cache)

    meas_payload = cache.get_intermediate("sess-1", "ccc")
    assert isinstance(meas_payload, pd.DataFrame)
    assert meas_payload.shape[0] > 0  # type: ignore[union-attr]


def test_bake_preview_cache_holds_no_image_references():
    """The cache must drop intermediate Images after rendering — only bytes.

    Uses a weakref against each :class:`Image` produced by
    :meth:`apply_with_intermediates`: after the cache is baked and our local
    references are dropped, the weakrefs must dereference to ``None``. If the
    cache (or any code reachable from it) still holds an intermediate alive,
    the corresponding ``ref()`` returns the live object and the test fails.
    """
    state = _seed_pipeline_state()
    pipeline = to_pipeline(state.root)
    result = pipeline.apply_with_intermediates(load_synth_yeast_plate())

    intermediate_refs = [
        weakref.ref(img)
        for img in result.intermediates.values()
        if img is not None
    ]
    assert intermediate_refs, "expected at least one intermediate Image"

    cache = IntermediatesCache()
    _bake_preview_cache(state, pipeline, result, "sess-1", cache)

    # Drop our local refs and force collection; only the cache could be
    # holding them now.
    del result
    gc.collect()

    live = [ref() for ref in intermediate_refs if ref() is not None]
    assert not live, (
        f"IntermediatesCache still holds {len(live)} intermediate Image(s); "
        "pre-baking should have dropped them after PNG encoding."
    )


def test_bake_preview_cache_renders_independently_of_run_preview_callback():
    """Smoke: same logic the callback invokes; no Dash app needed."""
    state = _seed_pipeline_state()
    pipeline = to_pipeline(state.root)
    result = pipeline.apply_with_intermediates(load_synth_yeast_plate())

    cache = IntermediatesCache()
    _bake_preview_cache(state, pipeline, result, "sess-2", cache)

    keys = cache.known_intermediate_keys("sess-2")
    assert set(keys) == {"aaa", "bbb", "ccc"}
    # Spot-check no PreviewRenderError leaked through on the happy path.
    for key in keys:
        assert not isinstance(
            cache.get_intermediate("sess-2", key), PreviewRenderError
        )


def test_failed_bake_cannot_publish_partial_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A staging failure leaves the prior complete generation untouched."""
    from phenotypic.gui.builder import _callbacks

    state = _seed_pipeline_state()
    pipeline = to_pipeline(state.root)
    result = pipeline.apply_with_intermediates(load_synth_yeast_plate())
    cache = IntermediatesCache()
    generation = _bake_preview_cache(
        state,
        pipeline,
        result,
        "sess-atomic",
        cache,
        pipeline_revision="revision-1",
    )
    old_keys = cache.known_intermediate_keys("sess-atomic")

    def _stage_then_fail(
        _state: object,
        _pipeline: object,
        _result: object,
        session_id: str,
        staging: object,
    ) -> None:
        staging.set_intermediate(session_id, "aaa", b"partial")  # type: ignore[attr-defined]
        raise RuntimeError("bake interrupted")

    monkeypatch.setattr(
        _callbacks,
        "_bake_preview_cache_legacy",
        _stage_then_fail,
    )

    with pytest.raises(RuntimeError, match="bake interrupted"):
        _bake_preview_cache(
            state,
            pipeline,
            result,
            "sess-atomic",
            cache,
            pipeline_revision="revision-2",
        )

    assert cache.preview_descriptor("sess-atomic") == ("revision-1", generation)
    assert cache.known_intermediate_keys("sess-atomic") == old_keys
    assert cache.get_preview(
        ("sess-atomic", "aaa", "revision-1", generation)
    ) != b"partial"


def test_measurement_failure_publishes_complete_error_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recoverable render failures still produce a complete node-key snapshot."""
    state = _seed_pipeline_state()
    pipeline = to_pipeline(state.root)
    result = pipeline.apply_with_intermediates(load_synth_yeast_plate())
    cache = IntermediatesCache()

    def _measure_failure(_self: object, _image: object) -> object:
        raise RuntimeError("measurement preview failed")

    monkeypatch.setattr(type(pipeline), "measure", _measure_failure)
    _bake_preview_cache(state, pipeline, result, "sess-measure-error", cache)

    assert set(cache.known_intermediate_keys("sess-measure-error")) == {
        "aaa",
        "bbb",
        "ccc",
    }
    cached = cache.get_intermediate("sess-measure-error", "ccc")
    assert isinstance(cached, PreviewRenderError)
    assert "measurement preview failed" in cached.message


# ---------------------------------------------------------------------------
# DAG-path regression: Phase 7 cache key migration
# ---------------------------------------------------------------------------
#
# Phase 7 retired the popover-era flow and split ``_bake_preview_cache`` into
# a legacy and a DAG branch (see :func:`_bake_preview_cache_dag`).  The DAG
# path must key the cache by the 32-char ``BlockNode.block_id`` (not the
# legacy 8-char ``StepNode.node_id`` slice) so the inspector's block-selection
# lookup hits without an id-translation step.  These tests pin that invariant.


def test_bake_preview_cache_dag_uses_32char_block_id_keys():
    """DAG path keys the cache by ``BlockNode.block_id`` (32-char hex).

    Regression test for the Phase 7 cache key migration. A DAG state run
    through ``_bake_preview_cache`` must populate cache slots keyed by
    the *full* ``block_id`` (not a truncated slice, not the class name,
    not the ``InputImage`` block which has no main-flow intermediate).
    """
    import json

    from phenotypic.gui.builder._conversion_dag import to_pipeline_dag
    from phenotypic.gui.builder._state import (
        INPUT_IMAGE_CLASS_NAME,
        state_from_json,
    )

    # ``linear_chain.json`` is a fully wired DAG fixture:
    # InputImage -> GaussianBlur -> OtsuDetector -> MeasureSize.
    fixture = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "builder_dag"
        / "linear_chain.json"
    )
    state = state_from_json(json.loads(fixture.read_text(encoding="utf-8")))
    pipeline = to_pipeline_dag(state)
    result = pipeline.apply_with_intermediates(load_synth_yeast_plate())

    cache = IntermediatesCache()
    _bake_preview_cache(state, pipeline, result, "sess-dag", cache)

    keys = set(cache.known_intermediate_keys("sess-dag"))

    # InputImage has no main-flow intermediate (Phase 7 skip rule).
    input_ids = {
        b.block_id for b in state.root.blocks
        if b.class_name == INPUT_IMAGE_CLASS_NAME
    }
    expected_block_ids = {
        b.block_id for b in state.root.blocks
        if b.class_name != INPUT_IMAGE_CLASS_NAME
    }

    assert keys == expected_block_ids, (
        f"DAG cache keys {keys} did not equal non-input block_ids "
        f"{expected_block_ids}"
    )
    # All DAG keys are 32-char lowercase hex (uuid4().hex contract).
    for key in keys:
        assert len(key) == 32, f"DAG cache key {key!r} is not 32 chars"
        assert all(c in "0123456789abcdef" for c in key), (
            f"DAG cache key {key!r} is not lowercase hex"
        )
    # And the InputImage block_id is *not* in the cache.
    assert keys.isdisjoint(input_ids)


def test_bake_preview_cache_legacy_uses_8char_node_id_keys():
    """Legacy path still keys the cache by 8-char ``StepNode.node_id``.

    Confirms ``_bake_preview_cache`` duck-types correctly: a legacy
    state (no ``selected_block_id`` attribute) routes to the legacy
    branch whose keys remain the short node_ids.
    """
    state = _seed_pipeline_state()
    assert not hasattr(state, "selected_block_id"), (
        "Test premise: legacy state must lack ``selected_block_id``."
    )

    pipeline = to_pipeline(state.root)
    result = pipeline.apply_with_intermediates(load_synth_yeast_plate())
    cache = IntermediatesCache()
    _bake_preview_cache(state, pipeline, result, "sess-legacy", cache)

    keys = set(cache.known_intermediate_keys("sess-legacy"))
    # Legacy keys are the 3-char node_ids from ``_seed_pipeline_state``.
    assert keys == {"aaa", "bbb", "ccc"}
