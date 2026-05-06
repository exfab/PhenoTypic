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

import pandas as pd

from phenotypic.data._synthetic_data import load_synth_yeast_plate
from phenotypic.gui.builder._callbacks import _bake_preview_cache
from phenotypic.gui.builder._session import IntermediatesCache, PreviewRenderError
from phenotypic.gui.builder._state import (
    BuilderScope,
    BuilderState,
    StepNode,
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
