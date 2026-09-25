"""The output-semantics revision in the process-mode work-id digest.

Spec ``2026-09-15-nested-gpu-staging`` §8.3: ``--mode process --layer objmap``
changed what it exports, and nothing in the work id tracked PhenoTypic's own
output semantics. A run interrupted before the change and resumed after it would
treat old-semantics PNGs as complete and publish a tree mixing two meanings.

**What would make these assertions pass on broken code.** Two shapes, and each
is ruled out by a control in the same test rather than by inspection:

* ``before != after`` is satisfied by any non-deterministic digest, whatever the
  revision does -- so :func:`test_the_digest_is_deterministic` pins that the
  same inputs hash the same twice.
* ``before == after`` is satisfied by a digest that ignores its arguments
  entirely -- so the full-run test also asserts that a real change to the base
  payload DOES move that digest.
"""

from phenotypic._cli import _cli_failure_tracker as tracker

_PROCESS_KWARGS = dict(
    image_type="Image",
    nrows=8,
    ncols=12,
    bit_depth=None,
    detect_mode="gray",
    process_only_layer="objmap",
    ext=".png",
    process_format="tiff",
    include_dataset_column=True,
    overlay_alpha=0.5,
    save_overlays=False,
)

_FULL_KWARGS = dict(_PROCESS_KWARGS, process_only_layer=None)


def _digest_at_revision(revision, **kwargs):
    """Hash *kwargs* with the module constant temporarily set to *revision*.

    Set and restored by hand rather than through ``monkeypatch`` so the
    restoration is visible at the call site; the constant is read from the
    module global inside ``processing_configuration_digest_from_values``, which
    is what makes the substitution take effect at all.
    """
    original = tracker.PROCESS_LAYER_SEMANTICS_REVISION
    try:
        tracker.PROCESS_LAYER_SEMANTICS_REVISION = revision
        return tracker.processing_configuration_digest_from_values(**kwargs)
    finally:
        tracker.PROCESS_LAYER_SEMANTICS_REVISION = original


def test_the_digest_is_deterministic():
    """Control for every ``before != after`` assertion in this file.

    Without this, a digest that folded in a timestamp or an object id would
    satisfy "bumping the revision changes the digest" while the revision was
    being ignored outright.
    """
    first = tracker.processing_configuration_digest_from_values(
        **_PROCESS_KWARGS
    )
    second = tracker.processing_configuration_digest_from_values(
        **_PROCESS_KWARGS
    )
    assert first == second


def test_bumping_the_revision_changes_the_digest():
    shipped = tracker.PROCESS_LAYER_SEMANTICS_REVISION
    before = _digest_at_revision(shipped, **_PROCESS_KWARGS)
    after = _digest_at_revision(shipped + 1, **_PROCESS_KWARGS)

    assert before != after, (
        "the digest ignores the semantics revision, so a process run resumed "
        "across an output-semantics change would reuse stale outputs"
    )


def test_a_full_run_digest_is_UNCHANGED_by_the_bump():
    """The change is scoped to process mode; full/measure must not cold-start.

    Putting the revision in the base payload would invalidate every in-flight
    full and measure continuation on the cluster for no correctness gain -- see
    the precedent comment beside ``process_format`` in
    ``processing_configuration_digest_from_values``.
    """
    shipped = tracker.PROCESS_LAYER_SEMANTICS_REVISION
    before = _digest_at_revision(shipped, **_FULL_KWARGS)
    after = _digest_at_revision(shipped + 1, **_FULL_KWARGS)

    assert before == after, (
        "a full-run digest must not depend on process-layer semantics"
    )

    # Control: the full-run digest is not simply insensitive to everything.
    # Without this, a constant-returning digest would pass the assertion above.
    moved = tracker.processing_configuration_digest_from_values(
        **dict(_FULL_KWARGS, detect_mode="rgb")
    )
    assert moved != before


def test_the_process_digest_still_separates_layers():
    """``objmap`` and ``gray`` exports remain distinct work.

    The revision is folded in as ``f"{layer}:{revision}"``. A revision spliced
    in on its own line would collapse nothing today, but the layer prefix is
    what keeps a future per-layer revision from being a payload change; this
    pins that the two layers do not hash alike.
    """
    objmap = tracker.processing_configuration_digest_from_values(
        **_PROCESS_KWARGS
    )
    gray = tracker.processing_configuration_digest_from_values(
        **dict(_PROCESS_KWARGS, process_only_layer="gray")
    )
    assert objmap != gray
