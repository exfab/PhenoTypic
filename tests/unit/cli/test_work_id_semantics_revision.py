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


# ------------------------------------ measurement-header revision (plan D9)
#
# Plan ``2026-09-23-texture-column-naming`` renamed the ``MeasureTexture``
# columns. The pipeline fingerprint is the user file's bytes, which the rename
# does not touch, so without a revision a run resumed across the upgrade would
# reuse stores measured under the old spelling and aggregate both spellings.
# D9 invalidates EVERY in-flight continuation, so -- unlike the process-layer
# revision above -- the full-run digest must move too.


def _digest_at_header_revision(revision, **kwargs):
    """Hash *kwargs* with ``MEASUREMENT_HEADER_REVISION`` set to *revision*."""
    original = tracker.MEASUREMENT_HEADER_REVISION
    try:
        tracker.MEASUREMENT_HEADER_REVISION = revision
        return tracker.processing_configuration_digest_from_values(**kwargs)
    finally:
        tracker.MEASUREMENT_HEADER_REVISION = original


def test_the_header_revision_moves_every_mode_digest():
    """Full/measure (one payload) and process digests all track the revision.

    ``test_the_digest_is_deterministic`` is the control that rules out a
    non-deterministic digest satisfying ``!=`` here.
    """
    shipped = tracker.MEASUREMENT_HEADER_REVISION
    for label, kwargs in (("full", _FULL_KWARGS), ("process", _PROCESS_KWARGS)):
        before = _digest_at_header_revision(shipped - 1, **kwargs)
        after = _digest_at_header_revision(shipped, **kwargs)
        assert before != after, (
            f"the {label} digest ignores MEASUREMENT_HEADER_REVISION, so a run "
            "resumed across a measurement-column rename would mix spellings"
        )


def test_the_shipped_digest_differs_from_the_pre_revision_payload():
    """The upgrade itself invalidates: the pre-D9 payload had no revision key.

    The reconstruction is checked against the shipped digest first (with the
    revision key added it must be EQUAL), so the final ``!=`` is attributable
    to the revision alone and not to a reconstruction that drifted from the
    real payload shape.
    """
    from phenotypic.sdk_._digests import canonical_digest

    pre_d9_base = {
        "image_type": _FULL_KWARGS["image_type"],
        "nrows": _FULL_KWARGS["nrows"],
        "ncols": _FULL_KWARGS["ncols"],
        "bit_depth": _FULL_KWARGS["bit_depth"],
        "detect_mode": _FULL_KWARGS["detect_mode"],
        "drop_originals": False,
        "include_dataset_column": _FULL_KWARGS["include_dataset_column"],
        "overlay_alpha": _FULL_KWARGS["overlay_alpha"],
        "save_overlays": _FULL_KWARGS["save_overlays"],
    }
    shipped = tracker.processing_configuration_digest_from_values(
        **_FULL_KWARGS
    )

    assert shipped == canonical_digest(
        dict(
            pre_d9_base,
            measurement_header_revision=tracker.MEASUREMENT_HEADER_REVISION,
        )
    ), "reconstruction no longer matches the shipped payload shape"
    assert shipped != canonical_digest(pre_d9_base)


def test_the_header_revision_moves_work_id_and_generation_in_every_mode(
    tmp_path, make_exec_config, monkeypatch
):
    """End to end: each mode's work id AND processing generation move.

    The generation is derived from ``per_image_config_digest``, which is this
    same digest (spec §5.4); a revision placed only in ``compute_work_id``
    would move the work id and leave the generation behind.
    """
    from phenotypic._cli._cli_identity import mint_run_identity

    image = tmp_path / "input" / "a.tiff"
    image.parent.mkdir()
    image.write_bytes(b"pixels")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")

    def _identity(**overrides):
        config = make_exec_config(
            pipeline_json=pipeline,
            input_path=image.parent,
            output_dir=tmp_path / "run",
            **overrides,
        )
        work_id = tracker.work_id_for_image(config, "plate", image)[0]
        generation = mint_run_identity(
            config, restart=False
        ).processing_generation
        return work_id, generation

    modes = {
        "full": {},
        "measure": {"measure_only": True},
        "process": {"process_only_layer": "gray"},
    }
    shipped = tracker.MEASUREMENT_HEADER_REVISION
    for mode, overrides in modes.items():
        after = _identity(**overrides)
        monkeypatch.setattr(tracker, "MEASUREMENT_HEADER_REVISION", shipped - 1)
        before = _identity(**overrides)
        monkeypatch.setattr(tracker, "MEASUREMENT_HEADER_REVISION", shipped)
        # Control: same revision, same identity (rules out an identity that
        # changes on every call).
        assert _identity(**overrides) == after
        assert before[0] != after[0], f"{mode} work id ignores the revision"
        assert before[1] != after[1], f"{mode} generation ignores the revision"
