"""The snapshot badge is a map over ``RunState.completion`` plus currency.

Spec §11 / P6 Task 1. The badge used to derive completion a second time from
``inspect_output_consistency`` and to compare a full-content SHA-256 of the
seven consumed-state deliverables on every 5-10 s poll. It now asks
``resolve_run_state`` for the completion and ``OutputRoot.snapshot_is_current``
for whether the bound snapshot still matches disk.

**Two axes, and the second one is not decoration** (CAN-18). A re-finalize
over an unchanged inventory leaves ``completion == "complete"`` while
rewriting the deliverables, so a badge driven by completion alone would read
"Current" over a stale mirror.

**The currency axis is deliberately the processing inventory, not the
consumed-state fingerprint.** Those two answer the same question about
different path sets, and they disagree on exactly one input: a write the GUI
itself made. The processing inventory excludes GUI-owned mutable state, so a
curation click no longer flips the badge to "Changed on disk" (audit S2).

**Excluding the mirror is REQUIRED, not a cost.** ``_curation_labels.py:846,848``
writes ``measurements.parquet`` and ``measurements.csv`` on every curation save,
so the mirror is GUI-owned: putting it back into the currency axis reintroduces
audit S2 exactly, one click at a time. An earlier draft of this docstring framed
its absence as a limitation, which would teach the next reader that restoring it
is an improvement.

The genuine residual, which is forced rather than chosen: an *external* rewrite
of the mirror alone -- nothing else touched -- is not detected by the badge. A
re-finalize IS detected, because it rewrites the master archive first and the
master is in the processing inventory.
"""

from __future__ import annotations

import hashlib
import typing
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from phenotypic.gui._snapshot_status import (
    _UNFINISHED_BADGE,
    snapshot_refresh_status,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import IMAGE
from phenotypic.sdk_._run_state import (
    Completion,
    RunDiagnostics,
    RunIdentity,
    RunState,
)
from tests._output_layout import write_master, write_measurements_mirror

_IDENTITY = RunIdentity(
    processing_generation="generation",
    restart_epoch=0,
    scheduler_epoch=None,
    owner_generation=None,
    inventory_digest="inventory",
    scientific_config_digest="scientific",
    finalization_input_digest="finalization",
)


def _run_state(completion: str) -> RunState:
    """A ``RunState`` carrying nothing the badge reads except *completion*."""
    return RunState(
        completion=completion,
        identity=_IDENTITY,
        images={},
        advisories=(),
        diagnostics=RunDiagnostics(accepted=0, verified=0, failed=0),
        depth="shallow",
        verified_at=datetime.now(timezone.utc),
    )


def _pin_completion(monkeypatch: pytest.MonkeyPatch, completion: str) -> None:
    """Answer the badge's completion question without a machine-state tree.

    Building a tree that genuinely resolves ``complete`` needs a run proof
    over a real inventory, which is ``resolve_run_state``'s own contract to
    test (``tests/unit/sdk_/``), not the badge's. Pinning it here keeps each
    test below about exactly one axis.
    """
    monkeypatch.setattr(
        "phenotypic.gui._snapshot_status.resolve_run_state",
        lambda output_dir, *, depth: _run_state(completion),
    )


class _FakeOutputRoot:
    """The four things the badge reads off an ``OutputRoot``, and no more."""

    def __init__(
        self,
        *,
        snapshot_current: bool = True,
        active_now: bool = False,
        bound_during_run: bool = False,
        output_root: Path | None = Path("/nonexistent/run"),
    ) -> None:
        self.layout = SimpleNamespace(output_root=output_root)
        self.snapshot = SimpleNamespace(active_run=bound_during_run)
        self._active_now = active_now
        self._snapshot_current = snapshot_current

    def active_run_is_currently_running(self) -> bool:
        return self._active_now

    def snapshot_is_current(self) -> bool:
        return self._snapshot_current


def _seed_full_run(root: Path) -> Path:
    """Seed the smallest tree ``OutputRoot.discover`` accepts as a full run."""
    frame = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"],
            str(IMAGE.IMAGE_NAME): ["plate"],
            "Object_Label": [1],
            "Shape_Area": [100.0],
        }
    )
    write_master(root, frame)
    write_measurements_mirror(root, frame)
    measurements = root / "results" / "d1" / "measurements"
    measurements.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture()
def discovered_root(tmp_path: Path) -> OutputRoot:
    """A real ``OutputRoot`` bound to a real tree, for the currency axis."""
    output = _seed_full_run(tmp_path / "output")
    return OutputRoot.discover(
        output,
        cache_root=tmp_path / "viewer-cache",
    )


@pytest.mark.parametrize(
    (
        "completion",
        "snapshot_current",
        "refresh_supported",
        "expected_color",
        "expected_fragment",
    ),
    [
        ("complete", True, True, "success", "Current"),
        ("complete", True, False, "success", "Current"),
        ("incomplete", True, True, "warning", "Run incomplete"),
        ("failed", True, True, "danger", "Run failed"),
        ("active", True, True, "warning", "Active run detected"),
        # CAN-18: complete, but the bound snapshot no longer matches disk.
        ("complete", False, True, "danger", "Changed on disk"),
    ],
)
def test_the_badge_is_a_map_over_completion_and_currency(
    completion: str,
    snapshot_current: bool,
    refresh_supported: bool,
    expected_color: str,
    expected_fragment: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§11: ~30 lines mapping ``completion`` -> badge.

    Replaces two fingerprints and a full re-hash of seven deliverables per
    poll. The ``("complete", False, ...)`` row is the one that fires if the
    currency axis is ever collapsed into ``completion``.
    """
    _pin_completion(monkeypatch, completion)
    label, color, _disabled = snapshot_refresh_status(
        _FakeOutputRoot(snapshot_current=snapshot_current),
        refresh_supported=refresh_supported,
    )
    assert color == expected_color
    assert expected_fragment in label


def test_every_completion_literal_has_a_badge() -> None:
    """A fifth ``Completion`` literal must be given a badge deliberately.

    Fires the moment someone adds a literal to ``Completion`` without
    deciding what the badge shows for it. Without this the new literal falls
    through ``_UNFINISHED_BADGE.get`` into the currency branch and an
    unrecognized run state silently reads "Current".
    """
    assert set(_UNFINISHED_BADGE) | {"complete", "active"} == set(
        typing.get_args(Completion)
    )


def test_a_rewritten_processing_product_shows_changed_on_disk(
    discovered_root: OutputRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAN-18, against a real tree.

    ``completion`` stays ``complete`` across a re-finalize because the
    inventory did not change, but the mirror the viewer is holding is stale.
    A re-finalize writes ``master_measurements.parquet`` before everything
    else it does (``finalize_post_master_outputs``), and the master is a
    processing product, so the currency axis sees it.
    """
    _pin_completion(monkeypatch, "complete")
    write_master(
        discovered_root.root,
        pl.DataFrame(
            {
                "Metadata_Dataset": ["d1", "d1"],
                str(IMAGE.IMAGE_NAME): ["plate", "plate"],
                "Object_Label": [1, 2],
                "Shape_Area": [100.0, 200.0],
            }
        ),
    )
    label, color, _disabled = snapshot_refresh_status(
        discovered_root,
        refresh_supported=True,
    )
    assert color == "danger"
    assert "Changed" in label


def test_a_curation_click_leaves_the_badge_current(
    discovered_root: OutputRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit S2: the viewer must not report its own write as external drift.

    ``CurationLabels.mark`` rewrites the mirror, the labels parquet and the
    custom-category file -- three of the seven paths the retired
    ``refresh_state_is_current`` hashed. Reinstating that call in the badge
    turns this green into a ``danger`` / "Changed on disk".
    """
    from phenotypic.gui.results_viewer._curation_labels import CurationLabels

    _pin_completion(monkeypatch, "complete")
    labels = CurationLabels.load(
        discovered_root.layout,
        discovered_root.clean_master_df,
    )
    labels.mark("plate", 1, "debris")

    label, color, _disabled = snapshot_refresh_status(
        discovered_root,
        refresh_supported=True,
    )
    assert (label, color) == ("Current", "success")


def test_no_badge_refresh_hashes_a_deliverable(
    discovered_root: OutputRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tick used to full-content SHA-256 seven deliverables. Per tab.

    ``measurements.parquet``, ``measurements.csv``, ``pipeline.json``,
    ``curation_labels.parquet``, ``custom_categories.json``, ``qc.duckdb``
    and ``review_state.json``. The currency axis stats an inventory instead,
    so the badge constructs no hasher at all; reinstating
    ``refresh_state_is_current`` makes ``_cancellable_paths_fingerprint``
    build one and this assertion fail.
    """
    _pin_completion(monkeypatch, "complete")
    calls = {"n": 0}
    real = hashlib.sha256

    def _counting_sha256(*args: object, **kwargs: object) -> object:
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", _counting_sha256)
    snapshot_refresh_status(discovered_root, refresh_supported=True)
    assert calls["n"] == 0


def test_a_standalone_bundle_skips_the_completion_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A portable bundle has no machine state, so it has no completion.

    ``BundleLayout`` resolves ``output_root`` to ``None`` for a bundle.
    Handing that to ``resolve_run_state`` raises ``TypeError`` out of a
    status poll; reporting it as ``incomplete`` badges every bundle as an
    unfinished run forever. Neither is acceptable, so the badge falls
    straight through to the currency question -- which is what the retired
    ``inspect_output_consistency`` did for a bundle too.
    """

    def _must_not_be_called(output_dir: object, *, depth: str) -> RunState:
        raise AssertionError(
            f"a standalone bundle has no run state to resolve: {output_dir!r}"
        )

    monkeypatch.setattr(
        "phenotypic.gui._snapshot_status.resolve_run_state",
        _must_not_be_called,
    )
    label, color, disabled = snapshot_refresh_status(
        _FakeOutputRoot(output_root=None, snapshot_current=True),
        refresh_supported=False,
    )
    assert (color, disabled) == ("success", True)
    assert "Current" in label


@pytest.mark.parametrize(
    ("active_now", "bound_during_run", "refresh_supported", "expected"),
    [
        (True, True, True, ("Active run snapshot", "warning", True)),
        (
            True,
            False,
            True,
            ("Active run detected · refresh snapshot", "warning", False),
        ),
        (
            False,
            True,
            True,
            ("Run finished · refresh snapshot", "info", False),
        ),
        (False, False, True, ("Current", "success", False)),
        (
            True,
            True,
            False,
            (
                "Active run detected · restart app after it finishes",
                "warning",
                True,
            ),
        ),
        (
            True,
            False,
            False,
            (
                "Active run detected · restart app after it finishes",
                "warning",
                True,
            ),
        ),
        (
            False,
            True,
            False,
            ("Run finished · restart standalone app", "info", True),
        ),
        (
            False,
            False,
            False,
            ("Current · restart app to refresh", "success", True),
        ),
    ],
)
def test_liveness_and_the_binding_are_separate_axes(
    active_now: bool,
    bound_during_run: bool,
    refresh_supported: bool,
    expected: tuple[str, str, bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whether a run is live and whether *this* binding caught it differ.

    ``active_run_is_currently_running()`` is a fact about the tree
    (``_output_root.py``: the GUI launch-owner record). ``snapshot.active_run``
    is a fact about the binding this session is holding, frozen at discovery.
    Their four combinations produce three badges and -- the part that makes
    this an axis rather than a nuance -- **two different ``disabled``
    values**: rows two and one above agree on label-stem and colour family and
    disagree on whether Refresh is offered at all.

    Collapsing the badge onto ``(completion, snapshot_is_current)`` therefore
    does not merely reword a label. It greys out Refresh for a live run whose
    snapshot predates it -- the one case where refreshing is exactly what the
    user wants -- or offers Refresh for a snapshot already captured mid-run,
    which rebinds to another moving target. This test is what fails if the
    third axis is dropped.
    """
    _pin_completion(monkeypatch, "complete")
    assert (
        snapshot_refresh_status(
            _FakeOutputRoot(
                active_now=active_now,
                bound_during_run=bound_during_run,
                snapshot_current=True,
            ),
            refresh_supported=refresh_supported,
        )
        == expected
    )


def test_a_gui_launched_run_is_live_even_when_completion_says_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two liveness sources are different owners, so both are read.

    ``active_run_is_currently_running()`` reads the **GUI** launch-owner
    record (``gui_launch_owner_path``); ``completion == "active"`` reads the
    **CLI** machine state. A run launched from the run console against a tree
    whose last CLI proof still covers the current inventory is live by the
    first and ``complete`` by the second. Dropping the GUI half badges that
    run "Current" while it is writing.
    """
    _pin_completion(monkeypatch, "complete")
    _label, color, _disabled = snapshot_refresh_status(
        _FakeOutputRoot(active_now=True, snapshot_current=True),
        refresh_supported=True,
    )
    assert color == "warning"


def test_a_cli_run_is_live_with_no_gui_owner_record_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The converse of the test above, so neither half can be dropped quietly.

    A plain ``python -m phenotypic`` run writes no GUI launch-owner record, so
    ``active_run_is_currently_running()`` is ``False`` throughout it and
    ``completion`` is the only witness that anything is happening.
    """
    _pin_completion(monkeypatch, "active")
    _label, color, disabled = snapshot_refresh_status(
        _FakeOutputRoot(active_now=False, snapshot_current=True),
        refresh_supported=True,
    )
    assert (color, disabled) == ("warning", False)


def test_the_completion_question_is_asked_shallow_at_the_run_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``depth="shallow"`` is the whole performance claim, so pin it.

    A ``"deep"`` poll re-reads every declared artifact's content on a 5-10 s
    tick, which is the cost this task exists to remove -- worse than the
    seven-file SHA-256 it replaces, not better. Counting hashers cannot catch
    that regression because a deep pass is entitled to hash; the argument is.
    """
    seen: dict[str, object] = {}

    def _capture(output_dir: object, *, depth: str) -> RunState:
        seen["output_dir"] = output_dir
        seen["depth"] = depth
        return _run_state("complete")

    monkeypatch.setattr(
        "phenotypic.gui._snapshot_status.resolve_run_state",
        _capture,
    )
    root = Path("/nonexistent/run")
    snapshot_refresh_status(
        _FakeOutputRoot(output_root=root),
        refresh_supported=True,
    )
    assert seen == {"output_dir": root, "depth": "shallow"}
