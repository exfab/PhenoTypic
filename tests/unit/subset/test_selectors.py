"""The ``SubsetSelector`` hierarchy — §10.3.

Three properties are load-bearing and each has a test that dies without it:

* **Seeding is real**, so a recorded seed reproduces a selection.
* **``group_filter`` lives on the ABC**, applied before ``_select``, so a
  selector that knows nothing about metadata still composes with it.
* **An unimplemented selector fails loudly**, never degrading to random. A
  placeholder that quietly returned a random sample would stamp
  ``method: EmbeddingSubsetSelector`` onto an artifact with none of the claimed
  visual coverage, and nothing downstream could contradict it.
"""

from __future__ import annotations

import pytest


# --------------------------------------------------------------------------
# RandomSubsetSelector
# --------------------------------------------------------------------------


def test_random_is_seeded_and_reproducible(image_refs):
    from phenotypic.subset import RandomSubsetSelector

    a = RandomSubsetSelector(n=6, seed=0).select(image_refs)
    b = RandomSubsetSelector(n=6, seed=0).select(image_refs)
    assert a.images == b.images
    assert len(a.images) == 6


def test_random_differs_between_seeds(image_refs):
    """Reproducibility alone passes against an implementation ignoring the seed."""
    from phenotypic.subset import RandomSubsetSelector

    a = RandomSubsetSelector(n=6, seed=0).select(image_refs)
    b = RandomSubsetSelector(n=6, seed=17).select(image_refs)
    assert a.images != b.images


def test_random_selects_parent_relative_paths(image_refs):
    """Bare filenames cannot disambiguate two datasets (§10.2)."""
    from phenotypic.subset import RandomSubsetSelector

    result = RandomSubsetSelector(n=6, seed=0).select(image_refs)
    assert set(result.images) <= {ref.relative_path for ref in image_refs}
    assert any("/" in image for image in result.images)


def test_random_records_its_own_provenance(image_refs):
    from phenotypic.subset import RandomSubsetSelector

    result = RandomSubsetSelector(n=6, seed=3).select(image_refs)
    assert result.method == "RandomSubsetSelector"
    assert result.seed == 3
    assert result.params["n"] == 6
    assert result.rationale


def test_n_larger_than_the_candidate_set_takes_it_whole(image_refs):
    from phenotypic.subset import RandomSubsetSelector

    result = RandomSubsetSelector(n=999, seed=0).select(image_refs)
    assert len(result.images) == len(image_refs)


def test_n_must_be_at_least_one():
    import pydantic

    from phenotypic.subset import RandomSubsetSelector

    with pytest.raises(pydantic.ValidationError):
        RandomSubsetSelector(n=0)


def test_selectors_are_keyword_only():
    from phenotypic.subset import RandomSubsetSelector

    with pytest.raises(TypeError):
        RandomSubsetSelector(6)


# --------------------------------------------------------------------------
# MetadataGroupSubsetSelector
# --------------------------------------------------------------------------


def test_metadata_group_equal_allocation(batches_csv, image_refs):
    """``equal`` gives the two-image ``rare`` batch the same share as ``common``."""
    from phenotypic.subset import MetadataGroupSubsetSelector

    sel = MetadataGroupSubsetSelector(
        n=4, seed=0, grouping_metadata=str(batches_csv),
        group_key="Metadata_Batch", allocation="equal")
    result = sel.select(image_refs)

    assert len(result.images) == 4
    assert result.method == "MetadataGroupSubsetSelector"
    rare = {ref.relative_path for ref in image_refs[:2]}
    assert len(rare & set(result.images)) == 2


def test_metadata_group_proportional_allocation_mirrors_group_sizes(
    batches_csv, image_refs
):
    """The two allocations must not be the same code path with a different name."""
    from phenotypic.subset import MetadataGroupSubsetSelector

    sel = MetadataGroupSubsetSelector(
        n=6, seed=0, grouping_metadata=str(batches_csv),
        group_key="Metadata_Batch", allocation="proportional")
    result = sel.select(image_refs)

    assert len(result.images) == 6
    rare = {ref.relative_path for ref in image_refs[:2]}
    # 2 of 12 candidates are rare, so a proportional share of 6 is 1.
    assert len(rare & set(result.images)) == 1


def test_min_per_group_floors_a_rare_group(batches_csv, image_refs):
    from phenotypic.subset import MetadataGroupSubsetSelector

    sel = MetadataGroupSubsetSelector(
        n=6, seed=0, grouping_metadata=str(batches_csv),
        group_key="Metadata_Batch", allocation="proportional", min_per_group=2)
    result = sel.select(image_refs)

    rare = {ref.relative_path for ref in image_refs[:2]}
    assert len(rare & set(result.images)) == 2


def test_metadata_group_reports_its_allocation(batches_csv, image_refs):
    from phenotypic.subset import MetadataGroupSubsetSelector

    sel = MetadataGroupSubsetSelector(
        n=4, seed=0, grouping_metadata=str(batches_csv),
        group_key="Metadata_Batch", allocation="equal")
    result = sel.select(image_refs)

    assert result.allocation == {"common": 2, "rare": 2}


def test_group_key_absent_from_the_csv_raises(batches_csv, image_refs):
    """Never a silent fall-through to a weaker grouping."""
    from phenotypic.subset import GroupKeyNotInMetadata, MetadataGroupSubsetSelector

    sel = MetadataGroupSubsetSelector(
        n=4, seed=0, grouping_metadata=str(batches_csv),
        group_key="Metadata_Nonexistent")
    with pytest.raises(GroupKeyNotInMetadata) as excinfo:
        sel.select(image_refs)
    assert "Metadata_Batch" in excinfo.value.available_columns


def test_group_key_resolves_through_the_metadata_canonicalizer(
    batches_csv, image_refs
):
    """``Batch`` and ``Metadata_Batch`` are the same column, by the central helper."""
    from phenotypic.subset import MetadataGroupSubsetSelector

    bare = MetadataGroupSubsetSelector(
        n=4, seed=0, grouping_metadata=str(batches_csv),
        group_key="Batch", allocation="equal").select(image_refs)
    prefixed = MetadataGroupSubsetSelector(
        n=4, seed=0, grouping_metadata=str(batches_csv),
        group_key="Metadata_Batch", allocation="equal").select(image_refs)

    assert bare.images == prefixed.images


def test_metadata_group_requires_its_csv():
    import pydantic

    from phenotypic.subset import MetadataGroupSubsetSelector

    with pytest.raises(pydantic.ValidationError):
        MetadataGroupSubsetSelector(n=4, group_key="Metadata_Batch")


def test_metadata_group_joins_on_the_parent_relative_path(tmp_path, image_refs):
    """A CSV keyed by bare filename does not silently half-match.

    ``plateA/plateA_01.tif`` and a CSV row saying ``plateA_01.tif`` are not the
    same key: accepting the second would make two datasets containing the same
    filename indistinguishable, which is why §10.2 records relative paths.
    """
    from phenotypic.subset import GroupKeyNotInMetadata, MetadataGroupSubsetSelector

    csv = tmp_path / "bare.csv"
    rows = "\n".join(
        f"{ref.relative_path.rsplit('/', 1)[-1]},common" for ref in image_refs
    )
    csv.write_text(f"image,Metadata_Batch\n{rows}\n")

    sel = MetadataGroupSubsetSelector(
        n=4, seed=0, grouping_metadata=str(csv), group_key="Metadata_Batch")
    with pytest.raises(GroupKeyNotInMetadata):
        sel.select(image_refs)


# --------------------------------------------------------------------------
# group_filter — on the ABC (USER-24)
# --------------------------------------------------------------------------


def test_group_filter_restricts_a_selector_that_knows_no_metadata(
    species_csv, image_refs
):
    """Proves the filter is on the ABC, not inside a metadata-aware selector."""
    from phenotypic.subset import RandomSubsetSelector

    sel = RandomSubsetSelector(
        n=4, seed=0, grouping_metadata=str(species_csv),
        group_filter={"Metadata_Species": "A_nidulans"})
    result = sel.select(image_refs)

    assert len(result.images) == 4
    assert all(image.startswith("plateA/") for image in result.images)


def test_group_filter_is_recorded_on_the_selection(species_csv, image_refs):
    """§10.2 records it at top level *and* inside ``selection.params``."""
    from phenotypic.subset import RandomSubsetSelector

    filter_ = {"Metadata_Species": "A_nidulans"}
    result = RandomSubsetSelector(
        n=4, seed=0, grouping_metadata=str(species_csv),
        group_filter=filter_).select(image_refs)

    assert result.group_filter == filter_
    assert result.params["group_filter"] == filter_


def test_group_filter_naming_an_absent_column_raises(species_csv, image_refs):
    """Never a silent selection from everything."""
    from phenotypic.subset import GroupFilterColumnNotFound, RandomSubsetSelector

    sel = RandomSubsetSelector(
        n=4, seed=0, grouping_metadata=str(species_csv),
        group_filter={"Metadata_Nonexistent": "x"})
    with pytest.raises(GroupFilterColumnNotFound) as excinfo:
        sel.select(image_refs)
    assert excinfo.value.column == "Metadata_Nonexistent"
    assert "Metadata_Species" in excinfo.value.available_columns


def test_group_filter_matching_nothing_raises_rather_than_selecting_nothing(
    species_csv, image_refs
):
    """An empty subset passes every downstream shape check and studies nothing."""
    from phenotypic.subset import GroupFilterMatchesNothing, RandomSubsetSelector

    sel = RandomSubsetSelector(
        n=4, seed=0, grouping_metadata=str(species_csv),
        group_filter={"Metadata_Species": "A_flavus"})
    with pytest.raises(GroupFilterMatchesNothing):
        sel.select(image_refs)


def test_group_filter_is_conjunctive(tmp_path, image_refs):
    from phenotypic.subset import GroupFilterMatchesNothing, RandomSubsetSelector

    csv = tmp_path / "two_columns.csv"
    rows = "\n".join(
        f"{ref.relative_path},"
        f"{'A_nidulans' if ref.dataset == 'plateA' else 'A_niger'},"
        f"{'day1' if ref.dataset == 'plateB' else 'day2'}"
        for ref in image_refs
    )
    csv.write_text(f"image,Metadata_Species,Metadata_Day\n{rows}\n")

    both_match = RandomSubsetSelector(
        n=3, seed=0, grouping_metadata=str(csv),
        group_filter={"Metadata_Species": "A_niger", "Metadata_Day": "day1"},
    ).select(image_refs)
    assert all(image.startswith("plateB/") for image in both_match.images)

    # Each pair matches on its own; together they match nothing.
    sel = RandomSubsetSelector(
        n=3, seed=0, grouping_metadata=str(csv),
        group_filter={"Metadata_Species": "A_niger", "Metadata_Day": "day2"})
    with pytest.raises(GroupFilterMatchesNothing):
        sel.select(image_refs)


def test_group_filter_resolves_through_the_metadata_canonicalizer(
    species_csv, image_refs
):
    """Never ``startswith("Metadata_")`` or prefix splitting (project rule)."""
    from phenotypic.subset import RandomSubsetSelector

    bare = RandomSubsetSelector(
        n=4, seed=0, grouping_metadata=str(species_csv),
        group_filter={"Species": "A_nidulans"}).select(image_refs)
    assert all(image.startswith("plateA/") for image in bare.images)


def test_group_filter_without_a_csv_is_a_construction_error():
    """Not an empty result — a filter no selector can evaluate must not exist."""
    import pydantic

    from phenotypic.subset import RandomSubsetSelector

    with pytest.raises(pydantic.ValidationError):
        RandomSubsetSelector(n=4, group_filter={"Metadata_Species": "A_nidulans"})


def test_group_filter_composes_with_metadata_stratification(tmp_path, image_refs):
    """``allocation`` stratifies *inside* the filtered set, not around it."""
    from phenotypic.subset import MetadataGroupSubsetSelector

    csv = tmp_path / "combined.csv"
    rows = "\n".join(
        f"{ref.relative_path},"
        f"{'A_nidulans' if ref.dataset == 'plateA' else 'A_niger'},"
        f"{'rare' if index % 7 == 0 else 'common'}"
        for index, ref in enumerate(image_refs)
    )
    csv.write_text(f"image,Metadata_Species,Metadata_Batch\n{rows}\n")

    result = MetadataGroupSubsetSelector(
        n=4, seed=0, grouping_metadata=str(csv), group_key="Metadata_Batch",
        allocation="equal", group_filter={"Metadata_Species": "A_nidulans"},
    ).select(image_refs)

    assert len(result.images) == 4
    assert all(image.startswith("plateA/") for image in result.images)
    # plateA holds indices 0..6, so exactly one of its images is ``rare``.
    assert result.allocation == {"common": 3, "rare": 1}


# --------------------------------------------------------------------------
# EmbeddingSubsetSelector — the placeholder that must never degrade
# --------------------------------------------------------------------------


def test_embedding_availability_is_false_with_a_reason(image_refs):
    from phenotypic.subset import EmbeddingSubsetSelector

    available, why = EmbeddingSubsetSelector(n=4, seed=0).availability()
    assert available is False
    assert "not implemented" in why.lower()


def test_embedding_select_raises_and_never_degrades(image_refs):
    """A placeholder that silently returned random would stamp
    ``method: EmbeddingSubsetSelector`` onto an artifact with none of the
    claimed visual coverage, and nothing downstream could contradict it."""
    from phenotypic.subset import EmbeddingSubsetSelector

    with pytest.raises(NotImplementedError):
        EmbeddingSubsetSelector(n=4, seed=0).select(image_refs)


def test_embedding_private_select_also_raises(image_refs):
    """The second barrier.

    ``select()`` refuses on ``availability()``, so a mutation to ``_select``
    alone does not change what ``select()`` does. This asserts the inner guard
    directly, so *both* halves of the refusal are pinned.
    """
    from phenotypic.subset import EmbeddingSubsetSelector

    with pytest.raises(NotImplementedError):
        EmbeddingSubsetSelector(n=4, seed=0)._select(image_refs)


def test_select_refuses_when_a_selector_reports_itself_unavailable(image_refs):
    """The ABC's gate, distinct from the subclass's second barrier.

    ``EmbeddingSubsetSelector`` raises from ``_select`` too, so its tests
    cannot tell the two apart: deleting the template's availability check
    leaves every one of them green. This stub reports unavailable while its
    ``_select`` happily returns a selection, so only the gate can refuse it.
    """
    from phenotypic.subset import SubsetSelector

    class _StubUnavailable(SubsetSelector):
        def availability(self):
            return False, "stub backend absent"

        def _select(self, candidates):
            return [ref.relative_path for ref in candidates]

    with pytest.raises(NotImplementedError, match="stub backend absent"):
        _StubUnavailable(n=2).select(image_refs)


def test_select_deduplicates_orders_and_drops_non_candidates(image_refs):
    """The template owns the shape of ``images``, not the subclass.

    A selector returning duplicates, arbitrary order, or a path outside the
    candidate set must not be able to put any of that on the artifact — an
    image the subset does not contain would be staged, symlinked, and tuned on.
    """
    from phenotypic.subset import SubsetSelector

    first, second = image_refs[0].relative_path, image_refs[1].relative_path

    class _StubSloppy(SubsetSelector):
        def _select(self, candidates):
            return [second, first, second, "not/in/the/parent.tif"]

    result = _StubSloppy(n=3).select(image_refs)
    assert result.images == tuple(sorted({first, second}))


def test_embedding_cost_class_is_w2_even_unimplemented():
    from phenotypic.subset import EmbeddingSubsetSelector

    assert EmbeddingSubsetSelector(n=4).cost_class() == "W2"


def test_cheap_selectors_are_w0():
    """``cost_class`` must not be a constant."""
    from phenotypic.subset import RandomSubsetSelector

    assert RandomSubsetSelector(n=4).cost_class() == "W0"


# --------------------------------------------------------------------------
# Serialization, resolution, and the user_named path
# --------------------------------------------------------------------------


def test_selectors_resolve_by_bare_class_name():
    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        SerializablePipeline,
    )

    assert SerializablePipeline._find_class_in_phenotypic("RandomSubsetSelector")


def test_selectors_round_trip_through_class_and_params(batches_csv):
    """``{class, params}`` in, the same selector back — like every other
    extensible class here, so adding a fourth needs no tool signature change."""
    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        SerializablePipeline,
    )
    from phenotypic.subset import MetadataGroupSubsetSelector

    original = MetadataGroupSubsetSelector(
        n=4, seed=2, grouping_metadata=str(batches_csv),
        group_key="Metadata_Batch", allocation="equal")
    payload = {
        "class": type(original).__name__,
        "params": original.model_dump(mode="json"),
    }

    cls = SerializablePipeline._find_class_in_phenotypic(payload["class"])
    assert cls is not None
    assert cls.model_validate(payload["params"]) == original


def test_a_user_named_selection_can_carry_a_group_filter():
    """GEN-37: hand-picking one group's plates is the path USER-24 recommends.

    ``group_filter`` was a *selector* field, so a ``user_named`` subset could
    never carry one — and a full-scope ack given for one group would then be
    spendable across every group with every digest still matching.
    """
    from phenotypic.subset import SubsetSelection

    selection = SubsetSelection(
        images=("plateA/plateA_01.tif", "plateA/plateA_02.tif"),
        method="user_named",
        group_filter={"Metadata_Species": "A_nidulans"},
        rationale="the two plates I trust",
    )
    assert selection.group_filter == {"Metadata_Species": "A_nidulans"}
    assert selection.method == "user_named"


def test_a_selection_rejects_rebinding_a_field():
    """The artifact is written from it; it must not be edited after the fact."""
    import pydantic

    from phenotypic.subset import SubsetSelection

    selection = SubsetSelection(images=("a.tif",), method="user_named")
    with pytest.raises(pydantic.ValidationError):
        selection.images = ("b.tif",)


def test_a_selections_group_filter_cannot_be_edited_in_place():
    """``frozen=True`` blocks assignment and **nothing else**.

    Rebinding ``.images`` — a tuple — passes against a model whose dict fields
    are wide open, so it proved nothing about the field that matters. USER-21
    binds a human's approval to a specific ``group_filter``; if one item
    assignment can retarget it, an ack given for one group is spendable on
    another with every digest still matching.
    """
    from phenotypic.subset import SubsetSelection

    selection = SubsetSelection(
        images=("plateA/plateA_01.tif",),
        method="user_named",
        group_filter={"Metadata_Species": "A_niger"},
    )

    with pytest.raises(TypeError):
        selection.group_filter["Metadata_Species"] = "TAMPERED"
    with pytest.raises(TypeError):
        selection.group_filter["Metadata_Batch"] = "injected"
    assert selection.group_filter == {"Metadata_Species": "A_niger"}


def test_a_selections_params_are_frozen_through_the_nesting():
    """``params`` is the selector's ``model_dump``, so it *contains* a filter.

    A shallow freeze would leave ``params["group_filter"]`` editable — the
    same retargeting one level down, in the copy the artifact records as what
    actually ran.
    """
    from phenotypic.subset import RandomSubsetSelector, SubsetSelection

    selection = SubsetSelection(
        images=("plateA/plateA_01.tif",),
        method="user_named",
        params=RandomSubsetSelector(n=1).model_dump(mode="json"),
    )
    assert "group_filter" in selection.params

    with pytest.raises(TypeError):
        selection.params["injected"] = True
    with pytest.raises(TypeError):
        selection.params["group_filter"]["Metadata_Species"] = "TAMPERED"


def test_a_selections_allocation_cannot_be_edited_in_place():
    """The recorded stratification is provenance too."""
    from phenotypic.subset import SubsetSelection

    selection = SubsetSelection(
        images=("a.tif",), method="user_named", allocation={"rare": 1}
    )
    with pytest.raises(TypeError):
        selection.allocation["rare"] = 99


def test_a_frozen_selection_still_serializes_to_plain_json(batches_csv, image_refs):
    """Immutability is an internal choice; the artifact must not notice it.

    A ``mappingproxy`` is not JSON-serializable, so a freeze that reached the
    artifact writer would break the thing it exists to protect.
    """
    import json

    from phenotypic.subset import MetadataGroupSubsetSelector

    selection = MetadataGroupSubsetSelector(
        n=4, seed=0, grouping_metadata=str(batches_csv), group_key="Metadata_Batch"
    ).select(image_refs)

    payload = json.loads(json.dumps(selection.model_dump(mode="json")))
    assert payload["params"]["group_filter"] == {}
    assert isinstance(payload["allocation"], dict)
    assert selection.group_filter == {}


def test_expected_vs_detected_keeps_its_shipped_field_name():
    """Guards the rename two reviewers proposed. It does not exist.

    Asserting only that ``ExpectedVsDetectedCount(expected_counts_csv="x.csv")``
    raises proves nothing: it raises for the **missing required ``metadata``**,
    and would keep raising if someone added ``expected_counts_csv`` as an alias
    — the exact change this test claims to guard.
    """
    from phenotypic.analysis.qc import ExpectedVsDetectedCount

    assert "expected_counts_csv" not in ExpectedVsDetectedCount.model_fields
    assert "metadata" in ExpectedVsDetectedCount.model_fields


# --------------------------------------------------------------------------
# n is the compute budget, min_per_group is a statistical floor
# --------------------------------------------------------------------------


def test_min_per_group_never_overruns_n(batches_csv, image_refs):
    """The budget is a ceiling, and infeasible floors are refused, not absorbed.

    ``n=2`` with ``min_per_group=3`` over two groups used to return **6**
    images -- three times the budget -- because the reduction loop refused to
    go below the floors and broke out. Bounding compute is the whole reason
    the subset boundary exists, so silently overrunning it defeated the
    boundary with every downstream check green.
    """
    from phenotypic.subset import (
        GroupFloorsExceedTarget,
        MetadataGroupSubsetSelector,
    )

    sel = MetadataGroupSubsetSelector(
        n=2, seed=0, grouping_metadata=str(batches_csv),
        group_key="Metadata_Batch", min_per_group=3,
    )
    with pytest.raises(GroupFloorsExceedTarget) as raised:
        sel.select(image_refs)

    assert raised.value.n == 2
    assert raised.value.min_per_group == 3
    assert raised.value.required == 5  # rare has only 2 candidates to floor
    assert "n=2" in str(raised.value)


def test_the_floor_is_clamped_to_a_group_smaller_than_it(batches_csv, image_refs):
    """A group of 2 cannot yield 3, so the floor it contributes is 2.

    Feasibility is judged on the *clamped* floors. Judging it on
    ``min_per_group * len(groups)`` would refuse selections that are perfectly
    satisfiable — here ``rare`` has 2 candidates, so floors need 2 + 3 = 5.
    """
    from phenotypic.subset import MetadataGroupSubsetSelector

    result = MetadataGroupSubsetSelector(
        n=5, seed=0, grouping_metadata=str(batches_csv),
        group_key="Metadata_Batch", min_per_group=3,
    ).select(image_refs)

    assert len(result.images) == 5
    assert result.allocation == {"common": 3, "rare": 2}


def test_a_feasible_floor_is_still_honoured(batches_csv, image_refs):
    """The refusal must not have turned every floor into an error."""
    from phenotypic.subset import MetadataGroupSubsetSelector

    result = MetadataGroupSubsetSelector(
        n=6, seed=0, grouping_metadata=str(batches_csv),
        group_key="Metadata_Batch", allocation="proportional", min_per_group=2,
    ).select(image_refs)

    assert len(result.images) == 6
    assert result.allocation["rare"] == 2


def test_a_selection_never_exceeds_n_across_the_shipped_selectors(
    batches_csv, image_refs
):
    """One assertion covering the property the budget actually needs.

    No floor here: with ``min_per_group=1`` and two groups, ``n=1`` is the
    infeasible case that now raises, which is a different property.
    """
    from phenotypic.subset import (
        MetadataGroupSubsetSelector,
        RandomSubsetSelector,
    )

    for n in (1, 3, 7, 12, 20):
        assert len(RandomSubsetSelector(n=n, seed=0).select(image_refs).images) <= n
        stratified = MetadataGroupSubsetSelector(
            n=n, seed=0, grouping_metadata=str(batches_csv),
            group_key="Metadata_Batch",
        ).select(image_refs)
        assert len(stratified.images) <= n
