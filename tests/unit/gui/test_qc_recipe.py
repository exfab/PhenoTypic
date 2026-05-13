"""Unit tests for :mod:`phenotypic.gui._qc_recipe`.

Covers the sidecar persistence contract for the results-viewer QC tab:

* :meth:`QcRecipe.load` behaviour on missing / corrupt / unresolved-class
  files.
* :meth:`QcRecipe.save` atomic ``.tmp`` + :func:`os.replace` write.
* :meth:`QcRecipe.add` / :meth:`remove` / :meth:`update` mutation +
  immediate save semantics.
* ``instance_id`` shape + uniqueness under tight-loop generation.
* Thread-safe concurrent ``add()`` from two writers.
* Round-trip JSON schema fidelity.
* :meth:`QcRecipe.instantiate` filtering + warning-collection behaviour.
"""
from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from phenotypic.analysis import ExpectedVsDetectedCount, ReplicateAgreement
from phenotypic.analysis.abc_ import QualityCheck
from phenotypic.gui._qc_recipe import (
    QC_RECIPE_FILENAME,
    VIEWER_CACHE_DIRNAME,
    QcRecipe,
    QcRecipeEntry,
    QcRecipeLoadWarning,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def output_root(tmp_path: Path) -> Path:
    """Empty output-root path; ``.viewer_cache`` is NOT pre-created."""
    return tmp_path


@pytest.fixture()
def recipe_path(output_root: Path) -> Path:
    """Absolute path the recipe sidecar would land at."""
    return output_root / VIEWER_CACHE_DIRNAME / QC_RECIPE_FILENAME


@pytest.fixture()
def metadata_csv(tmp_path: Path) -> Path:
    """A 96-row plate-layout CSV usable by :class:`ExpectedVsDetectedCount`."""
    path = tmp_path / "metadata.csv"
    df = pd.DataFrame(
        {
            "Metadata_ImageFile": ["plate1.png"] * 96,
            "ObjectLabel": list(range(96)),
        }
    )
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Load behaviour
# ---------------------------------------------------------------------------


class TestLoadMissingFile:
    def test_load_missing_file_returns_empty_recipe(
        self, output_root: Path, recipe_path: Path
    ) -> None:
        assert not recipe_path.exists()

        recipe = QcRecipe.load(output_root)

        assert recipe.entries == []
        assert recipe.load_warnings == []
        # Missing file ⇒ no eager creation; first save() is required.
        assert not recipe_path.exists()


class TestLoadCorruptJson:
    def test_load_corrupt_json_returns_load_warning_and_empty_entries(
        self, output_root: Path, recipe_path: Path
    ) -> None:
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncated invalid JSON.
        truncated = '{"version": 1, "checks": [{'
        recipe_path.write_text(truncated, encoding="utf-8")

        recipe = QcRecipe.load(output_root)

        assert recipe.entries == []
        assert len(recipe.load_warnings) == 1
        warning = recipe.load_warnings[0]
        assert warning.instance_id == "__file__"
        assert "invalid JSON" in warning.reason

        # The on-disk file must be UNCHANGED — recovery from VCS still works.
        assert recipe_path.read_text(encoding="utf-8") == truncated


class TestLoadUnresolvedClass:
    def test_load_unresolved_class_records_warning_and_skips(
        self, output_root: Path, recipe_path: Path
    ) -> None:
        recipe_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "checks": [
                {
                    "instance_id": "qc-Ghost-deadbeef",
                    "class": "DoesNotExist",
                    "enabled": True,
                    "params": {},
                }
            ],
        }
        recipe_path.write_text(json.dumps(payload), encoding="utf-8")

        recipe = QcRecipe.load(output_root)

        # The bad entry is dropped from the live recipe …
        assert recipe.entries == []
        # … but a warning naming the missing class is preserved.
        assert len(recipe.load_warnings) == 1
        warning = recipe.load_warnings[0]
        assert warning.class_name == "DoesNotExist"
        assert warning.instance_id == "qc-Ghost-deadbeef"


# ---------------------------------------------------------------------------
# add / remove / update — mutation + save
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_creates_entry_and_saves(
        self,
        output_root: Path,
        recipe_path: Path,
        metadata_csv: Path,
    ) -> None:
        recipe = QcRecipe.load(output_root)
        assert recipe.entries == []

        new_id = recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
                "on": "ObjectLabel",
                "severity_warn": 0.05,
                "severity_fail": 0.10,
            },
        )

        assert isinstance(new_id, str)
        assert len(recipe.entries) == 1
        assert recipe.entries[0].instance_id == new_id

        # On-disk JSON must reflect the new entry.
        assert recipe_path.exists()
        on_disk = json.loads(recipe_path.read_text(encoding="utf-8"))
        assert on_disk["version"] == 1
        assert len(on_disk["checks"]) == 1
        assert on_disk["checks"][0]["instance_id"] == new_id
        assert on_disk["checks"][0]["class"] == "ExpectedVsDetectedCount"


class TestRemove:
    def test_remove_drops_entry_and_saves(
        self,
        output_root: Path,
        recipe_path: Path,
        metadata_csv: Path,
    ) -> None:
        recipe = QcRecipe.load(output_root)
        first_id = recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
            },
        )
        recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
            },
        )
        assert len(recipe.entries) == 2

        removed = recipe.remove(first_id)

        assert removed is True
        assert len(recipe.entries) == 1
        assert recipe.entries[0].instance_id != first_id

        # On-disk JSON is rewritten with one entry remaining.
        on_disk = json.loads(recipe_path.read_text(encoding="utf-8"))
        assert len(on_disk["checks"]) == 1
        assert on_disk["checks"][0]["instance_id"] != first_id

    def test_remove_returns_false_for_unknown_id(
        self, output_root: Path
    ) -> None:
        recipe = QcRecipe.load(output_root)
        assert recipe.remove("never-existed") is False


class TestUpdate:
    def test_update_params_and_enabled(
        self,
        output_root: Path,
        recipe_path: Path,
        metadata_csv: Path,
    ) -> None:
        recipe = QcRecipe.load(output_root)
        new_id = recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
                "severity_warn": 0.05,
            },
        )

        updated = recipe.update(
            new_id,
            params={
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
                "severity_warn": 0.99,
            },
            enabled=False,
        )

        assert updated is True
        assert recipe.entries[0].params["severity_warn"] == 0.99
        assert recipe.entries[0].enabled is False

        # On-disk JSON reflects the changes.
        on_disk = json.loads(recipe_path.read_text(encoding="utf-8"))
        assert on_disk["checks"][0]["params"]["severity_warn"] == 0.99
        assert on_disk["checks"][0]["enabled"] is False


# ---------------------------------------------------------------------------
# Revision contract — drives STORE_QC_RECIPE_REVISION dcc.Store
# ---------------------------------------------------------------------------


def test_revision_bumps_on_mutation(
    output_root: Path,
    recipe_path: Path,
    metadata_csv: Path,
) -> None:
    """Every ``add`` / ``remove`` / ``update`` rewrites the on-disk file.

    The QC tab's card-list-render callback subscribes to
    ``STORE_QC_RECIPE_REVISION``. The store is bumped by the same callbacks
    that mutate the recipe, so a working "revision" proxy is the on-disk
    JSON's serialized payload: every successful mutation rewrites it with
    a new entries list, and a no-op (``remove`` / ``update`` with an
    unknown id) leaves it unchanged.

    Spec §1228 — `STORE_QC_RECIPE_REVISION` row in FEATURES.md.
    """
    recipe = QcRecipe.load(output_root)
    # No file yet ⇒ no initial revision payload.
    assert not recipe_path.exists()

    # ``add`` writes the file for the first time.
    first_id = recipe.add(
        ExpectedVsDetectedCount,
        {
            "metadata": str(metadata_csv),
            "groupby": ["Metadata_ImageFile"],
        },
    )
    payload_after_add = recipe_path.read_text(encoding="utf-8")
    assert json.loads(payload_after_add)["checks"][0]["instance_id"] == first_id

    # A second ``add`` produces a strictly different payload.
    second_id = recipe.add(
        ExpectedVsDetectedCount,
        {
            "metadata": str(metadata_csv),
            "groupby": ["Metadata_ImageFile"],
        },
    )
    payload_after_second_add = recipe_path.read_text(encoding="utf-8")
    assert payload_after_second_add != payload_after_add
    assert len(json.loads(payload_after_second_add)["checks"]) == 2

    # ``update`` bumps the revision when it actually changes the entry.
    assert recipe.update(second_id, enabled=False) is True
    payload_after_update = recipe_path.read_text(encoding="utf-8")
    assert payload_after_update != payload_after_second_add

    # ``update`` for an unknown id is a no-op — payload stays unchanged.
    assert recipe.update("never-existed", enabled=True) is False
    assert recipe_path.read_text(encoding="utf-8") == payload_after_update

    # ``remove`` bumps the revision when it actually drops an entry.
    assert recipe.remove(first_id) is True
    payload_after_remove = recipe_path.read_text(encoding="utf-8")
    assert payload_after_remove != payload_after_update
    assert len(json.loads(payload_after_remove)["checks"]) == 1

    # ``remove`` for an unknown id is a no-op — payload stays unchanged.
    assert recipe.remove("never-existed") is False
    assert recipe_path.read_text(encoding="utf-8") == payload_after_remove


# ---------------------------------------------------------------------------
# instance_id contract
# ---------------------------------------------------------------------------


_INSTANCE_ID_RE = re.compile(r"^qc-[A-Za-z]+-[0-9a-f]{8}$")


class TestInstanceId:
    def test_instance_id_format(
        self, output_root: Path, metadata_csv: Path
    ) -> None:
        recipe = QcRecipe.load(output_root)
        new_id = recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
            },
        )
        # The check's class name attribute is "Count".
        assert re.match(r"^qc-Count-[0-9a-f]{8}$", new_id), new_id

    def test_instance_id_uniqueness_under_tight_loop(
        self, output_root: Path
    ) -> None:
        """Generate 10000 IDs via the private helper.

        Calling ``add()`` 10000 times would round-trip the file 10000
        times; the private ``_new_instance_id`` helper is exercised
        directly here so the test stays I/O-free and fast.
        """
        recipe = QcRecipe.load(output_root)
        ids = {recipe._new_instance_id("Count") for _ in range(10_000)}
        # No collisions in 10k draws (8 hex chars ⇒ 4-billion-space).
        assert len(ids) == 10_000


# ---------------------------------------------------------------------------
# Atomic save
# ---------------------------------------------------------------------------


class TestAtomicSave:
    def test_atomic_save_via_tmp_and_replace(
        self,
        output_root: Path,
        recipe_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        metadata_csv: Path,
    ) -> None:
        captured: list[tuple[str, str]] = []
        real_replace = os.replace

        def tracking_replace(src: Any, dst: Any) -> None:
            captured.append((str(src), str(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(
            "phenotypic.gui._qc_recipe.os.replace", tracking_replace
        )

        recipe = QcRecipe.load(output_root)
        recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
            },
        )

        assert len(captured) >= 1
        src, dst = captured[-1]
        assert src.endswith(".tmp")
        assert dst == str(recipe_path)
        # The temp file no longer exists after the replace.
        assert not Path(src).exists()
        assert Path(dst).exists()


class TestSaveCreatesViewerCacheDir:
    def test_save_creates_viewer_cache_dir_if_missing(
        self, output_root: Path, recipe_path: Path
    ) -> None:
        assert not recipe_path.parent.exists()

        recipe = QcRecipe.load(output_root)
        recipe.save()

        assert recipe_path.parent.is_dir()
        assert recipe_path.exists()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrentWriters:
    def test_concurrent_save_from_two_threads_leaves_valid_json(
        self, output_root: Path, recipe_path: Path, metadata_csv: Path
    ) -> None:
        recipe = QcRecipe.load(output_root)
        params: dict[str, Any] = {
            "metadata": str(metadata_csv),
            "groupby": ["Metadata_ImageFile"],
        }

        per_thread = 50

        def worker() -> None:
            for _ in range(per_thread):
                recipe.add(ExpectedVsDetectedCount, params)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # On-disk JSON parses cleanly.
        on_disk = json.loads(recipe_path.read_text(encoding="utf-8"))
        assert len(on_disk["checks"]) == per_thread * 2
        assert len(recipe.entries) == per_thread * 2

        ids_on_disk = {item["instance_id"] for item in on_disk["checks"]}
        ids_in_memory = {entry.instance_id for entry in recipe.entries}
        assert len(ids_on_disk) == per_thread * 2
        assert len(ids_in_memory) == per_thread * 2
        # In-memory and on-disk views agree on the IDs they hold.
        assert ids_on_disk == ids_in_memory


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_round_trip_json_schema(
        self, output_root: Path, metadata_csv: Path
    ) -> None:
        recipe = QcRecipe.load(output_root)
        first_params = {
            "metadata": str(metadata_csv),
            "groupby": ["Metadata_ImageFile"],
            "severity_warn": 0.05,
            "severity_fail": 0.10,
        }
        second_params = {
            "on": "Size_Area",
            "groupby": ["Metadata_Plate"],
            "severity_warn": 0.10,
            "severity_fail": 0.20,
            "min_replicates": 3,
        }
        first_id = recipe.add(ExpectedVsDetectedCount, first_params)
        second_id = recipe.add(
            ReplicateAgreement, second_params, enabled=False
        )

        reloaded = QcRecipe.load(output_root)

        assert reloaded.load_warnings == []
        assert len(reloaded.entries) == 2

        round_tripped = {entry.instance_id: entry for entry in reloaded.entries}
        assert set(round_tripped) == {first_id, second_id}

        first_entry = round_tripped[first_id]
        assert first_entry.cls is ExpectedVsDetectedCount
        assert first_entry.enabled is True
        assert first_entry.params == first_params

        second_entry = round_tripped[second_id]
        assert second_entry.cls is ReplicateAgreement
        assert second_entry.enabled is False
        assert second_entry.params == second_params


# ---------------------------------------------------------------------------
# instantiate()
# ---------------------------------------------------------------------------


class TestInstantiate:
    def test_instantiate_returns_qualitycheck_instances_for_enabled_only(
        self, output_root: Path, metadata_csv: Path
    ) -> None:
        recipe = QcRecipe.load(output_root)
        a_id = recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
            },
        )
        b_id = recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
            },
        )
        c_id = recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
            },
        )

        # Disable the middle one.
        recipe.update(b_id, enabled=False)

        built = recipe.instantiate()

        assert len(built) == 2
        returned_ids = [iid for iid, _ in built]
        assert returned_ids == [a_id, c_id]
        for _iid, instance in built:
            assert isinstance(instance, QualityCheck)
            assert isinstance(instance, ExpectedVsDetectedCount)

    def test_instantiate_failed_construction_records_warning_not_raises(
        self, output_root: Path, metadata_csv: Path
    ) -> None:
        recipe = QcRecipe.load(output_root)
        recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                # Column "Missing_Col" is not in the metadata CSV —
                # __init__ will raise KeyError.
                "groupby": ["Missing_Col"],
            },
        )
        good_id = recipe.add(
            ExpectedVsDetectedCount,
            {
                "metadata": str(metadata_csv),
                "groupby": ["Metadata_ImageFile"],
            },
        )

        before = len(recipe.load_warnings)
        built = recipe.instantiate()

        # The surviving entry is returned …
        assert len(built) == 1
        assert built[0][0] == good_id
        # … and a load warning was recorded for the failed one.
        assert len(recipe.load_warnings) > before
        new_warning = recipe.load_warnings[-1]
        assert new_warning.class_name == "ExpectedVsDetectedCount"
        assert "instantiation failed" in new_warning.reason


# ---------------------------------------------------------------------------
# Direct dataclass round-trip
# ---------------------------------------------------------------------------


class TestEntryRoundTrip:
    """Cover :meth:`QcRecipeEntry.to_dict` + :meth:`from_dict` directly."""

    def test_to_dict_then_from_dict_yields_equivalent_entry(self) -> None:
        original = QcRecipeEntry(
            cls=ExpectedVsDetectedCount,
            params={"groupby": ["A"], "on": "ObjectLabel"},
            instance_id="qc-Count-12345678",
            enabled=False,
        )

        roundtripped = QcRecipeEntry.from_dict(original.to_dict())

        assert isinstance(roundtripped, QcRecipeEntry)
        assert roundtripped.cls is ExpectedVsDetectedCount
        assert roundtripped.params == original.params
        assert roundtripped.instance_id == original.instance_id
        assert roundtripped.enabled is False

    def test_from_dict_unknown_class_returns_warning(self) -> None:
        result = QcRecipeEntry.from_dict(
            {
                "class": "DoesNotExist",
                "instance_id": "qc-bad-deadbeef",
                "enabled": True,
                "params": {},
            }
        )
        assert isinstance(result, QcRecipeLoadWarning)
        assert result.instance_id == "qc-bad-deadbeef"
        assert result.class_name == "DoesNotExist"
