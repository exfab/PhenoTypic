"""Pipeline-backed QC recipe: the ``qc`` section of ``pipeline.json``.

A *QC recipe* is an ordered list of configured
:class:`~phenotypic.analysis.abc_.QualityCheck` instances. Smart QC stores
that list as the ``qc`` array inside the canonical
``<output>/pipeline.json`` (sibling to ``operations``/``post``/``filters``/
``model``), so the checks travel with the pipeline and are recomputed by
the CLI on every recompile/remeasure.

This module provides three things:

* :class:`QcRecipeEntry` — one ``{instance_id, class, enabled, params}``
  entry. This is the unit :class:`~phenotypic._core._image_pipeline.ImagePipeline`
  stores in its ``qc`` list (``pipeline.get_qc()`` returns these), and the
  unit the pipeline (de)serializer reads/writes. Instantiation into a
  concrete :class:`QualityCheck` is **lazy** (see :meth:`QcRecipe.instantiate`
  and :func:`phenotypic.sdk_._qc_recipe._runner.run_qc`) so a single un-resolvable or
  un-constructable entry never blocks pipeline load or a QC run.
* :class:`QcRecipeLoadWarning` — why an entry was skipped at load or
  instantiate time, surfaced to the GUI as a banner.
* :class:`QcRecipe` — a thin adapter over a *file* (``pipeline.json``)
  that performs a **scoped** atomic read-modify-write of only the ``qc``
  array (operations/post/filters/model are preserved byte-for-byte) with
  an mtime-staleness guard mirroring
  :class:`phenotypic.gui.analysis._recipe_state.RecipeState`. It also
  exposes an explicitly authorized legacy-sidecar migration. Merely loading
  or binding a viewer never performs that migration.

Import hygiene: at module load this module imports only
:class:`~phenotypic.analysis.abc_.QualityCheck` from the analysis layer
and the ``DIR_DELIVERABLES`` constant from :mod:`phenotypic.sdk_` (plus
stdlib). ``phenotypic.sdk_`` does not import this submodule, so that edge
is safe. It must **not** import ``_core``/``_cli``/``gui`` at module load —
``_core`` imports :class:`QcRecipeEntry` from here, so a back-edge would
create a cycle. The path helpers (``pipeline_json_path`` /
``resolve_pipeline_config_path``) are lazy-imported inside :meth:`QcRecipe.load`
        and the atomic-write helper inside :meth:`QcRecipe._write_qc_array`.
"""

from __future__ import annotations

import inspect
import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from phenotypic.analysis.abc_ import QualityCheck
from phenotypic.sdk_ import DIR_DELIVERABLES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from phenotypic.sdk_ import BundleLayout

logger = logging.getLogger(__name__)

#: On-disk filename for the *legacy* QC recipe sidecar inside
#: ``<output>/.viewer_cache/``. Folded into ``pipeline.json``'s ``qc``
#: array by :meth:`QcRecipe.migrate_from_sidecar` and then ignored.
QC_RECIPE_FILENAME: str = "qc_recipe.json"

#: Hidden directory inside an output root that held viewer-managed state
#: (including the legacy QC sidecar). Kept local so this module stays free
#: of GUI imports.
VIEWER_CACHE_DIRNAME: str = ".viewer_cache"

#: Schema version of a legacy sidecar payload. Only consulted by the
#: one-time migration.
QC_RECIPE_VERSION: int = 1

#: Key under which the QC entries live in ``pipeline.json``.
_QC_KEY: str = "qc"


def _resolve_check_class(class_name: str) -> type[QualityCheck] | None:
    """Find a :class:`QualityCheck` subclass by name in ``phenotypic.analysis``.

    Walks :mod:`phenotypic.analysis` via :func:`inspect.getmembers`,
    matching the discovery pattern used by the GUI operation registry and
    the pipeline (de)serializer's class resolution.

    Args:
        class_name: The unqualified class name to resolve (e.g.
            ``"ExpectedVsDetectedCount"``).

    Returns:
        The matching :class:`QualityCheck` subclass, or ``None`` if no
        such class is exported from :mod:`phenotypic.analysis`.
    """
    import phenotypic.analysis as analysis_module

    for name, obj in inspect.getmembers(analysis_module, inspect.isclass):
        if name != class_name:
            continue
        if not issubclass(obj, QualityCheck) or obj is QualityCheck:
            continue
        return obj
    return None


@dataclass
class QcRecipeEntry:
    """One configured :class:`QualityCheck` in a pipeline's ``qc`` list.

    Each entry pairs a check subclass with the constructor kwargs the user
    picked. The instance is *not* eagerly built — :meth:`instantiate`
    constructs the concrete :class:`QualityCheck` on demand so an
    un-constructable params dict downgrades to a load warning instead of
    breaking pipeline load or a QC run.

    Attributes:
        cls: The :class:`QualityCheck` subclass.
        params: Constructor kwargs as plain JSON-friendly values (e.g. a
            metadata *path* string, never a DataFrame).
        instance_id: Stable identifier of the form ``f"qc-{name}-{8-hex}"``.
            Generated once at "add" time and persisted across the
            pipeline's lifetime so GUI per-card component IDs and
            ``review_state.json`` keys can attach to it.
        enabled: When :data:`False`, the entry is preserved in
            ``pipeline.json`` but skipped by :meth:`instantiate` /
            :func:`run_qc`. Lets the user toggle a check off without
            losing its config.
    """

    cls: type[QualityCheck]
    params: dict[str, Any]
    instance_id: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the on-disk ``{instance_id, class, enabled, params}`` shape.

        This is the single source of truth for the JSON shape of a ``qc``
        entry; the pipeline (de)serializer reuses it so the two never drift.

        The ``params`` are **canonicalized through the check's own pydantic
        ``model_dump(mode="json")``** whenever the entry can be instantiated.
        This guarantees the persisted params are exactly the check's
        JSON-serializable surface — most importantly it persists an
        :class:`ExpectedVsDetectedCount` ``metadata`` *path* under the
        ``metadata`` key (a resolved in-memory frame serializes to ``None``,
        i.e. it cannot round-trip), rather than echoing whatever raw kwargs
        the caller happened to pass. When the entry cannot be built (bad
        params), the
        raw :attr:`params` are emitted verbatim so a misconfigured-but-saved
        check still round-trips its config rather than vanishing.

        Returns:
            A JSON-native dict with keys ``instance_id``, ``class``,
            ``enabled``, ``params``.
        """
        return {
            "instance_id": self.instance_id,
            "class": self.cls.__name__,
            "enabled": self.enabled,
            "params": self._dump_params(),
        }

    def _dump_params(self) -> dict[str, Any]:
        """Return the JSON-serializable params for this entry.

        Prefers the check's ``model_dump(mode="json")`` (so path-vs-frame
        normalization and ``PrivateAttr`` exclusion apply); falls back to
        the raw :attr:`params` when the check cannot be instantiated.

        Returns:
            A JSON-native params dict.
        """
        try:
            return self.cls(**self.params).model_dump(mode="json")
        except Exception:  # noqa: BLE001 - keep raw config for unbuildable entries
            return dict(self.params)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "QcRecipeEntry | QcRecipeLoadWarning":
        """Inverse of :meth:`to_dict` with class-resolution fallback.

        Walks :mod:`phenotypic.analysis` for the named subclass. When the
        class cannot be resolved (renamed/removed since the entry was
        saved), a :class:`QcRecipeLoadWarning` is returned instead of
        raising — the caller (pipeline loader or GUI) drops the entry and
        surfaces the warning rather than failing the whole load.

        Args:
            data: Dict in the on-disk schema.

        Returns:
            Either a fresh :class:`QcRecipeEntry` or a
            :class:`QcRecipeLoadWarning` explaining why the entry was
            skipped.
        """
        class_name = data.get("class", "")
        instance_id = data.get("instance_id", "?")
        resolved_cls = _resolve_check_class(class_name)
        if resolved_cls is None:
            return QcRecipeLoadWarning(
                instance_id=instance_id,
                class_name=class_name,
                reason="class not found in phenotypic.analysis",
            )
        return cls(
            cls=resolved_cls,
            params=dict(data.get("params", {})),
            instance_id=instance_id,
            enabled=bool(data.get("enabled", True)),
        )

    def instantiate(self) -> QualityCheck:
        """Construct the concrete :class:`QualityCheck` from :attr:`params`.

        Returns:
            A live check instance.

        Raises:
            Exception: Whatever the check's pydantic validation raises on
                a bad params dict (e.g. a missing metadata file, a
                ``KeyError`` on an absent ``groupby`` column). Callers that
                want tolerance wrap this in try/except — see
                :func:`phenotypic.sdk_._qc_recipe._runner.run_qc` and
                :meth:`QcRecipe.instantiate`.
        """
        return self.cls(**self.params)


@dataclass
class QcRecipeLoadWarning:
    """Why a recipe entry failed to resolve at load or instantiate time.

    Surfaced to the QC tab so the user sees a banner explaining the
    problem rather than a silent skip. Flavors:

    * **Class-resolution failures** (:meth:`QcRecipeEntry.from_dict`): the
      stored class is no longer exported from :mod:`phenotypic.analysis`.
    * **Construction failures** (:meth:`QcRecipe.instantiate`): the class
      resolved but its constructor raised (e.g. bad metadata path,
      ``KeyError`` on a missing ``groupby`` column, or an
      ``ExpectedVsDetectedCount`` serialized from an in-memory frame whose
      ``metadata`` round-tripped as ``None`` and cannot be re-read).
    * **Whole-file corruption** (:meth:`QcRecipe.load`): the JSON itself is
      malformed. The synthetic ``instance_id="__file__"`` distinguishes
      this from per-entry failures.

    Attributes:
        instance_id: Identifier of the offending entry, or ``"__file__"``
            for whole-file failures.
        class_name: Class name from the entry. Empty string for whole-file
            failures.
        reason: Human-readable explanation suitable for banner display.
    """

    instance_id: str
    class_name: str
    reason: str


@dataclass
class QcRecipe:
    """Scoped, mtime-guarded view of the ``qc`` array in ``pipeline.json``.

    Loaded at viewer-app boot via :meth:`load` and mutated in place by QC
    tab callbacks (:meth:`add`, :meth:`remove`, :meth:`update`). Each
    mutating method performs a **scoped atomic read-modify-write**: it
    re-reads the on-disk ``pipeline.json``, replaces *only* the ``qc`` key
    (operations/post/filters/model are preserved exactly as written by the
    CLI), and atomically swaps the file in. An mtime guard refuses the
    write when the file changed since load (typical cause: a CLI
    recompile-mode run happened while the viewer was open), mirroring
    :class:`phenotypic.gui.analysis._recipe_state.RecipeState`.

    Unlike the legacy sidecar :class:`QcRecipe`, the recipe is *not* the
    sole owner of its file — the CLI also writes ``pipeline.json`` — so the
    staleness guard matters here.

    Attributes:
        path: Absolute path to ``<output>/pipeline.json``.
        entries: Ordered list of configured checks (the in-memory mirror of
            the ``qc`` array). Display + run order is preserved.
        seed_mtime_ns: Nanosecond mtime of :attr:`path` as last observed by
            this instance. ``None`` means the file did not exist at load.
            Refreshed by :meth:`load` and after each successful write.
        source_path: Optional path whose document seeded this recipe when it
            differs from :attr:`path` (currently legacy ``pipeline.json``).
        load_warnings: Class-resolution / construction / corruption
            failures collected during :meth:`load` and :meth:`instantiate`.
    """

    path: Path
    entries: list[QcRecipeEntry] = field(default_factory=list)
    seed_mtime_ns: int | None = None
    source_path: Path | None = None
    load_warnings: list[QcRecipeLoadWarning] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # ------------------------------------------------------------------ #
    # Load
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, output_root_path: Path) -> "QcRecipe":
        """Load the ``qc`` array from ``<output_root_path>/pipeline.json``.

        Reads the pipeline JSON and parses its ``qc`` array via
        :meth:`QcRecipeEntry.from_dict`. Behaviour by file state:

        * **Missing file** — empty recipe, no warnings. The file is not
          created until the first :meth:`save`-style mutation.
        * **Valid JSON** — each ``qc`` entry is parsed; class-resolution
          failures become :class:`QcRecipeLoadWarning` entries and the
          offending entry is dropped. A pipeline with no ``qc`` key (legacy
          or QC-free) loads as an empty recipe.
        * **Corrupt JSON** — empty recipe plus a single
          ``QcRecipeLoadWarning(instance_id="__file__", ...)``. The file is
          left untouched so the user can recover it.

        Args:
            output_root_path: Path to the results-viewer's output root
                (the dir that holds ``pipeline.json``).

        Returns:
            A :class:`QcRecipe` ready for in-place mutation.
        """
        from phenotypic.sdk_ import (
            pipeline_json_path,
            resolve_pipeline_config_path,
        )

        pipeline_path = pipeline_json_path(Path(output_root_path))
        read_path = resolve_pipeline_config_path(Path(output_root_path))
        return cls._load_from_paths(read_path, pipeline_path)

    @classmethod
    def from_layout(cls, layout: "BundleLayout") -> "QcRecipe":
        """Load the ``qc`` array from a resolved :class:`BundleLayout`.

        The :class:`BundleLayout`-aware sibling of :meth:`load`. Anchors the
        pipeline config on ``layout.deliverables_base`` directly, so a
        standalone deliverables bundle (whose ``output_root is None``) resolves
        ``pipeline.json`` *inside the bundle* rather than via
        ``deliverables_dir(output_root)`` — which would double-join when the
        viewer's ``root`` is already the deliverables folder.

        Args:
            layout: Resolved bundle topology.

        Returns:
            A :class:`QcRecipe` ready for in-place mutation.
        """
        return cls._load_from_paths(
            layout.resolved_pipeline_config_path, layout.pipeline_config_path
        )

    @classmethod
    def _load_from_paths(
        cls, read_path: Path, pipeline_path: Path
    ) -> "QcRecipe":
        """Build a recipe from explicit read + canonical-write paths.

        Shared core of :meth:`load` and :meth:`from_layout`. ``read_path`` is
        the existing config to parse (canonical typed or legacy ``.json``);
        ``pipeline_path`` is the canonical typed path future writes target.
        """
        if not read_path.exists():
            return cls(path=pipeline_path)

        try:
            payload = json.loads(read_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning(
                "pipeline config at %s could not be parsed; loading empty QC "
                "recipe. On-disk file left untouched. Error: %s",
                read_path,
                exc,
            )
            return cls(
                path=pipeline_path,
                load_warnings=[
                    QcRecipeLoadWarning(
                        instance_id="__file__",
                        class_name="",
                        reason=f"invalid JSON: {exc}",
                    )
                ],
            )
        except OSError as exc:
            logger.warning(
                "pipeline config at %s could not be read; loading empty QC "
                "recipe. Error: %s",
                read_path,
                exc,
            )
            return cls(
                path=pipeline_path,
                load_warnings=[
                    QcRecipeLoadWarning(
                        instance_id="__file__",
                        class_name="",
                        reason=f"read error: {exc}",
                    )
                ],
            )

        entries, warnings = cls._parse_entries(payload)

        try:
            mtime: int | None = read_path.stat().st_mtime_ns
        except OSError:
            mtime = None

        return cls(
            path=pipeline_path,
            entries=entries,
            seed_mtime_ns=mtime,
            source_path=read_path if read_path != pipeline_path else None,
            load_warnings=warnings,
        )

    @staticmethod
    def _parse_entries(
        payload: Any,
    ) -> tuple[list[QcRecipeEntry], list[QcRecipeLoadWarning]]:
        """Parse the ``qc`` array out of a parsed ``pipeline.json`` payload.

        Args:
            payload: The parsed JSON (expected to be a dict; anything else
                yields an empty recipe).

        Returns:
            ``(entries, warnings)`` — resolvable entries in order, plus one
            warning per dropped entry.
        """
        entries: list[QcRecipeEntry] = []
        warnings: list[QcRecipeLoadWarning] = []
        raw = payload.get(_QC_KEY, []) if isinstance(payload, dict) else []
        if not isinstance(raw, list):
            return entries, warnings
        for item in raw:
            if not isinstance(item, dict):
                continue
            parsed = QcRecipeEntry.from_dict(item)
            if isinstance(parsed, QcRecipeLoadWarning):
                warnings.append(parsed)
            else:
                entries.append(parsed)
        return entries, warnings

    # ------------------------------------------------------------------ #
    # Staleness + scoped write
    # ------------------------------------------------------------------ #

    def is_stale(self) -> bool:
        """Return ``True`` when ``pipeline.json`` changed since load.

        The CLI seeds ``pipeline.json`` on every aggregate run; if the user
        re-runs the CLI while a viewer session is open this returns
        ``True`` until they :meth:`reload`. Callers refuse to mutate until
        the staleness clears so a stale ``qc`` array never clobbers a fresh
        CLI seed (which may carry new operations/post/model).
        """
        if self.seed_mtime_ns is None:
            return False
        tracked_path = self._tracked_path()
        try:
            current = tracked_path.stat().st_mtime_ns
        except FileNotFoundError:
            return True
        return current != self.seed_mtime_ns

    def reload(self) -> None:
        """Re-read the on-disk ``qc`` array, replacing :attr:`entries`."""
        with self._lock:
            output_root = (
                self.path.parent.parent
                if self.path.parent.name == DIR_DELIVERABLES
                else self.path.parent
            )
            fresh = QcRecipe.load(output_root)
            self.entries = fresh.entries
            self.seed_mtime_ns = fresh.seed_mtime_ns
            self.source_path = fresh.source_path
            self.load_warnings = fresh.load_warnings

    def _tracked_path(self) -> Path:
        """Return the path whose mtime should be compared against the seed."""
        if self.path.exists() or self.source_path is None:
            return self.path
        return self.source_path

    def _document_read_path(self) -> Path:
        """Return the best existing document to merge before a scoped write."""
        if self.path.exists():
            return self.path
        if self.source_path is not None and self.source_path.exists():
            return self.source_path
        return self.path

    def _write_qc_array(self) -> bool:
        """Atomically rewrite *only* the ``qc`` key of ``pipeline.json``.

        Re-reads the current on-disk pipeline (so operations/post/filters/
        model are preserved exactly as the CLI last wrote them), substitutes
        the in-memory :attr:`entries` for the ``qc`` array, and atomic-writes
        the merged document. When ``pipeline.json`` does not yet exist, a
        minimal ``{"qc": [...]}`` document is created.

        Returns:
            ``True`` on a successful write (and :attr:`seed_mtime_ns` is
            refreshed); ``False`` when the file was stale (caller must
            :meth:`reload` first) or the atomic write failed. Failures
            other than staleness are logged at WARNING.
        """
        from phenotypic.sdk_ import (
            atomic_write_json,
            pipeline_publication_lock,
        )

        with self._lock:
            with pipeline_publication_lock(self.path):
                if self.is_stale():
                    logger.warning(
                        "Refusing to write qc array to %s — mtime changed "
                        "since load (likely a CLI recompile-mode run). Reload "
                        "before saving again.",
                        self.path,
                    )
                    return False

                document: dict[str, Any] = {}
                read_path = self._document_read_path()
                if read_path.exists():
                    try:
                        loaded = json.loads(
                            read_path.read_text(encoding="utf-8")
                        )
                    except (json.JSONDecodeError, OSError):
                        logger.warning(
                            "Could not re-read %s before scoped qc write; "
                            "refusing to replace the existing pipeline.",
                            read_path,
                            exc_info=True,
                        )
                        return False
                    if not isinstance(loaded, dict):
                        logger.warning(
                            "Pipeline at %s is not a JSON object; refusing "
                            "to replace it with a minimal QC document.",
                            read_path,
                        )
                        return False
                    document = loaded

                try:
                    document[_QC_KEY] = [
                        entry.to_dict() for entry in self.entries
                    ]
                    atomic_write_json(self.path, document, sort_keys=False)
                except Exception:
                    logger.warning(
                        "Atomic write of qc array failed for %s",
                        self.path,
                        exc_info=True,
                    )
                    return False

                try:
                    self.seed_mtime_ns = self.path.stat().st_mtime_ns
                except OSError:
                    self.seed_mtime_ns = None
                self.source_path = None
                return True

    # ------------------------------------------------------------------ #
    # Mutators (each performs a scoped atomic write)
    # ------------------------------------------------------------------ #

    def _new_instance_id(self, name: str) -> str:
        """Generate a fresh ``qc-<name>-<8 hex>`` instance ID.

        Uses :func:`secrets.token_hex` for collision resistance under
        test-harness parallelism (``time.time()`` suffixes would collide on
        back-to-back adds within the same second and break per-card IDs).

        Args:
            name: The check's class-level ``name`` (e.g. ``"Count"``).

        Returns:
            A new instance ID of the form ``qc-<name>-<8 hex chars>``.
        """
        return f"qc-{name}-{secrets.token_hex(4)}"

    def add(
        self,
        check_cls: type[QualityCheck],
        params: dict[str, Any],
        *,
        enabled: bool = True,
    ) -> str | None:
        """Append a new entry and persist the ``qc`` array.

        Args:
            check_cls: The :class:`QualityCheck` subclass to add.
            params: Constructor kwargs (JSON-friendly values only).
            enabled: Whether the new entry should run.

        Returns:
            The generated ``instance_id`` on success, or ``None`` when the
            scoped write was refused (stale file) — in which case the
            in-memory append is rolled back so memory and disk stay
            consistent.
        """
        with self._lock:
            instance_id = self._new_instance_id(check_cls.name)
            entry = QcRecipeEntry(
                cls=check_cls,
                params=dict(params),
                instance_id=instance_id,
                enabled=enabled,
            )
            self.entries.append(entry)
            if not self._write_qc_array():
                self.entries.pop()
                return None
            return instance_id

    def remove(self, instance_id: str) -> bool:
        """Remove the entry whose ``instance_id`` matches and persist.

        Args:
            instance_id: The identifier returned by a previous :meth:`add`.

        Returns:
            ``True`` if the entry existed and the scoped write succeeded.
            ``False`` if no entry matched, or if the write was refused
            (stale file) — in which case the removal is rolled back.
        """
        with self._lock:
            for index, entry in enumerate(self.entries):
                if entry.instance_id == instance_id:
                    removed = self.entries.pop(index)
                    if not self._write_qc_array():
                        self.entries.insert(index, removed)
                        return False
                    return True
            return False

    def update(
        self,
        instance_id: str,
        *,
        params: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> bool:
        """Mutate an entry's params and/or enabled flag in place and persist.

        Args:
            instance_id: Target entry's identifier.
            params: When not ``None``, replaces the entry's params wholesale.
            enabled: When not ``None``, replaces the entry's enabled flag.

        Returns:
            ``True`` if the entry existed and the scoped write succeeded.
            ``False`` if no entry matched, or the write was refused (stale
            file) — in which case the change is rolled back.
        """
        with self._lock:
            for entry in self.entries:
                if entry.instance_id != instance_id:
                    continue
                prev_params = entry.params
                prev_enabled = entry.enabled
                if params is not None:
                    entry.params = dict(params)
                if enabled is not None:
                    entry.enabled = enabled
                if not self._write_qc_array():
                    entry.params = prev_params
                    entry.enabled = prev_enabled
                    return False
                return True
            return False

    # ------------------------------------------------------------------ #
    # Instantiate (lazy, tolerant)
    # ------------------------------------------------------------------ #

    def instantiate(self) -> list[tuple[str, QualityCheck]]:
        """Build concrete :class:`QualityCheck` objects for enabled entries.

        Construction failures (bad metadata path, missing ``groupby``
        column, an ``ExpectedVsDetectedCount`` rebuilt from a frame-only
        dump) land in :attr:`load_warnings` rather than raising. Disabled
        entries are skipped silently.

        Returns:
            ``(instance_id, instance)`` tuples in recipe order. Entries that
            failed to instantiate are absent; the rest are still returned so
            one bad config does not block the others.
        """
        with self._lock:
            built: list[tuple[str, QualityCheck]] = []
            for entry in self.entries:
                if not entry.enabled:
                    continue
                try:
                    instance = entry.instantiate()
                except Exception as exc:  # noqa: BLE001 - surfaced as warning
                    logger.warning(
                        "Failed to instantiate QC check %s (%s): %s",
                        entry.instance_id,
                        entry.cls.__name__,
                        exc,
                    )
                    self.load_warnings.append(
                        QcRecipeLoadWarning(
                            instance_id=entry.instance_id,
                            class_name=entry.cls.__name__,
                            reason=f"instantiation failed: {exc}",
                        )
                    )
                    continue
                built.append((entry.instance_id, instance))
            return built

    # ------------------------------------------------------------------ #
    # One-time sidecar migration
    # ------------------------------------------------------------------ #

    @classmethod
    def migrate_from_sidecar(
        cls,
        output_root_path: Path,
        *,
        allow_write: bool = False,
    ) -> bool:
        """Explicitly fold a legacy QC sidecar into ``pipeline.json``.

        Smart QC's predecessor stored the QC recipe in a standalone
        ``<output>/.viewer_cache/qc_recipe.json`` sidecar. Binding and
        discovery historically called this method, so writes are now gated by
        the explicit ``allow_write`` argument. The default is a pure no-op.

        The migration is a no-op (returns ``False``) when:

        * ``allow_write`` is false,
        * the sidecar does not exist, or
        * ``pipeline.json`` already has a non-empty ``qc`` array (the
          pipeline is already the source of truth — never overwrite it).

        It is **atomic and scoped**: only the ``qc`` key of ``pipeline.json``
        is touched, via the same mtime-guarded path as the mutators. The
        sidecar is renamed only after the pipeline write succeeds.

        Args:
            output_root_path: Path to the output root holding both files.
            allow_write: Explicit authorization for the migration write.

        Returns:
            ``True`` when entries were migrated and persisted; ``False`` when
            not explicitly authorized, there was nothing to do, or the scoped
            write was refused.
        """
        if not allow_write:
            return False
        output_root = Path(output_root_path)
        sidecar = output_root / VIEWER_CACHE_DIRNAME / QC_RECIPE_FILENAME
        if not sidecar.exists():
            return False

        recipe = cls.load(output_root)
        if recipe.entries:
            # pipeline.json already carries qc entries — it wins. Retire the
            # stale sidecar so we don't keep re-checking it.
            cls._retire_sidecar(sidecar)
            return False

        sidecar_entries, sidecar_warnings = cls._read_sidecar(sidecar)
        recipe.load_warnings.extend(sidecar_warnings)
        if not sidecar_entries:
            cls._retire_sidecar(sidecar)
            return False

        recipe.entries = sidecar_entries
        if not recipe._write_qc_array():
            # Stale pipeline.json / write failure: leave the sidecar in
            # place so a later reload + retry can still migrate it.
            return False

        cls._retire_sidecar(sidecar)
        return True

    @staticmethod
    def _read_sidecar(
        sidecar: Path,
    ) -> tuple[list[QcRecipeEntry], list[QcRecipeLoadWarning]]:
        """Parse a legacy sidecar's ``checks`` array into entries.

        Args:
            sidecar: Path to ``.viewer_cache/qc_recipe.json``.

        Returns:
            ``(entries, warnings)``; an unreadable/corrupt sidecar yields
            empty entries plus a ``__file__`` warning.
        """
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Legacy QC sidecar at %s could not be read; skipping "
                "migration. Error: %s",
                sidecar,
                exc,
            )
            return [], [
                QcRecipeLoadWarning(
                    instance_id="__file__",
                    class_name="",
                    reason=f"sidecar unreadable: {exc}",
                )
            ]

        entries: list[QcRecipeEntry] = []
        warnings: list[QcRecipeLoadWarning] = []
        checks = payload.get("checks", []) if isinstance(payload, dict) else []
        for item in checks:
            if not isinstance(item, dict):
                continue
            parsed = QcRecipeEntry.from_dict(item)
            if isinstance(parsed, QcRecipeLoadWarning):
                warnings.append(parsed)
            else:
                entries.append(parsed)
        return entries, warnings

    @staticmethod
    def _retire_sidecar(sidecar: Path) -> None:
        """Rename a folded sidecar to ``*.migrated`` (idempotency marker).

        Best-effort: failure to rename is logged at WARNING but does not
        raise — the empty-pipeline.json guard still prevents a double-fold
        of the same entries.
        """
        try:
            sidecar.replace(sidecar.with_suffix(sidecar.suffix + ".migrated"))
        except OSError:
            logger.warning(
                "Could not retire migrated QC sidecar %s",
                sidecar,
                exc_info=True,
            )
