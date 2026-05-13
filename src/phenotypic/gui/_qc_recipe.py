"""Sidecar persistence for the results-viewer's QC tab recipe.

The results-viewer QC tab is configured by a *recipe*: an ordered list of
:class:`~phenotypic.analysis.abc_.QualityCheck` instances together with
their parameter dicts. The recipe lives on disk at
``<output>/.viewer_cache/qc_recipe.json`` and is loaded once at
``create_app()`` boot, then mutated in place by the QC tab callbacks
(add / remove / update) which call :meth:`QcRecipe.save` on every edit.

The on-disk schema is documented in the spec at
``docs/superpowers/specs/2026-05-12-qc-analysis-and-gui-design.md``
lines 619-733. Highlights:

* Atomic ``.tmp`` + :func:`os.replace` writes survive process kills.
* Class-name resolution walks :mod:`phenotypic.analysis` dynamically;
  classes renamed since the recipe was written produce
  :class:`QcRecipeLoadWarning` entries rather than blocking the boot.
* No mtime-staleness refusal pattern: the recipe is owned solely by the
  viewer (unlike ``pipeline.json``, which the CLI also writes), so
  :meth:`QcRecipe.save` is unconditional. Two concurrent viewer
  sessions writing to the same output dir is an unsupported
  configuration.

This module mirrors the API of
:class:`~phenotypic.gui.analysis._recipe_state.RecipeState` but is
deliberately simpler (no ``ImagePipeline`` round-tripping, no
``is_stale()``).
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from phenotypic.analysis.abc_ import QualityCheck

logger = logging.getLogger(__name__)

#: Hidden directory inside the results-viewer output root that holds
#: viewer-managed state (DZI tile caches, QC recipe sidecar, etc.).
#: Local copy of :data:`phenotypic.gui._config.VIEWER_CACHE_DIRNAME` to
#: keep this module free of GUI-shell imports — :mod:`_config` itself
#: imports nothing else from the GUI layer, but a top-level import is
#: still avoided so that tools (CLI fixtures, smoke tests) can import
#: :class:`QcRecipe` without a Dash dependency in play.
VIEWER_CACHE_DIRNAME: str = ".viewer_cache"

#: On-disk filename for the QC recipe sidecar inside
#: ``<output>/.viewer_cache/``.
QC_RECIPE_FILENAME: str = "qc_recipe.json"

#: Schema version embedded in the on-disk JSON. Bump whenever the
#: serialization format changes incompatibly; :meth:`QcRecipe.load`
#: should grow a migration branch at that time.
QC_RECIPE_VERSION: int = 1


def _resolve_check_class(class_name: str) -> type[QualityCheck] | None:
    """Find a :class:`QualityCheck` subclass by name within ``phenotypic.analysis``.

    Walks :mod:`phenotypic.analysis` via :func:`inspect.getmembers`,
    matching the discovery pattern used by
    :meth:`phenotypic.gui._operation_registry.OperationRegistry._discover_analyzers`.

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
    """One configured :class:`QualityCheck` instance in a :class:`QcRecipe`.

    Each entry pairs a check subclass with the constructor kwargs the
    user picked in the QC tab. The instance is *not* eagerly built —
    :meth:`QcRecipe.instantiate` constructs concrete
    :class:`QualityCheck` objects on demand so an unparseable params
    dict downgrades to a load warning instead of breaking the boot.

    Attributes:
        cls: The :class:`QualityCheck` subclass.
        params: Constructor kwargs as plain JSON-friendly values
            (DataFrame inputs are normalized to string paths by the GUI
            form before they reach this dataclass).
        instance_id: Stable identifier of the form
            ``f"qc-{name}-{8-hex}"``. Generated at "Add check" time and
            persisted across the recipe's lifetime so per-card Dash
            component IDs can attach to it.
        enabled: When :data:`False`, the entry is preserved on disk but
            :meth:`QcRecipe.instantiate` skips it. Lets the user toggle
            a check off without losing its config.
    """

    cls: type[QualityCheck]
    params: dict[str, Any]
    instance_id: str
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the on-disk JSON shape.

        Returns:
            A dict matching the per-check schema documented in the spec
            (``instance_id``, ``class``, ``enabled``, ``params``).
        """
        return {
            "instance_id": self.instance_id,
            "class": self.cls.__name__,
            "enabled": self.enabled,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, Any]
    ) -> "QcRecipeEntry | QcRecipeLoadWarning":
        """Inverse of :meth:`to_dict` with class-resolution fallback.

        Walks :mod:`phenotypic.analysis` looking for the named subclass.
        If the class cannot be resolved (typical cause: a check was
        renamed or removed since the recipe was last saved), a
        :class:`QcRecipeLoadWarning` is returned instead of raising —
        the QC tab surfaces the warning in a banner and leaves the
        on-disk file untouched until the user takes a UI action.

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


@dataclass
class QcRecipeLoadWarning:
    """Why a recipe entry failed to resolve at load (or instantiate) time.

    Surfaced to the QC tab so the user sees a banner explaining the
    problem rather than a silent skip. Two flavors of warning use this
    dataclass:

    * **Class-resolution failures** (``QcRecipeEntry.from_dict``): the
      JSON named a class that is no longer exported from
      :mod:`phenotypic.analysis`.
    * **Construction failures** (:meth:`QcRecipe.instantiate`): the
      class resolved but ``__init__`` raised (e.g. bad metadata path,
      ``KeyError`` on a missing ``groupby`` column).
    * **Whole-file corruption** (:meth:`QcRecipe.load`): the JSON
      itself is malformed. The synthetic ``instance_id="__file__"``
      distinguishes this case from per-entry failures.

    Attributes:
        instance_id: Identifier of the offending entry, or
            ``"__file__"`` for whole-file failures.
        class_name: Class name from the JSON entry. Empty string when
            the warning is whole-file (no specific entry).
        reason: Human-readable explanation suitable for banner display.
    """

    instance_id: str
    class_name: str
    reason: str


@dataclass
class QcRecipe:
    """In-memory + on-disk view of the QC tab's recipe sidecar.

    Loaded at viewer-app boot via :meth:`QcRecipe.load` and mutated in
    place by QC tab callbacks (:meth:`add`, :meth:`remove`,
    :meth:`update`). Each mutating method calls :meth:`save` immediately
    so the on-disk file always reflects the in-memory state — if the
    viewer crashes between events, no edits are lost.

    Unlike
    :class:`~phenotypic.gui.analysis._recipe_state.RecipeState`, this
    class does **not** implement an ``is_stale()`` / mtime-refusal
    pattern: the recipe is owned solely by the viewer, so there is no
    external writer to race against. See spec lines 727-733 for the
    rationale.

    Attributes:
        path: Absolute path to ``<output>/.viewer_cache/qc_recipe.json``.
        entries: Ordered list of configured checks. The order is the
            display order in the QC tab and is preserved across
            save/load round-trips.
        seed_mtime_ns: Nanosecond mtime of :attr:`path` as last observed
            by this instance. ``None`` means the file did not exist when
            the recipe was loaded. Refreshed by :meth:`load` and
            :meth:`save`.
        load_warnings: Class-resolution and construction failures
            collected during :meth:`load` and :meth:`instantiate`.
            Rendered as a banner by the QC tab.
    """

    path: Path
    entries: list[QcRecipeEntry] = field(default_factory=list)
    seed_mtime_ns: int | None = None
    load_warnings: list[QcRecipeLoadWarning] = field(default_factory=list)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False
    )

    @classmethod
    def load(cls, output_root_path: Path) -> "QcRecipe":
        """Load (or seed) the recipe under *output_root_path*.

        Resolves the on-disk path to
        ``output_root_path / VIEWER_CACHE_DIRNAME / QC_RECIPE_FILENAME``
        and reads it. Behaviour by file state:

        * **Missing file**: returns an empty recipe with no warnings.
          The file is *not* created until the first :meth:`save`.
        * **Valid JSON**: each entry is parsed via
          :meth:`QcRecipeEntry.from_dict`; class-resolution failures
          become :class:`QcRecipeLoadWarning` entries in
          :attr:`load_warnings` and the offending entry is dropped from
          :attr:`entries`.
        * **Corrupt JSON**: returns an empty recipe with a single
          ``QcRecipeLoadWarning(instance_id="__file__", ...)`` carrying
          the parser error. The on-disk file is **not** modified — the
          user gets a chance to recover the file from VCS or by hand
          before any UI action triggers a save.

        Args:
            output_root_path: Path to the results-viewer's output root.

        Returns:
            A :class:`QcRecipe` ready for in-place mutation +
            :meth:`save`.
        """
        recipe_path = Path(output_root_path) / VIEWER_CACHE_DIRNAME / QC_RECIPE_FILENAME

        if not recipe_path.exists():
            return cls(path=recipe_path)

        try:
            raw_text = recipe_path.read_text(encoding="utf-8")
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.warning(
                "QC recipe at %s could not be parsed; loading empty recipe. "
                "On-disk file left untouched. Error: %s",
                recipe_path,
                exc,
            )
            warning = QcRecipeLoadWarning(
                instance_id="__file__",
                class_name="",
                reason=f"invalid JSON: {exc}",
            )
            return cls(path=recipe_path, load_warnings=[warning])
        except OSError as exc:
            logger.warning(
                "QC recipe at %s could not be read; loading empty recipe. "
                "Error: %s",
                recipe_path,
                exc,
            )
            warning = QcRecipeLoadWarning(
                instance_id="__file__",
                class_name="",
                reason=f"read error: {exc}",
            )
            return cls(path=recipe_path, load_warnings=[warning])

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

        mtime: int | None
        try:
            mtime = recipe_path.stat().st_mtime_ns
        except OSError:
            mtime = None

        return cls(
            path=recipe_path,
            entries=entries,
            seed_mtime_ns=mtime,
            load_warnings=warnings,
        )

    def save(self) -> None:
        """Atomically write the in-memory recipe to :attr:`path`.

        Writes to ``<path>.tmp`` and uses :func:`os.replace` to swap the
        new file in place, so a crash mid-write cannot leave a
        half-written JSON. Mirrors
        :func:`phenotypic._cli._cli_output_manager._atomic_write` and
        the ``FilteredMeasurements._save_locked`` pattern.

        The parent directory (``<output>/.viewer_cache/``) is created
        if missing so the first save into a freshly-curated output dir
        does not fail.

        Holds :attr:`_lock` for the duration. Exceptions from the
        underlying I/O propagate; callers are expected to surface them
        to the user.
        """
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "version": QC_RECIPE_VERSION,
                "checks": [entry.to_dict() for entry in self.entries],
            }
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(payload, indent=2, sort_keys=False),
                encoding="utf-8",
            )
            os.replace(tmp_path, self.path)

            try:
                self.seed_mtime_ns = self.path.stat().st_mtime_ns
            except OSError:
                self.seed_mtime_ns = None

    def _new_instance_id(self, name: str) -> str:
        """Generate a fresh ``qc-<name>-<8 hex>`` instance ID.

        Uses :func:`secrets.token_hex` for collision resistance under
        test-harness parallelism. ``time.time()``-based suffixes are
        explicitly rejected (spec lines 720-725) because back-to-back
        adds within the same second would collide and break the
        per-card Dash component-ID pattern.

        Args:
            name: The check's class-level ``name`` attribute (e.g.
                ``"Count"`` or ``"SE"``).

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
    ) -> str:
        """Append a new entry and persist the updated recipe.

        Args:
            check_cls: The :class:`QualityCheck` subclass to add.
            params: Constructor kwargs (JSON-friendly values only).
            enabled: Whether the new entry should run when
                :meth:`instantiate` is called.

        Returns:
            The generated ``instance_id`` so the caller can reference
            the new entry from a Dash callback.
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
            self.save()
            return instance_id

    def remove(self, instance_id: str) -> bool:
        """Remove the entry whose ``instance_id`` matches.

        Args:
            instance_id: The identifier returned by a previous
                :meth:`add` call.

        Returns:
            ``True`` if the entry existed and was removed, ``False``
            otherwise. The on-disk file is only rewritten when an
            entry was actually removed.
        """
        with self._lock:
            for index, entry in enumerate(self.entries):
                if entry.instance_id == instance_id:
                    del self.entries[index]
                    self.save()
                    return True
            return False

    def update(
        self,
        instance_id: str,
        *,
        params: dict[str, Any] | None = None,
        enabled: bool | None = None,
    ) -> bool:
        """Mutate an existing entry's params and/or enabled flag in place.

        Args:
            instance_id: Target entry's identifier.
            params: When not :data:`None`, replaces
                :attr:`QcRecipeEntry.params` wholesale (the dict is
                copied so the caller's reference is decoupled).
            enabled: When not :data:`None`, replaces
                :attr:`QcRecipeEntry.enabled`.

        Returns:
            ``True`` if the entry existed and was updated, ``False``
            otherwise. The on-disk file is only rewritten when an
            entry was actually updated.
        """
        with self._lock:
            for entry in self.entries:
                if entry.instance_id != instance_id:
                    continue
                if params is not None:
                    entry.params = dict(params)
                if enabled is not None:
                    entry.enabled = enabled
                self.save()
                return True
            return False

    def instantiate(self) -> list[tuple[str, QualityCheck]]:
        """Build concrete :class:`QualityCheck` objects for every enabled entry.

        Construction failures (e.g. metadata path missing,
        ``KeyError`` on an absent ``groupby`` column) land in
        :attr:`load_warnings` rather than raising — spec lines 695-696.
        Disabled entries are skipped silently.

        Returns:
            ``(instance_id, instance)`` tuples in recipe order. Entries
            that failed to instantiate are absent; the surviving
            entries are still returned so a single bad config does not
            block the rest of the QC tab.
        """
        with self._lock:
            built: list[tuple[str, QualityCheck]] = []
            for entry in self.entries:
                if not entry.enabled:
                    continue
                try:
                    instance = entry.cls(**entry.params)
                except Exception as exc:  # noqa: BLE001 - surfaced via warning
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
