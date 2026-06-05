"""The robust-eval held-out split — calibration vs held-out plate partition.

Phase 4.5 part 1 builds only the **split derivation + persistence** layer; the
held-out *evaluation* pass that consumes it (and writes ``generalization.json``)
is part 2. The split answers one question reproducibly: which plates are tuned on
(calibration) and which are reserved to later estimate the winner's true
generalization gap (held out).

Determinism is the contract — the partition must be identical across processes
and across resume, regardless of the run's master seed:

- :func:`_dataset_identity` is an **order-independent** SHA-256 over the plate
  names, so re-loading the same plates (in any order) yields the same identity.
- :func:`_split_subseed` folds ``(master_seed, identity)`` into a
  :class:`numpy.random.SeedSequence`, so the random choice draws from a
  per-dataset stream that never touches the global RNG.

``numpy`` (including ``SeedSequence``) is a core dependency; this module is
optuna-free (the lazy-import lock).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from phenotypic.tools_ import _io_constants as _io

#: The held-out partition tier (robust-eval split policy):
#: - ``"group"``: a whole metadata group is held out (the strongest test of
#:   cross-batch generalization);
#: - ``"within_group"``: a fraction of plates within the (single) group is held
#:   out — a weaker test, flagged via ``within_group_caveat``;
#: - ``"none"``: too few plates to reserve any (data-poor skip; all calibration).
SplitKind = Literal["group", "within_group", "none"]


def _dataset_identity(images: list) -> str:
    """A stable, order-independent SHA-256 fingerprint of the plate set.

    The identity is the hash of the **sorted** newline-joined ``image.name``
    values, so two loads of the same plates (in any order) agree, while adding or
    removing a plate changes it — letting :func:`resolve_split` detect a dataset
    that no longer matches a persisted split.

    Args:
        images: The plates being tuned; only ``image.name`` is read.

    Returns:
        The 64-character hex SHA-256 digest of the sorted plate names.
    """
    joined = "\n".join(sorted(im.name for im in images))
    return hashlib.sha256(joined.encode()).hexdigest()


def _split_subseed(master_seed: int, identity: str) -> np.random.SeedSequence:
    """Derive a per-dataset :class:`numpy.random.SeedSequence` for the split.

    Folds the run's ``master_seed`` and the dataset ``identity`` (its first 16
    hex chars, as an int) into a spawned ``SeedSequence`` so the held-out choice
    draws from a stream unique to this ``(seed, dataset)`` pair and reproducible
    across processes. Uses no global RNG state.

    Args:
        master_seed: The run's master seed.
        identity: A :func:`_dataset_identity` digest.

    Returns:
        A spawned ``SeedSequence`` to seed ``numpy.random.default_rng``.
    """
    return np.random.SeedSequence([master_seed, int(identity[:16], 16)]).spawn(1)[0]


@dataclass(frozen=True)
class Split:
    """The reproducible calibration / held-out partition of a plate set.

    A frozen value object: tuning runs on ``calibration`` and the ``held_out``
    plates are reserved for the (part 2) generalization pass. The partition is
    fully determined by the plates' names + ``(master_seed, dataset_identity)``,
    so it round-trips through ``split.json`` and is reused verbatim on resume.

    Args:
        calibration: The calibration plate **names** (sorted), tuned on.
        held_out: The held-out plate names (sorted), reserved for the
            generalization pass; empty for ``kind="none"``.
        kind: The partition tier — :data:`SplitKind`.
        group_key: The metadata column that defined the groups, or ``None`` when
            grouping was unavailable / unused.
        dataset_identity: The :func:`_dataset_identity` digest of the plate set
            this split was derived for (a mismatch on reload means the dataset
            changed).
        within_group_caveat: ``True`` for ``kind="within_group"`` — a flag that
            the held-out plates share a group with calibration, so the
            generalization estimate is weaker (no cross-group test).
        seed_entropy: The :func:`_split_subseed` entropy (as a list of ints) that
            seeded the choice — persisted for auditability / reproducibility.
    """

    calibration: list[str]
    held_out: list[str]
    kind: SplitKind
    group_key: Optional[str]
    dataset_identity: str
    within_group_caveat: bool = False
    seed_entropy: list[int] = field(default_factory=list)


def _resolve_groups(images: list, group_key: Optional[str]) -> dict[str, list[str]]:
    """Group plate names by their ``group_key`` metadata value.

    Args:
        images: The plates; ``image.metadata.get(group_key)`` resolves the value.
        group_key: The metadata column naming each plate's group, or ``None``.

    Returns:
        Ordered ``{group_value: [plate_name, ...]}``; empty when ``group_key`` is
        ``None`` or no plate carries the column.
    """
    if group_key is None:
        return {}
    groups: dict[str, list[str]] = {}
    for image in images:
        value = image.metadata.get(group_key)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(image.name)
    return groups


def _entropy_list(subseed: np.random.SeedSequence) -> list[int]:
    """The ``SeedSequence`` entropy normalized to a JSON-friendly list of ints."""
    entropy = subseed.entropy
    if entropy is None:
        return []
    if isinstance(entropy, int):
        return [entropy]
    return [int(value) for value in entropy]


def derive_split(
    images: list,
    *,
    master_seed: int,
    group_key: Optional[str],
    held_out_fraction: float,
    min_heldout_plates: int,
) -> Split:
    """Partition ``images`` into calibration + held-out plates (3-tier policy).

    Tiers, in order:

    1. **Data-poor skip** — fewer than ``min_heldout_plates`` plates: reserve
       nothing (``kind="none"``, all calibration). Holding out a couple of
       unrepresentative plates would distort tuning more than it protects.
    2. **Whole-group hold-out** — ``group_key`` resolves **≥ 2** groups: hold out
       one whole group (``kind="group"``), the strongest cross-batch
       generalization test. The held-out group is chosen by the per-dataset RNG.
    3. **Within-group hold-out** — a single group, or no usable ``group_key``,
       but enough plates: hold out a ``held_out_fraction`` slice of plate names
       (``kind="within_group"``, ``within_group_caveat=True``) — a weaker test,
       flagged so part 2 can caveat the verdict.

    The choice draws from :func:`_split_subseed` so it is deterministic and
    reproducible across processes and resumes (independent of the run's master
    seed once persisted).

    Args:
        images: The plates being tuned; ``image.name`` + ``image.metadata`` read.
        master_seed: The run's master seed (folded into the per-dataset stream).
        group_key: The metadata column defining plate groups, or ``None``.
        held_out_fraction: The target held-out fraction for the within-group tier
            (``round(fraction * n)``, at least 1).
        min_heldout_plates: The data-poor floor — below this many plates, skip.

    Returns:
        The derived :class:`Split`.
    """
    identity = _dataset_identity(images)
    subseed = _split_subseed(master_seed, identity)
    entropy = _entropy_list(subseed)
    all_names = sorted(im.name for im in images)
    n_plates = len(all_names)

    # Tier 1 — data-poor skip.
    if n_plates < min_heldout_plates:
        return Split(
            calibration=all_names,
            held_out=[],
            kind="none",
            group_key=None,
            dataset_identity=identity,
            within_group_caveat=False,
            seed_entropy=entropy,
        )

    rng = np.random.default_rng(subseed)
    groups = _resolve_groups(images, group_key)

    # Tier 2 — whole-group hold-out (≥ 2 groups; leave ≥ 1 calibration group).
    if len(groups) >= 2:
        group_values = sorted(groups)
        chosen = group_values[int(rng.integers(0, len(group_values)))]
        held_out = sorted(groups[chosen])
        calibration = sorted(set(all_names) - set(held_out))
        return Split(
            calibration=calibration,
            held_out=held_out,
            kind="group",
            group_key=group_key,
            dataset_identity=identity,
            within_group_caveat=False,
            seed_entropy=entropy,
        )

    # Tier 3 — within-group hold-out (single group / no usable grouping).
    n_held = max(1, round(held_out_fraction * n_plates))
    n_held = min(n_held, n_plates - 1)  # always leave ≥ 1 calibration plate
    held_idx = rng.choice(n_plates, size=n_held, replace=False)
    held_set = {all_names[int(i)] for i in held_idx}
    held_out = sorted(held_set)
    calibration = sorted(set(all_names) - held_set)
    return Split(
        calibration=calibration,
        held_out=held_out,
        kind="within_group",
        group_key=group_key if groups else None,
        dataset_identity=identity,
        within_group_caveat=True,
        seed_entropy=entropy,
    )


def write_split(output_dir: Path, split: Split) -> Path:
    """Persist ``split`` to ``<output>/splits/split.json`` (creating ``splits/``).

    Args:
        output_dir: The run output directory.
        split: The derived :class:`Split`.

    Returns:
        The path the split was written to.
    """
    path = _io.split_assignment_path(Path(output_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(split), sort_keys=True, indent=2))
    return path


def read_split(output_dir: Path) -> Optional[Split]:
    """Reload a persisted :class:`Split`, or ``None`` when none exists.

    Args:
        output_dir: The run output directory.

    Returns:
        The reconstructed :class:`Split`, or ``None`` when
        ``<output>/splits/split.json`` is absent.
    """
    path = _io.split_assignment_path(Path(output_dir))
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return Split(
        calibration=list(payload["calibration"]),
        held_out=list(payload["held_out"]),
        kind=payload["kind"],
        group_key=payload["group_key"],
        dataset_identity=payload["dataset_identity"],
        within_group_caveat=bool(payload["within_group_caveat"]),
        seed_entropy=[int(v) for v in payload["seed_entropy"]],
    )


def resolve_split(
    output_dir: Path,
    images: list,
    *,
    master_seed: int,
    group_key: Optional[str],
    held_out_fraction: float,
    min_heldout_plates: int,
) -> Split:
    """Reuse a persisted split if present, else derive one and persist it.

    Read-if-exists-else-derive-and-write: a fresh run derives + persists; a
    resume reuses the persisted partition **regardless of the new master seed**,
    so the held-out plates never leak into calibration across restarts. (Whether
    a persisted split still matches the current dataset is the caller's check via
    :attr:`Split.dataset_identity`; this resolver always honors an existing
    ``split.json``.)

    Args:
        output_dir: The run output directory.
        images: The plates being tuned.
        master_seed: The run's master seed (used only on a fresh derive).
        group_key: The metadata column defining plate groups, or ``None``.
        held_out_fraction: The within-group target held-out fraction.
        min_heldout_plates: The data-poor floor.

    Returns:
        The persisted or freshly-derived :class:`Split`.
    """
    existing = read_split(output_dir)
    if existing is not None:
        return existing
    split = derive_split(
        images,
        master_seed=master_seed,
        group_key=group_key,
        held_out_fraction=held_out_fraction,
        min_heldout_plates=min_heldout_plates,
    )
    write_split(output_dir, split)
    return split
