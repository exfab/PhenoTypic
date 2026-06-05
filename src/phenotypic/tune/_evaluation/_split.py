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

import numpy as np


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
