"""A ``MeasureDinoEmbedding`` that fails one image on disk (#282 P4).

``prepare_regions`` fails when the file ``$PHENOTYPIC_TEST_FAULT_DIR/<image
name>`` exists. Stage 4's loader calls ``prepare_regions`` inside spawned
DataLoader workers, which import this class afresh and inherit the
environment but never see a ``monkeypatch``, so a per-image fault is planted
as a file (``tests._tiny_run.plant_stage4_fault``). The inline path prepares
through ``_prepare_inline`` and never faults.

The file's text picks the fault (#282 P3-P4 review M6):

- ``fault`` (the default): a :class:`ValueError`, the image's own failure;
- ``oom-once``: a :class:`MemoryError`, node state, and the trigger is
  removed first, so a plain re-run succeeds;
- ``die``: the loader worker process exits at once (``os._exit``), as a host
  out-of-memory kill would end it; the trigger is removed first. Only inside a
  spawned DataLoader worker: in the main process it raises instead, so a test
  that forgets the workers fails rather than killing pytest.
"""

from __future__ import annotations

import os
from pathlib import Path

from phenotypic.measure import MeasureDinoEmbedding

#: The directory whose files name the images to fail.
FAULT_DIR_ENV = "PHENOTYPIC_TEST_FAULT_DIR"

#: The exit status a ``die`` fault ends its loader worker with.
DIE_EXIT_STATUS = 9


def _in_loader_worker() -> bool:
    import torch.utils.data

    return torch.utils.data.get_worker_info() is not None


class FaultyDinoEmbedding(MeasureDinoEmbedding):
    """:class:`MeasureDinoEmbedding` whose ``prepare_regions`` fails a planted image."""

    def prepare_regions(self, image, **kwargs):  # type: ignore[override]
        fault_dir = os.environ.get(FAULT_DIR_ENV)
        trigger = Path(fault_dir) / str(image.name) if fault_dir else None
        if trigger is not None and trigger.exists():
            kind = trigger.read_text(encoding="utf-8").strip() or "fault"
            if kind == "oom-once":
                trigger.unlink(missing_ok=True)
                raise MemoryError(f"planted Stage-4 out-of-memory for {image.name}")
            if kind == "die":
                if not _in_loader_worker():
                    raise RuntimeError("a 'die' fault needs a spawned loader worker (--njobs 2 or more)")
                trigger.unlink(missing_ok=True)
                os._exit(DIE_EXIT_STATUS)
            raise ValueError(f"planted Stage-4 fault for {image.name}")
        return super().prepare_regions(image, **kwargs)
