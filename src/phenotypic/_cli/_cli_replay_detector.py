"""Stage-3 stand-in for a GpuDetector whose inference already happened."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from phenotypic.abc_ import ObjectDetector
from phenotypic.sdk_._operation_tree import substitute_at_path
from phenotypic.sdk_.typing_ import NdArrayField, OperationField

if TYPE_CHECKING:
    from phenotypic import ImagePipeline
    from phenotypic._core._image import Image

    from ._cli_pipeline_split import StagePlan

__all__ = ["ReplayDetector", "build_replay_pipeline"]


class ReplayDetector(ObjectDetector):
    """Write a PRE-RECORDED Stage-2 result in place of running the model.

    Stage 2 ran the real detector on a GPU node and retained its raw output.
    Stage 3 substitutes this stub at the detector's tree path so the enclosing
    operation -- a ``CompositeDetector``, say -- runs exactly as it would in a
    single-pass run, merging this branch's mask with its CPU siblings'.

    The write delegates to the real detector's ``_write_object_output``, which
    owns ``drop_frame_background`` and ``split_disconnected_labels``. Assigning
    ``objmap`` directly here would skip both and silently bridge every colony
    the background label touches.

    **Identity is delegated, not inherited.** The four provenance hooks below
    make the journal record the *wrapped* detector, because a staged run and a
    single-pass run of the same pipeline must produce the same journal. The
    ``parameters`` override is not only a parity concern: the stub holds an
    ``NdArrayField``, so the default ``model_dump`` would serialise the entire
    recorded objmap into the journal.

    Args:
        detector: The real detector whose inference Stage 2 performed. Supplies
            the output-writing behaviour and the provenance identity.
        result: The raw Stage-2 output for this image -- a ``uint16`` labeled
            map (``output_kind="instance"``) or a boolean mask (``semantic``),
            interpreted by the wrapped detector's ``output_kind``.
        detector_duration_seconds: Stage-2 inference wall time, recovered from
            the Stage-2 token. Added to the stub's own (merge-only) wall time
            when the journal entry is written, so the recorded cost is the
            whole detection rather than the merge.

    Examples:
        The wrapped operation must be the real ``GpuDetector`` -- the writing
        behaviour lives on it, not on this stub:

        >>> import numpy as np
        >>> from phenotypic.abc_ import GpuDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> class _StubGpu(GpuDetector):
        ...     def _ensure_model_loaded(self):
        ...         pass
        >>> image = load_synth_yeast_plate()
        >>> recorded = np.zeros(image.gray[:].shape, dtype=np.uint16)
        >>> recorded[20:60, 20:60] = 1
        >>> recorded[120:160, 120:160] = 2
        >>> detector = _StubGpu(drop_frame_background=False)
        >>> _ = ReplayDetector(detector=detector, result=recorded).apply(
        ...     image, inplace=True)
        >>> image.num_objects
        2
    """

    detector: OperationField
    result: NdArrayField
    detector_duration_seconds: float = 0.0

    def provenance_operation_name(self) -> str:
        """Report the WRAPPED detector's class name to the journal."""
        return type(self.detector).__name__

    def provenance_operation_class(self) -> str:
        """Report the WRAPPED detector's import path to the journal."""
        cls = type(self.detector)
        return f"{cls.__module__}.{cls.__qualname__}"

    def provenance_parameters(self) -> dict[str, Any]:
        """Report the WRAPPED detector's parameters to the journal."""
        return self.detector.model_dump(mode="json")

    def provenance_duration_offset(self) -> float:
        """Stage-2 inference time to add to this stub's measured wall time.

        The stub's own wall time covers the MERGE only; the GPU cost was paid
        in Stage 2 and travels here in the Stage-2 token.
        """
        return float(self.detector_duration_seconds)

    def _operate(self, image: "Image") -> "Image":
        """Write the recorded result through the wrapped detector's writer."""
        self.detector._write_object_output(image, self.result)
        return image


def build_replay_pipeline(
    plan: "StagePlan",
    result: np.ndarray,
    *,
    detector_duration_seconds: float = 0.0,
) -> "ImagePipeline":
    """``plan.post_pipeline`` with a :class:`ReplayDetector` at ``gpu_path``.

    The two consumers of a Stage-2 raw array -- Stage 3
    (``stage3_merge_measure_core``) and the ``--mode process --layer objmap``
    export (``_export_objmap_layer``) -- both need exactly this pipeline, so
    the substitution rule lives here once rather than in each of them.

    Substituting is not an optimisation over writing the raw array directly:
    ``post_pipeline`` is cut at the detector's TOP-LEVEL ANCESTOR, so for a
    nested detector it *contains the real detector*, and applying it as-is
    would re-run live GPU inference on a CPU node.

    Args:
        plan: The split plan, for ``post_pipeline``, ``gpu_path`` and the real
            ``gpu_detector`` whose writer and provenance identity the stub
            borrows.
        result: The raw Stage-2 output for this image.
        detector_duration_seconds: Stage-2 inference wall time from the
            Stage-2 token, added to the stub's own merge time in the journal
            entry. The objmap export leaves this at ``0.0``: it reads no token
            and persists no journal, so there is nothing for the offset to
            reach.

    Returns:
        A throwaway pipeline; ``plan.post_pipeline`` is not mutated.
    """
    stub = ReplayDetector(
        detector=plan.gpu_detector,
        result=result,
        detector_duration_seconds=detector_duration_seconds,
    )
    return substitute_at_path(plan.post_pipeline, plan.gpu_path, stub)
