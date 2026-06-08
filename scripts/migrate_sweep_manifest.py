"""Migrate a legacy ``generate_sweep_manifest`` JSON to a ``tuning_spec.json``.

The hard cutover (master §9) deletes ``sweep``; this converts an existing
manifest into the new ``TuningSpec``. MVP scope: a single config of flat +
presence sweeps (the shape ``generate_sweep_manifest`` produced) — it derives
the base pipeline (the op-richest variant), a ``SearchSpace`` (Categorical knobs
over the per-op param values observed; a ``__enabled__`` presence knob for ops
absent in some variants), and a ``GridConfig``. The user supplies the
``Scorer``. Nested-op manifests raise ``NotImplementedError``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.tune import (
    Categorical,
    Evaluator,
    GridConfig,
    Knob,
    QCScorer,
    SearchSpace,
)
from phenotypic.tune._evaluation import build_pipeline
from phenotypic.tune._scoring import Scorer
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune._strategies._enumerate import enumerate_grid


def _pipelines(manifest: dict) -> list[ImagePipeline]:
    pipes: list[ImagePipeline] = []
    for cfg in manifest["configs"].values():
        for pipe_dict in cfg["pipelines"].values():
            pipes.append(ImagePipeline.from_json(json.dumps(pipe_dict)))
    return pipes


def _hashable(value: Any) -> Any:
    try:
        hash(value)
        return value
    except TypeError as exc:
        raise NotImplementedError(
            f"non-hashable swept param value {value!r}; nested-op manifests are "
            "not supported by the MVP migration"
        ) from exc


def _pipe_signature(pipe: ImagePipeline) -> tuple[tuple[str, str], ...]:
    """Stable operation signature for exact manifest reproduction checks."""
    return tuple(
        (
            type(op).__name__,
            json.dumps(op.model_dump(mode="json"), sort_keys=True, default=str),
        )
        for op in pipe.get_ops().values()
    )


def _assert_exact_reproduction(spec: TuningSpec, pipes: list[ImagePipeline]) -> None:
    """Fail loudly when independent grid knobs over-generate legacy variants."""
    expected = {_pipe_signature(pipe) for pipe in pipes}
    migrated = {
        _pipe_signature(build_pipeline(spec.pipeline, combo))
        for combo in enumerate_grid(spec.search_space)
    }
    if migrated != expected:
        raise NotImplementedError(
            "cannot migrate this sweep manifest exactly: the new independent "
            "grid knobs would generate pipeline combinations that were not in "
            "the legacy manifest. This usually means duplicate-class or "
            "position-shifted operations have correlated parameter values; "
            "write the tuning_spec.json manually for this case."
        )


def migrate_manifest_to_spec(manifest: dict, *, scorer: Scorer) -> TuningSpec:
    """Convert a legacy manifest into a ``TuningSpec`` (see module docstring).

    Args:
        manifest: The parsed legacy ``generate_sweep_manifest`` JSON.
        scorer: The objective the user supplies for the migrated spec.

    Returns:
        A ``TuningSpec`` whose grid reproduces the manifest's op-combinations.

    Raises:
        ValueError: If the manifest carries no pipelines.
        NotImplementedError: For nested-op (non-hashable) swept values.
    """
    pipes = _pipelines(manifest)
    if not pipes:
        raise ValueError("manifest has no pipelines")

    # The op-richest variant is the base. Unique operation classes can be matched
    # by class across variants where optional earlier ops shift positions. When a
    # class appears multiple times, migration must be position-based to avoid
    # conflating duplicate slots.
    base_pipe = max(pipes, key=lambda p: len(p.get_ops()))
    base_ops = list(base_pipe.get_ops().values())
    base_classes = [type(op).__name__ for op in base_ops]
    base_class_counts = {
        cls_name: base_classes.count(cls_name) for cls_name in set(base_classes)
    }

    knobs: list[Knob] = []
    for position, _op in enumerate(base_ops):
        cls = base_classes[position]
        # Which variants contain this slot? Unique classes match by class because
        # optional preceding operations can shift positions in legacy manifests.
        # Duplicate classes match by position or fail loudly if insertion/deletion
        # makes the slot ambiguous.
        present_ops: list[Any] = []
        for pipe_index, pipe in enumerate(pipes):
            pipe_ops = list(pipe.get_ops().values())
            if base_class_counts[cls] == 1:
                matches = [
                    candidate
                    for candidate in pipe_ops
                    if type(candidate).__name__ == cls
                ]
                if matches:
                    present_ops.append(matches[0])
                continue
            if position < len(pipe_ops):
                candidate = pipe_ops[position]
                if type(candidate).__name__ == cls:
                    present_ops.append(candidate)
                    continue
            if cls in {type(candidate_op).__name__ for candidate_op in pipe_ops}:
                raise NotImplementedError(
                    "cannot migrate sweep manifest with inserted/deleted "
                    f"duplicate-class operations around position {position} "
                    f"({cls!r}) in pipeline {pipe_index}; migration is "
                    "position-based and this manifest is ambiguous"
                )
        optional = len(present_ops) < len(pipes)
        enabled_key = f"{position}.{cls}.__enabled__"
        if optional:
            knobs.append(Knob(
                key=enabled_key,
                domain=Categorical(choices=(True, False)),
                source="presence_optin",
            ))
        # per-field varying values across the present variants
        fields: dict[str, set] = {}
        for op_i in present_ops:
            for fname in type(op_i).model_fields:
                fields.setdefault(fname, set()).add(
                    _hashable(getattr(op_i, fname))
                )
        for fname, values in fields.items():
            if len(values) <= 1:
                continue  # constant → not a knob
            knob_kwargs: dict[str, Any] = dict(
                key=f"{position}.{fname}",
                # sorted by repr only for a deterministic, stable choice order
                # across heterogeneous types; the order is arbitrary (not
                # semantic) — the golden lock compares grids as a set.
                domain=Categorical(choices=tuple(sorted(values, key=repr))),
            )
            if optional:
                knob_kwargs["conditional_on"] = ((enabled_key, True),)
            knobs.append(Knob(**knob_kwargs))

    spec = TuningSpec(
        pipeline=base_pipe,
        search_space=SearchSpace(knobs=tuple(knobs)),
        scorer=scorer,
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )
    _assert_exact_reproduction(spec, pipes)
    return spec


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate a sweep manifest to a tuning_spec.json"
    )
    parser.add_argument("manifest", help="legacy manifest JSON")
    parser.add_argument(
        "-o", "--output", required=True, help="tuning_spec.json to write"
    )
    parser.add_argument(
        "--metadata", required=True, help="layout CSV/Parquet for the Count scorer"
    )
    parser.add_argument("--groupby", nargs="+", default=["Metadata_ImageName"])
    return parser


def main() -> None:
    """CLI entry point — convert a manifest path to a ``tuning_spec.json``."""
    args = _build_cli().parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    scorer = QCScorer(check=ExpectedVsDetectedCount(
        metadata=args.metadata, groupby=list(args.groupby)))
    spec = migrate_manifest_to_spec(manifest, scorer=scorer)
    Path(args.output).write_text(spec.model_dump_json(indent=2))
    print(f"Wrote {args.output} ({len(spec.search_space.knobs)} knobs)")


if __name__ == "__main__":
    main()
