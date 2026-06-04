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
from phenotypic.tune._scoring import Scorer
from phenotypic.tune._spec import Budget, TuningSpec


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

    # ops keyed by class name, in first-seen order; the op-richest variant is base.
    base_pipe = max(pipes, key=lambda p: len(p.get_ops()))
    base_ops = list(base_pipe.get_ops().values())
    base_classes = [type(op).__name__ for op in base_ops]

    knobs: list[Knob] = []
    for position, op in enumerate(base_ops):
        cls = base_classes[position]
        # which variants contain this position's class?
        present = [
            p for p in pipes
            if cls in {type(o).__name__ for o in p.get_ops().values()}
        ]
        optional = len(present) < len(pipes)
        enabled_key = f"{position}.{cls}.__enabled__"
        if optional:
            knobs.append(Knob(
                key=enabled_key,
                domain=Categorical(choices=(True, False)),
                source="presence_optin",
            ))
        # per-field varying values across the present variants
        fields: dict[str, set] = {}
        for p in present:
            op_i = next(
                o for o in p.get_ops().values() if type(o).__name__ == cls
            )
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

    return TuningSpec(
        pipeline=base_pipe,
        search_space=SearchSpace(knobs=tuple(knobs)),
        scorer=scorer,
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


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
