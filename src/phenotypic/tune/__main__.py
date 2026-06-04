"""``python -m phenotypic.tune`` — run a tuning spec, or auto-infer a space.

Two subcommands:

* ``run SPEC -i IMAGES [-o OUT]`` — load a ``tuning_spec.json`` and run the
  engine over an image directory (the Phase-1 behaviour).
* ``auto-space PIPELINE [-o OUT] [--unattended]`` — mine a pipeline JSON into a
  reviewable ``InferredSearchSpace`` proposal, written to
  ``deliverables/tuning_spec.json``; **no** engine run. File-only and
  non-blocking; ``--unattended`` is reserved for a future skip-the-prompt flow.

Back-compat: a bare ``SPEC`` positional with no subcommand defaults to ``run``,
so ``python -m phenotypic.tune spec.json -i … -o …`` keeps working.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional, Sequence

from phenotypic import ImagePipeline

from ._spec import TuningSpec
from ._strategies._config import PHENOTYPIC_TUNE_STORAGE_URL_ENV
from ._tune_cli._auto_space import _render_review_table, run_auto_space
from ._tune_cli._run import _load_images, run_tuning

#: The recognised subcommands; a leading token outside this set is treated as a
#: bare ``run`` spec path (Phase-1 back-compat).
_SUBCOMMANDS = ("run", "auto-space")


def _default_output(input_dir: str) -> Path:
    return Path(f"./{Path(input_dir).name}_tune")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.tune",
        description="Tune an ImagePipeline, or auto-infer a search space.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser(
        "run", help="run a tuning_spec.json over an image set (the engine)"
    )
    run_p.add_argument("spec", help="path to a tuning_spec.json")
    run_p.add_argument("-i", "--input", required=True, help="image directory")
    run_p.add_argument("-o", "--output", default=None, help="output directory")
    run_p.add_argument(
        "--strategy",
        choices=("grid", "random", "tpe", "cmaes", "gp", "nsga2"),
        default=None,
        help=(
            "override the spec's strategy: grid/random use the built-in configs; "
            "tpe/cmaes/gp/nsga2 build an OptunaConfig (needs the 'tune' extra)"
        ),
    )
    run_p.add_argument(
        "--n-trials", type=int, default=None, help="trial-budget override"
    )
    run_p.add_argument(
        "--screen",
        dest="screen",
        action="store_true",
        default=False,
        help="enable the two-round screening freeze",
    )
    run_p.add_argument(
        "--no-screen",
        dest="screen",
        action="store_false",
        help="disable the two-round screening freeze (the default)",
    )
    run_p.add_argument(
        "--storage-url",
        default=None,
        help=(
            "Optuna storage URL (sqlite:///… or postgresql+psycopg://…); falls "
            f"back to ${PHENOTYPIC_TUNE_STORAGE_URL_ENV}"
        ),
    )
    run_p.add_argument(
        "--slurm",
        action="store_true",
        default=False,
        help="submit a distributed worker fleet over SLURM instead of running locally",
    )

    auto_p = sub.add_parser(
        "auto-space",
        help="infer a reviewable search space from a pipeline (no engine run)",
    )
    auto_p.add_argument("pipeline", help="path to a pipeline JSON")
    auto_p.add_argument("-o", "--output", default=None, help="output directory")
    auto_p.add_argument(
        "--unattended",
        action="store_true",
        help="reserved: skip the interactive review prompt (currently a no-op)",
    )
    return parser


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    """Prepend ``run`` when the first token is a bare spec path (back-compat).

    A leading token that is not a recognised subcommand (and not an option flag)
    is the Phase-1 bare-``SPEC`` form: insert the implicit ``run`` subcommand.
    """
    args = list(argv)
    if args and args[0] not in _SUBCOMMANDS and not args[0].startswith("-"):
        return ["run", *args]
    return args


def _run_command(args: argparse.Namespace) -> None:
    """Load a spec, scan ``--input``, and run the engine (writes deliverables)."""
    spec = TuningSpec.model_validate_json(Path(args.spec).read_text())
    output_dir = Path(args.output) if args.output else _default_output(args.input)
    images = _load_images(Path(args.input))
    if not images:
        raise SystemExit(f"no images found under {args.input!r}")
    # --storage-url falls back to the env var so a SLURM array can export one
    # shared URL to every worker (psycopg3 / sqlite scheme).
    storage_url = args.storage_url or os.environ.get(PHENOTYPIC_TUNE_STORAGE_URL_ENV)
    run_tuning(
        spec,
        images,
        output_dir,
        strategy=args.strategy,
        n_trials=args.n_trials,
        screen=args.screen,
        storage_url=storage_url,
        slurm=args.slurm,
        spec_path=Path(args.spec),
        images_dir=Path(args.input),
    )


def _auto_space_command(args: argparse.Namespace) -> None:
    """Infer a search-space proposal from a pipeline JSON and print the review."""
    pipeline = ImagePipeline.from_json(Path(args.pipeline).read_text())
    output_dir = (
        Path(args.output) if args.output else _default_output(args.pipeline)
    )
    proposal = run_auto_space(pipeline, output_dir)
    print(_render_review_table(proposal))


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point. See ``--help``.

    Dispatches to ``run`` (the engine) or ``auto-space`` (infer-only). A bare
    spec path with no subcommand is routed to ``run`` for Phase-1 back-compat.
    """
    import sys

    raw = list(sys.argv[1:]) if argv is None else list(argv)
    args = _build_parser().parse_args(_normalize_argv(raw))
    if args.command == "auto-space":
        _auto_space_command(args)
    else:
        _run_command(args)


if __name__ == "__main__":
    main()
