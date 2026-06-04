"""``python -m phenotypic.tune`` — run a tuning spec over an image directory."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from ._spec import TuningSpec
from ._tune_cli._run import _load_images, run_tuning


def _default_output(input_dir: str) -> Path:
    return Path(f"./{Path(input_dir).name}_tune")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m phenotypic.tune",
        description="Tune an ImagePipeline's parameters over an image set.",
    )
    parser.add_argument("spec", help="path to a tuning_spec.json")
    parser.add_argument("-i", "--input", required=True, help="image directory")
    parser.add_argument("-o", "--output", default=None, help="output directory")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """CLI entry point. See ``--help``.

    Loads a ``tuning_spec.json``, scans ``--input`` for images, runs the engine,
    and writes the ``deliverables/`` artifacts under ``--output`` (default
    ``./<input-basename>_tune/``). Resumes if ``trials.parquet`` already exists
    in the output dir.
    """
    args = _parse_args(argv)
    spec = TuningSpec.model_validate_json(Path(args.spec).read_text())
    output_dir = Path(args.output) if args.output else _default_output(args.input)
    images = _load_images(Path(args.input))
    if not images:
        raise SystemExit(f"no images found under {args.input!r}")
    run_tuning(spec, images, output_dir)


if __name__ == "__main__":
    main()
