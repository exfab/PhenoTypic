#!/usr/bin/env python3
"""
Backwards compatibility wrapper for phenotypic-cli.py.

DEPRECATED: Please use 'python -m phenotypic' instead.

This file exists only for backwards compatibility with existing scripts that
directly reference phenotypic-cli.py. All functionality has been moved to the
main package and can be invoked using:

    python -m phenotypic --pipeline PIPELINE_JSON --input INPUT_DIR -o OUTPUT_DIR [OPTIONS]

This wrapper will be removed in a future release.
"""

import warnings

warnings.warn(
    "phenotypic-cli.py is deprecated. Please use 'python -m phenotypic' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from phenotypic.phenotypicCLI import phenotypic_cli  # noqa: E402

if __name__ == "__main__":
    phenotypic_cli()
