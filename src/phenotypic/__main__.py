"""
Enable running PhenoTypic as a module from the command line.

Usage:
    python -m phenotypic PIPELINE_JSON INPUT_DIR OUTPUT_DIR [OPTIONS]

Example:
    python -m phenotypic my_pipeline.json ./raw_images ./results --n-jobs 4
"""

from phenotypic.phenotypic_cli import main

if __name__ == "__main__":
    main()
