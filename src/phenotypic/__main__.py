"""
Enable running PhenoTypic as a module from the command line.

Usage:
    python -m phenotypic --pipeline PIPELINE_JSON --input INPUT_DIR -o OUTPUT_DIR [OPTIONS]

Example:
    python -m phenotypic --pipeline my_pipeline.json --input ./raw_images -o ./results --n-jobs 4
"""

from phenotypic.phenotypicCLI import phenotypic_cli

if __name__ == "__main__":
    phenotypic_cli()
