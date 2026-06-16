"""
Enable running PhenoTypic as a module from the command line.

Usage:
    python -m phenotypic --mode full --pipeline PIPELINE_JSON --input INPUT_DIR --output OUTPUT_DIR [OPTIONS]

Example:
    python -m phenotypic --mode full --pipeline my_pipeline.json --input ./raw_images --output ./results --njobs 4
"""

from phenotypic.phenotypicCLI import phenotypic_cli

if __name__ == "__main__":
    phenotypic_cli()
