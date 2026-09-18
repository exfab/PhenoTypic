#!/bin/bash
# Symlink N images per dataset from the Linzer test set into a fresh input root.
# The CLI scanner reads exactly one level of dataset folders, so keep that shape.
set -euo pipefail
PROJECT=/bigdata/exfab/anguy344/projects/ucr_033_e_d_Linzer_Ganoderma
SRC=$PROJECT/data/pipeline_dev/inputs/test_set_2026-09-15
DEST=${1:?usage: make_inputs.sh <dest-root> [per-dataset]}
PER=${2:-3}

[[ -e $DEST ]] && { echo "REFUSING: $DEST exists" >&2; exit 1; }
for ds in "$SRC"/*/; do
    name=$(basename "$ds")
    mkdir -p "$DEST/$name"
    find "$ds" -maxdepth 1 -name '*.tiff' | LC_ALL=C sort | head -n "$PER" |
        while read -r link; do
            ln -s "$(readlink -f "$link")" "$DEST/$name/$(basename "$link")"
        done
done
find "$DEST" -name '*.tiff' | LC_ALL=C sort
