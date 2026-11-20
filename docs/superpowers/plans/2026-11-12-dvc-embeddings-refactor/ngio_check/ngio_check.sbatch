#!/bin/bash
#SBATCH --job-name=ngio-check
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/bigdata/exfab/anguy344/gate-results/ngio-check/ngio_%j.log
# Spec D4: does ngio discover tables nested one level down (tables/<n>-<sha12>/measurements)?
# Store built by the shipped CLI at HEAD; the probe runs in a throwaway uv env with only ngio.
set -uo pipefail
W=/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/dvc-embeddings-refactor
D=/bigdata/exfab/anguy344/gate-results/ngio-check
rm -rf "$D/run"; mkdir -p "$D/run"
cd "$W"; echo "commit $(git rev-parse --short HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
"$W/.venv/bin/python" "$D/make_store.py" "$D/run"
S=$(find "$D/run/out" -name '*.ome.zarr' -type d | head -1)
echo "probing $S"
cd "$D"; uv run --no-project --isolated --with ngio --with "polars[rtcompat]" python -I "$D/ngio_probe.py" "$S"
