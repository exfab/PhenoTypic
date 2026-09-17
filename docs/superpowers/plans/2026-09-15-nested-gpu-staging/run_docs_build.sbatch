#!/bin/bash
#SBATCH --job-name=phenotypic-docs-build
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/docs-%j.log
set -uo pipefail

# Sphinx HTML build for a frozen checkout. Run it as a Slurm job, never inline:
# docs/source/conf.py sets `nbsphinx_execute = "auto"`, so a full build runs
# every notebook, and an inline build on an interactive node once spawned an
# ipykernel at 626% CPU and throttled the whole session.
#
# Usage:
#   PHENO_DOCS_TREE=/bigdata/exfab/anguy344/gate-trees/<sha>-docs \
#     sbatch run_docs_build.sbatch
#
# PHENO_DOCS_EXECUTE=never (the default) skips notebook execution. That is the
# right setting for checking markdown rendering; a full build that executes
# notebooks is a separate and much heavier job, so choose it deliberately.
#
# An exit code of 0 does NOT mean the pages are correct. MyST directives with
# malformed fences rendered as Python code blocks while this build succeeded.
# Read the generated HTML for any section you changed.

: "${PHENO_DOCS_TREE:?set PHENO_DOCS_TREE to a frozen checkout}"
EXECUTE=${PHENO_DOCS_EXECUTE:-never}
JOBS=${SLURM_CPUS_PER_TASK:-1}

cd "$PHENO_DOCS_TREE" || exit 1
echo "Date:    $(date -Is)"
echo "Node:    $(hostname -s)"
echo "Job:     ${SLURM_JOB_ID}"
echo "Tree:    ${PHENO_DOCS_TREE}"
echo "Commit:  $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
echo "Dirty:   $(git status --porcelain | wc -l) file(s)"
echo "Execute: nbsphinx_execute=${EXECUTE}"
echo "Jobs:    -j ${JOBS}   (explicit; -j auto reads the NODE's cores)"
echo

# The FULL group set. `uv sync` removes anything not requested, so syncing
# only `--group docs` would uninstall the dev and test-qt tooling this tree
# also needs.
uv sync --group dev --group test-qt --group docs --all-extras \
    >/dev/null 2>&1 || { echo "uv sync failed"; exit 1; }

rm -rf docs/_build/html
status=0
uv run sphinx-build -b html -j "$JOBS" -q \
    -D nbsphinx_execute="$EXECUTE" \
    docs/source docs/_build/html || status=$?

echo
echo "sphinx-build exit: ${status}"
echo "warnings: $(grep -c WARNING "${SLURM_JOB_ID:+/bigdata/exfab/anguy344/slurm_logs/docs-${SLURM_JOB_ID}.log}" 2>/dev/null || echo '?')"
exit "${status}"
