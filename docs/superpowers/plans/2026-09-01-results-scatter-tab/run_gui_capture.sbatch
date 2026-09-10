#!/bin/bash
#SBATCH --job-name=pht-gui-capture
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/%j.log
set -euo pipefail

# Regenerates the synthetic tutorial dataset, runs the CLI over it, boots the
# GUI hub plus two standalone apps, and drives Chromium through every
# workflow. Far past the head-node bar, and it needs a real X display: the
# deck.gl/Viv surfaces screenshot blank without one.
WORKTREE=/bigdata/exfab/anguy344/PhenoTypic/.worktrees/results-scatter-gui
cd "$WORKTREE"

echo "Date: $(date)  Node: $(hostname -s)  Job: ${SLURM_JOB_ID}"
echo "HEAD: $(git rev-parse --short HEAD) $(git log -1 --format=%s)"
_DIRTY="$(git status --porcelain | wc -l)"
if ((_DIRTY > 0)); then
    echo "TREE: DIRTY -- ${_DIRTY} uncommitted path(s); these images are NOT HEAD's"
    git status --porcelain | sed 's/^/      /'
else
    echo "TREE: clean -- images are HEAD's"
fi

# Playwright's own Chromium. The script drives it through Playwright, which
# finds its bundled build itself; BROWSER_PATH is exported for the kaleido
# PDF path, which does not.
_PW_CHROME=$(ls -d "$HOME"/.cache/ms-playwright/chromium-*/chrome-linux64/chrome 2>/dev/null | tail -1)
if [[ -n "${_PW_CHROME}" ]]; then
    export BROWSER_PATH="${_PW_CHROME}"
    echo "BROWSER_PATH=${BROWSER_PATH}"
else
    echo "WARNING: no vendored Chromium found under ~/.cache/ms-playwright" >&2
fi

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PLAYWRIGHT=1

# xvfb-run, not QT_QPA_PLATFORM=offscreen: the capture is a real browser
# driving WebGL canvases, not a Qt widget test.
xvfb-run -a --server-args="-screen 0 1920x1080x24" \
    uv run python scripts/capture_gui_tutorial_screenshots.py --force

echo "Finished: $(date)"
git status --porcelain docs/source/_static/gui_images/ | head -40
echo "PNG count: $(find docs/source/_static/gui_images -name '*.png' | wc -l)"
