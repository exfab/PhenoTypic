#!/bin/bash
# L1 cross-node gate (C7) for the MCP server's P1 JournalStorage backend.
#
# The single-node suite reported DISCRIMINATION: NONE on GPFS because
# multiprocessing places every worker on one host, where append atomicity comes
# from the local kernel regardless of the filesystem. This job places one worker
# per NODE with srun, so the shared filesystem — not one kernel — is what has to
# arbitrate the appends. That is the configuration a SLURM tune fleet actually
# runs in.
#
# Submit:
#   sbatch docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/run_l1_cross_node.sbatch
#
# Read the SUMMARY block at the end of the log, not the individual ok/LOST lines.
#
#SBATCH --job-name=mcp-l1-xnode
#SBATCH --partition=short
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/bigdata/exfab/anguy344/mcp-l1-gate/logs/xnode-%j.log
set -uo pipefail

REPO=/bigdata/exfab/anguy344/PhenoTypic
SCRIPT=$REPO/docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/optuna_journal_storage.py
WORK=/bigdata/exfab/anguy344/mcp-l1-gate/xnode-${SLURM_JOB_ID}
N_WORKERS=4
TRIALS=15

echo "Date: $(date)  Job: ${SLURM_JOB_ID}"
echo "Nodes allocated: ${SLURM_JOB_NODELIST}"
df -PT "$(dirname "$WORK")" | awk 'NR==2 {print "Filesystem:", $1, $2, $7}'
mkdir -p "$WORK"
cd "$REPO" || exit 1

# One full trial of the experiment under a given lock. Echoes a verdict line the
# summary parses. Exit 0 = every write survived; 2 = writes were lost.
run_lock () {
    local lock="$1"
    local journal="$WORK/${lock}.log"

    echo
    echo "==================== C7 lock=${lock} ===================="

    uv run --no-sync python "$SCRIPT" --role init --journal "$journal" --lock "$lock" || return 1

    # One task per node. --kill-on-bad-exit so a crashed worker does not leave
    # the verify step interpreting a partial run as data loss.
    #
    # --cpu-bind=none is required, not cosmetic: this partition hands out
    # non-contiguous CPU masks (observed 0x0040400000000000 on c07), and srun's
    # default binding then fails the whole step with "Unable to satisfy cpu bind
    # request" before a single worker starts. Binding buys nothing here — the
    # workload is one short-lived append per trial, not compute.
    srun --nodes=$N_WORKERS --ntasks=$N_WORKERS --ntasks-per-node=1 \
         --cpu-bind=none --kill-on-bad-exit=1 \
        uv run --no-sync python "$SCRIPT" \
            --role worker --journal "$journal" --lock "$lock" --trials-per-proc $TRIALS
    local worker_rc=$?
    if [[ $worker_rc -ne 0 ]]; then
        echo "  workers exited $worker_rc — see above"
        return 1
    fi

    uv run --no-sync python "$SCRIPT" --role verify --journal "$journal" --lock "$lock" \
        --n-procs $N_WORKERS --trials-per-proc $TRIALS --require-distinct-nodes
    return $?
}

run_lock symlink
RC_SYMLINK=$?

run_lock noop
RC_NOOP=$?

echo
echo "==================== C7 SUMMARY ===================="
echo "symlink lock: rc=${RC_SYMLINK}   noop control: rc=${RC_NOOP}"
echo "  (rc 0 = all writes survived, 2 = writes LOST, 1 = the phase broke)"
echo

if [[ $RC_SYMLINK -eq 1 || $RC_NOOP -eq 1 ]]; then
    echo "VERDICT: INCONCLUSIVE — a phase failed for a reason unrelated to the lock."
    echo "  Neither branch below applies. Read the log above before rerunning."
    exit 1
fi

if [[ $RC_SYMLINK -eq 0 && $RC_NOOP -eq 2 ]]; then
    echo "VERDICT: DISCRIMINATION. The lock does real work across nodes on this mount."
    echo "  C2a/C7 are now meaningful evidence -> P1 may ship on the journal backend."
    exit 0
fi

if [[ $RC_SYMLINK -eq 0 && $RC_NOOP -eq 0 ]]; then
    echo "VERDICT: NO DISCRIMINATION, cross-node. GPFS serializes the appends itself;"
    echo "  the symlink lock is redundant here rather than broken. Journal storage is"
    echo "  safe on this mount BECAUSE OF THE FILESYSTEM, not because of the lock —"
    echo "  which is a measured claim, and must be documented as filesystem-dependent."
    exit 0
fi

if [[ $RC_SYMLINK -eq 2 ]]; then
    echo "VERDICT: UNSAFE. The symlink-locked run LOST WRITES across nodes."
    echo "  The journal backend must not be enabled on this filesystem."
    echo "  Postgres remains the supported distributed path (spec 7, 4.3)."
    exit 3
fi

echo "VERDICT: UNEXPECTED combination rc_symlink=${RC_SYMLINK} rc_noop=${RC_NOOP}."
exit 1
