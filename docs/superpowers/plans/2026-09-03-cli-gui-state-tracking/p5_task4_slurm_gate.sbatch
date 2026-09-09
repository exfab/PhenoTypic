#!/bin/bash
#SBATCH --job-name=p5-fanout-gate
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/bigdata/exfab/anguy344/slurm_logs/p5_fanout_gate_%j.log
set -euo pipefail

# P5 Task 4 Step 4 -- the phase gate.
#
# Verifies against a real scheduler:
#   * AutonomousSLURMStrategy was selected (no staged role in the job ledger)
#   * exactly ONE dependent finalizer, no unexpected job beside the array
#   * the finalizer array is `--array=0-K` with the reserved TASK_FINALIZE index
#   * the master is non-empty, carries all 6 images, and is BYTE-IDENTICAL to a
#     local `--njobs 1` run
#
# ── Why a fresh CPU run and not the 12-image markers-era bed ─────────────────
# The bed's pipeline contains `Sam2`, a `GpuDetector`. `create_execution_strategy`
# (:1388-1427) routes on `uses_staged_gpu_strategy` == `pipeline_requires_gpu`
# ALONE, decided from the pipeline file BEFORE any work is enumerated -- so a
# continuation with nothing outstanding still selects `StagedSlurmStrategy`.
# That strategy finalizes through `_cli_staged_slurm._finalizer_body`, never
# `generate_terminal_finalizer_script`, which is called at exactly one site
# (`_cli_execution_strategies.py:1113`) and is the whole of what P5 changed.
# So the bed cannot gate this phase under any configuration: it would exercise
# `resolve_finalizer_shard_inputs -> None`, already covered by
# `test_no_manifest_means_no_fanout_rather_than_an_error`.
#
# ── What this gate CANNOT verify ────────────────────────────────────────────
# K = 1. `shard_count(6, 0.026, 2500) = 1`, and no dataset on this cluster
# reaches N ~= 34,600 -- the full Maresca tree is 6,657 and still yields 1.
# Reaching K > 1 would mean editing the code under test, which is refused.
# K > 1 is covered by the local byte-identity test (real stores, real shards,
# njobs 1/2/3/8) and the 30-case partition grid; the residual is SLURM's own
# dispatch of an `--array=0-N` directive `array_spec` already pins at K=1..2499.
#
# It also does NOT touch markers-era legacy promotion, and is not a
# continuation -- both were caveats of the abandoned fixture, not properties
# wanted here. Nothing about this gate is evidence about either.
#
# NEVER `preempt`: REQUEUE restarts from line one, so an evicted gate yields a
# union across two attempts -- the incoherence a gate exists to exclude.

GATE_SHA="${GATE_SHA:?set GATE_SHA to the commit to measure}"
N_IMAGES="${N_IMAGES:-6}"
REPO=/bigdata/exfab/anguy344/PhenoTypic
BED=/bigdata/exfab/anguy344/projects/ucr_029_e_d_Maresca/data/results/2026-08-11-state-tracking-subset
SRC_IMAGES=/rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/data/processed/color_corrected_no_denoise/7-24-26_redo_full
WORK=/bigdata/exfab/anguy344/slurm_logs/p5_gate_${SLURM_JOB_ID}
TREE="$WORK/gate-worktree"
DATASET=gate_plate
IN="$WORK/input"
PIPE="$WORK/round_peaks.json"

echo "Date: $(date)  Node: $(hostname)  Job: ${SLURM_JOB_ID}  SHA: ${GATE_SHA}"
mkdir -p "$WORK"

# Cleanup is a TERMINAL FINALIZER, pre-armed before any work, never a line at
# the end. This job holds while an inner array runs, so hitting the 2 h wall is
# its LIKELY failure -- and a job killed at the wall never reaches its own last
# line. `afterany` fires on success, failure, cancellation and timeout alike.
CLEANUP_ID=$(sbatch --parsable --dependency=afterany:"${SLURM_JOB_ID}" \
    --export=ALL,WORKTREE="$TREE",REPO="$REPO",GATE_WORK="$WORK" \
    "$REPO/.worktrees/cli-gui-state-tracking/docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/p5_task4_gate_cleanup.sbatch" 2>&1)
[[ "$CLEANUP_ID" =~ ^[0-9]+$ ]] || { echo "FATAL: cleanup submit failed: $CLEANUP_ID"; exit 1; }
echo "cleanup finalizer armed: ${CLEANUP_ID} (afterany:${SLURM_JOB_ID})"

# --- Gate worktree: the run must not observe an edit to the live checkout ----
git -C "$REPO" worktree add --detach "$TREE" "$GATE_SHA"
cd "$TREE"
export UV_CACHE_DIR=/bigdata/exfab/anguy344/.uv_cache
uv sync --quiet

# The venv trap: several worktrees sharing one prebuilt venv all import the
# SAME tree, and every result becomes a copy of one with nothing failing.
RESOLVED=$(uv run python -c "import phenotypic; print(phenotypic.__file__)")
case "$RESOLVED" in
    "$TREE"/*) echo "provenance OK: $RESOLVED" ;;
    *) echo "FATAL: phenotypic resolves OUTSIDE the gate worktree: $RESOLVED"; exit 1 ;;
esac

# --- The pipeline: CPU-only, serialized by the code under test --------------
uv run python - "$PIPE" <<'PY'
import sys
from pathlib import Path
from phenotypic.prefab import RoundPeaksPipeline
from phenotypic._cli._cli_validation import pipeline_requires_gpu

path = Path(sys.argv[1])
path.write_text(RoundPeaksPipeline().to_json(), encoding="utf-8")
# The ENTIRE reason this fixture replaced the bed's. Asserted rather than
# assumed: a GPU op here would silently route to StagedSlurmStrategy, which
# never calls the code this gate exists to measure, and every topology
# assertion below would then be describing a different engine.
if pipeline_requires_gpu(path):
    raise SystemExit("FATAL: pipeline requires a GPU; the gate would measure the staged engine")
print(f"pipeline written, CPU-only confirmed: {path}")
PY

# --- Inputs: N images, in a dataset subdirectory that survives into work_id --
# `--input` is the PARENT. `work_id` folds in the path relative to the input
# root, so pointing at the dataset directory itself would make every relative
# path a bare filename. Both runs share this directory; neither writes to it.
mkdir -p "$IN/$DATASET"
uv run python - "$BED" "$SRC_IMAGES" "$IN/$DATASET" "$N_IMAGES" <<'PY'
import json, shutil, sys
from pathlib import Path

bed, src, dst, count = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
state = json.loads((bed / ".phenotypic" / "processing_state.json").read_text())
# `datasets.<ds>.initial_images`, NOT `config.work_ids`: the work_ids KEYS are
# bare stems, `initial_images` carries the extension. The same confusion in the
# opposite direction cost a cycle when this bed was built -- that copy filtered
# `<stem>.tiff` against bare-stem keys and kept nothing. One root cause, two
# failures: which of the two shapes a given field holds.
completed = sorted(
    set(state["datasets"]["7-24-26_redo_full"].get("completed") or [])
)
names = [n if n.endswith(".tiff") else f"{n}.tiff" for n in completed][:count]
if len(names) != count:
    raise SystemExit(f"FATAL: wanted {count} images, the bed offers {len(names)}")
for name in names:
    source = src / name
    if not source.is_file():
        raise SystemExit(f"FATAL: input image missing: {source}")
    shutil.copy2(source, dst / name)
print(f"copied {len(names)} inputs")
PY
COUNT=$(find "$IN" -type f | wc -l)
[[ "$COUNT" -eq "$N_IMAGES" ]] || { echo "FATAL: input tree has $COUNT files, expected $N_IMAGES"; exit 1; }
echo "input guard OK: exactly $N_IMAGES images under $IN/$DATASET"

common_args() {
    printf '%s\0' --pipeline "$PIPE" --input "$IN" --output "$1" \
        --image-type Image --bit-depth 16 --ext .tiff
}

# --- Local reference run, njobs 1 -------------------------------------------
echo "=== LOCAL (--njobs 1) ==="
mapfile -d '' LOCAL_ARGS < <(common_args "$WORK/local_out")
LOCAL_START=$SECONDS
uv run python -m phenotypic "${LOCAL_ARGS[@]}" --njobs 1 --force-local
LOCAL_ELAPSED=$(( SECONDS - LOCAL_START ))
# Printed so the real per-image cost falls out of this run rather than being
# re-measured. The 165 s/image the wall was sized against was measured with
# `--image-type GridImage` at 16 CPUs; this runs `Image`, so it is expected to
# be lower, and the wall was sized in the safe direction.
echo "LOCAL ELAPSED: ${LOCAL_ELAPSED}s for ${N_IMAGES} images ($(( LOCAL_ELAPSED / N_IMAGES ))s/image)"

# --- The SLURM fan-out ------------------------------------------------------
# Ledger baseline: this tree is fresh so it is 0, but it is read rather than
# assumed -- reading someone else's rows as this run's topology is the defect
# the `pht-` grep had, and asserting the baseline keeps its fix honest.
PIPE_SHA=$(sha256sum "$PIPE" | cut -d' ' -f1)
echo "pipeline sha256 as passed: ${PIPE_SHA}"
LEDGER="$WORK/slurm_out/.phenotypic/progress/slurm_jobs.jsonl"
LEDGER_BEFORE=$( [[ -f "$LEDGER" ]] && wc -l < "$LEDGER" || echo 0 )
echo "job ledger baseline: ${LEDGER_BEFORE} rows"

echo "=== SLURM (fan-out) ==="
mapfile -d '' SLURM_ARGS < <(common_args "$WORK/slurm_out")
uv run python -m phenotypic "${SLURM_ARGS[@]}" \
    --slurm slurm_partition=short \
    --slurm slurm_cpus_per_task=4 \
    --slurm mem_gb=16 \
    --slurm slurm_time=00:30:00 \
    --wait

# --- The one observation no unit test can make -------------------------------
# Printed and asserted BEFORE the assertion block, deliberately. `set -e` means
# a non-zero exit there aborts the script, so anything after it is emitted only
# when everything already passed -- i.e. only when it is least needed. A
# topology failure is precisely the case where the generated array directive is
# the thing you want in the log.
#
# And its absence is a FAILURE, not a friendly message: the earlier
# `|| echo "no finalizer script found"` let a glob matching nothing pass green,
# having observed nothing at all.
echo "=== finalizer array directive ==="
shopt -s nullglob
FINALIZER_SCRIPTS=( "$WORK"/slurm_out/.phenotypic/slurm_scripts/*finalizer*.sh )
shopt -u nullglob
if [[ "${#FINALIZER_SCRIPTS[@]}" -ne 1 ]]; then
    echo "FATAL: expected exactly 1 finalizer script, found ${#FINALIZER_SCRIPTS[@]}"
    ls -la "$WORK"/slurm_out/.phenotypic/slurm_scripts/ || true
    exit 1
fi
ARRAY_LINE=$(grep -h -- '#SBATCH --array=' "${FINALIZER_SCRIPTS[0]}" || true)
echo "  ${FINALIZER_SCRIPTS[0]}"
echo "  ${ARRAY_LINE:-<no --array directive found>}"
[[ "$ARRAY_LINE" == *"--array=0-1"* ]] || {
    echo "FATAL: expected --array=0-1 (K=1 shard plus the reserved TASK_FINALIZE"
    echo "       index); got: ${ARRAY_LINE:-<none>}"
    exit 1
}

# --- Assertions -------------------------------------------------------------
echo "=== ASSERTIONS ==="
uv run python - "$WORK/slurm_out" "$WORK/local_out" "$LEDGER" "$LEDGER_BEFORE" "$N_IMAGES" "$PIPE_SHA" <<'PY'
import json, subprocess, sys
from pathlib import Path
import polars as pl
from phenotypic.sdk_ import master_measurements_parquet_path

slurm_out, local_out = Path(sys.argv[1]), Path(sys.argv[2])
ledger, baseline, expected_images = Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
pipe_sha = sys.argv[6]
failures: list[str] = []

# POSITIVE assertions first: a gate that only checks for the absence of bad
# things reads a run that did nothing as a run that went well.
for label, root in (("slurm", slurm_out), ("local", local_out)):
    path = master_measurements_parquet_path(root)
    if not path.is_file():
        failures.append(f"{label}: no master written")
        continue
    frame = pl.read_parquet(path)
    images = frame["Metadata_ImageName"].n_unique()
    print(f"{label}: {frame.height} rows, {images} distinct images")
    if frame.height == 0:
        failures.append(f"{label}: master is EMPTY")
    if images != expected_images:
        failures.append(f"{label}: {images} images, expected {expected_images}")

# Compared against the sha of the file PASSED on the command line, taken
# before either run -- not against whatever each tree recorded, compared only
# to the other. `:958-967` snapshots the pipeline into the output tree and
# REBINDS `config.pipeline_json` to that snapshot, so the recorded digest is a
# digest of the copy. It is `atomic_write_bytes(..., read_bytes())` today, so
# the copy is byte-exact and the shas agree; pinning it to the passed file is
# what would catch that becoming a re-serialisation, which is how this
# project's own fixture acquired two pipeline files that hash differently.
digests = {}
for label, root in (("slurm", slurm_out), ("local", local_out)):
    state = root / ".phenotypic" / "processing_state.json"
    if state.is_file():
        digests[label] = json.loads(state.read_text())["config"].get("pipeline_sha256")
if set(digests.values()) != {pipe_sha}:
    failures.append(
        f"recorded pipeline digests {digests} do not both equal the passed "
        f"file's {pipe_sha}"
    )
else:
    print(f"both runs recorded pipeline_sha256={pipe_sha} (the file passed)")

if not failures:
    a = master_measurements_parquet_path(slurm_out).read_bytes()
    b = master_measurements_parquet_path(local_out).read_bytes()
    if a != b:
        failures.append(f"masters differ: {len(a)} vs {len(b)} bytes")
    else:
        print(f"masters byte-identical ({len(a)} bytes)")

# --- Topology, by identity from this run's OWN ledger ------------------------
rows = [
    json.loads(line)
    for line in ledger.read_text(encoding="utf-8").splitlines()[baseline:]
    if line.strip()
] if ledger.is_file() else []
submitted = [r for r in rows if r.get("status") == "submitted" and r.get("job_id")]
roles: dict[str, list[str]] = {}
for row in submitted:
    roles.setdefault(str(row.get("role", "?")), []).append(str(row["job_id"]))

print(f"this run submitted {len(submitted)} job(s):")
for role, ids in sorted(roles.items()):
    print(f"  {role:28} {ids}")
    for job_id in ids:
        out = subprocess.run(
            ["sacct", "-j", job_id, "-n", "-P", "--format=JobID,JobName,State,ExitCode"],
            capture_output=True, text=True,
        ).stdout.strip()
        for line in out.splitlines():
            if ".batch" not in line and ".extern" not in line:
                print(f"      {line}")

# A staged role appearing at all means the pipeline was not CPU-only and this
# gate measured a different engine. Positive evidence that
# AutonomousSLURMStrategy was selected, which is the thing that can go wrong.
staged = sorted(r for r in roles if any(
    token in r for token in ("stage", "controller")
))
if staged:
    failures.append(f"staged-engine roles present -- wrong strategy: {staged}")

# PRECONDITION, asserted where the assertion that depends on it lives.
# `AutonomousSLURMStrategy` (:1127-1144) records `job_ids[1]` as role
# "finalizer" ONLY when there is no initial dispatcher -- i.e. only for a
# single-chunk run. Chunk it, and that row is recorded as "dispatcher" and no
# finalizer row exists at submission at all. So "exactly one finalizer" is
# valid only inside a range nothing else states, and a future N that chunks
# would fail with "expected 1 finalizer, got 0" -- a true message pointing at
# the wrong thing.
# `dispatcher`, NOT a count of chunk roles. Counting chunks is VACUOUS: the
# strategy records only `chunk-0` (`:1131-1135`; the comment at `:1146-1150`
# says later chunk ids are assigned by the dispatcher after each chunk
# completes), so `len(chunks) == 1` holds whether or not the run chunked and
# the check could never fire. A `dispatcher` role appears exactly when
# `has_initial_dispatcher` is true, which is exactly when `job_ids[1]` was
# recorded as the dispatcher instead of the finalizer.
if any("dispatcher" in r for r in roles):
    failures.append(
        "the fixture chunked (a dispatcher role is present); the "
        "one-finalizer assertion below does not hold above one chunk, and "
        "its failure would name the finalizer rather than the chunking"
    )
finalizers = [r for r in roles if "finaliz" in r]
if len(finalizers) != 1:
    failures.append(f"expected exactly 1 finalizer role, got {finalizers}")
unexpected = [
    r for r in roles
    if "finaliz" not in r and "chunk" not in r and "dispatcher" not in r
]
if unexpected:
    failures.append(f"unexpected job(s) beside the array: {unexpected}")

# THE reserved TASK_FINALIZE index, made observable in the real scheduler.
# This is the one thing the gate establishes that no unit test can: that
# `--array=0-K` was dispatched as K+1 tasks. Read from sacct rather than by
# eyeballing the generated script.
for role, ids in roles.items():
    if "finaliz" not in role:
        continue
    for job_id in ids:
        out = subprocess.run(
            ["sacct", "-j", job_id, "-n", "-P", "--format=JobID"],
            capture_output=True, text=True,
        ).stdout
        indices = sorted({
            line.split("_", 1)[1]
            for line in out.splitlines()
            if "_" in line and ".batch" not in line and ".extern" not in line
        })
        print(f"  finalizer {job_id} array indices: {indices}")
        if indices != ["0", "1"]:
            failures.append(
                f"finalizer {job_id} dispatched indices {indices}, expected "
                "['0', '1'] -- one shard plus the reserved TASK_FINALIZE entry"
            )

print("GATE FAILURES:", failures or "none")
sys.exit(1 if failures else 0)
PY


echo "gate complete: $(date)"
