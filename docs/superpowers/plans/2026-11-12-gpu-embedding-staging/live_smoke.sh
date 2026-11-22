#!/bin/bash
# #282 P5 Task 4: the live SLURM Stage-4 smoke on real GPUs.
#
#   live_smoke.sh setup  <commit> <zarr-dir>        once: the NEW output folder, the
#                                                   subset
#   live_smoke.sh pipelines <v>                     pipelines/<v>/{nodet,det}.json
#   live_smoke.sh probe <n> <v>                     one CPU job (probe-<n>/): a one-image
#                                                   per-step timing, a --dry-run of both
#                                                   runs, then a 2-image local run of each
#   live_smoke.sh launch <attempt> <v> <exfab|short_gpu> <nodet-time> <det-time>
#                                                   one attempt: both runs, each under
#                                                   its own `phenotypic-wait` job, and an
#                                                   `afterany` worktree cleanup
#
# Two runs per attempt, both `--gpu-shards 2`, CPU stages on `intel`:
#   nodet  OtsuDetector, MeasureSize, and MeasureDinoEmbedding(size="small",
#          revision=<pinned>) on objects and on a 2 x 2 ManualGridFinder grid. No
#          GpuDetector: Stage 4 comes from `--gpu-slurm` alone (`--embed-stage auto`),
#          phases ["stage3","stage4"].
#   det    the same with FakeGpuDetector in Otsu's place (PHENOTYPIC_PRELOAD_MODULES),
#          so all four stages run.
# Pipeline v1 is the detector alone; on these plates Otsu labels 65535 objects (the
# uint16 ceiling). v2 brackets the detector as the plates' own pipeline does:
# BlurGauss(sigma=5) before, RemoveSmallObjects(min_size=600) and
# RemoveBorderObjects(border_size=20) after. In v2 FakeGpuDetector reads its default
# `rgb` layer, which the blur never touches, and labels thin plate-spanning networks
# (median padded crop side 4171 px against Otsu's 151), so each image's crop prep costs
# ~100x nodet's. v3 keeps nodet as v2 and points the fake at the blurred `detect_mat`
# with threshold=0.6, which gave Otsu-like objects on three probed plates
# (probe-4-threshold/threshold_sweep.json).
# `<nodet-time>`/`<det-time>` are the `--gpu-slurm time=` minutes, sized so Stage-4
# round 1 hits its walltime after some images commit (report, "Walltime sizing").
#
# The inputs are read only. The subset is given to the CLI as `--input <zarr-dir>
# --image-manifest subset.txt`: no store is copied or linked, and nothing is
# written under <zarr-dir>. Every job runs in a detached worktree at <commit>, one
# per attempt, with the foundation venv and PYTHONPATH pointing into that worktree
# (the venv's editable install points at another tree, so each job prints where
# `phenotypic` resolved from). Never `preempt`; every captured job id is checked.
set -euo pipefail

ROOT=/bigdata/exfab/anguy344/gate-results/c16-live-smoke-20261121
REPO=/bigdata/exfab/anguy344/PhenoTypic
PY=/bigdata/exfab/anguy344/gate-results/c18/venv-foundation/bin/python
WAIT=/bigdata/exfab/anguy344/gate-results/c18/venv-foundation/bin/phenotypic-wait
REVISION=ed25f3a31f01632728cabb09d1542f84ab7b0056   # cached facebook/dinov2-small
N_SUBSET=400
CPU_PARTITION=intel

die() { echo "live_smoke: $*" >&2; exit 1; }

# submit <sbatch args...>: sbatch --parsable, refusing an empty or malformed id.
submit() {
  local out
  out=$(sbatch --parsable "$@" 2>&1) || die "sbatch failed: $out"
  out=${out%%;*}
  [[ $out =~ ^[0-9]+$ ]] || die "sbatch returned no job id: $out"
  echo "$out"
}

# env_block <tree> [det]: the environment every job of a run exports. SLURM's
# default --export=ALL carries it into the controller and every job it submits.
env_block() {
  local tree=$1
  cat <<EOF
export PYTHONPATH=$tree/src:$tree
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_OFFLINE=1
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
EOF
  [[ ${2:-} == det ]] && echo "export PHENOTYPIC_PRELOAD_MODULES=tests._fakes.register_fake_gpu"
  cat <<EOF
echo "host=\$(hostname -s) job=\${SLURM_JOB_ID:-} date=\$(date -Is)"
echo "phenotypic=\$($PY -c 'import phenotypic; print(phenotypic.__file__)')"
echo "tree_head=\$(git -C $tree rev-parse HEAD) dirty=\$(git -C $tree status --porcelain --untracked-files=no | wc -l)"
EOF
}

# run_args <pipeline> <output> <manifest> [slurm-profile]: the CLI argv after `-m phenotypic`.
run_args() {
  local pipeline=$1 output=$2 manifest=$3
  echo "--pipeline $pipeline --input $(cat "$ROOT/input/zarr_dir.txt") --image-manifest $manifest --output $output --image-type gridimage"
}

add_tree() {
  local tree=$1 sha=$2
  [[ -e $tree ]] && die "$tree already exists; refusing to share it"
  git -C "$REPO" worktree add --detach "$tree" "$sha" >/dev/null
  [[ $(git -C "$tree" rev-parse HEAD) == "$sha" ]] || die "$tree is not at $sha"
  local resolved
  resolved=$(cd "$ROOT" && PYTHONSAFEPATH=1 PYTHONPATH=$tree/src:$tree "$PY" -c 'import phenotypic; print(phenotypic.__file__)')
  [[ $resolved == "$tree/src/phenotypic/__init__.py" ]] || die "phenotypic resolves to $resolved, not $tree"
}

cmd_setup() {
  local sha zarr_dir
  sha=$(git -C "$REPO" rev-parse --verify "${1:?commit}^{commit}")
  zarr_dir=$(cd "${2:?zarr-dir}" && pwd -P)
  [[ -e $ROOT ]] && die "$ROOT already exists; the smoke writes only into a new folder"
  compgen -G "$HOME/.cache/huggingface/hub/models--facebook--dinov2-small/snapshots/$REVISION/model.safetensors" >/dev/null \
    || die "dinov2-small $REVISION is not cached"
  mkdir -p "$ROOT"/{input,pipelines}
  echo "$sha" > "$ROOT/input/sha.txt"
  echo "$zarr_dir" > "$ROOT/input/zarr_dir.txt"

  # The subset: the first N stores by name (Python's code-point sort), absolute paths.
  "$PY" - "$zarr_dir" "$N_SUBSET" "$ROOT/input" <<'PY'
import json, pathlib, sys
zarr_dir, n, out = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
names = sorted(p.name for p in zarr_dir.iterdir()
               if p.name.endswith(".ome.zarr") and not p.name.startswith(".") and p.is_dir())
subset = names[:n]
shapes = {}
for name in subset:
    meta = json.loads((zarr_dir / name / "rgb" / "0" / "zarr.json").read_text())
    shapes.setdefault(tuple(meta["shape"]), []).append(name)
(out / "subset.txt").write_text("".join(f"{zarr_dir / s}\n" for s in subset))
(out / "subset2.txt").write_text("".join(f"{zarr_dir / s}\n" for s in subset[:2]))
summary = {"zarr_dir": str(zarr_dir), "stores_in_dir": len(names), "subset": len(subset),
           "first": subset[0], "last": subset[-1],
           "rgb_level0_shapes": {"x".join(map(str, k)): len(v) for k, v in shapes.items()}}
(out / "subset_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary))
if len(shapes) != 1:
    sys.exit("the subset mixes plate shapes; the 2 x 2 grid needs one shape")
PY
  sha256sum "$ROOT/input/subset.txt" > "$ROOT/input/subset.txt.sha256"

  echo "setup done: $ROOT"
}

cmd_pipelines() {
  local v=${1:?pipeline version (v1|v2|v3)}
  [[ -d $ROOT/input ]] || die "run setup first"
  [[ -e $ROOT/pipelines/$v ]] && die "$ROOT/pipelines/$v already exists"
  local sha
  sha=$(cat "$ROOT/input/sha.txt")
  mkdir -p "$ROOT/pipelines/$v"
  # Written by the code under test (a throwaway tree, removed after).
  local tree=/bigdata/exfab/anguy344/gate-worktrees/c16-${sha:0:9}-pipe-$v
  add_tree "$tree" "$sha"
  local shape
  shape=$("$PY" -c "import json;print(next(iter(json.load(open('$ROOT/input/subset_summary.json'))['rgb_level0_shapes'])))")
  (cd "$ROOT" && PYTHONSAFEPATH=1 PYTHONPATH=$tree/src:$tree "$PY" - "$ROOT/pipelines/$v" "$REVISION" "$shape" "$v" <<'PY'
import pathlib, sys
import phenotypic
from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.grid import ManualGridFinder
from phenotypic.measure import MeasureDinoEmbedding, MeasureSize
from phenotypic.refine import RemoveBorderObjects, RemoveSmallObjects
from tests._fakes.fake_gpu_detector import FakeGpuDetector

out, revision, version = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[4]
_, height, width = (int(v) for v in sys.argv[3].split("x"))
def meas():
    return {
        "size": MeasureSize(),
        "emb_obj": MeasureDinoEmbedding(size="small", revision=revision),
        "emb_grid": MeasureDinoEmbedding(
            size="small", revision=revision, regions="grid",
            grid=ManualGridFinder(row_edges=[0, height // 2, height],
                                  col_edges=[0, width // 2, width]),
        ),
    }
def ops(detector):
    if version == "v1":
        return [detector]
    return {"blur": BlurGauss(sigma=5.0), "detect": detector,
            "small": RemoveSmallObjects(min_size=600),
            "border": RemoveBorderObjects(border_size=20)}
(out / "nodet.json").write_text(
    ImagePipeline(name=f"c16-nodet-{version}", ops=ops(OtsuDetector()), meas=meas()).to_json())
fake = (FakeGpuDetector(input_layer="detect_mat", threshold=0.6) if version == "v3"
        else FakeGpuDetector(threshold=0.5))
(out / "det.json").write_text(
    ImagePipeline(name=f"c16-det-{version}", ops=ops(fake), meas=meas()).to_json())
print("pipelines written by", phenotypic.__file__)
PY
  )
  git -C "$REPO" worktree remove --force "$tree"
  git -C "$REPO" worktree prune
  echo "pipelines $v written: $ROOT/pipelines/$v"
}

cmd_probe() {
  [[ -d $ROOT/input ]] || die "run setup first"
  local n=${1:?probe number} v=${2:?pipeline version} sha tree
  local P=$ROOT/probe-$n PIPES=$ROOT/pipelines/$v
  [[ -f $PIPES/nodet.json && -f $PIPES/det.json ]] || die "no pipelines $v" 
  [[ -e $P ]] && die "$P already exists; each probe gets a new folder"
  sha=$(cat "$ROOT/input/sha.txt")
  tree=/bigdata/exfab/anguy344/gate-worktrees/c16-${sha:0:9}-probe$n
  mkdir -p "$P"
  add_tree "$tree" "$sha"
  local gpu_profile="--gpu-slurm slurm_partition=short_gpu --gpu-slurm slurm_constraint=gpu_latest --gpu-slurm slurm_qos=short_gpu --gpu-slurm time=30 --gpu-slurm slurm_cpus_per_task=2 --gpu-slurm mem_gb=24"
  local cpu_profile="--slurm slurm_partition=$CPU_PARTITION --slurm slurm_cpus_per_task=2 --slurm mem_gb=16 --slurm time=60"
  {
    echo "#!/bin/bash"
    echo "set -uo pipefail"
    env_block "$tree" det
    echo "cd $P"
    # 0. one input image through each pipeline's ops and measurers, timed per step
    for run in nodet det; do
      echo "echo '== one-image timing $run =='"
      echo "$PY - $(head -1 "$ROOT/input/subset.txt") $PIPES/$run.json $P/one_image_$run.json <<'PY'"
      cat <<'PY'
import json, pathlib, sys, time
from phenotypic import GridImage, ImagePipeline
store, pipe, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
pipeline = ImagePipeline.from_json(pipe.read_text())
row = {"store": store.name}
t = time.perf_counter(); image = GridImage.imread(store); row["imread_s"] = round(time.perf_counter() - t, 2)
t = time.perf_counter(); image = pipeline.apply(image); row["ops_s"] = round(time.perf_counter() - t, 2)
row["objects"] = int(image.num_objects); print(json.dumps(row), flush=True)
t = time.perf_counter(); pipeline.meas["size"].measure(image); row["size_s"] = round(time.perf_counter() - t, 2)
for key in ("emb_obj", "emb_grid"):
    t = time.perf_counter(); prep = pipeline.meas[key].prepare_regions(image)
    row[f"{key}_prepare_s"] = round(time.perf_counter() - t, 2); row[f"{key}_regions"] = int(prep.n_regions)
print(json.dumps(row), flush=True)
out.write_text(json.dumps(row, indent=1) + "\n")
PY
      echo "PY"
    done
    for run in nodet det; do
      # 1. dry-run of the real 400-image SLURM command
      echo "echo '== dry-run $run (400, SLURM) =='"
      echo "$PY -m phenotypic $(run_args "$PIPES/$run.json" "$P/dryrun-$run" "$ROOT/input/subset.txt") $cpu_profile $gpu_profile --gpu-shards 2 --dry-run; echo \"exit=\$?\""
      echo "test -e $P/dryrun-$run && echo 'dry-run created its output' || echo 'dry-run created no output'"
    done
    for run in nodet det; do
      # 2. two images, local, CPU, inline embeddings
      echo "echo '== local 2-image $run =='"
      echo "/usr/bin/time -v $PY -m phenotypic $(run_args "$PIPES/$run.json" "$P/cpu2-$run" "$ROOT/input/subset2.txt") --njobs 2 --force-local; echo \"exit=\$?\""
    done
    # 3. the Stage-4 loader's per-image CPU cost on the probe's own stores
    echo "echo '== loader timing =='"
    echo "$PY - $P/cpu2-nodet $PIPES/nodet.json <<'PY'"
    cat <<'PY'
import json, pathlib, sys, time
from phenotypic import GridImage, ImagePipeline
out, pipe = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
pipeline = ImagePipeline.from_json(pipe.read_text())
embedders = {k: v for k, v in pipeline.meas.items() if k.startswith("emb")}
rows = []
for store in sorted(out.glob("results/*/zarr/*.ome.zarr")):
    t0 = time.perf_counter(); image = GridImage.load_zarr(store); t1 = time.perf_counter()
    row = {"store": store.name, "load_s": round(t1 - t0, 3), "objects": int(image.num_objects)}
    for key, emb in embedders.items():
        t2 = time.perf_counter(); prepared = emb.prepare_regions(image); t3 = time.perf_counter()
        row[f"{key}_prepare_s"] = round(t3 - t2, 3)
        row[f"{key}_regions"] = int(prepared.n_regions)
    rows.append(row)
print(json.dumps(rows, indent=1))
(out.parent / "loader_timing.json").write_text(json.dumps(rows, indent=1) + "\n")
PY
    echo "PY"
  } > "$P/probe.sbatch"
  local jid cid
  jid=$(submit --job-name=c16-probe --partition=$CPU_PARTITION --cpus-per-task=4 --mem=32G \
        --time=02:00:00 --output="$P/probe-%j.log" "$P/probe.sbatch")
  cid=$(submit --job-name=c16-probe-cleanup --partition=$CPU_PARTITION --cpus-per-task=1 --mem=2G \
        --time=00:15:00 --dependency=afterany:$jid --output="$P/probe-cleanup-%j.log" \
        --wrap="git -C $REPO worktree remove --force $tree; git -C $REPO worktree prune; test ! -e $tree && echo 'worktree removed: $tree'")
  echo "probe job $jid, cleanup $cid"
  echo "$jid $cid" > "$P/job_ids.txt"
}

cmd_launch() {
  local attempt=${1:?attempt} v=${2:?pipeline version} gpu=${3:?exfab|short_gpu} t_nodet=${4:?nodet minutes} t_det=${5:?det minutes}
  local PIPES=$ROOT/pipelines/$v
  [[ -f $PIPES/nodet.json && -f $PIPES/det.json ]] || die "no pipelines $v" 
  [[ -d $ROOT/input ]] || die "run setup first"
  local adir=$ROOT/attempt-$attempt
  [[ -e $adir ]] && die "$adir already exists; each attempt gets a new folder"
  local gpu_flags
  case $gpu in
    exfab) gpu_flags="--gpu-slurm slurm_partition=exfab --gpu-slurm slurm_account=exfab" ;;
    short_gpu)
      gpu_flags="--gpu-slurm slurm_partition=short_gpu --gpu-slurm slurm_constraint=gpu_latest --gpu-slurm slurm_qos=short_gpu"
      for t in $t_nodet $t_det; do [[ $t == - ]] || (( t <= 120 )) || die "short_gpu caps at 120 minutes"; done ;;
    *) die "GPU partition must be exfab or short_gpu (never preempt)" ;;
  esac
  local sha tree
  sha=$(cat "$ROOT/input/sha.txt")
  tree=/bigdata/exfab/anguy344/gate-worktrees/c16-${sha:0:9}-a$attempt
  mkdir -p "$adir"
  add_tree "$tree" "$sha"
  local cpu_profile="--slurm slurm_partition=$CPU_PARTITION --slurm slurm_cpus_per_task=2 --slurm mem_gb=16 --slurm time=60"
  local ids=() names=()
  for run in nodet det; do
    local t=$t_nodet
    [[ $run == det ]] && t=$t_det
    [[ $t == - ]] && continue   # "-" skips that run in this attempt
    local out=$adir/$run
    {
      echo "#!/bin/bash"
      echo "set -uo pipefail"
      env_block "$tree" "$run"
      echo "cd $adir"
      echo "$WAIT --output $out -- $PY -m phenotypic $(run_args "$PIPES/$run.json" "$out" "$ROOT/input/subset.txt") $cpu_profile $gpu_flags --gpu-slurm time=$t --gpu-slurm slurm_cpus_per_task=2 --gpu-slurm mem_gb=24 --gpu-shards 2"
      echo "rc=\$?; echo \"phenotypic-wait exit=\$rc\"; date -Is; exit \$rc"
    } > "$adir/wait-$run.sbatch"
    ids+=("$(submit --job-name=c16-wait-$run --partition=$CPU_PARTITION --cpus-per-task=2 --mem=8G \
            --time=3-00:00:00 --output="$adir/wait-$run-%j.log" "$adir/wait-$run.sbatch")")
    names+=("$run")
  done
  (( ${#ids[@]} )) || die "no run selected"
  # The waiter returns once the run is published, which can be before the run's
  # last pre-armed recovery controller has run (attempt 2's 32353546 died importing
  # a tree this cleanup was deleting). So the cleanup, after the waiters, also waits
  # until no job of any run's ledger is still queued or running.
  local cid
  cid=$(submit --job-name=c16-cleanup-a$attempt --partition=$CPU_PARTITION --cpus-per-task=1 --mem=2G \
        --time=02:00:00 --dependency=afterany:$(IFS=:; echo "${ids[*]}") --output="$adir/cleanup-%j.log" \
        --wrap="bash $ROOT/live_smoke.sh cleanup $tree $(for n in "${names[@]}"; do printf '%s ' "$adir/$n"; done)")
  {
    printf 'pipelines=%s gpu=%s nodet_time=%s det_time=%s\n' "$v" "$gpu" "$t_nodet" "$t_det"
    for i in "${!ids[@]}"; do printf 'wait_%s=%s\n' "${names[$i]}" "${ids[$i]}"; done
    printf 'cleanup=%s\ntree=%s\n' "$cid" "$tree"
  } | tee "$adir/launch.txt"
}

# cleanup <tree> <run-dir...>: wait until no job in the runs' ledgers is queued or
# running, then remove the worktree.
cmd_cleanup() {
  local tree=${1:?tree}; shift
  local ids
  while :; do
    ids=$(cat "${@/%//.phenotypic/progress/slurm_jobs.jsonl}" 2>/dev/null \
          | grep -o '"job_id": *"[0-9]*"' | grep -o '[0-9]*' | sort -u | paste -sd,)
    [[ -z $ids ]] && break
    [[ -z $(squeue -h -j "$ids" 2>/dev/null) ]] && break
    echo "$(date -Is) waiting on ledgered jobs: $(squeue -h -j "$ids" -o %i | paste -sd' ')"
    sleep 30
  done
  git -C "$REPO" worktree remove --force "$tree"
  git -C "$REPO" worktree prune
  test ! -e "$tree" && echo "worktree removed: $tree"
}

# verify <run-dir> <wait-log>: Step 3's checks over one finished run, read by an
# independent reader (stdlib, zarr, pyarrow; no `phenotypic` import). Prints a
# summary and writes <run-dir>/../verify-<run>.json. Exits non-zero on a failed check.
cmd_verify() {
  local run=${1:?run dir} wait_log=${2:?waiter log}
  [[ -d $run/.phenotypic ]] || die "$run is not a run"
  (cd /tmp && "$PY" -I - "$run" "$wait_log" "$ROOT/input/subset.txt" "$ROOT/input/sha.txt" <<'PY'
import collections, datetime as dt, hashlib, json, math, pathlib, re, subprocess, sys
import pyarrow.parquet as pq
import zarr

run, wait_log, subset_path, sha_path = (pathlib.Path(a) for a in sys.argv[1:5])
ph = run / ".phenotypic"
fails, out = [], {"run": str(run)}
def check(ok, what):
    if not ok:
        fails.append(what)
    return ok
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# --- the waiter's exit code
m = re.findall(r"^phenotypic-wait exit=(\d+)$", wait_log.read_text(), re.M)
out["phenotypic_wait_exit"] = int(m[-1]) if m else None

# --- scripts and controller config
scripts = sorted(p.name for p in (ph / "slurm_scripts").iterdir())
cfg = json.loads((ph / "slurm_scripts" / "staged_controller.json").read_text())
out["scripts"] = scripts
out["controller"] = {k: cfg.get(k) for k in ("version", "phases", "stage3_role")}

# --- the ledger: every row; each job's scheduler state, per array task
rows = [json.loads(l) for l in (ph / "progress" / "slurm_jobs.jsonl").read_text().splitlines() if l.strip()]
epoch = rows[0]["epoch"]
jobs = {}
for r in rows:
    if r.get("job_id"):
        jobs.setdefault(r["job_id"], (r["role"], r["token"]))
def sacct(job):
    text = subprocess.run(["sacct", "-j", job, "-n", "-P", "-X", "--format=JobID,State,Start,End,Elapsed,NodeList"],
                          capture_output=True, text=True).stdout
    return [dict(zip(("id", "state", "start", "end", "elapsed", "node"), l.split("|"))) for l in text.splitlines() if l]
ledger = []
for job, (role, token) in jobs.items():
    ledger.append({"job_id": job, "role": role, "token": token, "tasks": sacct(job)})
out["ledger"] = ledger
stage4 = [j for j in ledger if j["role"] == "stage4"]
out["stage4_rounds"] = [{"token": j["token"], "states": sorted({t["state"] for t in j["tasks"]})} for j in stage4]

# --- the records, one per image of the manifest, each checked against the bytes on disk
manifest = json.loads((ph / "slurm_scripts" / "staged_manifest.json").read_text())
entries = manifest["images"]
subset = [pathlib.Path(l).name for l in subset_path.read_text().splitlines() if l.strip()]
check(sorted(e["image_name"] for e in entries) == sorted(subset), "manifest is not the subset")
# The terminal-failure journal: an image is "failed" when a row of this epoch names
# its work id and it has no full record (spec U7: no record, and a journal row).
journal_path = ph / "terminal_failures.jsonl"
journal = [json.loads(l) for l in journal_path.read_text().splitlines() if l.strip()] if journal_path.is_file() else []
failed_ids = {r["work_id"] for r in journal if r.get("lifecycle_epoch") == epoch}
out["journal_rows"] = len(journal)
out["journal_by_stage_and_type"] = dict(collections.Counter(
    f"{r['failed_stage']}: {r['exception_message'].split(chr(60) + 'class ' + chr(39))[-1].split(chr(39))[0]}"
    for r in journal))
n_ok, n_failed, compute_seen, batches_seen = 0, 0, collections.Counter(), collections.Counter()
for e in entries:
    rec_path = ph / "progress" / "images" / e["dataset"] / f"{e['stem']}.json"
    if e["work_id"] in failed_ids:
        partial = json.loads(rec_path.read_text()) if rec_path.is_file() else {}
        n_failed += bool(check("artifacts" not in partial, f"a failed image has a full record: {e['stem']}"))
        continue
    if not check(rec_path.is_file(), f"no record: {e['stem']}"):
        continue
    rec = json.loads(rec_path.read_text())
    ok = (rec.get("version") == 1 and rec.get("work_id") == e["work_id"]
          and rec.get("lifecycle_epoch") == epoch and "stage3" in rec.get("stages", {}))
    for name, art in rec["artifacts"].items():
        p = run / art["path"]
        if art["kind"] == "store":
            ok &= art["sha256"] == "sha256:" + sha256_file(p / "zarr.json")
        else:
            ok &= art["sha256"] == sha256_file(p) and art["size"] == p.stat().st_size
    store = run / rec["artifacts"]["store"]["path"]
    root = zarr.open_group(str(store), mode="r")
    tables = root.attrs["phenotypic"]["tables"]
    latest = max(tables["runs"].values(), key=lambda r: r["order"])
    meas = pq.read_table(store / latest["measurements"]["path"])
    labels = sorted(meas.column("Object_Label").to_pylist())
    emb = latest.get("embeddings", {})
    ok &= sorted(emb) == ["dinov2-vits14/rgb/grid", "dinov2-vits14/rgb/objects"]
    for key, block in emb.items():
        c = block["compute"]
        ok &= block["stale"] is False and c["stage"] == "stage4" and c["device"] == "cuda" and bool(c["device_name"])
        if c.get("physical_batch") is None:
            ok &= isinstance(c.get("physical_batches"), list) and len(c["physical_batches"]) > 1
            batches_seen[tuple(c.get("physical_batches") or ())] += 1
        else:
            batches_seen[c["physical_batch"]] += 1
        compute_seen[(c["device_name"], c["capability"], c["torch"], c["cuda"])] += 1
        for f in block["files"]:
            ok &= f["sha256"] == sha256_file(store / f["path"])
            t = pq.read_table(store / f["path"])
            vec = t.column("embedding").to_pylist()
            ok &= all(len(v) == block["dim"] and all(math.isfinite(x) for x in v) for v in vec)
            if key.endswith("/objects"):
                ok &= sorted(t.column("Object_Label").to_pylist()) == labels
            else:
                ok &= t.num_rows == 4
    n_ok += bool(check(ok, f"record or store check failed: {e['stem']}"))
out["records_valid"] = n_ok
out["images"] = len(entries)
out["compute"] = [{"device_name": k[0], "capability": k[1], "torch": k[2], "cuda": k[3], "entries": v}
                  for k, v in compute_seen.items()]
out["physical_batch"] = {str(k): v for k, v in batches_seen.items()}

# --- Stage-4 signals left over (consumed after each commit)
out["stage4_signals_left"] = sum(1 for p in (ph / "progress" / "stage4_payload").rglob("*") if p.is_file())
# Stage-2 signals left over: allowed only for an image whose Stage 3 failed terminally.
failed_stems = {e["stem"] for e in entries if e["work_id"] in failed_ids}
left2 = [p for d in ("stage2_done", "stage2_raw") if (ph / "progress" / d).is_dir()
         for p in (ph / "progress" / d).rglob("*") if p.is_file()]
out["stage2_signals_left"] = len(left2)
check({p.name.split(".")[0] for p in left2} <= failed_stems, "Stage-2 signals left for an image that did not fail")
out["events"] = {f"{k[0]} {k[1]}": v for k, v in sorted(collections.Counter(
    (l.split("|")[7], l.split("|")[3]) for l in (ph / "processing_events.log").read_text().splitlines()
    if len(l.split("|")) > 7).items())}

# --- re-embedding: Stage-4 events per image, bucketed into the rounds by the tasks' run windows
t = dt.datetime.fromisoformat
windows = []
for j in stage4:
    for task in j["tasks"]:
        if task["start"] not in ("", "Unknown", "None"):
            end = task["end"] if task["end"] not in ("", "Unknown") else "2999-01-01T00:00:00"
            windows.append((j["token"], task["id"], t(task["start"]), t(end)))
events = [l.split("|") for l in (ph / "processing_events.log").read_text().splitlines()]
per_image = collections.defaultdict(lambda: collections.Counter())
for ev in events:
    if len(ev) > 7 and ev[7] == "stage4":
        when = t(ev[0])
        rnd = next((w[0] for w in windows if w[2] <= when <= w[3]), "outside")
        per_image[ev[2]][(rnd, ev[3])] += 1
def round_no(token):
    return int(token.rsplit("-", 1)[1]) if token.startswith("stage4-round-") else None
completed_by_round = collections.Counter()
started_after_completion = 0
for name, c in per_image.items():
    done = [round_no(r) for (r, s), n in c.items() if s == "completed" and round_no(r) is not None]
    for r in done:
        completed_by_round[f"stage4-round-{r}"] += 1
    if done:
        started_after_completion += sum(n for (r, s), n in c.items()
                                        if s == "started" and round_no(r) is not None and round_no(r) > min(done))
outside = sum(n for c in per_image.values() for (r, s), n in c.items() if r == "outside")
out["stage4_events_outside_any_task_window"] = outside
out["stage4_completions_per_round"] = dict(completed_by_round)
out["images_completed_twice"] = sum(1 for c in per_image.values()
                                    if sum(n for (r, s), n in c.items() if s == "completed") > 1)
out["started_again_after_completion"] = started_after_completion
check(out["images_completed_twice"] == 0, "an image completed Stage 4 twice")
check(started_after_completion == 0, "an image completed in one round was started in a later one")

# --- the shard logs' own summaries
logs = sorted((ph / "logs" / "slurm").glob("stage4_*.log"))
out["shard_summaries"] = {p.name: [l for l in p.read_text().splitlines() if l.startswith("Stage 4 shard")]
                          for p in logs}

# --- the read-only input: nothing under the Maresca stores changed after setup
setup_time = (sha_path).stat().st_mtime
newest = max((pathlib.Path(l) / "zarr.json").stat().st_mtime for l in subset_path.read_text().splitlines() if l.strip())
out["inputs_newest_root_mtime"] = dt.datetime.fromtimestamp(newest).isoformat()
check(newest < setup_time, "an input store's root changed after setup")

out["failed_journalled"] = n_failed
out["unresolved"] = out["images"] - out["records_valid"] - n_failed
check(out["unresolved"] == 0, "verified + failed does not account for every image")
check(len(failed_ids) == n_failed, "a journal row names no image of the manifest")
# The waiter's code: 0 with no terminal failure, 3 with some (RunExitCode).
check(out["phenotypic_wait_exit"] == (3 if n_failed else 0), "phenotypic-wait exit does not match the failures")
check(out["stage4_signals_left"] == 0, "Stage-4 signals were left behind")
out["failed_checks"] = fails
dest = run.parent / f"verify-{run.name}.json"
dest.write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "ledger"}, indent=1))
print("ledger:")
for j in ledger:
    print(" ", j["role"], j["token"], j["job_id"], [(x["id"], x["state"], x["elapsed"], x["node"]) for x in j["tasks"]])
sys.exit(1 if fails else 0)
PY
  )
}

case ${1:-} in
  setup) shift; cmd_setup "$@" ;;
  verify) shift; cmd_verify "$@" ;;
  cleanup) shift; cmd_cleanup "$@" ;;
  pipelines) shift; cmd_pipelines "$@" ;;
  probe) shift; cmd_probe "$@" ;;
  launch) shift; cmd_launch "$@" ;;
  *) die "usage: live_smoke.sh {setup <commit> <zarr-dir> | pipelines <v> | probe <n> <v> | launch <attempt> <v> <exfab|short_gpu> <nodet-min> <det-min>}" ;;
esac
