# GUI hub

Run the unified PhenoTypic web interface, including Browse, the pipeline builder,
results viewer, and run console, under a single URL from any machine, including HPC clusters
reached over SSH.

## What the hub is

The hub is one Dash application that mounts its tools under sub-paths of a
single server:

| Path | Tool |
|------|------|
| `/` | Home: sandbox tree, capability badges, tab navigation |
| `/browse/` | Browse source images as deep-zoom single, timeline, or compare views |
| `/builder/` | Pipeline builder (node-graph editor) |
| `/results/` | Results viewer (OpenSeadragon overlays, measurement tables) |
| `/run/` | Run console (submit local or SLURM jobs, tail logs, view dashboard) |

All pages share a top-bar tab strip and a sidebar showing the sandbox
directory tree. The server is single-user and frozen-at-launch: the file
browser can only see the directory you pass via `--root`.

## Launching the hub

Two equivalent entry points boot the same server:

```bash
uv run phenotypic-gui --root ./images --port 8050
uv run python -m phenotypic.gui --root ./images --port 8050
```

**Options:**

| Flag | Default | Effect |
|------|---------|--------|
| `--root PATH` | current working directory | Freezes the sandbox the file browser is allowed to see |
| `--host ADDR` | `127.0.0.1` | Interface to bind; loopback-only by default |
| `--port N` | `8050` | TCP port |
| `--url-prefix PREFIX` | `/` | Browser-visible path prefix for path-stripping proxies |
| `--debug` | off | Enable Dash auto-reload and verbose tracebacks |

The startup banner printed to stdout repeats the URL and the SSH-tunnel
command for the bound port.

### Optional libvips acceleration

Browse converts source images into Deep Zoom Image (DZI) pyramids as they are
opened. On macOS and Windows, the GUI extra includes the official
[`pyvips[binary]`](https://pypi.org/project/pyvips/) distribution with a
self-contained native libvips. The normal installation needs no separate
Homebrew package or loader-path configuration.

Linux and HPC installations keep the smaller `pyvips` binding. Install or load
native [libvips](https://www.libvips.org/install.html) when the faster DZI path
is wanted:

```bash
# Debian or Ubuntu
sudo apt install libvips-dev --no-install-recommends
```

On an HPC system, load the site's libvips module before launching the GUI if one
is provided.

Confirm that the native library is visible to the project environment:

```bash
uv run python -c "import pyvips; print('pyvips', pyvips.__version__, 'libvips', '.'.join(str(pyvips.version(i)) for i in range(3)))"
```

The bundled build includes common image loaders but omits optional PDF and
OpenSlide support. Users who intentionally choose a fuller Homebrew libvips can
install it with `brew install vips`. If `vips --version` then succeeds but the
command above reports that `libvips.42.dylib` cannot be found, run the GUI from
a shell with Homebrew's library directory exposed:

```bash
export DYLD_FALLBACK_LIBRARY_PATH="$(brew --prefix vips)/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"
```

PhenoTypic automatically uses its Pillow tiler when libvips cannot be loaded.
This preserves Browse and Results functionality, but preparation of large DZI
pyramids can be slower and use more memory.

### Browse preparation and cache

Browse identifies assets by the source file's path, size, timestamps, render
schema, and DZI parameters. It shows a progressive preview while the selected
revision is prepared, then opens the completed DZI in the existing viewer.
Navigation work is prioritized ahead of one bounded speculative worker, so a
dataset preparation job cannot monopolize conversion while the user is moving
between images.

Prepared revisions persist across restarts. Storage is selected in this order:

1. `<sandbox>/.phenotypic-gui/browse_cache`;
2. the platform user cache, namespaced by a hash of the sandbox root;
3. a temporary session cache when neither persistent location is writable.

The cache begins least-recently-used pruning above 10 GiB and stops at 8 GiB.
The Browse preparation details panel reports the selected backend and cache
usage. Its **Prepare dataset**, **Stop**, and **Clear prepared images** actions
remain lower priority than foreground navigation and protect the displayed or
in-flight revision from deletion.

### Open OnDemand path prefix

Open OnDemand proxies apps through a browser path such as
`/node/<node>/<port>/` and forwards stripped paths to the local server. Pass
that path, not the full URL, as `--url-prefix`:

```bash
uv run phenotypic-gui \
  --root /rhome/ejaco020 \
  --host 0.0.0.0 \
  --port 30099 \
  --url-prefix /node/hz01/30099/
```

Then open the full proxy URL in your browser:

```text
https://ondemand.hpcc.ucr.edu/node/hz01/30099/
```

The app does not infer node names, ports, or OOD domains. The default prefix is
`/`, so local launches keep the usual `/builder/`, `/results/`, and `/run/`
paths.

:::{warning}
**`phenotypic gui` (without a hyphen) is not supported.**

The existing `phenotypic` CLI is reserved for batch pipeline execution with
explicit path options, not subcommands. Always use the hyphenated form
`phenotypic-gui` or the module form `python -m phenotypic.gui`. Typing
`uv run phenotypic gui` will fail because `gui` is an unexpected positional
argument.
:::

## The hub pages

### Home (`/`)

The landing page. The sidebar shows the sandbox directory tree; each entry
carries zero or more capability badges:

| Badge | Meaning |
|-------|---------|
| `img` | Directory contains image files |
| `cfg` | File is a PhenoTypic pipeline JSON |
| `out` | Directory is a CLI output root (`deliverables/master_measurements.parquet` + `results/`) |
| `?` | Directory could not be listed due to a permission error |

Hidden files (names starting with `.`) and symlinks whose targets fall outside
the sandbox root are hidden by default. Two toggles in the sidebar header
surface them. The Refresh button flushes the classifier's LRU cache and
re-classifies all visible nodes — use it after dropping new files into a
directory.

### Browse (`/browse/`)

Browse source-image directories in a deep-zoom single view, acquisition-time
timeline, or synchronized compare view. Image access remains constrained to the
hub sandbox root.

### Builder (`/builder/`)

The existing node-graph pipeline builder, mounted under the hub prefix. Use it
to compose `ImagePipeline` objects, set per-operation parameters, and export
`pipeline.json`. The builder's file picker is scoped to the same sandbox root
as the rest of the hub.

### Results viewer (`/results/`)

The existing results viewer, accessible from the hub without a separate launch.
The viewer starts with an empty-state landing page; click a directory with the
`out` badge in the sidebar to load it.

**Memory note.** The viewer loads the master measurements parquet and per-image
overlays into memory on first access. Navigating away and releasing the viewer
frees the Python object graph, but process RSS may not return to the OS — the
CPython allocator pools freed pages rather than returning them immediately.
Subsequent access re-loads the data from disk.

### Run console (`/run/`)

Pick a pipeline JSON, an input directory, and an output directory from the
sidebar or the inline file pickers. Toggle **Local** or **SLURM** mode, then
click **Validate** (dry-run) or **Run**.

**Validate** runs the CLI with `--dry-run`: it validates the pipeline and lists
the images that would be processed without writing any output. The log tail
shows the result.

**Run (Local)** spawns `python -m phenotypic ...` as a subprocess of the GUI
process. See [CLI Batch Processing](../../tutorials/pages/cli_batch_processing.md) for
the full list of CLI flags that the form exposes. Only one local run can be active at a time;
the Run button is disabled while a local subprocess is live. SLURM submissions
have no equivalent cap.

**Run (SLURM)** shells out to the CLI's SLURM path. See
[SLURM Pipelines](slurm_pipelines.md). The submission runs on a background
thread so the UI stays responsive. Once `progress/job_metadata.json` is
written, the array primary job ID surfaces as `slurm-<id>` in the Recent Runs
panel.

The **Recent Runs** panel lists runs from the current session and historical
runs rehydrated from the sandbox at boot time. Click a row to re-point the
iframe at that run's dashboard.

## Log tail and iframe dashboard

Once a local run starts, stdout (and stderr, which is merged into stdout) is
teed to `<output_dir>/.gui_log/stdout.log` and to a 5000-line in-memory ring
buffer. The log-tail panel polls the ring buffer on a short interval.

Once `dashboard.html` lands in the output directory's `deliverables/`
subfolder, the iframe panel points at
`/runs/<rel>/deliverables/dashboard.html`. The `/runs/` route is registered on
the hub's Flask server directly — not under the dispatcher middleware — so the
URL works regardless of which tab is active. The iframed dashboard polls its own
progress files using relative URLs, which the `/runs/` route resolves inside the
sandbox. Local dashboards show progress without a tab bar; SLURM dashboards
retain Progress and Download tabs. Result exploration lives in the Results
Viewer and `/analysis/` app.

**Cancel** sends SIGTERM to the subprocess and waits up to 10 seconds. If the
subprocess does not exit within that window, SIGKILL is sent. The GUI also
registers an `atexit` hook so closing the hub with Ctrl-C propagates a SIGTERM
to any live subprocesses before the process exits.

## SSH-tunnel pattern

When the hub runs on an HPC cluster, forward the port to your workstation in a
separate terminal before opening the browser:

```bash
ssh -L 8050:localhost:8050 <user>@<cluster>
```

Then open `http://localhost:8050/` locally. The default `--host 127.0.0.1`
binding means the server only accepts connections from `localhost`, so the
tunnel is the only path in from outside the cluster.

(gui-hub-slurm-tunnel)=
### Running the GUI on a Slurm compute node

If you launch the GUI from inside a `srun` / `salloc` allocation, the server is
bound to `localhost` on the **compute node**, not on the login node you SSH'd
into. A single-hop tunnel to the login node will return
`channel N: open failed: connect failed: Connection refused` because nothing is
listening on the login node's port. You need a two-hop tunnel.

Pick the compute node hostname (visible in your shell prompt after the
allocation lands, e.g. `<user>@gpu12:~$`) and use one of the following:

**Option 1 — `ProxyJump` from your workstation (simplest, single command):**

```bash
ssh -J <user>@<cluster> -L 8050:localhost:8050 <user>@<compute-node>
```

This requires that the login node can SSH to the compute node without an
extra password; that is the default on most Slurm clusters while you have an
active allocation.

**Option 2 — Nested SSH (works even if direct SSH to the compute node is
blocked, as long as the login node permits forwarding):**

```bash
ssh -L 8050:localhost:8050 <user>@<cluster> \
    ssh -L 8050:localhost:8050 -N <compute-node>
```

**Option 3 — Two terminals (useful when you want to keep the allocation shell
interactive):**

1. Terminal A — start the GUI on the compute node:

   ```bash
   ssh <user>@<cluster>
   srun -A <account> -p <partition> -t 4:00:00 --pty -c 8 --mem=32g bash -l
   uv run python -m phenotypic.gui --root <project-dir> --port 8050
   ```

2. Terminal B — open the two-hop tunnel from your workstation:

   ```bash
   ssh -J <user>@<cluster> -L 8050:localhost:8050 -N <user>@<compute-node>
   ```

In all three options, open `http://localhost:8050/` on your workstation once
the GUI logs `Dash is running on http://127.0.0.1:8050/`.

To confirm the diagnosis if a tunnel keeps failing: from the **login node**,
run `curl http://localhost:8050/`. It will fail (the GUI is on the compute
node). From the **compute node**, the same `curl` succeeds. That mismatch is
the symptom that you need a two-hop tunnel rather than a single-hop one.

## Cloud deployment

Cloud deployment (multi-user, per-session sandbox roots, authentication) is a
non-goal in v1. The hub is single-user with no auth gate. A
`TODO(cloud-deploy)` in `src/phenotypic/gui/shell/_sandbox.py` marks the
hook point for a future auth layer.

Do not expose the hub on `0.0.0.0` without authentication — there is nothing
stopping another user from reading files or submitting jobs through the
unguarded API.

## Standalone tools

Each component can be launched independently for debugging:

```bash
# Pipeline builder only
uv run python -m phenotypic.gui.builder --image-root ./images

# Results viewer only
uv run python -m phenotypic.gui.results_viewer --output-root ./out

# Run console only
uv run python -m phenotypic.gui.run_console --root ./images
```

All three accept `--host`, `--port`, and `--debug` with the same defaults as
the hub launcher.
