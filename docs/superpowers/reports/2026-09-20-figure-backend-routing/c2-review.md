# C2 review — `_backends.py` and `_failures.py`, claim-by-claim

- **Date:** 2026-09-21
- **Worktree:** `.claude/worktrees/figure-backend-routing`, branch `fix/figure-backend-routing`
- **Cluster:** `af509476` (Chrome probe + shared `plotly.min.js`), `8acfc706` (durable
  failure record). Diff `be51a8e4..HEAD -- src tests`: 482 insertions, 0 deletions.
- **Subject:** spec `design.md` §2 and §3; plan Tasks 4 and 8.
- **Method:** no mutation testing. This cluster was already mutation-tested in depth by
  its implementer, with predictions stated before each run, and a second pass over the
  same surface would have reconfirmed the same three results. Instead: every docstring
  sentence, inline comment, `Returns:`/`Raises:` entry and commit-message claim in the
  two modules was enumerated and read against the code it describes, and nine runtime
  probes were run by the orchestrator to settle the claims that reading alone could not
  — including an 8-process race on both `ensure_plotlyjs_bundle` and
  `record_plot_failure`. Library behaviour was read out of the installed
  `.venv` (`kaleido` 1.2.0, `plotly` 6.6.0) rather than recalled.

---

## Verdict

**Safe to build C3–C6 on, with one fix first.**

Nothing in this cluster is wrong about what it *does* under the inputs it will actually
receive today. The concurrency design — the part most likely to be wrong and least
likely to be caught by unit tests — is correct, and was measured rather than argued: 8
processes racing a cold directory produced exactly one bundle write (one distinct
`mtime_ns`), and 8 processes × 25 appends produced 200 lines, none lost, none
interleaved, all parseable.

**D1 should land before C5 and C9, not after.** `record_plot_failure` silently discards
the entire record when any field is not JSON-serializable — a `Path`, a `numpy.int64`,
`bytes`. That is this module's stated purpose failing through its own safety net, it is
a two-word fix, and C5/C9 are precisely the commits that start feeding it
caller-supplied `dataset` / `image_stem` values from upstream. Fixing it after five more
commits means auditing five more call sites instead of one function.

**D4 should land before C4 adds the second caller.** The `reset_chrome_probe` fixture is
file-local to `test_backends.py`, and the memo is process-global. Measured: once
`chrome_available()` has run, monkeypatching `plotly.io.to_image` is inert. Task 7's own
planned tests call `preflight_plot_backends` without resetting, and Task 5 puts
`chrome_available()` on the publication path — so C4 and C5 will both leak a verdict
into whatever runs after them on the same xdist worker. There is no `pytest-randomly`
here, so this produces a *stable wrong answer* rather than a flake, which is worse.

The remaining findings are claims that overreach their code. None of them changes
behaviour today. They matter because this cluster's one real defect — the `__str__` hole
— was a claim that overreached its code, and the mutations could not find it because the
mutations and the guard came from the same mental model. The same shape is present again
in three places below (C1, C3, C5).

**Two corrections to the probe read-out**, since both change a recommendation:

- **P2 is not an unbounded hang in the general case.** kaleido 1.2.0 defaults
  `Kaleido.__init__(timeout=90)` (`kaleido/kaleido.py:106`) and `calc_fig` wraps the
  render in `asyncio.wait_for(..., self._timeout)` (`:412-417`). The render is bounded
  at 90 s. What is *not* bounded is everything before it: `_get_kaleido_tab` is a bare
  `await self.tabs_ready.get()` with no timeout (`:278`), and browser launch happens
  outside any kaleido timeout. So the correct statement is narrower and still
  substantive — see C1.
- **P3's "different roots" case is correct, not hazardous.** `/mnt/x/pages` →
  `/bigdata/y/plotly.min.js` giving `../../../bigdata/y/plotly.min.js` is the right
  lexical answer; `os.path.relpath` never touches the filesystem, so separate mount
  points are a non-issue on POSIX. Only *mixed* anchoring is a hazard — see C7.

---

## Claims that do not match the code

### C1 — `chrome_available` promises about raising, and the caller needs a promise about returning

`_backends.py:43` — *"``True`` if a PNG can be produced, ``False`` otherwise. Never
raises."*

"Never raises" is true for `Exception` and is the only guarantee offered. The guarantee a
caller on the CLI validation path needs is **"always returns"**, and the docstring
neither makes it nor disclaims it.

Measured: `plotly.io.to_image` exposes no `timeout` parameter, so `chrome_available`
cannot set one. Read from the installed kaleido 1.2.0: the render is bounded at 90 s by
default, but `Kaleido._get_kaleido_tab` (`kaleido/kaleido.py:278`) awaits a tab from an
unbounded queue, and the browser launch inside `calc_fig_sync` is outside that timeout.

**What makes this sharper than a generic "no timeout" note:** the spec chose the render
probe over a binary check *specifically* to catch the present-but-broken case —
`design.md:265`, *"a present-but-broken or sandboxed Chrome splits those"*. That case is
the one input for which this probe's cost is unmeasured and its termination unproven.
The measured 0.14 s is the `ChromeNotFoundError` path, which errors immediately; the
commit message is scrupulous about saying so. A machine where Chrome is installed but
cannot start costs up to 90 s if it fails cleanly, and blocks if it stalls before a tab
is ready — on the submitting process, before submission, per plan Task 7.

Not a defect today (no Chrome on this cluster, and the eager site does not exist yet).
It is a claim the plan's Step 6 gate should be reading, and it is not.

### C2 — "**Never raises.**" is false for `BaseException`, and the signature says so

`_failures.py:53`. Measured (P9): an `error` whose `__str__` raises `KeyboardInterrupt`
propagates straight out of `record_plot_failure`.

The parameter is typed `error: BaseException` — the author explicitly contemplated
non-`Exception` errors — while both guards (`_failures.py:33`, `:93`) catch `Exception`.

**Agreeing with the orchestrator's read and recording the reason:** do not widen the
catch. Swallowing `KeyboardInterrupt`/`SystemExit` is worse than the hole; a Ctrl-C
during `__str__`, or during a 30 s lock wait, must propagate. The correction belongs to
the sentence — "never raises an `Exception`" — which is both true and the guarantee the
`except` blocks in `_coordinator.py` actually need.

### C3 — "**Every** step is inside the one handler" — one step is outside it

`_failures.py:56-58`. The sentence is the load-bearing one in this module: it is the
generalisation drawn from the `__str__` defect, and it enumerates the steps considered
(directory, lock, open, write, entry build).

`from phenotypic.sdk_ import plot_failures_jsonl_path` at `_failures.py:72` is outside
the `try`. It is not in the enumeration.

It cannot fail in practice, and the reason is worth stating because it is not the reason
the docstring gives: `_failures.py:16` already imports
`phenotypic.sdk_._file_locking` at module level, which fully executes
`phenotypic/sdk_/__init__.py` — where `plot_failures_jsonl_path` is bound
(`sdk_/__init__.py:268`) — before this function can ever run. `phenotypic.sdk_` imports
nothing from `phenotypic.plotting`, so there is no cycle that could leave it partially
initialised. The line is a `sys.modules` lookup plus a `getattr`.

So: harmless, and the sentence is still inaccurate. This is listed not because the
import will fail but because **"every step" was the conclusion drawn from the one real
defect in this cluster, and it was stated one line too broadly on the very first
attempt.** That is the pattern worth noticing.

### C4 — `ensure_plotlyjs_bundle` says "once", and has no `Raises:`

`_backends.py:68,78`.

**"once" fails when the source is shorter than the floor.** Measured (P5) with
`get_plotlyjs` returning 8 bytes: three consecutive calls each took the lock, each
rewrote the file, and each returned a path that `_is_complete_bundle` classifies as
incomplete — with no error, no warning, and no convergence. Not reachable with plotly
6.6.0 (see "Checked and sound"), but the function's contract is "once" and its failure
mode is "forever".

**`Returns: Path to the bundle.`, with no `Raises:` section.** Measured (P9):
`ArtifactLockTimeout` escapes. So can `OSError` from `mkdir`/`write_text`/`os.replace`,
and `ImportError` from `plotly.offline`. The docstring's own framing —
*"Concurrent SLURM workers race to create it, so the write is locked"* — invites the
reader to believe the concurrent case is handled, and the concurrent case is the one
that produces `ArtifactLockTimeout`.

Contained at the one planned call site: plan `plan.md:1382` puts it inside
`except Exception` in `_render_page`. So this is a documentation gap, not a live
hazard — but it is a documentation gap that the next caller has no way to learn about
except by reading `_file_locking.py`.

### C5 — the `_MIN_BUNDLE_BYTES` rationale describes a failure this function cannot have

`_backends.py:21-24` — *"A bare ``is_file()`` check would accept the zero-byte remnant
of an interrupted write as a finished bundle."*

This function writes to `bundle.with_name(f".{bundle.name}.{uuid4}.tmp")` and publishes
with `os.replace` (`:92-95`). Because the temp file is a sibling, the rename is atomic
on the same filesystem, so **`ensure_plotlyjs_bundle` can never leave a zero-byte or
partial file at `bundle`.** An interruption leaves a `.tmp` or leaves the previous
complete file; never a truncated bundle.

The floor is still worth having — it guards writers that are *not* this function (a
truncated `cp`/`rsync` of a deliverables tree, a partially-synced GPFS copy, a user, and
plotly's own `"directory"` mode if it ever ran here). The comment attributes the danger
to the wrong author. Same shape as C3: a correct mechanism with a rationale that is
broader than what the code can actually do.

### C6 — "probing at most once" is per-process, not per-caller

`_backends.py:29`. `chrome_available` is not thread-safe: two threads can both pass
`if _CHROME is not None` and both run `pio.to_image`. Idempotent and harmless (worst
case, a duplicated probe), but the GUI hub is a threaded Dash server and
`_plot_refresh._coordinator` runs per request, so "at most once" is literally false
there. One clause, or a lock, closes it.

### C7 — `plotlyjs_src_for` has an unstated precondition

`_backends.py:106-120`, *"Returns: A relative POSIX path such as ``"../../plotly.min.js"``"*.

`os.path.relpath` is purely lexical and resolves **any relative argument against
`os.getcwd()`**. Measured (P3): `page_dir="out/plots/sym"` with an absolute bundle
produced a ten-level `../` chain anchored at the review worktree's cwd. The result is
correct only if both arguments are anchored the same way.

They are, at every planned call site: `directory = self._plots_base / safe_path_component(binding.id)`
(`_coordinator.py:335`) and `bundle = plotlyjs_bundle_path(plots_base)` both descend
from `PlotCoordinator._plots_base` (`_coordinator.py:71-75`), which is stored as given
and never `.resolve()`d — so they are consistently absolute or consistently relative.
**That is a property of the call sites, not of the function**, and C3/C5 are about to
add the callers. One `Args:` line stating the precondition costs nothing.

### C8 — `reset_chrome_probe` says "Tests only" and is exported

`_backends.py:62` says *"Tests only."*; `_pipeline/__init__.py:37,92` puts it in the
package `__all__`. `_pipeline` is private, so this is cosmetic — noted only because the
`__all__` entry is what a future reader will take as the sanctioned surface.

### C9 — spec §3 still specifies `plot_failures_jsonl_path(output_dir)`

`design.md:378-380`. Implemented as `plot_failures_jsonl_path(plots_base)`.

**The implementation is right and the spec is the stale document.** Verified: the GUI
passes `plots_base=layout.plots_dir` (`_gui/_plot_refresh.py:132`), and
`BundleLayout.plots_dir` is `deliverables_base / "plots"` with no `deliverables/`
segment for a portable bundle — pinned by
`tests/unit/sdk_/test_io_constants.py:200`. An `output_dir`-keyed helper could not serve
that call site. The plan records this as S6 (`plan.md:938-943`) and the commit message
restates it. Only the spec was not updated.

One consequence nobody has written down: a GUI-driven refresh and a CLI run of the same
pipeline write their `.failures.jsonl` to **different directories**. Correct per S6;
worth one line in C9's wiring so nobody looks for a run's failures in only one place.

### C10 — spec §2 says "the expected size"; the code uses a floor

`design.md:220` — *"skipping it when already present with the expected size"*, against a
spec that states the exact size (4,847,452 chars) one paragraph earlier.

The floor is the better choice and should be recorded as a deliberate deviation rather
than left to look like an approximation: an exact-size check against a hard-coded
constant would rewrite the bundle on **every call forever** after any plotly upgrade
that changes the bundle length. The floor trades that for C4's failure mode, which
requires plotly to ship a bundle 4.85× smaller.

### C11 — spec §2 says "the same `exclusive_path_lock` used for publication"

`design.md:222`. Implemented with a *distinct* lock file, `plots_base/.plotlyjs.lock`,
not publication's `directory/.publication.lock` (`_writer.py:84`).

**The implementation's reading — same helper, different lock — is the only workable
one.** The publication lock is per *page directory*; the bundle is one shared file above
all of them, so the publication lock could not serialise it. The spec sentence is
imprecise; no change needed to the code.

---

## Defects

### D1 — a non-`str` field silently discards the entire record  *(fix before C5/C9)*

`_failures.py:88`, `json.dumps(entry, sort_keys=True)`.

Measured (P4), each into a fresh directory:

| field | result |
|---|---|
| `dataset=Path("plate_a")` | **record silently lost — no file written** |
| `dataset=np.str_("plate_a")` | recorded |
| `image_stem=np.int64(3)` | **record silently lost — no file written** |
| `binding_id=b"sym"` | **record silently lost — no file written** |

`json.dumps` raises `TypeError` *inside* the handler, so `record_plot_failure` returns
normally, the caller's soft failure stays soft, and the entry that was the whole point
of the call does not exist. The module docstring's claim — *"carries the reason on disk
rather than only in a log nobody reads"* — is false for these inputs, and it fails
through the safety net installed to prevent exactly that.

This is the same shape as the `__str__` defect, one layer further out: the boundary was
widened to cover *building the entry*, and serialising the entry was left outside the
set of things that can fail. Every existing test passes a literal `str`.

**Why it will not stay hypothetical.** The fields are typed `str` but arrive from
callers, and mypy cannot help once a value comes from upstream — C5/C9 thread `dataset`
and `image_stem` through from the coordinator. `np.str_` passing while `np.int64` does
not is the kind of near-miss that survives review: a stem that happens to be numeric
(`"01"` read as a number somewhere upstream) is exactly the case that fails.

**Fix:** `json.dumps(entry, sort_keys=True, default=str)`. Two words, no behaviour
change for correct input, and it converts a lost record into a slightly-coerced one.
A regression test wants a `Path` and a `np.int64`, asserting the record **exists** —
not merely that the call returned, which is the assertion that let this through.

### D2 — the recorder's own failure path is a debug log

`_failures.py:93-96`. When recording fails for any reason — D1, an unwritable target,
`ArtifactLockTimeout` — the fallback is `logger.debug(...)`. The module docstring names
*"a log nobody reads"* as the thing this module exists to replace, and its failure mode
is a log strictly less visible than the `logger.warning` the swallowing call sites
already emit.

**Measured counter-evidence, which is why this is ranked below D1:** P8 ran 8 processes
× 25 appends against one lock and lost nothing — 200 expected, 200 actual, 0
unparseable. The 30 s `exclusive_path_lock` timeout is not a practical bottleneck at
that scale. The concern is that the loss is *unannounced* at any scale, not that loss is
likely. `logger.warning` for the failure of a durability mechanism would cost nothing.

### D3 — `lifecycle` is a documented closed set typed `str`

`_failures.py:47,67`. Four values are documented in prose (`"image"`,
`"measurements"`, `"analysis"`, `"qc"`) and nothing enforces them. C9 wires six call
sites (five `except` blocks in `_coordinator.py` plus `_writer.py:140`, per
`design.md:375-376`); a typo in one writes a garbage lifecycle into the durable record
and no test, type check or reader notices.

`lifecycle: Literal["image", "measurements", "analysis", "qc"]` makes mypy check all six
at the moment they are written. The repo's own convention (root `CLAUDE.md`, "Closed
Value Sets & Operation Parameters") points the same way; this is a private helper rather
than an operation parameter, so the rule does not strictly bind — but the cost here is
one annotation and the payoff is six call sites checked for free.

### D4 — the `_CHROME` memo leaks out of `test_backends.py`  *(fix before C4/C5)*

`_backends.py:19`. The memo is module-global; the autouse `reset_chrome_probe` fixture
is **file-local to `tests/unit/plotting/test_backends.py:16-20`**. There is no
`tests/unit/plotting/conftest.py`, and `reset_chrome_probe` appears nowhere else in the
tree.

Measured (P6), in one process:

```
real probe (cold)          -> False   (0.10 s)
after patching to succeed  -> False   <-- stale memo, the patch had no effect
after reset_chrome_probe() -> True
```

`test_backends.py` is protected in both directions by its own fixture. The leak is
outward, and C4/C5 are the commits that create it:

- Plan Task 7's tests call `preflight_plot_backends` **without** patching
  (`plan.md:1720`, `:1745`), so they will run the real probe and leave `_CHROME` set.
- Plan Task 5 calls `chrome_available()` inside `_render_page` (`plan.md:1398`), so
  every publication test sets it too.

A later test that monkeypatches `plotly.io.to_image` and expects `True` will get the
stale `False`. `pyproject.toml` configures no `pytest-randomly`, so file order within an
xdist worker is deterministic — this yields a **stable wrong result** that looks like a
real failure until someone renames a file or rebalances the shards, at which point it
moves. That is harder to diagnose than a flake.

**Fix:** move the autouse fixture into `tests/unit/plotting/conftest.py` now, and add
one wherever Task 7's tests land. A session-scoped autouse in `tests/conftest.py` would
cover all of it in one place.

---

## Simplifications available

- **Two lazy imports that defer nothing.** `_backends.py:80` and `_failures.py:72` defer
  `from phenotypic.sdk_ import ...`, but `_backends.py:14` and `_failures.py:16` already
  import `phenotypic.sdk_._file_locking` at module level, which fully initialises
  `phenotypic.sdk_` at import time. Both could be module-level imports at zero cost —
  and in `_failures.py` that also closes C3 mechanically rather than by rewording.
  (The *plotly* lazy imports at `_backends.py:50-51,90` are load-bearing and must stay:
  `plotly` is in both `HEAVY_STARTUP_MODULES` and `DEFERRED_RUNTIME_MODULES`.)
- **`_format_error`'s first guard is unreachable.** `_failures.py:31-34` wraps
  `type(error).__name__`. `error` is a `BaseException`, so `type(error)` is a class and
  `__name__` is a plain attribute lookup; reaching the `except` needs a metaclass with a
  raising `__name__` descriptor. No test covers it, and none can without constructing
  that metaclass. Four lines defending against nothing, inside the function whose point
  is that it defends against something real.

---

## Checked and sound

- **Concurrency, measured not argued.** 8 spawned processes released against a common
  wall-clock deadline on a cold directory: 0 exceptions, 1 distinct path, 1 distinct
  size (4,847,499), **1 distinct `mtime_ns`** — written exactly once. Directory
  afterwards: `['.plotlyjs.lock', 'plotly.min.js']`, no orphan `.tmp`.
- **The double-check interleaving is correct.** The loser acquires the lock, re-runs
  `_is_complete_bundle`, and returns from inside the `with`; `exclusive_path_lock`'s
  `finally: _release(handle)` still runs. Because the temp file is a sibling of the
  bundle, `os.replace` is atomic, so the *unlocked* fast-path check at `:83` can only
  ever see "absent" or "complete" — never a partial file.
- **Append integrity under contention.** 8 processes × 25 appends: 200 expected, 200
  actual, 0 missing, 0 unparseable, 200 distinct ids, trailing newline intact.
- **`_MIN_BUNDLE_BYTES` compares the right quantity.** plotly 6.6.0's
  `get_plotlyjs()` is 4,847,452 chars — matching the spec to the character — but
  **4,847,499 bytes in UTF-8, and not pure ASCII.** `_is_complete_bundle` compares
  `st_size`, i.e. bytes, which is the correct side of that 47-byte gap. Nobody should
  later "fix" it to compare against `len(get_plotlyjs())`. Headroom over the floor:
  4.85×. Forward note for C5: because the bundle is not ASCII, the page that references
  it must declare UTF-8 — plotly's own `write_html` full-page output does.
- **The helpers are actually called.** `ensure_plotlyjs_bundle` uses
  `plotlyjs_bundle_path(plots_base)` and `record_plot_failure` uses
  `plot_failures_jsonl_path(plots_base)` — a deviation from the plan's own draft code
  (`plan.md:1051`, `:2023`, both of which hand-joined the constant) toward the plan's own
  Step 3 requirement (`plan.md:1988`). Correct on both counts.
- **`_is_complete_bundle` is simpler than the plan specified** — it dedupes a condition
  the plan wrote out twice.
- **Startup-import discipline.** Neither module imports `plotly` at module level;
  `_pipeline/__init__.py` already imported `_writer` and `_coordinator`, so the two new
  eager submodule imports add no startup weight.
- **`_format_error` is total against `Exception`, and the entry build is inside the try
  for an independent reason** — the commit message's distinction between those two facts
  is correct and worth preserving if either is ever refactored.
- **`except Exception` instead of the spec's `RuntimeError`/`ChromeNotFoundError`
  (`design.md:227`) is the safe direction** — an `ImportError` from a broken plotly must
  also mean "cannot rasterise".
- **`# noqa: BLE001` is house style** — 177 occurrences under `src/`. Ruff here selects
  only the default `E4/E7/E9/F`, so neither BLE001 nor E501 is enforced and no CI
  workflow runs ruff at all; the three lines over the 79-char limit are not a gate.
- **No test or `src` code enumerates `plots/` exhaustively.** The only globs are
  `*.png` in `tests/unit/plotting/test_coordinator.py:98,141,143`, which the new
  `plotly.min.js`, `.failures.jsonl` and two `.lock` files do not disturb. (Lock files
  persist after use, consistent with the existing `.publication.lock`.)
- **`mypy src/phenotypic/plotting/_pipeline/_backends.py _failures.py`** — clean.
- **`__all__` ordering** in `_pipeline/__init__.py` is alphabetical as the plan asked.

---

## Forward risks for C3–C6

Not defects in C2 — things C2's shape makes easy to get wrong next, listed because this
is the gate for them.

1. **`base = plots_base if plots_base is not None else directory`** (`plan.md:1439`).
   Any caller of `publish_plot_output` that omits `plots_base` writes the 4.8 MB bundle
   **into the page directory** — silently reintroducing the exact per-directory
   duplication the whole design exists to prevent. `ensure_plotlyjs_bundle` has no way
   to detect it; the resulting `src` is even correct. Worth a test in C5 that a
   multi-page image plot produces exactly one `plotly.min.js` in the whole tree.
2. **Lock acquisition order.** C5 calls `ensure_plotlyjs_bundle` (→ `.plotlyjs.lock`)
   and C9 calls `record_plot_failure` (→ `.failures.lock`) from inside
   `_publish_plot_output_locked`, which already holds `directory/.publication.lock`.
   That is publication → {plotlyjs, failures}, with no reverse edge, so there is no
   cycle. Keep it that way: never take `.publication.lock` while holding either child.
   Separately, `flock` is per open-file-description, so a `record_plot_failure` called
   from inside a block already holding `.failures.lock` in the same process would block
   for the full 30 s and then lose the record.
3. **`chrome_available()` at `plan.md:1398` sits outside any `try`** in `_render_page` —
   the one place in that function not covered. Given C1, that is where a stalled browser
   would surface.
4. **The module docstring's isolation claim is about to be tested.** *"Kept apart from
   ``_writer`` so the capability probe can be imported by CLI validation without dragging
   in publication"* is true of `_backends.py` today, and true of Task 7's planned
   `from phenotypic.plotting._pipeline._backends import ...` (`plan.md:1720`). It is
   **not** true of `from phenotypic.plotting._pipeline import chrome_available`, which
   C2 just made available and which pulls in `_writer`, `_coordinator`, `_bindings` and
   `_analysis_registry`. Task 7 also plans to put `preflight_plot_backends` into this
   same module, so whatever *that* imports at module level becomes CLI-validation
   startup cost.

---

## Unverified — stated plainly

- **No mutation testing was run by me.** Per the scoping, the implementer's three
  mutations stand as the evidence for the handler boundary; I neither reproduced nor
  extended them. Every finding above rests on reading plus the nine probes, not on a
  killed mutant.
- **The Chrome-present success path remains unmeasured**, exactly as the spec and the
  commit message say. C1 is therefore a reasoned bound (read out of kaleido 1.2.0's
  source), not a measurement. Nobody has timed `chrome_available()` on a machine with a
  working Chrome, or with a broken one.
- **`os.path.relpath` on Windows raises `ValueError` for paths on different drives.**
  Reasoned from the documented behaviour, not run — this cluster was probed on Linux
  only. The project claims cross-platform support; whether a Windows user can produce a
  cross-drive `plots_base`/`page_dir` pair is a C5 question.
- **The concurrency probes were 8 processes on one node**, not a multi-node SLURM array.
  `exclusive_path_lock` uses `fcntl.flock`, whose behaviour across GPFS from multiple
  nodes is not established by P7/P8. This risk is *inherited*, not new — publication's
  `.publication.lock` has the same dependency — but "verified concurrent" here means
  single-node.
- **The `_gui` threaded path (C6) was read, not exercised.** No test drives
  `chrome_available` from two threads.
- **I did not audit C3–C6's code**, which does not exist, nor re-verify C1's findings.
  The "forward risks" section is read off the plan, not off an implementation.
