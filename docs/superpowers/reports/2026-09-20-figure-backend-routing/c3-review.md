# C3 gate review — `_render_page`, `save_html`, manifest `schema_version: 2`

Reviewed at `5dbdf503`. Cluster commits `e6f58f16`, `8a08bb67`, plus `0e06cb6a`
and `5dbdf503` (the two fixes this gate produced).

**Diff range correction — the lead's range was incorrect.** The brief gave
`ce5b9e8c..HEAD` as the cluster diff. That range **excludes `e6f58f16`**, the
commit that introduces `_render_page` — the central change of the cluster.
`git merge-base --is-ancestor e6f58f16 ce5b9e8c` returns true: `e6f58f16` is
behind `ce5b9e8c`, not after it, so `git log --oneline ce5b9e8c..HEAD` returns
a single commit and a 3-file / +153 diffstat.

The brief named both hashes correctly and then handed over a range omitting one
of them. This is the dangerous shape: reviewing that range produces a complete,
confident, internally consistent review of the *wrong* half, and nothing in the
output looks wrong. The range and the hashes disagreed, and only the hashes were
right.

The cluster diff is **`e04331d4..HEAD`**: 6 files, +518 / −33.

---

## Verdict

**Safe to build C4–C6 on, with one contract correction that C5 must make and one
spec promise that C5 cannot keep without a decision from you.**

`_render_page` is the right extraction. Its signature carries every argument the
flat image path has available, it writes no manifest and reads no manifest, and
it can be called from `_publish_image_value` exactly as plan Task 9 Step 6
describes. The `renderers` chain is correct over all eight reachable
backend/Chrome combinations. Nothing in the tree still reads the removed
per-page `"file"` key.

Two things qualify the verdict:

1. **C5 must not apply Task 9 Step 6's error-recording snippet literally.** The
   plan was written against `errors: list[str]`; C3 deliberately changed it to
   `list[BaseException]`. The snippet's `RuntimeError(message)` wrapper would
   write the wrong exception class into `.failures.jsonl` — the same defect
   class `e6f58f16`'s commit message boasts of removing, reintroduced one caller
   over. Details in **The C5 contract**.

2. **Spec §3's "a PNG-less plot directory explains itself" will not hold for
   image plots after C5**, because the flat path writes no manifest and
   `_render_page` emits no error when Chrome is simply absent. This is not a
   signature problem and C5 cannot fix it by calling `_render_page` differently.
   It needs a decision. Details below.

Two defects were found during this gate and fixed before the report. The live
one: neither production caller passed `plots_base`, so `plotly.min.js` was
written per page directory — **7.45 GB extrapolated to a 1,536-image plate**,
measured. Fixed in `0e06cb6a` with a whole-tree count guard. The documentation
one: `record_plot_failure`'s `lifecycle` closed set excluded a value the writer
passes, fixed in `5dbdf503`, which also surfaced a second out-of-set value
(`"qc dependency"`) that becomes live the moment C5 wires the recorder in.

---

## The C5 contract

### Does the signature work?

**Yes.** `_render_page(figure, directory, stem, *, plots_base, plot_id,
publication_guard, commit_guard)`. At `_coordinator.py:345` every one of the six
is in scope: `output.pages[0].figure`, `base`, `output_stem`,
`self._plots_base`, `binding.id`, `self._publication_guard` /
`self._commit_guard`. Nothing has to be synthesised or faked.

It is also correct on the two things that would have made it awkward:

- **It does not require the directory to pre-exist.** `_atomic_write` writes its
  temporary as a sibling of the destination and never calls `mkdir`, but both
  `FigureAdapter.save_html` and `FigureAdapter.save_png`
  (`_adapter.py:33`) do `path.parent.mkdir(parents=True, exist_ok=True)` on the
  path they are handed — which is the temporary, whose parent is `directory`.
  Plan Step 6's `base.mkdir(...)` is belt-and-braces, not load-bearing. Worth
  saying because the docstring does not state the precondition either way, so
  removing either adapter's `mkdir` later would break the flat path silently.
- **The bundle src is computed from `directory`, not from a hard-coded depth.**
  `plotlyjs_src_for(directory, bundle)` gives `../../../plotly.min.js` for
  `plots/<id>/<dataset>/`, which is one of the three layouts spec §2 enumerates.

### What C5 does with `errors` — **this is the correction**

Plan Task 9 Step 6 says:

```python
for message in errors:
    record_plot_failure(..., error=RuntimeError(message), ...)
```

That was written when plan Task 5 declared `-> tuple[dict[str, str], list[str],
str | None]`. `e6f58f16` changed the second element to `list[BaseException]`,
for a good reason it states in the commit message. **The plan snippet was not
updated, and the two no longer compose.**

Applied literally, `RuntimeError(exc)` wraps the real exception. `_format_error`
renders `f"{type(error).__name__}: {error}"`, and `str(RuntimeError(exc))` is
`str(exc)`, so the record reads:

| real exception from `_render_page` | recorded, if C5 wraps |
|---|---|
| `RuntimeError("raster exploded")` | `"RuntimeError: raster exploded"` — accidentally right |
| `TypeError("unsupported figure type ...")` | `"RuntimeError: unsupported figure type ..."` — **wrong class** |

The unsupported-figure case is not hypothetical: it is the one error
`_render_page` manufactures itself (`_writer.py:174`), and it is the case where
the class name is the whole diagnostic. That the RuntimeError case comes out
right by coincidence is what makes this survive a casual test — exactly the
shape of the doubled-prefix bug `e6f58f16` removed.

**C5 must pass `error=exc` directly.** Same for the raise at the end of the
step: `f"...: {errors[0] if errors else 'no renderer'}"` interpolates a
caller's exception into an f-string outside any handler, which is the hazard
`_format_error` exists to contain and which `_writer.py`'s own module-level
import comment calls out by name. Use `_format_error(errors[0])`.

### What C5 does with `backend` — it is worth more than the plan thinks

Plan Step 6 discards it (`_backend`), and for the plan's own purposes that costs
nothing: `backend is None` implies `files == {}`, which the plan's
`if not files: raise` already covers. So the discard is safe.

But `backend` is **the piece that makes the silent-no-PNG case detectable**, and
C5 is the only place that can detect it. Given the two guard conditions in
`_render_page` — HTML attempted iff `backend == "plotly"`, PNG attempted iff
`backend == "mpl" or chrome_available()` — the return triple is already
sufficient:

```python
backend == "plotly" and files == {"html"} and not errors   # ⟺ Chrome was absent
```

That is exact, not heuristic. If `files == {"html"}` then the HTML write
succeeded, so any exception in `errors` could only have come from the PNG
attempt; `errors` being empty therefore means the PNG block never ran, and for a
Plotly figure the only way that happens is `chrome_available()` returning False.
Every neighbouring case is separated cleanly:

| `backend` | `files` | `errors` | means |
|---|---|---|---|
| `"plotly"` | `{html}` | `[]` | **Chrome absent — PNG never attempted, nothing recorded** |
| `"plotly"` | `{html}` | `[exc]` | PNG attempted and failed — recorded |
| `"plotly"` | `{html, png}` | `[]` | complete |
| `"mpl"` | `{png}` | `[]` | complete — matplotlib has no HTML form |
| any | `{}` | any | total failure — plan's `raise` fires |

**This matters for the decision below.** It means option (b) — have C5 record
the capability verdict on the flat path — needs **no change to `_render_page`'s
signature and no fourth return element**. C5 can derive the verdict from what it
is already handed. That makes (b) substantially cheaper than it looks, and it is
the reason the contract as shipped is adequate even though the gap is real.

### Is anything in `_render_page` assuming a manifest will follow?

Structurally, no. It never touches `manifest.json`, and `files` / `errors` /
`backend` are returned rather than written. Three couplings are worth naming,
one of which is serious.

**1. Serious — the flat path has nowhere to put the capability verdict.**

`renderers` and `failed` are built in `_publish_plot_output_locked`, *after* the
page loop, not in `_render_page`. The flat image path writes no manifest at all,
before or after C5. So spec §3's central promise —

> `"renderers"` records the capability verdict **per published directory**, which
> is what makes a PNG-less plot directory self-explaining rather than merely empty.

— is not delivered for image plots, which is every plot whose `inspect` returns
a bare figure, i.e. the case B1 exists for and the case Task 9 Step 6 calls "the
commonest image plot".

It is worse than "no manifest", because the missing PNG leaves **no trace
anywhere**. Read `_writer.py:167`:

```python
if backend == "mpl" or chrome_available():
```

When Chrome is absent and the backend is `plotly`, the PNG block is *not
attempted*. No exception is raised, so `errors` is empty, so C5's
`record_plot_failure` loop records nothing and its `if not files` raise does not
fire. After C5, `plots/<id>/<dataset>/` on a Chrome-less machine contains
`<stem>-<hash>.html` files, no PNGs, an empty `.failures.jsonl`, and no manifest.
Nothing on disk says why. The writer path is self-explaining precisely because
`renderers["png"] = "unavailable: chrome not found"` covers this; the flat path
has no equivalent.

`.failures.jsonl` does not close this. It records *failures*, and an
unattempted renderer is not a failure.

**This is not fixable by changing `_render_page`'s signature**, so it is not a
C3 defect — but it is the thing that will bite, and it needs a decision before
C5 writes the step:

- **(a)** the flat path writes a small sidecar manifest per dataset directory;
- **(b)** C5 derives the verdict from the triple it is already handed — see
  **What C5 does with `backend`** above; this needs no signature change; or
- **(c)** you accept that spec §2's once-per-process announcement is the only
  record for image plots, and narrow §3's "The manifest carries the same fact
  durably" so it does not claim more than the code does.

(c) is legitimate and cheapest, but it is a spec change, so it is yours. (b) is
cheaper than it first appeared, now that the triple turns out to determine the
verdict exactly.

**2. Minor — the filenames written by the flat path are recorded nowhere.**

`files` is `{"html": ..., "png": ...}` — manifest vocabulary. C5 discards it.
Today the flat path's output is discoverable by convention
(`<stem>-<hash>.png`); after C5 it is `<stem>-<hash>.{html,png}` with *which*
present depending on backend and Chrome. No consumer reads these files today
(verified — see **Checked and sound**), so this costs nothing now. It is the
reason (a) or (b) above would be cheap if you ever want one.

**3. Minor — `PlotPublicationBlocked` means something different on the flat path.**

`_render_page`'s docstring says blocked "means the output snapshot changed and
this whole publication is void", and the writer honours that by propagating out
of `publish_plot_output`. On the flat path, `_publish_image_value` is called
inside `emit_image`'s `try`, whose handler catches `Exception` —
and `PlotPublicationBlocked` subclasses `RuntimeError`. So after C6 removes
`strict=True`, a blocked image publication is caught and recorded as a routine
plot failure, not treated as void. That is pre-existing for this path, not
introduced here, but the docstring now asserts otherwise for a caller it names.

### C5 checklist, from this gate

1. Pass `error=exc`, **not** `error=RuntimeError(message)` — the plan's snippet
   predates the `list[str]` → `list[BaseException]` change.
2. Use `_format_error(errors[0])` in the final `raise`, not f-string
   interpolation of an exception outside a handler.
3. Change `lifecycle="qc dependency"` at `_coordinator.py:303` to `"qc"`. The
   plan does not route you through this line; the bad value arrives by parameter
   forwarding into the handler you *are* editing.
4. Decide (a) / (b) / (c) on the flat path's capability verdict before writing
   Step 6, not after. (b) needs no signature change.
5. `plots_base=self._plots_base` on both `publish_plot_output` calls is
   **already done** (`0e06cb6a`) — Step 6's first instruction is satisfied; do
   not re-apply or assume it is missing.

---

## Claims that do not match the code

### 1. `publish_plot_output`: "Every production caller passes it" — was false for **both** callers *(found here, fixed in `0e06cb6a`)*

`plots_base`'s docstring said omitting it "writes a 4.8 MB bundle per page
directory — correct output, but the duplication this design exists to avoid.
Every production caller passes it." Neither `_publish_aggregate`
(`_coordinator.py:336`) nor the multi-page image branch (`:361`) passed it.
Measured in the tree at `8a08bb67`, three images:

```
_MultiPagePlotly/ds/img-A-2cf2d99c00f1/plotly.min.js  4,847,499 bytes
_MultiPagePlotly/ds/img-B-f96779ea0dff/plotly.min.js  4,847,499 bytes
_MultiPagePlotly/ds/img-C-aa9b76c4508b/plotly.min.js  4,847,499 bytes
```

14.5 MB for three images; **7.45 GB on a 1,536-image plate**. Every page's
relative `src` was correct, so each page opened fine and the duplication was
invisible from any single page.

The same fallback also put `.failures.jsonl` in the page directory rather than
`deliverables/plots/`, violating spec §3's single-file record. Both are fixed by
the one change in `0e06cb6a`; only the bundle count is guarded.

**Why no test caught it, which is the part worth keeping:** every figure in
`tests/unit/plotting/test_coordinator.py` is `plt.figure()` — matplotlib, for
which no bundle is ever written — and all three writer tests that assert bundle
location pass `plots_base=` explicitly. The default path was not overlooked; it
was **unreachable from any existing test's inputs**.

### 2. The `plots_base or directory` comment justifies a distinction that does not exist

```python
# NOT `plots_base or directory`. Defaulting to the page directory writes a
# 4.8 MB bundle into EVERY directory ...
base = plots_base if plots_base is not None else directory
```

`Path` defines neither `__bool__` nor `__len__` (verified), so **every** `Path`
instance is truthy — including `Path("")`, which normalises to `PosixPath('.')`.
There is no `Path` value for which `or` and `is not None` differ. The comment
attributes the gigabyte trap to a choice that has no behavioural consequence;
the trap came from callers not passing the argument at all, which is what
`0e06cb6a` fixed. (If anything `or` is marginally safer: it would coerce a
stray `""` to `directory` instead of propagating a `str` where a `Path` is
expected.)

### 3. `record_plot_failure(lifecycle="page")` violated its own docstring's closed set *(fixed in `5dbdf503`)*

`_failures.py:73` documented `lifecycle: "image", "measurements", "analysis", or
"qc"`; the writer passes `"page"` (`_writer.py:323`). Spec §3's record example
lists the same four. The C2-gate shape exactly: a prose-stated closed set that
the code walks out of, with nothing checking.

Resolved as a genuinely **orthogonal** value rather than a fifth peer, which is
the right call: the four are the coordinator's emit points (*when* a plot was
being refreshed), while `"page"` is the writer's own level — one page of a
multi-page output failed, inside whichever of the four was running, and the
writer cannot know which.

#### The generalisation: `"qc dependency"`, and why C5 will not see it coming

An AST sweep of every `lifecycle=` constant found a fourth value in use:
`"qc dependency"` at `_coordinator.py:303`. Not a live defect — the coordinator
calls `record_plot_failure` nowhere yet (`grep -c` returns 0), so it currently
reaches only a `logger.warning` format argument. C5 wires the recorder in, at
which point it becomes a JSON field value outside the closed set and the only
one containing a space.

**The part worth adding is how it will arrive.** `_emit_aggregate` takes
`lifecycle` as a **parameter** (`:323`) and its handler (`:328`) is one of the
five C5 edits. Plan Task 9 Step 4's snippet for that handler is
`self._record_failure(binding, exc, lifecycle=lifecycle)` — it forwards the
parameter. So a C5 implementer working through the five handlers **never sees
the string `"qc dependency"`**: it is supplied 25 lines away at the
`emit_dependent_qc` call site, which the plan does not ask them to touch. The
fix belongs at `:303`, not in the handler being edited. Recommend passing
`"qc"` there and keeping the dependency distinction in the log line, where it
already lives and where a space is harmless.

Two related facts for C5, from the same sweep:

- **`"analysis"` and `"image"` have no `lifecycle=` call site at all today.**
  The only three are `"measurements"` (`:115`), `"qc"` (`:258`) and
  `"qc dependency"` (`:303`); `emit_image` and `emit_analyses` have their own
  handlers (`:104`, `:177`) and supply no lifecycle string. C5 introduces both
  values as literals. So the documented set is presently aspirational in two of
  its four members — worth knowing before trusting it as a description.
- **`_emit_aggregate`'s handler swallows without re-raising**, so a failure
  inside it never reaches `emit_qc`'s handler at `:259`. After C5 that means one
  record per failure, not two — but it also means `:259` fires *only* for
  prelude failures, which is precisely the F4 window Task 9 Step 5 is about.
  Worth confirming during C6 rather than assuming.

### 4. `_render_page` docstring names a caller that does not exist

> It is called from `_publish_plot_output_locked` ... **and from
> `PlotCoordinator._publish_image_value`** for the flat single-page image path

`_render_page` has exactly one caller in the tree (`_writer.py:291`). The second
is C5's job. Written in the present tense, it reads as a statement about the
code rather than about the plan — and it is the one sentence a C5 implementer
would take as confirmation that the wiring already exists.

### 5. `save_html`: "standalone interactive HTML page"

The page is explicitly **not** standalone — its next sentence says so ("the
4.8 MB bundle is referenced rather than embedded"), and "standalone" is plotly's
own word for `include_plotlyjs=True`, the opposite mode. A reader who trusts the
first line will copy a page without its bundle and get a blank div.

### 6. `8a08bb67`: "Five test repairs, three of them mechanical `'file'` → `'files'`"

Five tests were repaired — that part holds. The rename count does not.

By AST parse of the repaired files, the `["files"]["png"]` reads land in
**four** tests (`test_output_adapter.py:80, :106, :178` and
`test_plot_meas_time_series.py:334`). An independent AST diff of the two
revisions, run by the lead, gives the same split from the other direction:
`test_output_adapter.py` has 3 changed tests, **2 rename-only**
(`collision_safe_names`, `hash_suffix_rechecked`) and 1 not (the concurrency
test); `test_plot_meas_time_series.py` has 2 changed, neither rename-only.

So: five repairs, **two** purely mechanical, **four** containing the rename.
Neither reading gives three. The third rename lives in a test that also took the
`chrome_available` change, so calling it mechanical is wrong on either count.

The commit is three back and this repo has no interactive rebase, so it is not
being amended; this is the correction of record.

*(For contrast, `e6f58f16`'s corresponding claim — "three assertions still read
the ... `"file"` key and one reads the `"renderers"` map" — **is exactly right**:
the parse of the test file at that commit shows 3 `["file"]` reads, at :80, :106
and :170, and 1 `["renderers"]` read at :249.)*

### 7. `publish_plot_output`'s summary line is now stale

"Destination directory for page **PNGs** and the manifest" (also HTML now), and
"A page failure is logged and omitted while sibling pages continue" — which was
the whole truth at `schema_version: 1` and is now two-thirds of it: a failure is
also appended to `.failures.jsonl` and listed under `failed`.

### The pattern

Findings 1, 2, 3, 4 and 5 are all the same failure: **an invariant asserted in
prose, in a codebase where nothing checks prose.** With the C2 gate's
`record_plot_failure` ("every step is inside the one handler" — a lazy import
sat outside it) and `_MIN_BUNDLE_BYTES` ("the zero-byte remnant of an
interrupted write" — `os.replace` makes that impossible), that is seven across
two gates, all in the same three modules. The comments in this cluster are
unusually good *as explanations* and unusually unreliable *as claims*; the
density of load-bearing prose is itself the risk. Finding 1 is the one that
mattered — 7.45 GB — and it was a docstring sentence stating the opposite of the
code two files away.

Two of these have now recurred on the *same parameter* across two gates
(`record_plot_failure`'s handler scope in C2, its `lifecycle` set here), which
suggests the module is not merely unlucky. The cheap structural answer, if one
is wanted: `lifecycle` is the only documented closed set in this package that is
a bare `str`. A `Literal` annotation would have made both the `"page"` and the
`"qc dependency"` findings a type error rather than a review finding, and the
project already has the convention for it (`adding-an-operation`, closed value
sets). That is a C5-or-later suggestion, not a gate condition.

---

## Defects

**D1 — `plots_base` was threaded by neither production caller.** Severity:
high. **Fixed in `0e06cb6a`**, guarded by
`test_a_multi_page_plotly_image_plot_writes_exactly_one_bundle`, which counts
bundles across the whole output tree (the only form of the assertion that
distinguishes hoisted from duplicated, since a per-directory bundle is still a
bundle that exists) and is machine-independent — it asserts the HTML page count
and the bundle count, both of which hold with or without Chrome.

**D2 — `.failures.jsonl` placement is unguarded.** `0e06cb6a` fixed its location
as a side effect of the same argument, but nothing asserts it. A future caller
that forgets `plots_base` again would scatter the record across page directories
and only the bundle guard would fire. One assertion in the D1 guard would close
it: `sorted(tmp_path.rglob(".failures.jsonl"))` is empty or singular and rooted
at `deliverables/plots`.

**D3 — `png_ok` probes Chrome even when no Plotly figure is present.**
`_publish_plot_output_locked` computes `png_ok = chrome_available()`
unconditionally, but all three of its uses sit under `has_plotly`
(`_writer.py:369, 372`; the fourth branch does not read it). So an
all-matplotlib publication launches Chrome — or pays the 0.14 s failed probe —
for a value it cannot use. Memoised per process, so the cost is bounded and this
is minor; but spec §2 accounted for the eager probe on the grounds that it buys
the user an early verdict, and here it buys nothing. `chrome_available() if
has_plotly else False`, computed after the loop, is equivalent.

**D4 — `renderers` is answering two different questions at once, and the map
cannot be both.**

Spec §3 defines it as a **capability** verdict: "`renderers` records the
capability verdict per published directory". The comment beside the `partial`
branch defends that branch on **outcome** grounds: "the matplotlib pages have a
PNG and the Plotly pages do not, so neither 'available' nor 'unavailable' is
true of the directory".

Those are different claims about the world, and each branch is correct under one
reading and wrong under the other:

- **Under capability**, `renderers` describes *what this machine can render*.
  `html: "available"` is then true whenever a Plotly figure is present, whether
  or not any HTML file was written — and `"partial: chrome not found"` is the
  odd one out, because capability does not vary across a directory; it is a
  property of the process. It is really a statement about *which figures in here
  needed Chrome*, which is an outcome.
- **Under outcome**, `renderers` describes *what is on disk*. `partial` is then
  exactly right — and `html: "available"` becomes false whenever every HTML
  write failed (a full disk, an `ArtifactLockTimeout` from
  `ensure_plotlyjs_bundle`). The `partial` comment's own standard is what
  convicts it: *"an overstatement here is worse than an absence — it stops them
  looking further."* That standard, applied to `html`, condemns the branch three
  lines above it.

So the defect is not a wrong value — every reachable combination is defensible
under *some* reading, and `failed` / `partial` carry the literal truth in every
case where they diverge. The defect is that a reader cannot know which question
they are getting an answer to, and the two branches were written by someone
answering different ones. That is precisely the condition under which a later
change "fixes" one branch to match the other and silently breaks a consumer of
the first.

**Recommendation:** commit to capability (it is what the spec says, and it is
the only reading under which the map is cheap to compute correctly), replace
`"partial: chrome not found"` with something that does not smuggle in an
outcome — `"available: matplotlib only; chrome not found"` — and state the
chosen reading in the manifest's own documentation. Low severity, but it should
be settled before C5 adds a second producer of this vocabulary on the flat path.

---

## Checked and sound

**The four-branch `renderers` chain, over all eight combinations.** Re-derived
the chain in isolation and compared each verdict against what is on disk. With
every write succeeding, **all eight are true of the directory**:

| plotly | mpl | chrome | `renderers` | true? |
|---|---|---|---|---|
| T | T | T | `{html: available, png: available}` | yes |
| T | T | F | `{html: available, png: partial: chrome not found}` | yes |
| T | F | T | `{html: available, png: available}` | yes |
| T | F | F | `{html: available, png: unavailable: chrome not found}` | yes |
| F | T | T | `{png: available}` | yes |
| F | T | F | `{png: available}` | yes — mpl needs no Chrome |
| F | F | T | `{}` | every page unsupported → all in `failed` |
| F | F | F | `{}` | same |

The neither-backend case the brief asked about produces an **empty** `renderers`
and routes every page to `failed`. That is an absence, not a false claim, and it
is consistent with the comment's own "an overstatement is worse than an
absence". `html` is never claimed when no Plotly figure is present. It *is*
claimed when a Plotly figure is present and its HTML write failed — the D4
semantics question, not a reachable wrong value under the spec's own framing.

The chain also fixes a real bug in the plan's two-`if` form, which let the
matplotlib branch overwrite the Plotly verdict so a mixed Chrome-less directory
claimed `png: available`. The commit message's mutation evidence for this is
consistent with what I derived.

**The `lambda dest:` closures.** Not a late-binding hazard, and the brief's
framing ("inside a loop") does not apply: **there is no loop inside
`_render_page`**. `figure` and `src` are ordinary locals of a function invoked
once per page from the loop in `_publish_plot_output_locked`, so each call has
its own frame. `_atomic_write` invokes `write(temporary)` synchronously on its
first statement, before anything else can rebind. Both halves hold.

**`_atomic_write`'s `finally: temporary.unlink(missing_ok=True)` after a
successful `os.replace`.** Harmless and masking nothing. `os.replace` is a
rename — after it succeeds the temporary path no longer exists, so
`missing_ok=True` makes the call a no-op costing one syscall. It cannot remove
the destination (different path) and cannot hide a failed write (a raising
`write` propagates; the `finally` only cleans up after it).

**S11's orphan cleanup path under stem de-duplication.** Correct.
`files["html"]` is set to `f"{stem}.html"` from the same `stem` parameter the
HTML was written under, so `directory / files["html"]` resolves to the
collision-hashed name when one was assigned. It is keyed off the recorded
filename, not recomputed from the label — which is the form that stays right.

Scope worth noting: S11 protects *within* a page. If page 2's PNG is blocked,
page 1's already-published files remain on disk and the manifest is never
written, so the directory can hold an older manifest describing different files.
That is pre-existing blocked-publication behaviour, not introduced here.

**Nothing reads the removed per-page `"file"` key.** Swept `.py`, `.md` and
`.js` across the tree. Every surviving `"file"` hit is an unrelated
`kind == "file"` / `scope == "file"` discriminator (GUI directory trees,
`_metadata_migration.py`) or prose in the spec/plan/review documents describing
the change itself. The only `manifest.json` readers in `src/` are the CLI
progress/dashboard manifests, which are a different file.

**And nothing reads the published plot *files*, either**, which is what makes
the version bump as cheap as the spec claims. `_gui/analysis/_render.py:128`
uses `FigureAdapter.backend_name` on an **in-memory** figure and builds a Dash
component; it never touches `deliverables/plots/`. The two documentation strings
spec §2 names (`_cli_interactive.py:132`, `_cli_readme_generator.py:106`)
describe the tree generically ("Configured plot outputs and page manifests") and
need no update for HTML.

**`FigureAdapter.backend_name` is not dead** after the writer stopped calling
it — `_gui/analysis/_render.py:128` and `tests/unit/abc_/plotting/test_output.py`
still use it. Removing the inert patch from the concurrency test rather than
leaving it beside the new one was the right call for the reason `8a08bb67`
gives.

**The second `figure_backend_of` call on closed matplotlib figures.** Safe, and
verifiable rather than assumed: `figure_backend_of`
(`abc_/plotting/_output.py:29-36`) reads only `type(figure).__module__` and
`type(figure).__name__`. Neither is affected by `plt.close`.

**The two import styles are each justified by a real test.** Module-level
`figure_backend_of` is patched as `_writer.figure_backend_of`
(`test_output_adapter.py:141`); function-scope `chrome_available` is patched as
`_backends.chrome_available` at five sites. The "do not harmonise" comment is
accurate, and harmonising either way would break live tests.

**`e6f58f16` is correct that the tree is RED at that commit**, and the count is
right (3 + 1, verified by parse). One nuance the message does not mention: at
that commit `test_concurrent_plot_publications_do_not_mix_generations` fails at
`generations.pop()` with `KeyError` *before* reaching its `["file"]` read, for
the separate reason the next commit's message explains. The four assertions are
real; one of them is unreachable at the moment it is described.

**The `save_html` → `include_plotlyjs=<str>` behaviour** is pinned by
`test_the_html_references_the_hoisted_bundle`, which asserts both
`src="../plotly.min.js"` in the page and the absence of a sibling bundle.

---

## Unverified

- **I ran no pytest.** Every finding here is from reading, from AST parse, and
  from the one runtime probe (bundle placement) the lead executed. The
  cluster's own green/red status is taken from the commit messages, not
  re-measured. The full `tests/unit/plotting/` run belongs to the C4 gate or to
  Task 14.
- **The Chrome-present path is unmeasured**, here as in the spec. Every
  combination in the truth table with `chrome=True` is derived from the branch
  structure, not observed — this cluster has no Chrome. Spec §2 already gates on
  measuring the success path in Task 4 Step 6; nothing in C3 changes that debt,
  and D3's "launches a browser for nothing" is a prediction, not a measurement.
- **`.failures.jsonl` under genuine SLURM concurrency** is untested — the lock
  is exercised only single-process. Out of scope for C3; flagging because the
  writer is now a second producer of that file alongside the coordinator, so the
  contention shape changed even though the locking did not.
- **`0e06cb6a`'s guard was mutation-checked by the lead, not by me.** I read the
  test and the reported failure message; I did not run the mutation.
- **The `8a08bb67` rename count is corroborated, not merely repeated.** My AST
  parse counted `["files"]["png"]` reads in the post-repair files; the lead's
  AST diff counted changed tests between the two revisions. They were run
  independently and agree (four tests carry the rename, two repairs are
  rename-only). Both are static; neither ran the tests.
- **`5dbdf503`'s claims I did verify directly**: the `lifecycle=` sweep returns
  four constants (`"page"`, `"measurements"`, `"qc"`, `"qc dependency"`), and
  `grep -c record_plot_failure src/phenotypic/plotting/_pipeline/_coordinator.py`
  returns 0, so `"qc dependency"` is confirmed inert today. The claim that C5
  makes it live follows from plan Task 9 Step 4 as written, which is a statement
  about the plan, not a measurement.
