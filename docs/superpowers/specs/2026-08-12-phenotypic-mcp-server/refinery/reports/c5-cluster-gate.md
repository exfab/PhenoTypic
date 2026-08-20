# C5 cluster gate — T13 / T14 / T17

**Scope:** `c640f605` (T13 `directory_digest`), `4bd7a2ea` (T14 `SubsetSelector`
hierarchy), `7dc9e916` (T17 subset staging), on `feat/mcp-server`.
**Method:** read + mutation testing in an isolated `git archive` copy under
scratch (`PYTHONPATH` override), plus live filesystem probes on this cluster's
`xfs`/`gpfs` pair. The shared working tree was never modified.

---

## Verdict

**Not sound as committed — one blocker.** `subset_digest` keys the staging
directory on parent-*relative* paths only, so two different parents with the
same internal layout stage onto the same directory and the second caller silently
receives the **first parent's images**. Reproduced end to end (§B-1). T13 and T14
are sound; their remaining findings are non-blocking.

Everything the gate was asked to verify directly:

| Asked | Result |
|---|---|
| I1a fixed, and the fixed test discriminates | **Yes** — `test_selectors.py:486-497` asserts `model_fields` membership, exactly as the correction specified |
| `group_filter` applied before any selector runs | **Yes, proven** — mutating `_select(filtered)` → `_select(pool)` fails 2 tests |
| `group_filter` recordable on a `user_named` subset | **Yes** at the level Phase 1b covers (`SubsetSelection` is directly constructible and carries it); `subset_put` is a Phase-2 tool and does not exist yet |
| Digest: contents-only change | **Invisible** — documented tradeoff, mtime catches every non-adversarial case |
| Digest: add / remove / rename / move between datasets | All detected |
| Digest: listing order | Stable; mutation-proven |
| Digest: `/rhome` ↔ `/bigdata` symlink pair | **Stable** — same tree via a symlinked root gives the same digest |
| Staging: symlink vs copy, cross-filesystem | Symlinks by default; copy fallback works and is reported; cross-filesystem `cache_root` fine (links are absolute) |
| Staging: `flat/` basename collision | Handled and tested (`dataset__name.tif`, suffix preserved) |
| Staging: idempotent on a second call | Behaviour correct, **but the test that claims to prove it cannot fail** (§C-1) |
| Staging: partially-written directory detectable | **Yes** — temp build + `os.replace` + `.complete` written last; mutation-proven |

---

## A. Blocker

### B-1 — `subset_digest` is parent-blind: two datasets collide on one staging tree

`src/phenotypic/_services/staging.py:135-140` hashes only the sorted
parent-relative paths. `SubsetToStage.parent` is carried on the dataclass and
used by the fidelity check, but contributes **nothing** to the key.

Two independent experiments that happen to share a layout — the normal case for
a plate-imaging lab, where `plateA/plateA_01.tif` is a naming convention, not an
identity — collide:

```
digest exp1 : sha256:c1f0c615bf0f1d86…
digest exp2 : sha256:c1f0c615bf0f1d86…
same root   : True
exp2 staged bytes: [b'EXPERIMENT-ONE']
exp2 symlink -> : …/experiment1/plates/plateA/plateA_01.tif
```

`stage_subset(exp2, …)` finds `exp1`'s `.complete` marker, takes the `_adopt`
early return (`:199-200`), and hands back a tree of symlinks into `experiment1`.
The tune/deploy run then trains on the wrong dataset with every downstream check
green — the fidelity check never runs, because nothing is built.

This is the exact failure the subset boundary exists to prevent, and it is
*worse* than a plain collision: it is silent, and one workspace `cache_root`
shared across a campaign is the design's normal configuration.

**Why no test catches it:** every staging fixture builds its parent under a fresh
`tmp_path`, and no test stages two *different* parents into one `cache_root`.
`test_a_different_image_set_stages_somewhere_else` varies the image list, never
the parent. This is the `tmp_path`-is-always-fresh pattern the brief predicted.

**Fix.** Mix per-image content identity into `subset_digest` — hash each chosen
image's `file_fingerprint` (or `(size, mtime_ns)`) alongside its relative path.
That keeps the property `test_the_digest_ignores_where_the_parent_lives` asserts
(a preserving copy at another mount is the same subset) while separating two
genuinely different parents. Cost is bounded: a subset is tens of images, not the
480-image parent — which is the reason `directory_digest` refuses to hash bytes,
and the reason it is safe to do so here.

Do **not** fix it by folding in `parent.resolve()`: that breaks the mount-alias
property on purpose asserted at `test_subset_staging.py:442-461`.

**Test that must fail before the fix:** two parents, identical relative layouts,
different bytes, one `cache_root`; assert the second staging's files read back as
the second parent's bytes.

---

## B. False greens

### C-1 — `test_restaging_the_same_digest_is_a_noop` cannot fail

`tests/unit/services/test_subset_staging.py:125-130` asserts only
`first.flat == second.flat`. Both sides are `subset_staging_dir(cache_root,
subset_digest(subset))` — a pure function of the inputs. It is true whether or not
staging is idempotent, and true even if the second call rebuilds the whole tree.

Mutation: delete the completion-marker early return (`staging.py:199-200`) so
every call rebuilds → **27/27 still pass.**

Blast radius of the real bug it fails to guard is bounded (the mid-build marker
check at `:210` still adopts, so it wastes work rather than corrupting), but the
test named "is a noop" proves nothing about no-op. Replace with an observable:
drop a sentinel file inside `staged.root`, restage, assert the sentinel survives
and the flat entries' `st_ino`/`lstat().st_mtime_ns` are unchanged.

### C-2 — `SubsetSelection` is not frozen in the way its docstring claims

`_selector.py:167-201` says *"Frozen because the artifact is written from it: a
selection that could be edited afterwards is a recorded provenance that need not
match what actually ran."* `model_config = ConfigDict(frozen=True)` blocks
attribute **assignment** only. `group_filter` and `params` are plain mutable
dicts:

```python
s = SubsetSelection(images=("a.tif",), method="user_named",
                    group_filter={"Metadata_Species": "A_niger"})
s.group_filter["Metadata_Species"] = "TAMPERED"   # succeeds
s.params["injected"] = True                        # succeeds
```

`test_a_selection_is_frozen` (`:475-483`) only rebinds `.images`, a tuple field —
so it passes against a model where the two dict fields are wide open. Given
USER-21 (an ack bound to one group's images must not be spendable on another's)
this is the field that matters most.

**Fix:** annotate both as an immutable mapping and coerce in a validator, or state
the weaker guarantee in the docstring. Then assert the in-place mutation raises.

### C-3 — `test_link_mode_is_reported` and `test_the_umask_…` cannot discriminate

- `:228-229` asserts `staged.link_mode in {"symlink", "copy"}` — true for any
  hardcoded value. (Hardcoding *is* caught, by
  `test_copy_fallback_is_used_and_reported`; the test itself is just inert.)
- `:464-467` checks `os.access(…, R_OK|X_OK)` as the **owner**, who always has
  access to a directory they just created. The stated risk ("a fleet worker reads
  this directory") is a *different user or a restrictive umask*, and neither is
  reachable through `os.access` from the creating process.

### C-4 — the length prefix in `directory_digest` is unexercised

`_io_constants.py:353-358` length-prefixes each name so the `(name, size, mtime)`
stream is unambiguously parseable. Removing the prefix leaves **9/9 green**. The
property is correct and worth keeping; it is simply untested. Low severity — an
actual collision needs adversarial control of sizes and timestamps.

---

## C. Non-blocking findings

### N-1 — `directory_digest` silently skips an unreadable subdirectory

`pathlib.rglob` swallows `PermissionError`, so a subtree the caller cannot read
contributes nothing and the digest simply comes out different, with no error:

```
unreadable subdir : SILENTLY SKIPPED (digest differs, no error)
```

Since the digest decides "is this resume compatible" and "does the human's
approval still apply", two users — or the same user across a permission change —
get two identities for one tree and nothing says why. Consider walking with
`os.walk(root, onerror=…)` and raising, or recording the unreadable directory in
the digest. `src/phenotypic/sdk_/_io_constants.py:339-348`.

### N-2 — `relative_to` naming a non-ancestor silently falls back to absolute paths

`_io_constants.py:345-348`:

```python
try:
    name = path.relative_to(anchor).as_posix()
except ValueError:
    name = path.as_posix()
```

Pass a `relative_to` that is not an ancestor of `root` and every recorded name
becomes an **absolute** path — the digest silently becomes mount-dependent, which
is precisely what `test_digest_is_relative_so_a_moved_directory_keeps_its_identity`
exists to prevent. Verified: `relative_to a non-ancestor → SILENTLY ABSOLUTE`.
The fallback should raise (`ValueError: relative_to must be an ancestor of root`).
No caller exists yet — `directory_digest` is currently exported and nothing else.

### N-3 — a pre-1970 mtime raises `OverflowError`

`mtime_ns.to_bytes(16, "big")` (`:357`) rejects negatives:
`OverflowError: can't convert negative int to unsigned`. Reachable from a
scanner/camera with a mis-set clock or a restored archive. One-word fix:
`to_bytes(16, "big", signed=True)`.

### N-4 — `link_mode` disagrees with itself between a build and an adopt

`_build_layouts` returns `"copy"` if **any** of the `2N` link operations fell back
(`staging.py:256-269`), while `_observed_link_mode` (`:315-319`) reports based on
whichever single entry `flat.iterdir()` yields first — and never looks at
`nested/` at all. Same tree, two answers:

```
first.link_mode : copy
second.link_mode: symlink   (same tree, re-adopted)
```

`link_mode` is a diagnostic, not a correctness field, hence non-blocking — but it
should be recorded in the `.complete` marker at publish time rather than
re-inferred.

### N-5 — `_link_or_copy`'s blanket `except OSError` turns "destination exists" into "copy"

`staging.py:307-312` catches every `OSError`, including `FileExistsError`, and
responds by `shutil.copy2`-ing over the destination. With duplicate `ImageRef`s in
`SubsetToStage.images` this surfaces as a bare
`shutil.SameFileError` from deep inside the builder, and the collision-detection
in `_flat_names` also mis-fires (a duplicated ref reads as a cross-dataset
collision and mangles the flat filename). `SubsetSelection.select()` deduplicates,
but `SubsetToStage` accepts any tuple. Catch only the symlink-unsupported errors,
and dedupe `subset.images` on construction.

### N-6 — `min_per_group` silently overrides `n`

`MetadataGroupSubsetSelector(n=2, min_per_group=3)` over two groups returns
**6** images — 3× the target — because `_allocate`'s reduction loop refuses to go
below the floors and `break`s (`_selectors.py:207-212`). The ABC docstring's
"`n` is a target, not a contract" covers taking *fewer* than `n`; it does not
cover silently taking three times as many, and bounding compute is the point of
the subset boundary. Either clamp, or raise when the floors are infeasible, and
test it.

### N-7 — `group_filter` is applied before the availability gate

`select()` (`_selector.py:329-334`) runs `_apply_group_filter` first, so an
unavailable selector with a malformed filter reports the *filter* error rather
than `NotImplementedError`, after doing CSV I/O it did not need. Cosmetic
ordering; flagged because the two error classes route to different tool-layer
codes.

### N-8 — `MetadataGroupSubsetSelector` re-reads its CSV up to three times per `select()`

`_apply_group_filter`, `_select`→`_groups_of`, and `_allocation_of`→`_groups_of`
each call `_read_grouping_metadata` (`_selector.py:357`, `_selectors.py:145,232`).
Harmless at plate scale; a one-line memo on the instance would remove it.

### N-9 — cluster note: `cp -a` and `rsync -a` do preserve `mtime_ns` here

Measured on this cluster (`xfs` scratch → `gpfs` `/bigdata` and `/rhome`, GNU
coreutils 8.30): a preserving copy reproduces `mtime_ns` **exactly** for any file
at least ~10 s old, so `directory_digest` survives `cp -a`/`rsync -a` for real
image datasets. The one exception is a file copied in the same instant it was
written, where write-back moves the destination mtime ~10-25 ms forward. Not a
defect — recorded so the docstring's "a copy made without preserving timestamps
reads as a different set" is not read as a broader warning than it is.

**Types and lint.** `mypy` over `src/phenotypic/subset`,
`_services/staging.py` and `sdk_/_io_constants.py` reports **0 errors in C5's own
files** (the run surfaces 391 pre-existing errors in followed imports, all in
`gui/` and unrelated modules).

---

## D. What was mutation-proven sound

Each mutation below was applied to an isolated copy and **did** turn the suite red:

| Mutation | Caught by |
|---|---|
| `_select` receives the unfiltered pool | `test_group_filter_restricts_a_selector_that_knows_no_metadata`, `test_group_filter_composes_with_metadata_stratification` |
| `_apply_group_filter` becomes a passthrough | 6 tests |
| `link_mode` hardcoded | `test_copy_fallback_is_used_and_reported` (+9) |
| Fidelity check compares dataset *names* only | `test_fidelity_check_rejects_images_filed_under_the_wrong_dataset` |
| Build straight into `root`, no temp + `os.replace` | `test_a_refused_tree_leaves_nothing_marked_usable`, `test_a_builder_that_loses_the_race_adopts_the_winners_tree` |
| Drop the mid-build "loser adopts the winner" check | `test_a_builder_that_loses_the_race_adopts_the_winners_tree` |
| `directory_digest` drops the sort | `test_digest_ignores_listing_order` |
| `directory_digest` ignores `relative_to` | `test_relative_to_anchors_the_recorded_names` |
| `directory_digest` hashes directories too | `test_digest_is_relative_so_a_moved_directory_keeps_its_identity` |

Also verified by probe: the fidelity check correctly refuses a two-level relative
path (which `scan_directory_structure` cannot represent) and an `ImageRef` whose
`path` points outside the parent; `scan_directory_structure` rejects a mixed
parent before staging; adding `"phenotypic.subset"` to `PHENOTYPIC_CLASS_MODULES`
does **not** leak selectors into the operation catalog
(`_discovery_targets` has no entry for it) and leaves `skipped_imports` empty; and
the `test_one_shared_module_list` relaxation from equality to an ordered *prefix*
preserves the 10a correction's intent, with
`test_the_selector_subpackage_is_appended_not_interleaved` pinning the
append-only rule that makes the relaxation safe.
