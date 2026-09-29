# Deferred from the CLI preflight change

Each entry was considered for `design.md` and left out on purpose. The reason is recorded
so a later change can pick it up without re-deriving the argument.

## Make a swallowed post-op failure visible at run time

`_apply_post_to_master` (`_cli/_cli_output_manager.py:873-944`) logs any non-integrity post
failure at WARNING and publishes the clean master, discarding the output of every post op
in the chain (claim-verification report §7). The preflight's `PF-POST-COLUMN` catches the
static cause, a column that cannot exist, but not a data-dependent one, such as an
`ExpandMetadata` split whose part count differs on some rows.

The runtime fix is to record the failure durably and surface it in the run's summary. It
is deferred because it changes the finalizer's publication contract, and the finalizer is
the sole publisher of the SLURM completion marker (`_cli/CLAUDE.md`). A change there needs
its own review of what "complete" means when post output is missing. The
figure-backend-routing change (`docs/superpowers/specs/2026-09-20-figure-backend-routing/`)
is the precedent for how to do it: a durable failure record and a manifest that cannot be
mistaken for success.

## A strict mode that promotes warnings to errors

A `--strict-preflight` flag would let a production pipeline refuse on any warning, for
example unmatched metadata rows. It is deferred until users have seen the warnings on real
runs, so that the set a strict mode should promote is chosen from evidence rather than
guessed.

## Validating the `qc` and `plots` slots

`find_operations` does not walk `get_qc()` or `get_plots()`
(`sdk_/_operation_tree.py`; `_image_pipeline_core.py:659`, `:687`). QC operations and plot
bindings could carry requirements of their own. Plot backends already have a preflight
(`preflight_plot_backends`), and no QC operation is known to fail on a static
incompatibility, so there is no finding to close yet.

## A bounded Chrome probe

`validate_pipeline` calls `preflight_plot_backends`, which may call `chrome_available()`,
and that function's own docstring says it can hang. The preflight inherits this unchanged.
A timeout belongs in the plotting module, not here.

## Persisting the preflight report

Writing the report to `.phenotypic/preflight.json` after the mutating half begins would
give a run an audit record of what was checked. It is deferred because nothing reads it
yet, and a file nothing reads is a file nobody keeps correct.

## The dry-run size estimate

`execute_dry_run` estimates output size from hard-coded megabytes per image
(`_cli_interactive.py:236-250`) that predate the per-image OME-Zarr layout. Correcting it
needs measured store sizes, which this change does not produce.

## Declaring `tifffile` as a direct dependency

Spec §7 planned to add `tifffile` to `[project] dependencies`, since it is
imported directly but arrives only transitively through scikit-image. Plan
Task 9 attempted it on 2026-09-24 and `uv lock` (uv 0.8.17) could not
re-resolve the project in that environment: the Windows x86_64 split fails on
`gudhi==3.13.0` (the `topology` extra), which has no compatible wheel there.
The failure is independent of `tifffile` -- any re-resolution hits it, while
the committed lockfile verifies with `uv lock --locked` because nothing forces
a re-resolve. Editing `uv.lock` by hand is not an option. `tifffile` remains
installed through scikit-image, so nothing breaks today; add the direct
dependency in the change that next re-resolves the lockfile successfully.
