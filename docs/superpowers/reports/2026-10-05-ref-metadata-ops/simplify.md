# Simplify pass — ref-metadata-ops

Scope: `git diff 8159b8819e18c02e432b70ff7fee8a82e87ae2c8 HEAD -- src`, lines this
branch added only. No on-disk format, digest, public API, message, PF-REF code,
Dash id, catch set or lazy-import placement changed. No tests were edited.

## Applied

- **`src/phenotypic/_core/_reference_context.py` (`_read_table`).** The
  `.csv`/`.parquet` suffix check now sits above the `try`, which removes the
  `except ReferenceTableError: raise` pass-through clause. *Preserved:* the
  unsupported-suffix `ReferenceTableError` has the same message and is raised
  at the same point, before any read. The `except Exception` wrap still covers
  exactly `pl.read_csv` / `pl.read_parquet`. A `.parquet` suffix now reaches
  `read_parquet` through `else`, since the guard has already excluded every
  other suffix.

- **`src/phenotypic/_cli/_cli_preflight.py` (`check_reference_metadata`) +
  `src/phenotypic/_cli/_cli_reference.py`.** The preflight's hand-written
  `isinstance(op, RefMetadata) and op._ref_columns()` filter over
  `operations_in_scope(context)` was a copy of
  `_cli_reference._reference_operations_with_paths`. That helper is now named
  `reference_operations_with_paths`, and the preflight calls it.
  *Preserved:*
  - (a) Imports: the preflight imports `_cli_reference` only inside the
    function. `_cli_reference` imports `_cli_preflight` only inside
    functions, plus `RunMode` under `TYPE_CHECKING`. Neither module imports
    the other at module level, so there is no cycle.
  - (b) Output: `operations_in_scope(context)` is literally
    `operations_run_in_mode(context.pipeline, context.mode)`, and the helper
    applies the same filter to that same call. It returns the same raw
    tuple paths, unjoined, in the same depth-first order. The preflight
    still does its own `"/".join(path)`. So the `PF-REF-NO-TABLE` /
    `PF-REF-COLUMN` subjects and messages are unchanged, including the
    separator and the `ops[0]` / `meas:<key>` segment spellings that come
    from `walk_operations`.
  - (c) Rename: a grep of `src/` and `tests/` found
    `_reference_operations_with_paths` only in `_cli_reference.py`. No test
    patches it, either by name or as a string. All three uses are updated.

- **`src/phenotypic/phenotypicCLI.py` (`_reference_operation_paths_for`).**
  Replaced a nested ternary that mapped `cli_mode` to an identical string
  with `cast(RunMode, cli_mode)`. *Preserved:* the guard two lines above,
  `if pipeline_json is None or cli_mode not in ("full", "process", "measure"):
  return []`, admits exactly the three `RunMode` values. The ternary was
  therefore the identity on every value that reaches it.

- **`src/phenotypic/_cli/_cli_reference.py` (new private-module helper
  `reference_table_snapshot_path`) + `src/phenotypic/phenotypicCLI.py`
  (`_refuse_unusable_reference_table`).** The fallback-snapshot choice was
  written twice: process mode uses `.phenotypic/reference_metadata.csv`,
  every other mode uses `deliverables/metadata.csv`. The copies were in
  `resolve_reference_table_path` and in the CLI refusal, and the refusal's
  docstring promised they agree. Both now call the one helper. *Preserved:*
  each caller passes the same boolean it used before
  (`config.process_only_layer is not None` and `process_mode`). Each still
  does its own `is_file()` check, error message and ordering. The helper
  imports `sdk_._io_constants` lazily, so `_cli_reference` still imports only
  the standard library at module level.

- **`src/phenotypic/_core/_image_parts/_image_io_handler.py`
  (`_from_stored_matrix`).** The out-of-range warning path computed
  `np.nanmin` / `np.nanmax` twice, once in the condition and once in the
  message. They are now computed once, as `lo, hi`. *Preserved:* the
  condition (`float` dtype, non-empty, `lo < 0 or hi > 1`), the warning
  text, the category and the `stacklevel` are unchanged. The old code
  short-circuited `nanmax` only when `nanmin < 0`. Both reductions are pure,
  and an all-NaN array emitted numpy's RuntimeWarning from both calls
  before as well. So no observable side effect changes.

## Proposed and dropped

- **`_image_color_handler.py` (`if not explicit_gamma:` in place of the
  repeated `isinstance(gamma, _Unset)`).** I dropped this one during
  implementation. The `isinstance` checks are what narrow `gamma` and
  `illuminant` away from `_Unset` for mypy, for the
  `self.illuminant: Literal[...] = illuminant` assignment and the gamma
  coercion. A boolean flag does not narrow, so the "simplification" would
  introduce type errors.

## Considered, left alone

- `_cli_reference._parse_manifest` and `_reference_context._read_image` are
  trivial wrappers, but tests monkeypatch both.
- The repeated `ReferencePin(source_image_stem(p), identity.reference_digest)`
  and `(identity.work_id, identity.relative_path)` constructions appear at
  three or four sites. A helper would add API surface for little gain.
