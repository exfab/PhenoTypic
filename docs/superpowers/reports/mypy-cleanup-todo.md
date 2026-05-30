# mypy Cleanup — Todo List

**Baseline:** `uv run mypy src/phenotypic` → **780 errors across 224 files** (502 source
files checked). Captured at commit `3cb49c8e` on `feature/smart-qc-gui`.

Config is minimal (`[tool.mypy] plugins = ["pydantic.mypy"]`, no strictness flags), so
this is default-strictness mypy.

Categories 1–3 are the highest-leverage / lowest-risk work and are being executed now.
Categories 4–6 are catalogued for follow-up.

---

## Error counts by code (baseline)

| Code | Count | Code | Count | Code | Count |
|---|---|---|---|---|---|
| `override` | 213 | `misc` | 15 | `valid-type` | 5 |
| `import-untyped` | 138 | `index` | 14 | `no-redef` | 4 |
| `arg-type` | 121 | `overload-cannot-match` | 13 | `import-not-found` | 4 |
| `attr-defined` | 71 | `syntax` | 9 | `annotation-unchecked` | 4 |
| `assignment` | 64 | `call-arg` | 9 | `name-defined`/`dict-item`/`call-overload` | 3 ea |
| `union-attr` | 48 | `operator` | 6 | `str`/`return`/`abstract` | 2 ea |
| `return-value` | 27 | `var-annotated` | 5 | `type-var` | 1 |

---

## ✅ Category 1 — `_operate`/`apply` overload design (≈206 errors) — DONE

**Root cause.** ABCs declare `_operate` and `apply` with two `@overload`s,
`(Image) -> Image` and `(GridImage) -> GridImage`. Two consequences:

- `overload-cannot-match` (13): `GridImage` subclasses `Image`, so the `Image`
  overload is broader and shadows the `GridImage` one — it can never be selected.
- `override` (193 of the 213 are `_operate`): each concrete op implements only
  `def _operate(self, image: Image) -> Image`, which doesn't satisfy the
  `(GridImage) -> GridImage` overload, so mypy flags every operation subclass.

Not a runtime bug — a type-modeling choice mypy rejects.

**Fix.**

- [x] Collapse the internal `_operate` to a single `def _operate(self, image: Image) -> Image`
      (remove the two `@overload` stubs) in:
  - [x] `abc_/_image_operation.py`
  - [x] `abc_/_object_detector.py`
  - [x] `abc_/_object_refiner.py`
  - [x] `abc_/_image_enhancer.py`
  - [x] `abc_/_image_corrector.py`
  - [x] `abc_/_threshold_detector.py`
  - [x] `abc_/_gpu_detector.py`
- [x] Reorder the public `apply` overloads so the narrower `GridImage` overload comes
      first (preserves precise `GridImage -> GridImage` typing, kills overload-cannot-match) in:
  - [x] `abc_/_image_operation.py`
  - [x] `abc_/_object_detector.py`
  - [x] `abc_/_object_refiner.py`
  - [x] `abc_/_image_enhancer.py`
  - [x] `abc_/_image_corrector.py`
  - [x] `correction/_color_correction/_color_corrector.py`
- [x] Remove now-unused `from typing import overload`/`GridImage` from the `_operate`-only
      files (`_threshold_detector.py`, `_gpu_detector.py`).
- [x] Re-run mypy; `overload-cannot-match` 13→0, `_operate` override 193→13.

**Result:** `override` 213→47, `overload-cannot-match` 13→0. Overlapping-overload design
eliminated. No other error category increased (`return-value` even dropped 33→30).
Verified: full layer test suite green (796 passed), `apply()` round-trip detects 121
objects on the synthetic plate.

**Residual (NOT this category, → Category 6):** 47 `override` errors remain from unrelated
root causes — `_operate` ×13 (measure/grid subclasses with genuinely different supertype
signatures, e.g. `GridFinder._operate` vs `MeasureFeatures`), `_apply2group_func` ×4,
`apply` ×3 (grid-narrowing in `_grid_operation`/`_grid_corrector`/etc.), `show` ×2,
`imsave` ×2, `get_headers`, `from_json`, `_estimate_edges`, `shape`, `histogram`.

---

## ✅ Category 2 — Missing third-party stubs (142 errors) — DONE

**Root cause.** No code is wrong; mypy can't find type info. pandas (59), plotly (28),
napari (13), h5py (5), bm3d (5), joblib (4), ipywidgets (3), sklearn, sam2, rawpy,
numba, mahotas, vispy, ruptures, pooch, param, tqdm (2), psutil (2).

**Fix (config only, no source changes).**

- [x] `uv add --group dev pandas-stubs types-tqdm types-psutil` (pulled `types-requests`
      transitively).
- [x] Added a `[[tool.mypy.overrides]]` block with `ignore_missing_imports = true` for the
      stub-less libraries (napari, plotly, h5py, bm3d, joblib, ipywidgets, sklearn,
      sam2, rawpy, numba, mahotas, vispy, ruptures, pooch, param, qtpy) plus the optional
      `micro_sam.*` (conda-only) and `habana_frameworks.*` (Intel Gaudi) backends.
- [x] Re-run mypy: `import-untyped` 138→0, `import-not-found` 4→0.

**Note:** `arg-type`/`assignment`/`return-value` ticked up slightly (e.g. arg-type
121→128) — expected, because `pandas-stubs` now lets mypy *see into* pandas calls and
surface genuine imprecisions previously masked by the untyped import. Net still down.

---

## ✅ Category 3 — Genuine annotation typos (14 errors) — DONE

Real mistakes (harmless at runtime since annotations aren't enforced, but wrong).

- [x] `-> (plt.Figure, plt.Axes):` → `-> tuple[plt.Figure, plt.Axes]:` — all 9 `syntax`
      sites: `analysis/_tukey_outlier.py` (×3), `analysis/_mad_outlier.py` (×3),
      `_core/.../color_space_accessors/_hsv_accessor.py` (×2),
      `_core/.../accessors/_objmap_accessor.py` (×1). (The 9 syntax errors were spread
      wider than the original sample suggested — `_edge_correction.py` had none.)
- [x] `dict[str, any]` → `dict[str, Any]` (+ added `from typing import Any`) in
      `_tukey_outlier.py`, `_mad_outlier.py`, `_edge_correction.py`,
      `_cli/_cli_directory_scanner.py`
- [x] `int | float | np.nan = np.nan` → `int | float = np.nan` in
      `abc_/_measure_features.py:1039`
- [x] Re-run mypy: `syntax` 9→0, `valid-type` 5→0.

---

## ⬜ Category 4 — Genuine bugs surfaced (follow-up)

- [ ] **`OperationIntegrityError(...)` missing required arg — REAL BUG.**
      `tools_/funcs_.py:217,223` call it with one string, but
      `__init__(self, opname, component, image_name=None)` needs two positional args. If
      these guard paths fire they raise `TypeError` instead of the intended error.
- [ ] **`_DagBuilderState(selected_node_id=...)` / `.selected_node_id`**
      (`gui/builder/_callbacks.py:2120,2163`, `_layout.py:3805`) — state exposes
      `selected_block_id`/`selected_edge_id`, not `selected_node_id`. Verify whether the
      path is live; if so it's a runtime `AttributeError`/`TypeError`.

---

## ⬜ Category 5 — Pydantic-alias & Qt-binding false positives (≈13 errors, follow-up)

- [ ] **`Unexpected keyword argument "desc" for "ImagePipeline"`** (`call-arg`, 4 sites).
      Field is `desc_value` with a `desc` alias — accepted at runtime. The pydantic mypy
      plugin builds `__init__` from the field name. Declare the alias so the plugin sees it.
- [ ] **`Qt.UserRole` / `QTableWidget.NoEditTriggers` / `QHeaderView.ResizeToContents`**
      (`attr-defined`, ~8 in `gui/sweep/`). Code uses `qtpy` (runtime-normalized); stubs
      expose Qt6 nested enums. Use the nested path or `ignore_missing_imports` for `qtpy.*`.

---

## ⬜ Category 6 — Local type imprecisions (≈385 errors, follow-up, file-by-file)

| Group | Count | Pattern & fix |
|---|---|---|
| `arg-type` | 121 | Mostly `param = None` vs non-Optional (`objmap: ArrayLike = None` ×12 in `_measure_features.py`) → `ArrayLike \| None = None`; `Literal` mismatches in `nn/_checkpoint_manager.py`; Dash `**dict[str,str]` splats into `Div(...)`. |
| `attr-defined` | 71 | `_napari_sweep_viewer.py` (21) `self.viewer=None` then `.layers` — guard/type it; `schema/_measurement_info.py` (7) enum `.label/.desc/.pair` set via `__new__` → declare class-level annotations; plus Qt/DAG items above. |
| `assignment` | 64 | default-`None` vs typed, or rebinding a name to a new type → `X \| None` / guards. |
| `union-attr` | 48 | attribute access on `X \| None` → `assert`/early-return guards or fix declared type. |
| `return-value`+`return` | 29 | annotation wider/narrower than real return, or missing `return`. |
| `index`+`operator` | 20 | numpy union types (`float64 \| ndarray`) in `_grid_inference_mixin.py` → `np.asarray`/assert. |
| `misc` | 15 | `"int" object is not iterable` (real annotation mismatch); accessor abstract/overload decorator combos; unpack-count in `hdf_.py`. |
| `var-annotated` | 5 | empty containers need a type. Trivial. |
| tail | ~20 | `no-redef`, `dict-item`, `call-overload`, `str`, `abstract`, `type-var`, `name-defined` (`msvcrt` on darwin — `--platform` or per-line ignore). |
| residual `override` | ~20 | `_apply2group_func` ×4, `show`/`imsave` ×2 ea, `get_headers`, `from_json`, `_estimate_edges`, `_grid_operation.apply` — distinct signature mismatches. |

---

## Progress log

- `3cb49c8e` baseline: **780 errors / 224 files**.
- After Categories 2 + 3: **677 errors / 194 files** (import-untyped 138→0,
  import-not-found 4→0, syntax 9→0, valid-type 5→0).
- After Category 1: **491 errors / 131 files** (override 213→47,
  overload-cannot-match 13→0). **Net −289 (−37%).**
- Verification: `uv run pytest tests/unit/{abc_,detect,enhance,refine,analysis}` →
  **796 passed**, 0 failed. `apply()` round-trip on the synthetic plate detects 121
  objects. `ruff check` on changed files clean (one pre-existing unrelated `F841` in
  `_hsv_accessor.py:184` left untouched).

## Final error profile (491 remaining — all Category 4–6)

| Code | Count | Code | Count |
|---|---|---|---|
| `arg-type` | 128 | `misc` | 15 |
| `attr-defined` | 74 | `call-overload` | 12 |
| `assignment` | 73 | `operator` | 11 |
| `union-attr` | 49 | `call-arg` | 9 |
| `override` (residual) | 47 | `var-annotated` | 5 |
| `return-value` | 30 | tail (`no-redef`, `dict-item`, `name-defined`, `abstract`, …) | ~18 |
| `index` | 20 | tail (`no-redef` 4, `dict-item` 4, `annotation-unchecked` 4, `name-defined` 3, `type-var`/`str`/`return`/`bool`/`abstract` 2 ea, `list-item` 1) | 26 |
