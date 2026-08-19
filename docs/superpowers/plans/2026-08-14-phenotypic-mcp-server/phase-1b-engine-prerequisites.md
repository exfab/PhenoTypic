# Phase 1b — Engine prerequisites (P3–P7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Reviewers run at CLUSTER boundaries, not per task.** See the plan README's
> *Review protocol* and `execution.md`. A cluster with an unaddressed correctness
> finding does not hand off to the next one.

**Implements:** §7 P3, P4, P5, P6, P7. **Spec:**
[`../../specs/2026-08-12-phenotypic-mcp-server/07-prerequisites.md`](../../specs/2026-08-12-phenotypic-mcp-server/07-prerequisites.md)

**Goal:** Close the five engine-side gaps that make the v1 tool surface
reachable. None of these is MCP code; each stands on its own and is verifiable by
the existing suite.

**Depends on:** Phase 1a complete and its exit gate green.

**These tasks are NOT mutually independent** — an earlier draft of this header
claimed they were, and it was wrong in three ways. `15`, `16` and `18` all edit
`tune/_tune_cli/_run.py`; `14` edits the same literal `10a` lifts; `11` needs
`10`'s reconciliation and `12` extends the file `11` creates. The real shape is
two chains — `10 → 11 → 12` and `10 → 14 → 17` — plus the `_run.py` group. See
[execution.md](execution.md) for the clustering built from the corrected DAG.

---

## MANDATORY CORRECTIONS — read before executing any task below

A plan review found defects in the task bodies of this document. **Where a task
below conflicts with this section, this section wins.** Each was verified against
the code; see [review-findings.md](review-findings.md) for the evidence.

### Task 10 splits into 10a / 10b / 10c (B5)

The task as written **cannot pass its own tests**, for three separate reasons:

1. `discover()` (`_services/registry.py`) is **not** eight symmetric module walks.
   It is seven `(module, category, base_class)` triples through
   `_discover_from_module`, filtered `issubclass(obj, base_class)` where
   `base_class: Type[ImageOperation]` — **plus** `analysis` through a separate
   `_discover_analyzers` walking the `SetAnalyzer` hierarchy. The new modules fit
   neither: `FilamentousFungiPipeline(PrefabPipeline)` is not an `ImageOperation`,
   and the scorers are on the scorer hierarchy. Adding module names to a list
   therefore discovers **nothing**, and all three test assertions fail.
2. **`detect.nn` stays invisible regardless.** `_discover_from_module` uses
   `inspect.getmembers(module, inspect.isclass)`, which reads `dir(module)`.
   `detect/nn/__init__.py` is a module-level `__getattr__` lazy loader **with no
   `__dir__`**, so `MicroSamDetector` is in `__all__` but never in the module dict
   until touched. It needs an `__all__`-driven getattr walk — and then the
   per-module `try/except ImportError` the task proposes sits at the wrong level,
   because the failure lands at *getattr* time inside the heavy imports.
3. **The proposed tuple reorders `detect.nn`**, three lines after instructing the
   implementer to preserve order because resolution is first-match. The real order
   is `detect, measure, enhance, refine, grid, correction, analysis, prefab, post,
   detect.nn, tune, tune.score, tune.strategy`. **Copy it from the file, not from
   the task body.**

Execute as three tasks:

| | Scope | Test |
|---|---|---|
| **10a** | Lift the `submodules` literal to `PHENOTYPIC_CLASS_MODULES`; both consumers read it. **Zero behaviour change.** | `test_one_shared_module_list` only |
| **10b** | Give prefabs, scorers and strategies their category + base class so `discover()` can actually find them | `test_registry_reaches_prefabs...`, `test_scorers_and_strategies...` |
| **10c** | `__all__`-driven getattr walk so lazy modules (`detect.nn`) are discoverable; guard at getattr level | the `MicroSamDetector` assertion |

10a is the cheap one and unblocks Task 14. **Do not start 10b/10c until 10a is committed** — it is the only part with no behavioural risk.

### Task 15: the tests do not match `run_tuning`, and the guard is misplaced (B9)

- `run_tuning(spec, images, output_dir, *, ...)` — **`images` is a required
  positional.** All three tests omit it, so they raise `TypeError` instead of
  exercising the assertion under test. Add it.
- `--slurm` additionally requires `spec_path` and `images_dir`
  (`_validate_slurm_request`), and there is an `assert effective_storage_url is not None`
  before the branch. A test that gets past the guard must satisfy those.
- `test_screen_alone_still_works` as written runs a **real local tuning study**.
  That is not a unit test; stub the engine.
- **Placement:** the task raises immediately before `if slurm:`, but
  `_write_run_marker` has already run by then, so a *refused* run leaves artifacts
  behind. Put the guard in `_validate_slurm_request` — whose own docstring says
  "Reject unsupported SLURM combinations **before any run artifact is written**" —
  and give it a `screen: bool` parameter.

### Task 16: pick one merge point (B4), and the CLI is argparse (B3)

The task's Step 3 says to merge the four legacy flags **inside**
`_submit_slurm_fleet`, while `test_legacy_flags_still_work` asserts the merge
already happened **above** that boundary. Both cannot hold.

**Decided:** merge in `run_tuning`; `_submit_slurm_fleet` takes one `slurm_args`
dict, deleting its four `slurm_*` parameters and the `if ... is not None` chain.
This changes `run_tuning`'s signature and its call site — say so in the commit.

**Also (B3, decided by the user):** the tune CLI is **argparse, not Click** —
there is no `cli` object, so every `CliRunner().invoke(...)` in the task is
unrunnable; rewrite against `main([...])` / `_build_parser().parse_args(...)`.
`--slurm` becomes `action="append"` with presence implying submission, matching
`python -m phenotypic`. **A bare `--slurm` must keep working and keep meaning
"submit"** — it ships today as a boolean and scripts rely on it. Wrap
`parse_slurm_args`' `click.BadParameter` into `parser.error(...)`.

**Edit `src/phenotypic/_services/argv.py`, not `gui/tune/_run_argv.py`** — Task 8
already promoted the tune argv builder, and the GUI path is now a 15-line shim.

### Task 18: re-load the calibration images (B7), and one test cannot fail (I4)

`_finalize_generalization` takes `(winner, spec, output_dir, split, images,
images_by_name)` — the last three being **loaded `GridImage` objects**.
`finalize_distributed_study(output_dir)` has none of them. **Decided:** re-load
them; the run marker records what is needed to re-scan. Dropping the step would
leave `generalization.json` unwritten and make the held-out `gap` permanently
null for every distributed study, which §8.3 relies on to detect an arm that won
by overfitting.

Also: `order.index(...) == 2` passes **even if step 4 is silently dropped** — add
`assert len(order) == 4`. And note `_finalize_pareto_outputs` **does** have a test
call site (`tests/unit/tune/test_run_tuning_pareto.py`), contrary to the task's
claim, so moving the functions breaks it.

### Task 14: one test cannot fail for its stated reason (I1a)

`test_expected_vs_detected_keeps_its_shipped_field_name` asserts that
`ExpectedVsDetectedCount(expected_counts_csv="x.csv")` raises. It does — but for
**missing required `metadata`**, and it would still raise if someone added
`expected_counts_csv` as an alias, which is the exact change the test claims to
guard. Replace with:

```python
assert "expected_counts_csv" not in ExpectedVsDetectedCount.model_fields
assert "metadata" in ExpectedVsDetectedCount.model_fields
```

### Task 13: the directory is `tests/unit/sdk_`, with a trailing underscore

Mirrors `src/phenotypic/sdk_`. `tests/unit/sdk` does not exist.


---

## File structure this phase creates

| File | Responsibility |
|---|---|
| `src/phenotypic/_services/catalog.py` | JSON-serializable operation descriptor + `header_scheme()`-dispatching column derivation |
| `src/phenotypic/subset/__init__.py` | Public exports: `SubsetSelector`, the three selectors, `SubsetSelection` |
| `src/phenotypic/subset/_selector.py` | `SubsetSelector` ABC + `SubsetSelection` result type |
| `src/phenotypic/subset/_selectors.py` | `RandomSubsetSelector`, `MetadataGroupSubsetSelector`, `EmbeddingSubsetSelector` |
| `src/phenotypic/_services/staging.py` | Subset staging: `flat/` + `nested/` symlink trees, digest-keyed |
| `src/phenotypic/tune/_tune_cli/_finalize.py` | The distributed finalize entry point |
| `src/phenotypic/sdk_/_io_constants.py` | Gains `directory_digest` |
| `src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py` | Gains the shared module-list constant |

---

## P3 — Catalog reconciliation and the JSON descriptor

### Task 10: One enumeration list

**Files:**
- Modify: `src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py:645` (DR2 — note the `_pipeline_parts/` segment the spec omits)
- Modify: `src/phenotypic/_services/registry.py:198-205`
- Test: `tests/unit/services/test_catalog_reconciliation.py`

**The bug this fixes.** `OperationRegistry.discover()` walks eight modules —
`enhance, detect, refine, correction, measure, grid, post, analysis` (verified at
`:198-205`). `_find_class_in_phenotypic` resolves those **plus** `prefab`,
`tune`, `tune.score`, `tune.strategy`, and `detect.nn`. So a class the pipeline
loader can deserialize is invisible to the catalog the agent browses. Without
`detect.nn` the entire staged-GPU path is unreachable from the server — a
headline CLI capability silently amputated — and without `prefab` the
prefab-first procedure (§9.4) has nothing to list.

**Interfaces:**
- Produces: `PHENOTYPIC_CLASS_MODULES: tuple[str, ...]` — the single source both
  consumers read.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_catalog_reconciliation.py
"""One list, two consumers. Two lists is how detect.nn went missing."""

def test_one_shared_module_list():
    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        PHENOTYPIC_CLASS_MODULES,
    )

    for expected in (
        "phenotypic.enhance",
        "phenotypic.detect",
        "phenotypic.detect.nn",
        "phenotypic.prefab",
        "phenotypic.tune.score",
        "phenotypic.tune.strategy",
    ):
        assert expected in PHENOTYPIC_CLASS_MODULES

def test_registry_reaches_prefabs_and_nn_detectors():
    from phenotypic._services.registry import get_registry

    names = {op.name for op in get_registry().all_operations()}
    assert "FilamentousFungiPipeline" in names, "prefabs unreachable from the catalog"
    assert "MicroSamDetector" in names, "detect.nn unreachable — staged GPU is invisible"

def test_scorers_and_strategies_are_catalog_citizens():
    """§3.1: without these, an agent authoring a spec can only guess."""
    from phenotypic._services.registry import get_registry

    names = {op.name for op in get_registry().all_operations()}
    assert {"QCScorer", "SupervisedScorer", "ReferenceFreeScorer"} <= names
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/services/test_catalog_reconciliation.py -v`
Expected: FAIL — `ImportError: cannot import name 'PHENOTYPIC_CLASS_MODULES'`.

- [ ] **Step 3: Define the constant and point both consumers at it**

In `_serializable_pipeline.py`, lift the `submodules = [...]` literal at `:645`
to a module-level constant and keep `_find_class_in_phenotypic` reading it:

```python
PHENOTYPIC_CLASS_MODULES: tuple[str, ...] = (
    "phenotypic.detect",
    "phenotypic.detect.nn",
    "phenotypic.measure",
    "phenotypic.enhance",
    "phenotypic.refine",
    "phenotypic.grid",
    "phenotypic.correction",
    "phenotypic.analysis",
    "phenotypic.prefab",
    "phenotypic.post",
    "phenotypic.tune",
    "phenotypic.tune.score",
    "phenotypic.tune.strategy",
)
```

Preserve the existing order for any module already listed — resolution is
first-match, so reordering can change which class a duplicate name resolves to.
Then rewrite `discover()`'s eight hard-coded `import phenotypic.X as X_module`
lines to iterate `PHENOTYPIC_CLASS_MODULES` via `importlib.import_module`,
keeping the laziness Task 3 preserved. Guard each import with `try/except
ImportError` and record the skip: `detect.nn` pulls optional heavy dependencies,
and the catalog must degrade to "that family is unavailable here" rather than
failing to build at all.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/services -q && uv run pytest tests/unit/core -q`
Expected: PASS.

- [ ] **Step 5: Prove it can fail**

Delete `"phenotypic.detect.nn"` from the tuple; confirm
`test_registry_reaches_prefabs_and_nn_detectors` FAILS. Restore.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "fix(catalog): reconcile discovery with the loader's module list"
```

---

### Task 11: The JSON operation descriptor

**Files:**
- Create: `src/phenotypic/_services/catalog.py`
- Test: `tests/unit/services/test_operation_descriptor.py`

**Interfaces:**
- Produces: `describe_operation(name: str, *, verbose: bool = False) -> dict`,
  `list_operations(category: str | None, query: str | None, limit: int) -> dict`

**Three corrections over the spec's first draft, each already verified there —
do not re-litigate them:**

1. **`constraints` uses JSON Schema keywords, not pydantic `Field()` kwargs.**
   `FlattenIllumination.sigma`, declared `Field(200.0, gt=0.0)`, reports
   `"exclusiveMinimum": 0.0`. Pass the real keywords through; do not invent a
   `gt`/`le` spelling.
2. **Descriptions are long** — `BlurGauss.sigma`'s runs ~180 characters over four
   sentences, and `FilamentousFungiDetector` has 20 params. Default to the
   **first sentence**; full text only behind `verbose`.
3. **No `tunable` field.** `ParamInfo` carries no suggested domain, and both
   `infer_search_space` and `pipeline_targets` require a *positioned* pipeline —
   knobs are keyed `"<position>.<field>"`. Tunability is a property of an
   operation *in a pipeline*, so it belongs to `tune_space` (Phase 2B), not here.

Two gaps the raw schema cannot express, filled from `OperationInfo`/`ParamInfo`:
`OperationField` erases to `Any` (so operation-valued params appear as untyped
`{}` — use the `_OperationFieldMarker` walk for `is_operation`/`is_pipeline`),
and `NdArrayField`'s schema is a bare `{"type":"array","items":{}}` with no shape
or dtype (flag it `type: "ndarray"`; it is not agent-authorable in practice).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_operation_descriptor.py
def test_constraints_are_json_schema_keywords():
    """Field(200.0, gt=0.0) reports exclusiveMinimum, not gt."""
    from phenotypic._services.catalog import describe_operation

    desc = describe_operation("FlattenIllumination")
    sigma = next(p for p in desc["params"] if p["name"] == "sigma")
    assert sigma["constraints"] == {"exclusiveMinimum": 0.0}
    assert "gt" not in sigma["constraints"]

def test_description_defaults_to_first_sentence():
    from phenotypic._services.catalog import describe_operation

    terse = describe_operation("BlurGauss")
    verbose = describe_operation("BlurGauss", verbose=True)
    t = next(p for p in terse["params"] if p["name"] == "sigma")["description"]
    v = next(p for p in verbose["params"] if p["name"] == "sigma")["description"]
    assert t.count(".") == 1
    assert len(v) > len(t)

def test_no_tunable_field_on_a_class():
    from phenotypic._services.catalog import describe_operation

    for param in describe_operation("BlurGauss")["params"]:
        assert "tunable" not in param
        assert "suggested_domain" not in param

def test_operation_valued_params_are_flagged():
    """OperationField erases to Any; the schema alone cannot say this."""
    from phenotypic._services.catalog import describe_operation

    desc = describe_operation("FilamentousFungiDetector")
    nested = [p for p in desc["params"] if p["is_operation"]]
    assert nested, "nested operation params must be discoverable"

def test_json_schema_is_verbatim():
    from phenotypic._services.catalog import describe_operation
    from phenotypic.enhance import BlurGauss

    assert describe_operation("BlurGauss")["json_schema"] == BlurGauss.model_json_schema()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/services/test_operation_descriptor.py -v`
Expected: FAIL — `No module named 'phenotypic._services.catalog'`.

- [ ] **Step 3: Write the projection**

```python
# src/phenotypic/_services/catalog.py
def describe_operation(name: str, *, verbose: bool = False) -> dict:
    """Project an OperationInfo into the agent-facing descriptor.

    Args:
        name: Operation, scorer, or strategy class name.
        verbose: Return each parameter's full docstring text instead of its
            first sentence.

    Returns:
        A JSON-serializable dict carrying the verbatim ``model_json_schema()``
        plus the two facts that schema cannot express — whether a parameter
        takes an operation, and whether it is a raw array.
    """
```

Implement `name`, `category`, `module`, `doc`, `json_schema`, `params[]`, and
`layers_modified`. Each param carries `name`, `type`, `default`, `required`,
`description`, `constraints`, `is_operation`, `is_pipeline`, `is_list`,
`is_optional`, `choices`, `column_ref`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/services/test_operation_descriptor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(catalog): add the JSON operation descriptor projection"
```

---

### Task 12: Column derivation that dispatches on `header_scheme()`

**Files:**
- Modify: `src/phenotypic/_services/catalog.py`
- Test: `tests/unit/services/test_column_derivation.py`

**A blanket `get_headers()` is wrong, and for one measurer it raises.**
`TEXTURE` (`schema/_texture.py:144-181`) overrides
`get_headers(scale, matrix_name=None)` with **no default for `scale`**:

```
SIZE.header_scheme()     -> "static"   -> get_headers()  -> ['Size_Area', …]
TEXTURE.header_scheme()  -> "texture"  -> get_headers()  -> TypeError:
                                          missing 1 required positional argument: 'scale'
TEXTURE.get_headers(5)   -> ['Texture_AngularSecondMoment-deg000-scale05', …]
```

`MeasureTexture` emits one header per (member × angle × scale) — **130 columns**
for `scale=[5,10]`, not the 13 base labels. So the derivation must read
`header_scheme()` per class and dispatch: `static` → `get_headers()`; `texture` →
`get_headers(scale, matrix_name)` once per entry in the **live measurer
instance's** `scale` list, merged; `metric_qualified` → the qualified-header path.

The class list comes from the public instance method
`MeasureFeatures.get_measurement_infoclasses()` (`abc_/_measure_features.py:333`),
which is genuinely instance-dependent — `MeasureColor` includes or excludes
members based on `self.include_XYZ` / `self.include_xy`.

**Do not model this on the README generator.** `_cli_readme_generator.py:140-235`
iterates `pipeline._meas` and renders `member.value` directly, never expanding
texture headers — it under-reports texture columns in generated READMEs **today**.
Reusing it inherits the bug. (Worth fixing there separately; out of scope.)

**Interfaces:**
- Produces: `derive_columns(pipeline) -> list[str]` — what Phase 2A's
  `produces_columns` calls.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_column_derivation.py
import pytest

def test_texture_headers_expand_per_scale():
    from phenotypic import ImagePipeline
    from phenotypic._services.catalog import derive_columns
    from phenotypic.measure import MeasureTexture

    pipe = ImagePipeline(meas=[MeasureTexture(scale=[5, 10])])
    cols = derive_columns(pipe)

    assert len(cols) == 130, f"expected 130 expanded texture columns, got {len(cols)}"
    assert "Texture_AngularSecondMoment-deg000-scale05" in cols
    assert any("scale10" in c for c in cols)

def test_static_scheme_still_works():
    from phenotypic import ImagePipeline
    from phenotypic._services.catalog import derive_columns
    from phenotypic.measure import MeasureSize

    assert "Size_Area" in derive_columns(ImagePipeline(meas=[MeasureSize()]))

def test_blanket_get_headers_would_have_raised():
    """Pin the reason this dispatch exists, so nobody 'simplifies' it back."""
    from phenotypic.schema import TEXTURE

    with pytest.raises(TypeError, match="scale"):
        TEXTURE.get_headers()

def test_color_columns_follow_the_instance_not_the_class():
    from phenotypic import ImagePipeline
    from phenotypic._services.catalog import derive_columns
    from phenotypic.measure import MeasureColor

    with_xyz = derive_columns(ImagePipeline(meas=[MeasureColor(include_XYZ=True)]))
    without = derive_columns(ImagePipeline(meas=[MeasureColor(include_XYZ=False)]))
    assert len(with_xyz) > len(without)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/services/test_column_derivation.py -v`
Expected: FAIL — `cannot import name 'derive_columns'`.

- [ ] **Step 3: Implement the dispatch**

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/services/test_column_derivation.py -v`
Expected: PASS — in particular the 130-column assertion, which is the one that
catches a regression to a blanket `get_headers()`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(catalog): derive measurement columns via header_scheme dispatch"
```

---

### Task 13: A directory-level digest helper

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py`
- Test: `tests/unit/sdk_/test_directory_digest.py` (note the trailing underscore —
  the directory is `tests/unit/sdk_`, mirroring `src/phenotypic/sdk_`)

**Nothing today can do this.** `bytes_fingerprint` and `file_fingerprint`
(`:154,166`) and `pipeline_content_digest` are all **single-file**, and
`TuningSpec` records no dataset at all (`tune/_spec.py:162-171`). Without a
directory digest, `campaign_status.comparable` (§8.3) cannot detect two arms
tuned against different image sets, and the subset artifact (§10.2) has no
`parent.digest` to verify a promotion against.

**Digest format matters.** `bytes_fingerprint` / `file_fingerprint` return
`f"sha256:{...}"` while `pipeline_content_digest`
(`_cli/_cli_staged_resume.py:64-66`) returns a **bare** hexdigest. They are the
same hash over the same bytes but **do not string-compare equal**. Match the
`sha256:` prefixed form here, and document the mismatch at the call site.

**Interfaces:**
- Produces: `directory_digest(root: Path, *, relative_to: Path | None = None) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/sdk_/test_directory_digest.py
def test_digest_is_stable_and_prefixed(tmp_path):
    from phenotypic.sdk_._io_constants import directory_digest

    (tmp_path / "a.tif").write_bytes(b"aaa")
    (tmp_path / "b.tif").write_bytes(b"bbb")

    first = directory_digest(tmp_path)
    assert first.startswith("sha256:")
    assert first == directory_digest(tmp_path)

def test_digest_changes_when_a_file_is_added(tmp_path):
    from phenotypic.sdk_._io_constants import directory_digest

    (tmp_path / "a.tif").write_bytes(b"aaa")
    before = directory_digest(tmp_path)
    (tmp_path / "c.tif").write_bytes(b"ccc")
    assert directory_digest(tmp_path) != before

def test_digest_ignores_listing_order(tmp_path, monkeypatch):
    """Sorted by relative path, so filesystem order cannot leak in."""
    from phenotypic.sdk_ import _io_constants

    (tmp_path / "z.tif").write_bytes(b"z")
    (tmp_path / "a.tif").write_bytes(b"a")
    forward = _io_constants.directory_digest(tmp_path)

    real_glob = _io_constants.Path.rglob
    monkeypatch.setattr(
        _io_constants.Path, "rglob",
        lambda self, pat: reversed(list(real_glob(self, pat))),
    )
    assert _io_constants.directory_digest(tmp_path) == forward
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_directory_digest.py -v`
Expected: FAIL — `cannot import name 'directory_digest'`.

- [ ] **Step 3: Implement it**

A stable digest over sorted `(relative path, size, mtime_ns)` per file is
sufficient and cheap — it does not read file contents, which matters on a
480-image parent.

- [ ] **Step 4: Run the tests** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(sdk): add a directory-level content digest"
```

---

### Task 14: The `phenotypic/subset/` subpackage

**Files:**
- Create: `src/phenotypic/subset/{__init__,_selector,_selectors}.py`
- Create: `tests/unit/subset/__init__.py` (empty — test subdirs are packages here)
- Modify: `_serializable_pipeline.py` — add `"phenotypic.subset"` to `PHENOTYPIC_CLASS_MODULES`
- Test: `tests/unit/subset/test_selectors.py`

Selectors follow the same pattern as every other extensible class here: a
pydantic ABC, concrete subclasses, `{class, params}` serialization, resolution by
bare class name. Adding a fourth is then a subclass plus one `__init__.py`
export — no tool signature changes, no schema bump.

**Interfaces:**
- Produces: `SubsetSelector` (`.select`, `.availability`, `.cost_class`,
  **`group_filter`**), `SubsetSelection` (carrying `group_filter`),
  `RandomSubsetSelector`, `MetadataGroupSubsetSelector`, `EmbeddingSubsetSelector`

```python
class SubsetSelector(BaseModel, ABC):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n: int = Field(..., ge=1)
    seed: int = 0
    group_filter: dict[str, str] = Field(default_factory=dict)

    @abstractmethod
    def _select(self, candidates: list[ImageRef]) -> list[str]: ...

    def availability(self) -> tuple[bool, str]: ...
    def cost_class(self) -> Literal["W0", "W1", "W2"]: ...

    def select(self, candidates: list[ImageRef]) -> SubsetSelection:
        """Template: apply group_filter to the candidates, check
        availability, delegate to _select, then dedup, order, and record the
        rationale so the artifact explains itself."""
```

**`group_filter` ships on the ABC in this task, not later** (spec §10.3,
USER-24). It is a `{metadata column: value}` map applied to the candidate set
**before** `_select` runs, so it is implemented once in `select()`'s template and
no subclass can skip it or reimplement it.

Three reasons it cannot be deferred to Phase 2:

1. `model_config = ConfigDict(extra="forbid")` means it is **not addable as an
   extra key**. Adding it later is a model change to the ABC and to every
   selector's serialized `params`, i.e. a schema bump on an artifact that will by
   then exist on disk.
2. Spec §10.2 records it on the subset artifact and §5.4's plan token **binds**
   it at `scope:"full"` — USER-21's guarantee that an ack given for one group's
   images cannot be spent on another's. Without the field there is nothing to
   record and nothing to bind.
3. It is the *only* multi-group primitive that survived USER-24's offload of
   grouping strategy to the agent. If it does not land here it does not land.

Contract:

| Aspect | Rule |
|---|---|
| Application point | `select()`'s template, before `_select`; conjunctive over all `{column: value}` pairs |
| Column source | The same `grouping_metadata` CSV, joined by parent-relative path. A non-empty `group_filter` with no CSV configured is a construction error |
| Comparison | String comparison against the CSV cell, after the `Metadata_` canonicalization — **never** `startswith("Metadata_")` or prefix splitting (project rule) |
| Column absent | Raise so the tool layer maps it to `group_filter_column_not_found`, carrying the CSV's column list |
| Matches nothing | Raise so the tool layer maps it to `group_filter_matches_nothing`. **Not** an empty selection: an empty subset passes every downstream shape check and produces a study of nothing |
| Recorded | Copied onto `SubsetSelection` and thence to the artifact's top-level `group_filter` *and* `selection.params` (§10.2) |

Two tests belong in `test_selectors.py` alongside the existing three: a filtered
`RandomSubsetSelector` selects only from the filtered candidates (proving the
filter is on the ABC and composes with a selector that knows nothing about
metadata), and a filter naming an absent column raises rather than silently
selecting from everything.

**`MetadataGroupSubsetSelector` performs its own CSV→filename join. It does NOT
reuse `_resolve_groups`.** Verified by reproduction: `_resolve_groups`
(`tune/_evaluation/_split.py:114-133`) is a pure in-memory
`image.metadata.get(group_key)` lookup with no CSV and no join, and a freshly
read image carries only `MetadataImage_{BitDepth,FileSuffix,ImageName,ImageType,UUID}`
— so `img.metadata.get("Metadata_Batch")` returns `None` and `derive_split`
falls through to a weaker tier **silently**. External CSV columns reach data only
via `join_metadata`, which operates on the measurement DataFrame after a full
run. The selector reads `grouping_metadata`, joins rows to images by
parent-relative path, and stratifies on that.

**The field is `grouping_metadata`, and this naming does NOT extend to
`QCScorer.check.metadata`.** Three different CSVs appear in this design and
passing the wrong one at the scorer produces a meaningless objective rather than
an error — hence the distinct name here. But `ExpectedVsDetectedCount.metadata`
**ships today with no alias** (`analysis/qc/_expected_vs_detected.py:208`), so it
keeps its name. Verified: `ExpectedVsDetectedCount(metadata=…)` succeeds and
`ExpectedVsDetectedCount(expected_counts_csv=…)` raises `ValidationError`. **Two
separate reviewers proposed "fixing" this. Do not.**

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/subset/test_selectors.py
import pytest

def test_random_is_seeded_and_reproducible(image_refs):
    from phenotypic.subset import RandomSubsetSelector

    a = RandomSubsetSelector(n=6, seed=0).select(image_refs)
    b = RandomSubsetSelector(n=6, seed=0).select(image_refs)
    assert a.images == b.images
    assert len(a.images) == 6

def test_metadata_group_equal_allocation(tmp_path, image_refs):
    from phenotypic.subset import MetadataGroupSubsetSelector

    csv = tmp_path / "batches.csv"
    csv.write_text("image,Metadata_Batch\n" + "\n".join(
        f"{r.relative_path},{'rare' if i < 2 else 'common'}"
        for i, r in enumerate(image_refs)))

    sel = MetadataGroupSubsetSelector(
        n=4, seed=0, grouping_metadata=str(csv),
        group_key="Metadata_Batch", allocation="equal")
    result = sel.select(image_refs)
    assert len(result.images) == 4
    assert result.method == "MetadataGroupSubsetSelector"

def test_embedding_selector_raises_and_never_degrades(image_refs):
    """A placeholder that silently returns random would stamp
    method: EmbeddingSubsetSelector onto an artifact with none of the
    claimed visual coverage, and nothing downstream could contradict it."""
    from phenotypic.subset import EmbeddingSubsetSelector

    sel = EmbeddingSubsetSelector(n=4, seed=0)
    available, why = sel.availability()
    assert available is False
    assert "not implemented" in why.lower()
    with pytest.raises(NotImplementedError):
        sel.select(image_refs)

def test_embedding_cost_class_is_w2_even_unimplemented():
    from phenotypic.subset import EmbeddingSubsetSelector

    assert EmbeddingSubsetSelector(n=4).cost_class() == "W2"

def test_selectors_resolve_by_bare_class_name():
    from phenotypic._core._pipeline_parts._serializable_pipeline import (
        SerializablePipeline,
    )

    assert SerializablePipeline._find_class_in_phenotypic("RandomSubsetSelector")

def test_expected_vs_detected_keeps_its_shipped_field_name():
    """Guards the rename two reviewers proposed. It does not exist."""
    import pydantic

    from phenotypic.analysis.qc import ExpectedVsDetectedCount

    with pytest.raises(pydantic.ValidationError):
        ExpectedVsDetectedCount(expected_counts_csv="x.csv")
```

Add an `image_refs` fixture to `tests/unit/subset/conftest.py` producing ~12
`ImageRef`s across two parent-relative subdirectories.

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/subset -v`
Expected: FAIL — `No module named 'phenotypic.subset'`.

- [ ] **Step 3: Implement the ABC and three selectors**

- [ ] **Step 4: Run the tests** — Expected: PASS.

- [ ] **Step 5: Prove the embedding guard can fail**

Make `EmbeddingSubsetSelector._select` return a random sample; confirm
`test_embedding_selector_raises_and_never_degrades` FAILS. Revert.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(subset): add the SubsetSelector hierarchy"
```

---

## P4 — Close the `--screen` + `--slurm` silent no-op

### Task 15: Make the dropped combination an error

**Files:**
- Modify: `src/phenotypic/tune/_tune_cli/_run.py:593-595,623`
- Test: `tests/unit/tune/test_screen_slurm_guard.py`

**The bug, verified on this branch.** `run_tuning` hits
`if slurm: return _submit_slurm_fleet(...)` at `:593-595` and returns **before**
reaching `if screen:` at `:623`; `_worker.py`'s `run_worker` constructs
`TuningEngine(...).optimize(...)` with no `ScreeningController` at all. So
`--screen --slurm` today silently drops screening — no error, no warning, the
full unscreened space runs on the fleet.

This must become an explicit error **before** the MCP server exposes screening,
since an agent that asked for screening and got an unscreened fleet has been
given silently different behaviour than it requested.

**Interfaces:**
- Produces: a `ValueError` (surfaced by Phase 2B as
  `screening_unsupported_on_slurm`)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_screen_slurm_guard.py
import pytest

def test_screen_plus_slurm_is_refused(minimal_tuning_spec, tmp_path):
    from phenotypic.tune._tune_cli._run import run_tuning

    with pytest.raises(ValueError, match="screen.*slurm|slurm.*screen"):
        run_tuning(spec=minimal_tuning_spec, output_dir=tmp_path,
                   screen=True, slurm=True)

def test_screen_alone_still_works(minimal_tuning_spec, tmp_path):
    from phenotypic.tune._tune_cli._run import run_tuning

    run_tuning(spec=minimal_tuning_spec, output_dir=tmp_path, screen=True, slurm=False)

def test_slurm_alone_still_submits(minimal_tuning_spec, tmp_path, fake_sbatch):
    from phenotypic.tune._tune_cli._run import run_tuning

    run_tuning(spec=minimal_tuning_spec, output_dir=tmp_path, screen=False, slurm=True)
    assert fake_sbatch.called
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/tune/test_screen_slurm_guard.py -v`
Expected: FAIL — no exception raised; the call silently submits an unscreened
fleet. **That failure is the bug, reproduced.**

- [ ] **Step 3: Add the guard**

Raise **before** the `if slurm:` early return at `:593`, so the message names
both flags and the fact that screening is not implemented on the fleet path:

```python
if slurm and screen:
    raise ValueError(
        "--screen is not supported with --slurm: the fleet path returns before "
        "the screening round runs, and the SLURM worker constructs no "
        "ScreeningController. Run screening locally, or drop --screen."
    )
```

- [ ] **Step 4: Run the tests** — Expected: PASS, all three.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "fix(tune): refuse --screen with --slurm instead of dropping it silently"
```

---

## P5 — Give the tune CLI a `--slurm key=value` surface

### Task 16: One profile, both engines

**Files:**
- Modify: `src/phenotypic/tune/__main__.py:104-125`
- Modify: `src/phenotypic/tune/_tune_cli/_run.py:724-736,798-804` (DR3 — offsets shifted from the spec's `797-805`)
- Test: `tests/unit/tune/test_tune_slurm_kv.py`

**Why this moved into Phase 1 (decision D2).** `python -m phenotypic` accepts
free-form repeated `--slurm key=value` (`phenotypicCLI.py:795`);
`python -m phenotypic.tune` accepts only four discrete flags —
`--slurm-partition`, `--slurm-mem`, `--slurm-time`, `--slurm-constraint` — and
`_submit_slurm_fleet` builds its `slurm_args` from those alone. So **`account`,
`qos`, `cpus_per_task`, and `gpus_per_node` cannot reach a tune fleet at all.**

On UCR HPCC that is not cosmetic: `--account` is **mandatory** for the `exfab`
and `preempt` partitions, so today no tune fleet can reach the GPU node or the
preempt pool. Where `account` is optional the work is silently billed to the
default account instead.

Both engines already funnel into the same `format_sbatch_directives`
(`sdk_/slurm/_sbatch.py:102`), which handles arbitrary keys — the narrowing is
purely in the tune CLI's argument surface. Landing this collapses one profile to
both paths and makes §5.2.1's expressibility check vestigial, so Phase 2B builds
one fewer guard.

**Interfaces:**
- Produces: `--slurm key=value` (repeatable) on the tune CLI; the four existing
  flags stay as sugar.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_tune_slurm_kv.py
import pytest
from click.testing import CliRunner

def test_account_reaches_the_fleet(monkeypatch, minimal_spec_file, tmp_path):
    """The whole point: exfab and preempt are unreachable without --account."""
    from phenotypic.tune.__main__ import cli

    captured = {}
    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run._submit_slurm_fleet",
        lambda **kw: captured.update(kw) or {},
    )
    result = CliRunner().invoke(cli, [
        "run", str(minimal_spec_file), "-i", str(tmp_path), "-o", str(tmp_path / "out"),
        "--slurm", "slurm_account=exfab",
        "--slurm", "slurm_partition=exfab",
        "--slurm", "slurm_cpus_per_task=8",
    ])
    assert result.exit_code == 0, result.output
    assert captured["slurm_args"]["slurm_account"] == "exfab"
    assert captured["slurm_args"]["slurm_cpus_per_task"] == 8

def test_legacy_flags_still_work(monkeypatch, minimal_spec_file, tmp_path):
    from phenotypic.tune.__main__ import cli

    captured = {}
    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run._submit_slurm_fleet",
        lambda **kw: captured.update(kw) or {},
    )
    result = CliRunner().invoke(cli, [
        "run", str(minimal_spec_file), "-i", str(tmp_path), "-o", str(tmp_path / "out"),
        "--slurm-partition", "batch", "--slurm-mem", "16G",
    ])
    assert result.exit_code == 0, result.output
    assert captured["slurm_args"]["slurm_partition"] == "batch"

def test_explicit_kv_wins_over_the_sugar_flag(monkeypatch, minimal_spec_file, tmp_path):
    """Both spellings present: the specific one wins, and it is documented."""
    from phenotypic.tune.__main__ import cli

    captured = {}
    monkeypatch.setattr(
        "phenotypic.tune._tune_cli._run._submit_slurm_fleet",
        lambda **kw: captured.update(kw) or {},
    )
    CliRunner().invoke(cli, [
        "run", str(minimal_spec_file), "-i", str(tmp_path), "-o", str(tmp_path / "out"),
        "--slurm-partition", "batch", "--slurm", "slurm_partition=epyc",
    ])
    assert captured["slurm_args"]["slurm_partition"] == "epyc"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/tune/test_tune_slurm_kv.py -v`
Expected: FAIL — `no such option: --slurm`.

- [ ] **Step 3: Add the option and widen the plumbing**

Add `--slurm` as `multiple=True` on the tune CLI, parsed by the **existing**
`parse_slurm_args` (`_cli/_cli_utils.py:336-375`) so both engines share one
parser. Widen `_submit_slurm_fleet` to accept a `slurm_args: dict` and merge the
four legacy flags into it, with explicit `key=value` taking precedence.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/tune -q`
Expected: PASS.

- [ ] **Step 5: Update the docs**

`tune_distributed_hpcc.md` documents the four-flag surface. Add the `key=value`
form and an `exfab` example carrying `slurm_account=exfab`.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(tune): accept --slurm key=value so account and qos reach the fleet"
```

---

## P6 — Subset staging

### Task 17: Materialize a subset as two directory layouts

**Files:**
- Create: `src/phenotypic/_services/staging.py`
- Test: `tests/unit/services/test_subset_staging.py`

**Neither engine accepts a file list**, so §10's whole subset boundary depends on
this. Verified: `tune`'s `-i` is documented "image directory"
(`tune/__main__.py:49`) and `_load_images` is a **non-recursive**
`Path(input_dir).iterdir()` (`_run.py:235-279`); the forward CLI's `-i` is a
single `click.Path` with **no `multiple=True`** (`phenotypicCLI.py:721-730`)
feeding `scan_directory_structure`. There is no manifest flag and no repeated
`-i` on either. `--sample N` only randomly *thins* datasets already discovered.

**Two layouts, because the two engines want opposite things:**

```
<workspace>/.phenotypic-mcp/subset-staging/<subset-digest>/
├── flat/                 # tune — _load_images is a NON-RECURSIVE iterdir
│   ├── plateA_01.tif -> …/data/plates/plateA/plateA_01.tif
│   └── plateB_03.tif -> …
└── nested/               # deploy — Metadata_Dataset comes from subdir names
    ├── plateA/plateA_01.tif -> …
    └── plateB/plateB_03.tif -> …
```

**A single layout cannot serve both.** At the root of a *nested* directory
`_load_images` sees only subdirectories, matches zero images, and the run dies on
`SystemExit("no images found under …")` (`tune/__main__.py:202-204`).
Conversely `scan_directory_structure` derives `Metadata_Dataset` from
subdirectory names, so handing *deploy* a flat directory silently relabels every
row's dataset to the staging folder name — the exact corruption nesting prevents.

**Fidelity is a check, not just a property.** Nothing in the engines can catch a
mismatch: `scan_directory_structure` only rejects *internally* inconsistent
directories (root images **and** subdirectories together,
`_cli_directory_scanner.py:97-103`) and has no way to know what the parent looked
like. So the check lives in the builder or nowhere.

**Interfaces:**
- Produces: `stage_subset(subset, *, cache_root) -> StagedSubset` with
  `.flat: Path`, `.nested: Path`, `.link_mode: Literal["symlink", "copy"]`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/services/test_subset_staging.py
def test_flat_layout_is_a_non_recursive_iterdir_match(staged):
    """What tune's _load_images actually does."""
    files = [p for p in staged.flat.iterdir() if p.is_file()]
    assert len(files) == 3

def test_nested_layout_preserves_dataset_names(staged):
    from phenotypic._cli._cli_directory_scanner import scan_directory_structure

    datasets = scan_directory_structure(staged.nested)
    assert set(datasets) == {"plateA", "plateB"}

def test_restaging_the_same_digest_is_a_noop(subset, tmp_path):
    from phenotypic._services.staging import stage_subset

    first = stage_subset(subset, cache_root=tmp_path)
    second = stage_subset(subset, cache_root=tmp_path)
    assert first.flat == second.flat

def test_link_mode_is_reported(staged):
    assert staged.link_mode in {"symlink", "copy"}

def test_fidelity_check_rejects_a_mismatched_layout(subset, tmp_path, monkeypatch):
    """The builder must verify its own output round-trips."""
    import pytest

    from phenotypic._services import staging

    monkeypatch.setattr(staging, "_dataset_of", lambda ref: "wrong")
    with pytest.raises(ValueError, match="fidelity|dataset"):
        staging.stage_subset(subset, cache_root=tmp_path)
```

Add `subset` and `staged` fixtures building a parent with `plateA/` (2 images)
and `plateB/` (1 image).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/services/test_subset_staging.py -v`
Expected: FAIL — `No module named 'phenotypic._services.staging'`.

- [ ] **Step 3: Implement staging**

Four required properties: mirrors the parent's dataset substructure; symlinks by
default with a **copy fallback** (Windows symlink creation needs elevated
privileges or Developer Mode, and this project supports Windows) and the mode
reported; keyed by subset digest so concurrent arms share one directory rather
than racing; lives under `.phenotypic-mcp/` so `--restart`/`--overwrite`
semantics can never reach the parent images through it.

- [ ] **Step 4: Run the tests** — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(services): stage a subset as flat and nested symlink trees"
```

---

## P7 — The distributed finalize

### Task 18: A re-runnable finalize entry point

**Files:**
- Create: `src/phenotypic/tune/_tune_cli/_finalize.py`
- Test: `tests/unit/tune/test_distributed_finalize.py`

**There is no code to build from.** Verified: no `finalize`/recompute entry point
exists anywhere in `tune/` or `gui/tune/`; `_finalize_outputs` (`:945`),
`_finalize_best_params` (`:705`), `_finalize_pareto_outputs` (`:982`), and
`_finalize_generalization` (`:907`) have **no call sites outside `run_tuning`**
(only `:631` and `:637`); and the `--recompile` referenced by a stale docstring
(`_run.py:744`) **does not exist** on the tune CLI.

Consequence: a SLURM-launched study never writes `best_params.json`, and
`prepare_best_from_run` hard-requires it (`gui/tune/_export.py:75-77`, raising
`FileNotFoundError`) — so the plain export path raises on **every** distributed
study.

**The order is load-bearing** (`_run.py:628-641`):

1. `_finalize_outputs` → `trials.parquet`, `param_importance.json`, `best_pipeline.json`
2. `_finalize_pareto_outputs` → Pareto front, per-axis winners, and it
   **overwrites** `best_pipeline.json` with the knee
3. `_finalize_best_params` → `best_params.json` — **last, deliberately**
4. `_finalize_generalization` → `generalization.json`

`best_params.json` is written last **because it is the de-facto completion
marker**. Writing it first would leave an interrupted finalize looking exportable
when it is not.

**Two interruption hazards the order does not close**, both reported rather than
hidden: a kill *inside* step 2 leaves `best_pipeline.json` holding the scalar
best from step 1, which a later export could mislabel `pareto_knee` — so write a
`finalize_in_progress` marker at step 1 and clear it after step 4, and refuse
with `finalize_incomplete` if it is found. And finalize is **not** safe against a
still-running study: two concurrent calls would each compute a different
`_headline_winner` as trials land and overwrite each other — so gate on the study
being terminal (budget drained or no live scheduler jobs) and refuse with
`study_not_finished`.

**Interfaces:**
- Produces: `finalize_distributed_study(output_dir, *, force=False) -> FinalizeResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/tune/test_distributed_finalize.py
import pytest

def test_finalize_writes_what_the_slurm_branch_skipped(finished_distributed_study):
    from phenotypic.sdk_._io_constants import best_params_path
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    out = finished_distributed_study
    assert not best_params_path(out).is_file(), "fixture precondition"

    finalize_distributed_study(out)

    assert best_params_path(out).is_file()
    assert (out / "trials.parquet").is_file()

def test_best_params_is_written_last(finished_distributed_study, monkeypatch):
    """It is the completion marker; writing it early breaks export gating."""
    from phenotypic.tune._tune_cli import _finalize

    order: list[str] = []
    for name in ("_finalize_outputs", "_finalize_pareto_outputs",
                 "_finalize_best_params", "_finalize_generalization"):
        real = getattr(_finalize, name)
        monkeypatch.setattr(
            _finalize, name,
            lambda *a, _n=name, _r=real, **k: (order.append(_n), _r(*a, **k))[1],
        )
    _finalize.finalize_distributed_study(finished_distributed_study)
    assert order.index("_finalize_best_params") == 2

def test_refuses_a_running_study(running_distributed_study):
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    with pytest.raises(RuntimeError, match="not finished|still running"):
        finalize_distributed_study(running_distributed_study)

def test_interrupted_finalize_leaves_a_marker(finished_distributed_study, monkeypatch):
    from phenotypic.tune._tune_cli import _finalize

    monkeypatch.setattr(_finalize, "_finalize_pareto_outputs",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        _finalize.finalize_distributed_study(finished_distributed_study)

    assert (finished_distributed_study / "finalize_in_progress").exists()

    with pytest.raises(RuntimeError, match="finalize_incomplete|incomplete"):
        _finalize.finalize_distributed_study(finished_distributed_study)

def test_finalize_is_rerunnable(finished_distributed_study):
    from phenotypic.tune._tune_cli._finalize import finalize_distributed_study

    finalize_distributed_study(finished_distributed_study)
    finalize_distributed_study(finished_distributed_study)  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/tune/test_distributed_finalize.py -v`
Expected: FAIL — `No module named 'phenotypic.tune._tune_cli._finalize'`.

- [ ] **Step 3: Implement the entry point**

Open the store, gate on terminal, write the marker, run the four steps **in the
order above**, clear the marker. `_finalize_best_params` silently no-ops when the
winner is `None` (`_run.py:712-713`), which would otherwise surface later as a
misleading `FileNotFoundError` — detect and report it instead.

Writing `trials.parquet` here resolves OQ-4.3 affirmatively: the finalize already
holds the store open, so the marginal cost is one parquet write, and it is what
makes a distributed study directory readable offline and openable by the GUI's
parquet-only degradation path.

- [ ] **Step 4: Run the tests** — Expected: PASS, all five.

- [ ] **Step 5: Prove the ordering test can fail**

Move `_finalize_best_params` to first in the sequence; confirm
`test_best_params_is_written_last` FAILS. Revert.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(tune): add a re-runnable finalize for distributed studies"
```

---

## Phase 1b exit gate — and the Phase 2 entry gate

- [ ] `uv run pytest tests/unit -q` — green.
- [ ] `uv run pytest tests/integration -q` — green.
- [ ] `uv run pytest tests/gui -q` — green. **`tests/gui` IS in `testpaths`**
      (`pyproject.toml:200`), so CI runs it.
- [ ] CI ledger gates green: `FEATURES.md`, `WORKFLOWS.md`, smoke-capture.
- [ ] `uv run mypy src/phenotypic` — no new errors.
- [ ] The import-purity gate still collects one case per `_services` module and
      passes — `catalog.py` and `staging.py` joined the tier this phase.
- [ ] Every "prove it can fail" step was run, with the failure observed.
- [ ] **Spec updated for DR1–DR5** (README drift register), so §1.4, §3.2, §5.2.1,
      §4.2, and §10.1 match the code Phase 1 produced.

Phase 2A is written against the signatures this phase produced — `describe_operation`,
`derive_columns`, `directory_digest`, `stage_subset`, `build_array_script_spec`,
`finalize_distributed_study`, and the `_services` modules — not invented ahead of them.
