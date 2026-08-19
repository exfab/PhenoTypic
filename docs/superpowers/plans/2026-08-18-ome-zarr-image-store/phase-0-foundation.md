# Phase 0 — Foundation: dependencies, Python floor, CI, vendored NGFF schemas

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §6, §7.

**Why first:** every later phase imports `zarr`, and the conformance harness in Phase 2
imports `jsonschema` and reads the vendored schemas. Nothing else can start until the
resolution universe contains them.

**Blocks:** Phases 1–7.

---

### Task 0.1: Raise the Python floor, add `zarr`, `jsonschema` and `xmlschema`, update CI

**Files:**
- Modify: `pyproject.toml` (`requires-python` line 25, classifiers lines 27–34,
  `dependencies` line 45 region, `[dependency-groups]`, `[[tool.mypy.overrides]]`)
- Modify: `uv.lock` (regenerated, not hand-edited)
- Modify: `.github/workflows/run-pytest.yml` (header prose lines 4–7; matrix `3.10` entry)
- Modify: `.github/workflows/run-pytest-full.yml` (header prose line 10; matrix `3.10`
  entry at line 46)
- Modify: `.github/workflows/package-integrity.ci.yml` (comment line 43; matrix line 44)
- Modify: `.github/workflows/publish_to_pypi.yml` (lines 17 and 20)
- Test: `tests/unit/test_packaging_floor.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: an environment in which `import zarr` and `import jsonschema` succeed, and
  `zarr.__version__` starts with `3.`. Every later task assumes both.

**Constraints specific to this task:**
- `jsonschema` is currently **transitive only** — it is not named anywhere in
  `pyproject.toml` (verified). Spec §7 forbids a conformance check that skips on a
  missing dependency, so it must become a **declared** dependency of the test group, not
  left to chance.
- Do **not** add `ome-zarr` or `ome-zarr-models` to any group (Global Constraints).
- Ruff sets no `target-version` and mypy no `python_version`; both follow
  `requires-python`, so raising the floor may surface new `UP` lints. Fix them in this
  task, with `uv run ruff check --fix` on **explicit paths only**.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_packaging_floor.py`:

```python
"""Guards on the declared dependency universe for the OME-Zarr store.

These are packaging assertions, not behaviour tests: they fail loudly if a
future edit reintroduces Python 3.10, adopts an ome-zarr package, or lets the
NGFF conformance dependency drift back to transitive-only.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_requires_python_floor_is_311(pyproject: dict) -> None:
    assert pyproject["project"]["requires-python"] == ">=3.11, <3.13"


def test_classifiers_drop_310(pyproject: dict) -> None:
    classifiers = pyproject["project"]["classifiers"]
    assert "Programming Language :: Python :: 3.10" not in classifiers
    assert "Programming Language :: Python :: 3.11" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers


def test_zarr_is_a_runtime_dependency(pyproject: dict) -> None:
    deps = pyproject["project"]["dependencies"]
    assert any(dep.startswith("zarr") for dep in deps), deps


def test_h5py_is_retained_for_migration(pyproject: dict) -> None:
    deps = pyproject["project"]["dependencies"]
    assert any(dep.split(">")[0].split("=")[0].strip() == "h5py" for dep in deps)


def test_ome_zarr_packages_are_not_adopted_anywhere(pyproject: dict) -> None:
    """`ome-zarr-models` pins pydantic<2.13; uv resolves one universe."""
    banned = {"ome-zarr", "ome-zarr-models"}
    pools: list[list[str]] = [list(pyproject["project"]["dependencies"])]
    for group in pyproject.get("dependency-groups", {}).values():
        pools.append([item for item in group if isinstance(item, str)])
    for extra in pyproject["project"].get("optional-dependencies", {}).values():
        pools.append(list(extra))
    for pool in pools:
        for requirement in pool:
            name = (
                requirement.split(";")[0]
                .split("[")[0]
                .split(">")[0]
                .split("<")[0]
                .split("=")[0]
                .strip()
                .lower()
            )
            assert name not in banned, requirement


@pytest.mark.parametrize("package", ["jsonschema", "xmlschema"])
def test_conformance_deps_are_declared_not_transitive(
    pyproject: dict, package: str
) -> None:
    """Spec §7: a conformance check may never skip on a missing dependency.

    Parametrized, not two functions: both gates fail the same way (green
    locally, red in CI) and a new conformance dependency should be one list
    entry, not a copied test. Ledger GEN-24.
    """
    groups = pyproject.get("dependency-groups", {})
    declared = {
        requirement.split(";")[0].split(">")[0].split("<")[0].split("=")[0].strip().lower()
        for group in groups.values()
        for requirement in group
        if isinstance(requirement, str)
    }
    assert "jsonschema" in declared


def test_zarr_v3_is_importable_at_runtime() -> None:
    import zarr

    assert zarr.__version__.startswith("3."), zarr.__version__
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
uv run pytest tests/unit/test_packaging_floor.py -v
```

Expected: `test_requires_python_floor_is_311`, `test_classifiers_drop_310`,
`test_zarr_is_a_runtime_dependency`, `test_conformance_deps_are_declared_not_transitive`, and
`test_zarr_v3_is_importable_at_runtime` all FAIL (the last with `ModuleNotFoundError:
No module named 'zarr'`). `test_h5py_is_retained_for_migration` and
`test_ome_zarr_packages_are_not_adopted_anywhere` PASS already — that is correct; they are
regression guards, not new work.

- [ ] **Step 3: Edit `pyproject.toml`**

Line 25:

```toml
requires-python = ">=3.11, <3.13"
```

Classifiers — delete the `3.10` line, keep `3.11` and `3.12`:

```toml
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12"
```

Add to `[project].dependencies`, beside the existing `h5py` entry:

```toml
    "zarr>=3.0",
```

Leave `"h5py"` in place — it is the `--mode migrate` read path (Phase 5).

Add **both** conformance dependencies to the **`dev`** dependency group:

```toml
    "jsonschema>=4.0",
    "xmlschema>=3.0",
```

> **`xmlschema` is not optional and must be declared here (ledger GEN-24).** Task 2.5's
> `_ome_xsd()` imports it, and the OME-XML gate is required to *fail* rather than skip when
> its dependency is missing (spec §7) — so a transitive-only install is a CI break waiting to
> happen: the conformance tests pass locally, where something else pulled it in, and fail in
> CI. An earlier draft named the package only in Task 0.2's prose, whose `Files:` list touches
> `pyproject.toml` solely for `[tool.ruff] extend-exclude`. This is GEN-1's failure mode with
> a different package.
>
> It also needs no network at test time: `ome.xsd`'s single remote `xsd:import`
> (`w3.org/2001/xml.xsd`) resolves against `xmlschema`'s bundled fallback
> (`xmlschema/locations.py:121`), so vendoring the one file is sufficient offline. That is
> **not** a repeat of PRE-B3.

> **There is no `test` group.** Verified: `pyproject.toml` defines only `dev`,
> `test-qt`, and `docs`, and every CI lane runs
> `uv sync --group dev --group test-qt --all-extras` (`run-pytest.yml:147`,
> `run-pytest-full.yml:83,119,154`). Creating a new `test` group would leave
> `jsonschema` uninstalled in **every** CI lane — so the conformance harness
> would fail there while passing locally, which spec §7 forbids. Note that
> `test_conformance_deps_are_declared_not_transitive` scans all groups and passes
> either way, so it does not catch this. If a new group is ever added, the
> `uv sync` line in all four workflows must be updated with it.

Add a mypy override beside the existing `h5py` / `mahotas` ones so an untyped `zarr`
does not fail the type gate:

```toml
[[tool.mypy.overrides]]
module = [
    "zarr",
    "zarr.*",
]
ignore_missing_imports = true
```

- [ ] **Step 4: Edit the four CI workflows**

`.github/workflows/run-pytest.yml` — header prose (lines 4–7) and the matrix. Replace the
`3.10` floor entry with `3.11` in **both** places:

```yaml
# Matrix: Linux x Python {3.11 (floor), 3.12 (ceiling)}. Windows, macOS, and
# any intermediate Python move to the nightly full lane in
# ``run-pytest-full.yml``.
```

```yaml
          # Ubuntu: floor (3.11) + ceiling (3.12).
          - os: ubuntu-latest
            python-version: "3.11"
          - os: ubuntu-latest
            python-version: "3.12"
```

`.github/workflows/run-pytest-full.yml` — line 10 prose becomes
`#   * Linux x Python {3.11, 3.12}`; delete the `python-version: "3.10"` matrix entry at
line 46 together with its sibling `os:` key.

`.github/workflows/package-integrity.ci.yml` lines 43–44:

```yaml
        # Matches requires-python (>=3.11, <3.13). The <3.13 ceiling is
        # mahotas 1.4.18 (no cp313 wheel), not zarr.
        python-version: ["3.11", "3.12"]
```

`.github/workflows/publish_to_pypi.yml` lines 17 and 20:

```yaml
       - name: Set up Python 3.11
         uses: actions/setup-python@v4
         with:
           python-version: '3.11'
```

Keep `@v4` — the file pins it at line 18, and bumping the action inside a
Python-version edit is an unrequested change riding along.

Leave the testmon cache key at `run-pytest.yml:153` alone — it already keys on
`hashFiles('uv.lock')`, and Step 5 changes `uv.lock`, which invalidates it correctly
without an edit.

- [ ] **Step 5: Regenerate the lock and sync**

```bash
uv lock
uv sync --group dev --group test-qt --group docs --extra gui --extra napari
uv run python -c "import zarr, jsonschema, xmlschema; print(zarr.__version__)"
```

Expected: a `3.x` version string. Markers resolve zarr to 3.1.6 on 3.11 and 3.3.x on
3.12 with no pinning; do **not** pin it.

- [ ] **Step 6: Run the packaging test and the lint/type gates**

```bash
uv run pytest tests/unit/test_packaging_floor.py -v
uv run ruff check --fix pyproject.toml src/phenotypic tests/unit/test_packaging_floor.py
uv run mypy src/phenotypic
```

Expected: all packaging tests PASS. Ruff may report new `UP` lints now that the floor is
3.11 (e.g. `UP007` union syntax) — fix them; that churn belongs to this task, not to a
later one. If ruff rewrites files outside `src/phenotypic`, `git status` and revert them
before committing.

- [ ] **Step 7: Run the full unit suite to catch 3.11-floor fallout**

```bash
uv run pytest tests/unit -x -q
```

Expected: PASS. A failure here is real fallout from the floor raise (removed 3.10
compatibility shims), not from zarr, which nothing imports yet.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .github/workflows tests/unit/test_packaging_floor.py src/phenotypic
git commit -m "build: raise Python floor to 3.11 and add zarr>=3.0

Drops Python 3.10 across pyproject, the four CI workflows, and the lock.
Adds zarr>=3.0 as a runtime dependency and promotes jsonschema from a
transitive to a declared test dependency, because spec §7 forbids a
conformance check that skips on a missing dependency. h5py is retained
for the --mode migrate read path. The <3.13 ceiling is mahotas 1.4.18,
not zarr; that is now stated where the cap is spelled."
```

---

### Task 0.2: Vendor the NGFF 0.5 JSON schemas as read-only reference material

**Files:**
- Create: `tests/fixtures/ngff/0.5/image.schema`
- Create: `tests/fixtures/ngff/0.5/label.schema`
- Create: `tests/fixtures/ngff/0.5/ome.schema`
- Create: `tests/fixtures/ngff/0.5/_version.schema`
- Create: `tests/fixtures/ome/2016-06/ome.xsd`
- Create: `tests/fixtures/ngff/0.5/SOURCE.md`, `tests/fixtures/ome/2016-06/SOURCE.md`
- Modify: `pyproject.toml` (`[tool.ruff] extend-exclude`)
- Test: `tests/unit/test_ngff_schema_fixtures.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `tests/fixtures/ngff/0.5/{image,label,ome,_version}.schema` — read by the
  conformance harness `assert_store_conforms(...)` introduced in Task 2.5.

**Constraints specific to this task:**
- **A fifth vendored file: `ome.xsd`.** NGFF §2.2.3 makes `OME/METADATA.ome.xml` a
  conditional MUST — it *"MUST adhere to the OME-XML specification but MUST use
  `<MetadataOnly/>` elements"* — and the plan now emits it and validates against it
  (user ruling; ledger **ALGO-1**). Fetch
  `http://www.openmicroscopy.org/Schemas/OME/2016-06/ome.xsd` into
  `tests/fixtures/ome/2016-06/`, under the same read-only rules as the JSON schemas, with
  its own `SOURCE.md` recording the URL, date, and sha256.

  `xmlschema` is the validator; **the dependency itself is declared in Task 0.1** (ledger
**GEN-44** — an earlier draft issued the same `pyproject.toml` edit from both tasks). This
task vendors the schema file under the same read-only rules as the JSON schemas
  (same reasoning: spec §7 forbids a check that skips on a missing dependency, and there is
  no `test` group). It is pure-Python and pulls only `elementpath`.
- **Four JSON schemas, not three.** All three of `image`, `label`, and `ome` carry exactly one
  **remote** `$ref` — `https://ngff.openmicroscopy.org/0.5/schemas/_version.schema`
  (verified by parsing the downloaded files). `jsonschema` >= 4.18 does **not** fetch remote
  refs; it raises `referencing.exceptions.Unresolvable`, which is not a `ValidationError`, so
  the harness would **error** rather than fail and offline CI would have no fallback.
  `_version.schema` is 280 bytes: `{"type": "string", "enum": ["0.5"]}`. Task 2.5 resolves it
  through a `referencing.Registry` keyed on each file's `$id`.
- These are **vendored upstream sources**. Per CLAUDE.md they must stay byte-identical to
  upstream: never lint, format, autofix, tidy, or "fix" them. Add the directory to
  `[tool.ruff] extend-exclude` in the same commit that adds the files, so no later bare
  `ruff check --fix` can touch them.
- `SOURCE.md` records the exact upstream URL, the retrieval date, and the sha256 of each
  file, so a future reader can prove the copy is unmodified.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ngff_schema_fixtures.py`:

```python
"""The vendored NGFF 0.5 schemas must be present, parseable, and unmodified.

Spec §7 forbids a conformance check that skips on a missing fixture, so the
absence of these files is a hard failure here rather than a skip downstream.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

SCHEMA_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "ngff" / "0.5"
SCHEMA_NAMES = ("image.schema", "label.schema", "ome.schema", "_version.schema")


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_is_present_and_parses(name: str) -> None:
    path = SCHEMA_DIR / name
    assert path.is_file(), f"vendored NGFF schema missing: {path}"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_matches_recorded_digest(name: str) -> None:
    """SOURCE.md pins each file's sha256; a mismatch means someone edited it."""
    recorded = dict(
        re.findall(
            r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|",
            (SCHEMA_DIR / "SOURCE.md").read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )
    actual = hashlib.sha256((SCHEMA_DIR / name).read_bytes()).hexdigest()
    assert recorded.get(name) == actual, (
        f"{name} does not match the digest recorded in SOURCE.md; the vendored "
        "upstream copy must stay byte-identical."
    )


def test_every_schema_is_rooted_at_the_attributes_object() -> None:
    """All three are ``{"required": ["ome"], "properties": {"ome": …}}``.

    This is what the conformance harness must validate against: the whole
    ``attributes`` mapping, NOT ``attributes["ome"]``. Passing the inner block
    fails with "'ome' is a required property" on every store.
    """
    for name in ("image.schema", "label.schema", "ome.schema"):
        payload = json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))
        assert payload["required"] == ["ome"], name
        assert list(payload["properties"]) == ["ome"], name
        assert payload["description"] == "The zarr.json attributes key", name


def test_ome_schema_requires_series() -> None:
    """Stricter than the prose — §7 calls this out explicitly."""
    payload = json.loads((SCHEMA_DIR / "ome.schema").read_text(encoding="utf-8"))
    assert payload["properties"]["ome"]["required"] == ["series", "version"]


def test_label_schema_requires_image_label() -> None:
    payload = json.loads((SCHEMA_DIR / "label.schema").read_text(encoding="utf-8"))
    assert payload["properties"]["ome"]["required"] == ["image-label", "version"]


def test_image_label_does_not_require_exhaustive_colors() -> None:
    """Pins the fact that re-graded P1: `colors` is OPTIONAL.

    `$defs/image-label` has no `required` list at all, so nothing obliges one
    entry per unique label value. The spec's §2.3 "MUST" is a PhenoTypic
    invention, not an NGFF rule.
    """
    payload = json.loads((SCHEMA_DIR / "label.schema").read_text(encoding="utf-8"))
    image_label = payload["$defs"]["image-label"]
    assert "required" not in image_label
    assert "colors" in image_label["properties"]


def test_every_remote_ref_is_vendored() -> None:
    """A remote $ref raises Unresolvable, which is not a ValidationError."""
    import re

    ids = {
        json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))["$id"]
        for name in SCHEMA_NAMES
    }
    for name in SCHEMA_NAMES:
        raw = (SCHEMA_DIR / name).read_text(encoding="utf-8")
        for ref in re.findall(r'"\$ref"\s*:\s*"(https?://[^"]+)"', raw):
            assert ref in ids, f"{name} references un-vendored {ref}"
```

- [ ] **Step 2: Run it to confirm it fails**

```bash
uv run pytest tests/unit/test_ngff_schema_fixtures.py -v
```

Expected: FAIL with `vendored NGFF schema missing: .../image.schema`.

- [ ] **Step 3: Fetch the schemas**

```bash
mkdir -p tests/fixtures/ngff/0.5
BASE=https://ngff.openmicroscopy.org/0.5/schemas
for name in image label ome _version; do
  curl -fsSL "$BASE/$name.schema" -o "tests/fixtures/ngff/0.5/$name.schema"
done
sha256sum tests/fixtures/ngff/0.5/*.schema
```

Do not reformat the downloaded bytes. If a URL 404s, resolve the correct one from
<https://ngff.openmicroscopy.org/0.5/> and record what you used in `SOURCE.md` — do
**not** hand-write a schema.

- [ ] **Step 4: Write `SOURCE.md`**

```markdown
# Vendored NGFF 0.5 JSON schemas

Read-only upstream reference material. **Never lint, format, autofix, or edit
these files.** They are the artifact every conformance assertion resolves
against; editing one silently invalidates every claim ever checked against it.

- Upstream: <https://ngff.openmicroscopy.org/0.5/schemas/>
- Retrieved: 2026-08-18

| file | sha256 |
|---|---|
| `image.schema` | `<paste from sha256sum>` |
| `label.schema` | `<paste from sha256sum>` |
| `ome.schema` | `<paste from sha256sum>` |
| `_version.schema` | `<paste from sha256sum>` |

Three facts about these files that the spec gets wrong or omits:

- `ome.schema` **requires** `["series", "version"]`, though the prose presents
  named series as optional.
- `label.schema` **requires** `["image-label", "version"]`, though the prose says
  SHOULD — **but `$defs/image-label` has no `required` list**, so `colors` is
  optional and nothing requires one entry per unique label value. The spec's
  §2.3 "MUST" is a PhenoTypic policy, not an NGFF rule.
- `$defs/omero` requires only `["channels"]`; the channel item has no `required`
  list and `color` is an unconstrained string. Only `window`, **if present**,
  requires all four of `start`/`min`/`end`/`max`. Emitting the full block is
  PhenoTypic policy too.
- All three reference `_version.schema` remotely, which is why it is vendored
  here and resolved through a `referencing.Registry` rather than fetched.
```

- [ ] **Step 5: Exclude the directory from ruff (documentation, not protection)**

⚠️ **This exclusion is inert and must not be mistaken for the guard.** Ruff visits only
`.py`, `.pyi`, and `.ipynb` files; this directory holds `.schema` JSON, which ruff would
never touch anyway. The thing that actually protects these bytes is the sha256 digest test
in `test_ngff_schema_fixtures.py`. Add the exclusion as a statement of intent for human
readers, but do not treat it as the mechanism.

In `pyproject.toml`, extend the existing `[tool.ruff] extend-exclude` list (the one
already protecting `docs/superpowers/**/refs`):

```toml
extend-exclude = [
    "docs/superpowers/**/refs",
    "tests/fixtures/ngff",
]
```

- [ ] **Step 6: Run the test to verify it passes**

```bash
uv run pytest tests/unit/test_ngff_schema_fixtures.py -v
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/ngff pyproject.toml tests/unit/test_ngff_schema_fixtures.py
git commit -m "test: vendor the NGFF 0.5 JSON schemas as read-only fixtures

Conformance is validated against the published schemas via jsonschema
rather than ome-zarr-models, which pins pydantic<2.13. SOURCE.md pins a
sha256 per file and a test asserts the digests still match, so a stray
formatter cannot silently invalidate every conformance assertion. The
directory is added to ruff's extend-exclude in the same commit."
```

---

## Phase 0 exit criteria

- [ ] `uv run python -c "import zarr, jsonschema, xmlschema; print(zarr.__version__)"`
      prints a `3.x` version — **all three imports**, since both conformance gates are
      required to fail rather than skip when their dependency is absent (ledger **GEN-24**).
- [ ] `uv run pytest tests/unit/test_packaging_floor.py tests/unit/test_ngff_schema_fixtures.py -v` is all green.
- [ ] `grep -rn "3\.10" .github/workflows/` returns no `python-version` matches.
- [ ] `uv run pytest tests/unit -q` passes at the raised floor.
- [ ] `uv run mypy src/phenotypic` passes.
