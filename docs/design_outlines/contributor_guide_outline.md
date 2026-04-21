# Contributor Guide TODO Outline

- [ ] Audience & Scope (drafted content)
  - [x] Audience: Python contributors focused on scientific imaging/arrayed colony phenotyping, pipeline designers, documentation writers, and reviewers.
  - [x] Purpose: PhenoTypic is a modular bio-image analysis framework for agar plate phenotyping with high-level Image/GridImage abstractions, accessor-driven data access, and chainable operations/pipelines.
  - [x] Expectations: Comfortable with Python 3.10–3.12, NumPy stack, basic image processing concepts; familiarity with microbiology plate layouts helpful.

- [ ] Community & Governance (drafted content)
  - [x] Code of Conduct: Add `CODE_OF_CONDUCT.md` (not yet present) and reference it here; include reporting email/issue labels for conduct concerns.
  - [x] Decision roles: Identify maintainers who approve PRs and releases; describe how release owners are assigned and how many reviews are required for merge.
  - [x] Triage: Document who labels new issues, expected first response SLA (e.g., 5 business days), and escalation path for regressions/security.

- [ ] Prerequisites & Support Matrix (drafted content)
  - [x] Python/OS: Support Python >=3.10,<3.13; targets macOS, Windows, Linux (CI covers these). Note Windows excludes some optional deps.
  - [x] Optional deps: ExifTool required for full metadata; `rawpy` and `pympler` not available on Windows—guard imports and provide fallbacks.
  - [x] Hardware: Guidance for CPU-first workflows; note memory considerations for high-res plate images; call out GPU not required.

- [ ] Tooling Setup (drafted content)
  - [x] Primary path: `uv sync --group dev` (or `uv sync --group dev --group docs` for docs); activate via `source .venv/bin/activate` or prefix commands with `uv run`.
  - [x] Alternatives: Editable install `uv add --editable .`; `pip install phenotypic` for users; `uv add "phenotypic[gui]"` for notebook/GUI work.
  - [x] External tools: Install ExifTool for raw metadata; include a validation command example (e.g., `exiftool -ver`) and note impact if missing.

- [ ] Repository Tour (drafted content)
  - [x] Modules: Brief map of `src/phenotypic` (analysis, detect, enhance, grid, measure, refine, prefab, correction) and their purposes.
  - [x] Architecture: Image class composition (handlers/accessors), ABC operation system, pipelines with lazy execution and benchmarking.
  - [x] Layout: Tests mirror modules under `tests/` with resources in `tests/resources`; docs in `docs/source/`; design notes in `docs/design_outlines/`; CI workflows in `.github/workflows`.

- [ ] Development Environment Workflow (drafted content)
  - [x] Clone/setup: `git clone ... && cd PhenoTypic`; add upstream remote if forking; branch naming convention (see below).
  - [x] Dependencies: `uv sync --group dev`; rerun `uv sync` when `uv.lock` changes; prefer `uv run ...` for Python commands.
  - [x] Smoke tests: `uv run pytest -k <pattern>` for quick checks before full suite; emphasize using sample images in `tests/resources`.

- [ ] Coding Standards (drafted content)
  - [x] Accessors: Never mutate raw image arrays directly; use handlers and return new Image instances (immutability philosophy).
  - [x] ABCs: Extend the correct base (`ImageOperation`, `GridOperation`, `MeasureFeatures`, etc.), implement `_operate`, single-responsibility.
  - [x] Docstrings: Google style with microbiology intuition, runnable examples, caveats (per `.cursor/rules/ops-doc-rules.mdc` and `phenotypic.enhance` examples).
  - [x] Typing: Prefer duck typing for array-like inputs; avoid over-constraining scientific data structures.
  - [x] Plotting/logging: Use explicit Matplotlib interfaces; leverage BaseOperation logging/benchmark hooks instead of ad-hoc prints.

- [ ] Image/Data Handling Discipline (drafted content)
  - [x] Color: Use accessor pipeline for color spaces (RGB/XYZ/Lab/HSV); note gamma handling via `ImageColorHandler`.
  - [x] Data: Store sample data under `tests/resources`; preserve metadata when possible; document acceptable formats (PNG/TIFF/JPEG).
  - [x] Performance: Encourage caching via accessors; highlight memory tips for large plates and avoiding in-place mutations.
  - [x] Privacy/provenance: Require provenance notes for contributed datasets; scrub sensitive metadata in shared images.

- [ ] Adding New Features (drafted content)
  - [x] Operations: Inherit appropriate ABC, implement `_operate`, update module `__init__.py` exports, add docstring with microbiology context.
  - [x] Tests/docs: Add targeted tests under mirrored `tests/` module path; include example usage in docstring; update relevant docs pages.
  - [x] Pipelines: Document serialization expectations (`to_yaml`/`from_yaml`), benchmarking hooks, and batch processing notes.
  - [x] Cross-platform: Guard optional deps; document fallbacks/feature flags when a dependency is absent.

- [ ] Testing & Quality Gates (drafted content)
  - [x] Commands: `uv run pytest`, `uv run pytest -n auto`; single test via `uv run pytest tests/test_image.py::test_image_load`.
  - [x] Type checks: `uv run mypy src/phenotypic`; guidance on minimal suppressions with justification.
  - [x] Style: Manual formatting, Google-style docstrings, follow `.cursor/rules/`; no autoformatter configured.
  - [x] Fixtures: How to add regression images to `tests/resources`; avoid oversized binaries; document seeds and parameters.
  - [x] CI parity: Mention GitHub Actions matrices (Linux/Windows/macOS; Python 3.10–3.12) and expectation to keep PRs green.

- [ ] Documentation Contributions (drafted content)
  - [x] Build: From repo root `cd docs && uv run sphinx-build -b html source build`; for live reload `uv run sphinx-autobuild source build`.
  - [x] Examples: Prefer docstring examples with microbiology context over notebooks; when notebooks are needed, follow docs style and tooling.
  - [x] Artifacts: Update diagrams/design outlines under `docs/design_outlines` and `docs/diagrams`; keep scientific clarity and labeling.
  - [x] Releases/changelog: Decide whether to add release notes section; if so, outline format and location.

- [ ] Contribution Workflow (drafted content)
  - [x] Issues: Provide template expectations (not yet present)—require repro steps, images, environment, seeds, operation parameters.
  - [x] PR checklist: Tests passing, `mypy` clean, docs updated, benchmarks for performance-sensitive ops, note optional dep impacts.
  - [x] Reviews: Reviewers check accessor usage, ABC compliance, reproducibility, cross-platform guards, docstring quality.
  - [x] Deprecation policy: Adopt semantic deprecation notices with warnings and documented migration; prefer two-minor-release grace period before removal.

- [ ] Reproducibility & Scientific Rigor (drafted content)
  - [x] Seeds: Require fixed random seeds for stochastic steps and document them in examples/tests.
  - [x] Benchmarking: Encourage using built-in timing/memory tracking; share results when optimizing algorithms.
  - [x] Validation: Describe expected metrics/visual checks for colony detection/refinement; encourage comparison against provided sample images.

- [ ] Release & Packaging Notes (drafted content)
  - [x] Versioning: State chosen scheme (recommend semver) and branch/tag naming for releases.
  - [x] Packaging: How to build sdist/wheel (`uv run python -m build`); dependency pinning philosophy from `pyproject.toml`.
  - [x] Compatibility: Verify pipeline serialization across versions; include backward-compatibility checklist before release.

- [ ] Support Channels & Next Steps (drafted content)
  - [x] Support: Point to GitHub issues/discussions; add contact email if available.
  - [x] Quickstart checklist: Outline minimal steps before PR (sync deps, run smoke tests, add docs/tests, ensure accessor compliance).

## Branch and PR conventions (step 2 draft)
- [x] Branch names: `feature/<topic>`, `bugfix/<issue#-short>`, `docs/<topic>`, `chore/<topic>`, `hotfix/<issue>`, `release/<version>`.
- [x] Commit hygiene: Prefer small, reviewable commits with descriptive messages; avoid force-pushes after review without coordination.
- [x] PR expectations: Link to issue, describe behavior change, include before/after imagery if visual impact, list commands run (`pytest`, `mypy`, docs build).
- [x] Review bar: At least one maintainer approval; require green CI; blocking comments resolved; note optional dependency behavior and platform checks.
- [x] Deprecation handling: Introduce warnings and doc notes; avoid breaking changes without a deprecation period; document migration path in PR description and docs.
