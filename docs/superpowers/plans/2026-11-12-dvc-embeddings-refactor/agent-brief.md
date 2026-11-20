# Brief for every implementation agent (#281)

Read this first, then your cluster's tasks. It is not part of the rendered plan.

## Where
- Worktree: `/bigdata/exfab/anguy344/PhenoTypic/.claude/worktrees/dvc-embeddings-refactor`,
  branch `worktree-dvc-embeddings-refactor`. Run every command from there. Never
  `cd` to the repo root and never touch another worktree.
- Plan: `docs/superpowers/plans/2026-11-12-dvc-embeddings-refactor/` (README.md has
  Global Constraints, Review Focus, Conventions, the expected-red GUI test list).
- Spec: `docs/superpowers/specs/2026-11-12-dvc-embeddings-refactor/design.md`. The
  spec is authoritative; the plan implements it.
- Root `CLAUDE.md` and the module guides (`src/phenotypic/_cli/CLAUDE.md`,
  `sdk_/CLAUDE.md`, `abc_/CLAUDE.md`, `schema/CLAUDE.md`) bind you.

## How
- **TDD per the plan:** write the task's failing tests, run them and record the
  failing names, implement, run them green.
- **Tests:** `env QT_QPA_PLATFORM=offscreen uv run pytest -q --no-header -p no:randomly -o addopts= -m "not slow" <files>`.
  Never `-n auto`; this node has 8 allocated cores, so `-n 8` at most. Anything
  over a few minutes is a Slurm job (`slurm-job` skill; never the `preempt`
  partition; always `--mem` and `--time`; logs on `/bigdata`, never `/scratch`).
- **Lint:** `uv run ruff check --fix <paths you changed>`. Never bare.
- **Commits:** you commit your own work, explicit paths only (`git add <paths>`),
  never `git add -A` / `git add .` / `git commit -a`. Other agents may share the
  worktree. Never `git stash`, never push, never amend or rebase commits you did
  not make. End each message with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01BcdQZe33xFFxycFrk1x25u
  ```
- **No file under `src/phenotypic/_gui/` changes**, and you do not edit, xfail or
  skip GUI tests. The README lists which GUI tests are expected to go red; any
  other red GUI test is a real failure you must fix outside `_gui/`.
- Heavy imports (torch, transformers, colour, …) stay inside functions; the
  startup-import guards in `tests/unit/ci/` stay green. `sdk_` never imports
  `_cli`.
- Store changes get an independent-reader check (zarr-python + pyarrow), per the
  `working-with-ome-zarr` skill.

## Stop and report instead of deciding
If the plan and the spec disagree, the code contradicts a plan assumption in a way
that changes behaviour, or you face a design choice the plan and spec do not
settle, **stop and report it** as an open question. Do not pick silently. A
cited `file:line` that has merely drifted is not a question; find the right line.

Report anything that looks like a pre-existing bug separately instead of fixing it
silently.

## Your final report (keep it under 400 words)
- commit SHA(s), each with one line on what it does;
- the failing test names you saw before each fix;
- the exact test commands you ran last and their summary lines;
- deviations from the plan, each with its reason;
- open questions, if any (or "none").
