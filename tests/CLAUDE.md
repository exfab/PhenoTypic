# Testing

## Running Tests

**Never run the full test suite.** It is computationally heavy and takes too long. Full regression testing is offloaded to GitHub Actions runners. Instead, use one of these approaches:

1. **Targeted tests** (preferred) — run only tests related to your changes:
   ```bash
   pixi run -e dev pytest -p no:napari tests/unit/<category>/test_<module>.py
   ```
2. **Testmon** — automatically runs only tests affected by changed code:
   ```bash
   pixi run -e dev pytest --testmon -p no:napari
   ```
3. **Smoke test** — run a small subset to verify nothing is obviously broken:
   ```bash
   pixi run -e dev pytest -p no:napari tests/unit/<category>/test_<module>.py -x --timeout=60
   ```

- Don't mix `--testmon` with `-n auto` (SQLite concurrency conflicts)

### Sandbox Mode (Claude Code)

Napari fixtures cause `PermissionError` in sandbox. Always disable with `-p no:napari`:

```bash
pixi run -e dev pytest -p no:napari tests/unit/path/to/test.py
```

---

## Configuration

- `conftest.py` sets `VALIDATE_OPS = True` and debug logging for `ImagePipeline`
- Fixtures in `tests/unit/test_fixtures.py`, shared via `pytest_plugins = ["tests.unit.test_fixtures"]`
- pytest config in `pyproject.toml` under `[tool.pytest.ini_options]`
- Image test data from `phenotypic.data` (e.g., `load_synth_yeast_plate()`)

---

## Writing New Tests

- File naming: `test_<module_name>.py` in `tests/unit/<category>/`
- Use `load_synth_yeast_plate()` for image data
- Import fixtures from `test_fixtures.py` as needed
- Run: `pixi run -e dev pytest -p no:napari tests/unit/<category>/test_<module_name>.py`
