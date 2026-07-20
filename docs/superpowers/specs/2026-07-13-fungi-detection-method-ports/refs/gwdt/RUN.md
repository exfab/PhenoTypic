# APP2 GWDT reference harness

The source authority is Vaa3D commit
`475e4ca92d4e51de10f1c05d80cef6615432c087`. The selected entry point is
`vaa3d/app2/fastmarching_dt.h:33-199`; the one-slice connectivity reduction is
exercised through `cnn_type=1` and `cnn_type=2`.
`HISTORY.md` proves the selected function is identical to the paper-era `a7210aa5`
revision before the later directory rename.

Regenerate the committed fixture from the repository root:

```bash
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/gwdt/generate_fixture.py
```

The generator compiles `source_harness.cpp` with `c++ -std=c++17 -O0`, invokes the
unaltered Vaa3D `fastmarching_dt` template, and stores complete distance maps for both
2-D connectivities. The COST records are derived executions of the active fixed
`givals[int(normalized*255)]` macro/table using a robust min/max scan; `fastmarching_dt`
does not expose COST, and the active tree overload's coupled `if max ... else if min`
scan is undefined on strictly positive increasing flattened input. Compiler progress
output is not fixture data.

Run the independent numerical and mutation gates:

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/gwdt.py
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/gwdt/run_mutations.py
```

Verify every pinned source, harness, fixture, and validation artifact:

```bash
uv run python \
  docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/gwdt/verify_checksums.py
```
