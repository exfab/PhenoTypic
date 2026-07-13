# Tensor-voting reference execution

The oracle is the unmodified `source/` archive from MATLAB Central File Exchange 21051,
version 1.0.0.0. The harness adapts `(response, theta)` to the source's rank-1 tensor input,
calls only `calc_vote_stick`, and saves every input, accumulated tensor component, eigenvalue,
and saliency output.

The locally verified runtime is MATLAB `23.2.0.2668659 (R2023b) Update 9`:

```bash
/Applications/MATLAB_R2023b.app/bin/matlab -batch \
  "run('docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/tensor_voting/source_harness.m')"
```

The command writes
`tests/fixtures/reconnect/tensor_voting/linton_calc_vote_stick_r2023b.mat` in MATLAB v7
format so SciPy can load it. MATLAB's UI is disabled by the harness before the unmodified source
opens its progress waitbar.

MATLAB v7 container headers include nondeterministic metadata, so raw MAT-file hashes are not
reproducible. Verify the manifest's canonical decoded-content hash after generation by running:

```bash
uv run python \
  docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/tensor_voting.py
```

Canonicalization sorts the required keys; hashes each UTF-8 key, array rank, and shape; encodes
numeric arrays as little-endian dtype plus C-order bytes; and encodes strings as length-prefixed
UTF-8. The source ZIP retains its separate raw SHA-256 in `PROVENANCE.json`.

Reproduce every isolated mutation check without modifying the reviewed source tree:

```bash
uv run python \
  docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/tensor_voting/run_mutations.py
```

The runner loads the baseline, copies source into a temporary directory, proves each textual
diff is exactly the declared replacement, injects TV-M01 through TV-M15 one at a time, executes
the named killing probe, rejects survivors, and verifies the production source SHA-256 is
unchanged after the temporary directory is discarded.

No wrapper oracle exists. The wrapper and image-to-orientation estimator are intentionally
deferred.
