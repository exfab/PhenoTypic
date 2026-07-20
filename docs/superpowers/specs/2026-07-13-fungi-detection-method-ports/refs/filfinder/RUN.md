# A10 FilFinder oracle commands

Verify the pinned runtime and regenerate the external fixture:

```bash
uv run --with fil-finder==1.8 python -c \
  "import fil_finder; assert fil_finder.__version__ == '1.8'"
PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/filfinder/upstream \
  uv run --with fil-finder==1.8 python \
    docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/filfinder/generate_fixture.py
git diff --exit-code -- tests/fixtures/reconnect/filfinder/oracle.json
shasum -a 256 tests/fixtures/reconnect/filfinder/oracle.json
```

The expected raw schema-v3 fixture SHA-256 is
`fabf4ddd818d51f7f376de85035b83f2c9393a55dbc5d1d91b8946f68e511106`. Its directory `.gitattributes`
forces LF, and the generator itself rejects missing child-process warnings, an overbroad supplied-
mask warning filter, unexpected worker stderr, or an empty-input worker launch.

Schema v3 is adapter-reachable evidence: every synthetic source image is coerced through
`float32`, then copied into the same `float64` buffer the wrapper passes to FilFinder. The threshold
boundary uses adjacent float32 values around 0.5. Native float64 synthetic values are not a valid
wrapper oracle because `ImageData` cannot retain them.

The standalone logic suite nevertheless includes one native `float64` predecessor control. It
first proves the value compares below 0.5 directly, then proves the required ImageData-style
`float32` coercion rounds it to equality and therefore foreground. This is a mutation control for
the precision seam, not an external-oracle input.

Run the independent adapter logic suite:

```bash
uv run python \
  docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/filfinder.py
uv run python \
  docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/filfinder/verify_checksums.py
```

`verify_checksums.py` canonicalizes CRLF to LF only for explicitly classified text evidence and
hashes binary evidence byte for byte. This makes the manifest independent of checkout EOL policy
without weakening binary artifact identity.

Artifact authority:

- sdist SHA-256: `a253d2217d1d5eef8311100e00766eed71207f3fc9d79cc8b378896a9b89a9e8`
- wheel SHA-256: `8d7a2b2844346c037685c2af198566344426fbef9ca04833c9261175d75c833e`
- tag commit: `22539cf2176ad9b717658652e8da749158597f4d`
- local source corpus: 59 files, aggregate SHA-256
  `73db4ddb96269a1602a66f1afdc9d6b036faf79f3d374861cf4544761a590174`

The root integrator, not the algorithm agent, adds `fil-finder==1.8` to the `topology` extra and
updates `uv.lock` after G0 approval.
