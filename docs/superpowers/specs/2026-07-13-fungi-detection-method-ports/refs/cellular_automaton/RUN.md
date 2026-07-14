# A06 TrickTrack Reference Commands

Run from the `cellular_automaton` reference directory. These commands reconstruct the exact
pinned source tree and execute the upstream integration oracle without importing `phenotypic`.

```bash
rm -rf /tmp/TrickTrack-b164fad
mkdir -p /tmp/TrickTrack-b164fad
tar -xzf TrickTrack-b164fad.tar.gz -C /tmp/TrickTrack-b164fad --strip-components=1
c++ -std=c++14 \
  -I/tmp/TrickTrack-b164fad/include \
  /tmp/TrickTrack-b164fad/tests/integration/test_integration.cpp \
  /tmp/TrickTrack-b164fad/src/tricktrack/CMGraphUtils.cpp \
  -o /tmp/tricktrack-a06-oracle
/tmp/tricktrack-a06-oracle
```

Expected stdout:

```text
TrickTrack integration test successful
```

The command was verified on macOS with Apple Clang 21.0.0. The upstream project declares C++14
(`source/CMakeLists.txt` inside the complete archive). No third-party library is needed for this
specific integration executable, although the full upstream CMake project requires Eigen.

Verify the pinned artifacts before use:

```bash
shasum -a 256 -c CHECKSUMS.sha256
```

Regenerate the instrumented all-output fixture from the pinned archive:

```bash
uv run python generate_fixture.py
```

Run the independent oracle and required mutation matrix from the repository root:

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/cellular_automaton.py
uv run python docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/refs/cellular_automaton/run_mutations.py
```
