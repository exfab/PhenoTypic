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
