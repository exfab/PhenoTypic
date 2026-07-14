# Selected source corpus

The executable authority is official GUDHI tag `tags/gudhi-release-3.13.0` at commit
`5dbe510bed5d8d700ec7243fd915769fb67964a2`. The downloaded tag archive had SHA-256
`7640d716d6a118b7787602826e4144bfa1ad5903a31f4a760a7f3a1e68cfdc02`.

The local `upstream/` subset contains every file used to establish or review the A11 runtime
contract:

- `LICENSE` and `CMakeGUDHIVersion.txt` for license/version identity;
- `src/python/gudhi/cubical_complex.py` and `_cubical_complex.cc` for the public Python API,
  Fortran flattening, persistence call, pair-coface interface, and extension binding;
- `src/python/gudhi/sklearn/cubical_persistence.py` for independent confirmation of dimensions,
  field 11, and strict `min_persistence` semantics;
- `src/python/doc/cubical_complex_ref.rst` and `cubical_complex_user.rst` for the release docs;
- `src/python/test/test_cubical_complex.py` and `test_cubical_persistence_low_dim.py` for upstream
  behavioral controls;
- `Bitmap_cubical_complex.h`, `Bitmap_cubical_complex_base.h`, and
  `Persistent_cohomology.h` for the invoked C++ implementation path.

The aggregate digest formed by hashing the sorted local files and then hashing that checksum
stream is `6c1c21e26de626126b24457fd0425a91b3f59d1ad330571addf189d8a92a3087`.

The official CPython 3.12 macOS universal wheel is committed beside the corpus so fixture
regeneration does not depend on a mutable resolver result. Its digest is independently present in
the official PyPI JSON at `pypi-gudhi-3.13.0.json:191-206`.

No GUDHI source archive is vendored in the installable package. The selected readable source is
review evidence under `docs/`; the compiled wheel is an oracle artifact only.
