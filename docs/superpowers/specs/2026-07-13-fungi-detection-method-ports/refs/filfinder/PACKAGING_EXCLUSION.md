# A10 reference packaging exclusion

The pinned source, paper, generator, fixture, and standalone logic script are development/audit
evidence and must not ship in PhenoTypic release artifacts. The current setuptools configuration
discovers packages only below `src/`, and `MANIFEST.in` includes only license/notice material.

`verify_packaging_exclusion.py` builds a fresh wheel and sdist, enumerates every member, and rejects
`docs/superpowers`, `refs/filfinder`, `upstream/fil_finder`, or
`tests/fixtures/reconnect/filfinder`. This gate must run again after production and topology-extra
integration. The legitimate runtime package is the separately installed `fil-finder==1.8`
dependency, not the local reference corpus.
