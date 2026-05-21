"""Golden-equivalence harness for the pydantic v2 operation migration.

This package captures bit-exact golden outputs of every PhenoTypic
operation and analyzer on the *current, unmigrated* code, then re-runs
each operation and asserts identical results. It is the immovable
correctness reference for the pydantic migration: as long as
``tests/migration`` stays green, the migration changed no numerical
behavior.

Modules:
    _inputs: Frozen input artifacts (raw/detected plates and grids, a
        reference measurement frame) and their reconstruction helpers.
    _scenarios: The scenario registry -- auto-default scenarios for every
        discovered operation plus curated non-default extras.
    _runner: Scenario execution and golden I/O shared by the capture
        script and the equivalence test.
    test_equivalence: The pytest harness, parametrized over scenarios.
"""
