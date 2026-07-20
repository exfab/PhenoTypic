"""Re-derive the conservative CED sweep selection from exported evidence."""

from __future__ import annotations

import csv
from pathlib import Path


SPEC_DIR = (
    Path(__file__).resolve().parents[2]
    / "specs"
    / "2026-07-15-orientation-field"
)
AGGREGATE_CSV = (
    SPEC_DIR
    / "artifacts"
    / "twok_ced_literal_crossing_parameter_sweep_aggregate.csv"
)


def verify_ced_parameter_selection() -> None:
    """Assert the documented guard-based CED selection from the sweep CSV."""
    with AGGREGATE_CSV.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 26:
        raise AssertionError(f"Expected 26 CED configurations, got {len(rows)}")

    def value(row: dict[str, str], key: str) -> float:
        return float(row[key])

    if any(value(row, "MeanRoughnessReduction") > 0.0 for row in rows):
        raise AssertionError("Sweep unexpectedly contains a roughness improvement")

    eligible = [
        row
        for row in rows
        if value(row, "MeanRoughnessReduction") >= -0.01
        and value(row, "WorstCrossingDeviation") <= 0.02
    ]
    selected = max(eligible, key=lambda row: value(row, "MeanCoherenceGain"))
    if selected["ConfigID"] != "CED24":
        raise AssertionError(f"Expected CED24, selected {selected['ConfigID']}")
    expected = {
        "sigma": 2.5,
        "rho": 5.0,
        "num_iter": 30.0,
        "C": 95.0,
        "dt": 0.1,
        "alpha": 0.001,
    }
    for key, expected_value in expected.items():
        if value(selected, key) != expected_value:
            raise AssertionError(
                f"CED24 {key}={selected[key]}, expected {expected_value}"
            )

    print(
        "PASS: all 26 settings fail to reduce mean branch-interior roughness; "
        "CED24 uniquely maximizes coherence under the documented <=1% "
        "roughness and <=2% crossing-count guards"
    )


if __name__ == "__main__":
    verify_ced_parameter_selection()
