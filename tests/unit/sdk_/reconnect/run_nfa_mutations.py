"""Run required source-free A07 mutations against named focused tests."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "src" / "phenotypic" / "sdk_" / "reconnect" / "_nfa.py"
TEST = ROOT / "tests" / "unit" / "sdk_" / "reconnect" / "test_nfa.py"


@dataclass(frozen=True)
class Mutation:
    """One textual mutation and the named test required to kill it."""

    name: str
    replacements: tuple[tuple[str, str], ...]
    test_name: str


MUTATIONS = (
    Mutation(
        "sf-k-instead-of-k-minus-one",
        (("k_values[general] - 1,", "k_values[general],"),),
        "test_golden_fixture_rejects_sf_k_instead_of_sf_k_minus_one",
    ),
    Mutation(
        "omit-n-tests",
        (("log10_nfa = log10_n_tests + log10_tail", "log10_nfa = log10_tail.copy()"),),
        "test_source_fixture_exact_fields_and_score_convention",
    ),
    Mutation(
        "natural-log-reported-as-log10",
        (("_LOG_10 = math.log(10.0)", "_LOG_10 = 1.0"),),
        "test_exact_rational_small_tail_grid_stays_within_calibrated_256_ulp",
    ),
    Mutation(
        "reverse-score-sign",
        (("score = -log10_nfa", "score = log10_nfa"),),
        "test_source_fixture_exact_fields_and_score_convention",
    ),
    Mutation(
        "strict-meaningful-threshold",
        (("meaningful = log10_nfa <= math.log10(epsilon_value)", "meaningful = log10_nfa < math.log10(epsilon_value)"),),
        "test_exact_edges_and_inclusive_equality_skip_backend",
    ),
    Mutation(
        "accept-more-than-one-million-trials",
        (("if np.any(n_broadcast > _MAX_TRIALS):", "if np.any(n_broadcast > np.iinfo(np.int64).max):"),),
        "test_invalid_inputs_raise",
    ),
    Mutation(
        "accept-fractional-test-family",
        (("n_tests_value = _validated_scalar(n_tests, name=\"n_tests\", minimum=1.0)", "n_tests_value = _validated_scalar(n_tests, name=\"n_tests\", minimum=0.0)"),),
        "test_invalid_inputs_raise",
    ),
    Mutation(
        "accept-invalid-backend-result",
        (
            (
                "if np.any(np.isnan(backend_log_tail)) or np.any(backend_log_tail > 0.0):",
                "if False:",
            ),
            (
                "if np.any(np.isnan(log_tail)) or np.any(log_tail > 0.0):",
                "if False:",
            ),
        ),
        "test_invalid_backend_log_survival_raises",
    ),
    Mutation(
        "multiplicative-subnormal-ratio",
        (
            (
                """log_ratio = (
            math.log(n - j + 1)
            - math.log(j)
            + log_p
            - log_one_minus_p
        )""",
                "log_ratio = math.log((n - j + 1) * p / (j * (1.0 - p)))",
            ),
        ),
        "test_minimum_positive_probability_uses_additive_log_ratio",
    ),
    Mutation(
        "omit-k-zero-shortcut",
        (("log_tail[k_zero | p_one] = 0.0", "log_tail[p_one] = 0.0"),),
        "test_exact_edges_and_inclusive_equality_skip_backend",
    ),
    Mutation(
        "omit-k-equals-n-shortcut",
        (("    log_tail[k_zero | p_one] = 0.0", "    exact_all_successes &= False\n    log_tail[k_zero | p_one] = 0.0"),),
        "test_exact_edges_and_inclusive_equality_skip_backend",
    ),
    Mutation(
        "omit-p-zero-and-p-one-shortcuts",
        (
            ("log_tail[k_zero | p_one] = 0.0", "log_tail[k_zero] = 0.0"),
            ("log_tail[p_zero_impossible] = -np.inf", "log_tail[p_zero_impossible] = np.nan"),
        ),
        "test_exact_edges_and_inclusive_equality_skip_backend",
    ),
    Mutation(
        "trust-backend-negative-infinity",
        (
            (
                "fallback = np.isneginf(log_tail) & (p_values > 0.0) & (p_values < 1.0)",
                "fallback = np.zeros(log_tail.shape, dtype=bool)",
            ),
        ),
        "test_million_trial_underflow_uses_finite_bounded_fallback",
    ),
)


def _apply_mutation(source: str, mutation: Mutation) -> str:
    mutated = source
    for old, new in mutation.replacements:
        count = mutated.count(old)
        if count != 1:
            raise AssertionError(
                f"{mutation.name}: expected one mutation site, observed {count}"
            )
        mutated = mutated.replace(old, new)
    return mutated


def run_required_mutations() -> None:
    """Require every frozen A07 mutant to fail its named focused test."""

    source = SOURCE.read_text(encoding="utf-8")
    for mutation in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="a07-nfa-mutant-") as temporary:
            package = Path(temporary) / "src" / "phenotypic" / "sdk_" / "reconnect"
            package.mkdir(parents=True)
            for parent in (package.parents[1], package.parent, package):
                (parent / "__init__.py").write_text("", encoding="utf-8")
            (package / "_nfa.py").write_text(
                _apply_mutation(source, mutation),
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(Path(temporary) / "src")
            environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-x",
                    "-c",
                    os.devnull,
                    "--confcutdir",
                    str(TEST.parent),
                    str(TEST),
                    "-k",
                    mutation.test_name,
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                raise AssertionError(f"SURVIVED {mutation.name}: {mutation.test_name}")
            print(f"KILLED   {mutation.name}: {mutation.test_name}")


if __name__ == "__main__":
    run_required_mutations()
