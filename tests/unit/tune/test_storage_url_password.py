"""``_resolve_storage_url`` rejects a password-bearing storage URL (B2).

The single chokepoint that resolves the Optuna storage URL also refuses an
inline password: that secret would otherwise be persisted in plaintext into the
GUI-discovery ``run.json`` marker AND the generated SLURM worker script, both
world-readable on a shared cluster filesystem. A password-less Postgres URL
(libpq resolves the secret from ``~/.pgpass`` / ``$PGPASSWORD`` / ``PGSERVICE``)
and the local SQLite fallback must still resolve cleanly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.sdk_ import _io_constants as io
from phenotypic.tune import (
    Categorical,
    Evaluator,
    Knob,
    SearchSpace,
)
from phenotypic.tune.score import QCScorer
from phenotypic.tune.strategy import OptunaConfig
from phenotypic.tune._spec import Budget, TuningSpec
from phenotypic.tune.strategy._config import PHENOTYPIC_TUNE_STORAGE_URL_ENV
from phenotypic.tune._tune_cli._run import _resolve_storage_url, run_tuning


def _optuna_spec(tmp_path: Path, *, storage_url: str) -> TuningSpec:
    csv = tmp_path / "layout.csv"
    import pandas as pd

    pd.DataFrame(
        {
            "Metadata_ImageName": ["Synthetic96PlateWithObjects"] * 96,
            "Object_Label": list(range(96)),
        }
    ).to_csv(csv, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops=[BlurGauss(sigma=1.0), OtsuDetector()]),
        search_space=SearchSpace(
            knobs=(
                Knob(
                    key="1.ignore_zeros",
                    domain=Categorical(choices=(True, False)),
                ),
            )
        ),
        scorer=QCScorer(
            check=ExpectedVsDetectedCount(
                metadata=str(csv), groupby=["Metadata_ImageName"]
            )
        ),
        evaluator=Evaluator(),
        strategy=OptunaConfig(n_trials=2, storage_url=storage_url),
        budget=Budget(),
    )


def test_rejects_explicit_password_bearing_url(tmp_path):
    with pytest.raises(ValueError, match="inline password"):
        _resolve_storage_url(
            "postgresql+psycopg://user:s3cret@db.example.org:5432/tune",
            tmp_path,
        )


def test_rejects_password_bearing_url_from_env(tmp_path, monkeypatch):
    # The env-var fallback is the same chokepoint — a secret there is rejected too.
    monkeypatch.setenv(
        PHENOTYPIC_TUNE_STORAGE_URL_ENV,
        "postgresql+psycopg://user:s3cret@db.example.org:5432/tune",
    )
    with pytest.raises(ValueError, match="inline password"):
        _resolve_storage_url(None, tmp_path)


def test_password_less_postgres_url_passes(tmp_path):
    url = "postgresql+psycopg://user@db.example.org:5432/tune"
    assert _resolve_storage_url(url, tmp_path) == url


def test_sqlite_fallback_has_no_password_and_passes(tmp_path):
    # The local study.db fallback carries no password; it must always resolve.
    resolved = _resolve_storage_url(None, tmp_path)
    assert resolved.startswith("sqlite:///")


def test_error_message_is_actionable(tmp_path):
    # The message must point at the libpq alternatives so a user can fix it.
    with pytest.raises(ValueError) as excinfo:
        _resolve_storage_url(
            "postgresql://user:pw@host/db", tmp_path
        )
    msg = str(excinfo.value)
    assert ".pgpass" in msg
    assert "PGPASSWORD" in msg
    assert "PGSERVICE" in msg


def test_resolved_sqlite_url_round_trips_without_a_password():
    # A sqlite path with a ':' in it (Windows-ish) still has no URL password.
    from urllib.parse import urlsplit

    url = f"sqlite:///{Path('/tmp/a/study.db')}"
    assert urlsplit(url).password is None


def test_spec_password_url_fails_before_any_run_artifact_is_written(tmp_path):
    out = tmp_path / "out"
    spec = _optuna_spec(
        tmp_path,
        storage_url="postgresql+psycopg://user:s3cret@db.example.org:5432/tune",
    )

    with pytest.raises(ValueError, match="inline password"):
        run_tuning(spec, [load_synth_yeast_plate()], out)

    assert not io.tuning_spec_path(out).exists()
    assert not io.tune_cache_run_marker_path(out).exists()
    assert not (out / "slurm_scripts").exists()
    assert not io.slurm_scripts_dir(out).exists()
