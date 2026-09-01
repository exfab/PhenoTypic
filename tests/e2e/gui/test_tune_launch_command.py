"""Playwright coverage for the Tune Launch command-card browser callback."""
from __future__ import annotations

import shlex
from pathlib import Path

from playwright.sync_api import Page


def test_browser_command_remains_portable_after_form_edits(page: Page) -> None:
    """Browser form edits must retain the portable command prefix and flags."""
    asset = (
        Path(__file__).parents[3]
        / "src"
        / "phenotypic"
        / "gui"
        / "tune"
        / "_assets"
        / "tune_launch.js"
    )
    page.set_content("<main id='tune-launch-command'></main>")
    page.add_script_tag(path=str(asset))
    initial, edited = page.evaluate(
        """(states) => states.map(
            (values) => window.dash_clientside.tune_launch.renderCommand(
                ...values,
            ),
        )""",
        [
            [
                "tpe",
                50,
                None,
                [],
                [],
                {
                    "spec": "/run/tuning.json",
                    "input": "/run/images",
                    "output": "/run/out",
                },
            ],
            [
                "random",
                7,
                "journal:///shared/study.log",
                ["on"],
                ["on"],
                {
                    "spec": "/run/edited tuning.json",
                    "input": "/run/edited images",
                    "output": "/run/edited out",
                },
            ],
        ],
    )

    assert shlex.split(initial) == [
        "uv",
        "run",
        "phenotypic-tune",
        "run",
        "/run/tuning.json",
        "-i",
        "/run/images",
        "-o",
        "/run/out",
        "--strategy",
        "tpe",
        "--n-trials",
        "50",
    ]
    assert shlex.split(edited) == [
        "uv",
        "run",
        "phenotypic-tune",
        "run",
        "/run/edited tuning.json",
        "-i",
        "/run/edited images",
        "-o",
        "/run/edited out",
        "--strategy",
        "random",
        "--n-trials",
        "7",
        "--storage-url",
        "journal:///shared/study.log",
        "--screen",
        "--slurm",
    ]
