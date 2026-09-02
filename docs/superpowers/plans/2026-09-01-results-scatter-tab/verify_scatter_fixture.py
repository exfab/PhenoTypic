"""Acceptance check for the Scatter tab against the real verification run.

This is Task 14 Step 5's smoke run, done headlessly. It drives the tab's
OWN code path -- ``index_frame`` -> ``plottable`` -> ``section_values`` ->
``plan_facets`` -> ``build_scatter_figure`` -> ``resolve_click`` ->
``export_sections_pdf`` -- over the run's measurements mirror, plus one
real crop off one real store.

**It breaks this directory's usual contract, deliberately.** Its neighbour
``crop_uint16_scaling.py`` follows the rule in ``CLAUDE.md``: stdlib plus
numpy only, never importing ``phenotypic``, re-deriving a numeric claim
from scratch so the claim and the code are independent witnesses. This
script imports ``phenotypic`` throughout, because the two artifacts prove
different things. ``crop_uint16_scaling.py`` proves the *arithmetic* of
the mod-256 bug without reference to any implementation; this proves the
*shipped code* produces the right numbers on data nobody wrote for it. A
from-scratch re-derivation of ``plan_facets`` and ``resolve_click`` would
be a second implementation agreeing with itself, which is the failure the
stricter rule exists to prevent, not an instance of it.

Why a script and not a browser: a machine-checked run is repeatable, fails
loudly, and can be committed beside the change it verifies. Booting a live
viewer at this run is *safe* -- ``migrate_legacy_qc`` has no GUI caller,
``test_discover_leaves_legacy_qc_and_viewer_sidecar_byte_identical`` pins
discovery as byte-neutral, and ``_external_cache_dir`` refuses a cache root
inside the output -- it is simply worse evidence.

**Nothing here writes into the run.** The mirror is read with
``pl.read_parquet`` and one store layer with ``crop_store_rgb``; the only
file created is a temporary PNG under ``$TMPDIR``, deleted after use.

What it does NOT cover, stated so the result is not read as more than it
is: the browser-only affordances -- the settings popover, the splitter
drag, the Contours control as a DOM control -- which are pinned by
``tests/gui/results_viewer/test_splitter_browser.py`` and the layout and
callback unit tests. And it checks that the crop is *smooth* rather than a
mod-256 sawtooth, which is the bug it exists to catch; it cannot tell a
smooth colony from a smooth patch of agar. Only a person looking at the
inspector can do that.

Usage::

    uv run python docs/superpowers/logic_validation_scripts/2026-09-01-results-scatter-tab/verify_scatter_fixture.py
    uv run python <this file> --root /path/to/another/run

Exit codes: ``0`` every check passed (or the run is absent / mid-republish
and there was nothing to check -- see ``_run_is_readable``), ``1`` a check
failed, ``2`` the run is present and readable but has no mirror.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
from pathlib import Path

import numpy as np
import polars as pl

#: The run every number below was measured against. Not a repo fixture: it
#: is a real results tree, and other jobs republish it.
DEFAULT_ROOT = Path(
    "/rhome/anguy344/bigdata_exfab/projects/ucr_029_e_d_Maresca/data/results/"
    "2026-08-11-migration-test"
)

# ---------------------------------------------------------------------------
# Expected counts, split by the FRAME each was measured on
# ---------------------------------------------------------------------------
#
# The split is the whole point of this section, not bookkeeping. An earlier
# revision kept one flat table and asserted `images` and `null_strain_rows`
# against the plottable frame while their values had been measured on the
# mirror. Both numbers were real measurements of this real run; the boundary
# they crossed was four lines of code, and the only thing that caught it was
# running the script. Scope is therefore a required argument to `expect`
# below rather than a comment, so the same mistake cannot be written again
# without naming which frame it means.
#
# The two frames differ by exactly the phantom filter: 121 metadata-only rows
# that cover 4 images entirely and carry 1 of the mirror's 82 null strains.

#: Properties of the MIRROR -- the whole frame the viewer loads, phantoms
#: included. This is what `OutputRoot.master_df` holds and what a click
#: index is anchored to.
EXPECTED_MIRROR = {
    "rows": 844,
    "images": 36,
    "null_strain_rows": 82,
}

#: Properties of the PLOTTABLE frame -- what survives `plottable`, which is
#: what the tab actually draws and pages.
EXPECTED_PLOTTABLE = {
    "rows": 723,
    "images": 32,
    "null_strain_rows": 81,
    "sections": 22,
}

#: Rows the phantom filter removes, i.e. the difference between the two.
EXPECTED_PHANTOMS = 121

_SCOPES = {"mirror": EXPECTED_MIRROR, "plottable": EXPECTED_PLOTTABLE}

_failures: list[str] = []
_checks = 0


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Record one check, print its result, and remember any failure.

    Args:
        label: What the check asserts, phrased as the property.
        ok: Whether it holds.
        detail: Measured values, printed either way so a passing run is
            still evidence rather than a bare "PASS".

    Returns:
        ``ok``, so a caller can branch on it.
    """
    global _checks
    _checks += 1
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail else ""))
    if not ok:
        _failures.append(label)
    return ok


def _run_is_readable(root: Path) -> bool:
    """Is the run present *and* published, i.e. would discovery accept it?

    Presence alone is not enough. This is a live results tree the CLI
    republishes: a run in progress clears its aggregate publication
    marker, rewrites ``master_measurements.parquet``, then republishes.
    Inside that window the directory exists and every count below is
    meaningless. Copied from ``_run_is_readable`` in
    ``tests/unit/gui/results_viewer/test_measurement_join_migration_run.py``,
    which was added after a routine republish turned three green tests red
    at 22:07 with no code change since 21:59.

    Any failure to determine readability also reads as "not readable":
    this script's value is the assertions, none of which is about
    discovery's own gate.

    Args:
        root: The run directory.

    Returns:
        True when the run can be read coherently right now.
    """
    if not root.is_dir():
        return False
    try:
        from phenotypic.gui.results_viewer._output_consistency import (
            inspect_output_consistency,
        )
        from phenotypic.sdk_ import BundleLayout

        return inspect_output_consistency(BundleLayout.detect(root)).core_readable
    except Exception:  # noqa: BLE001 - unreadable for any reason means skip
        return False


#: Where Playwright vendors the browsers the e2e suite already uses, in the
#: order kaleido can drive them. `run_scatter_tests.sbatch` resolves the same
#: cache the same way.
_PLAYWRIGHT_CHROME_GLOBS = (
    ".cache/ms-playwright/chromium-*/chrome-linux64/chrome",
    ".cache/ms-playwright/chromium-*/chrome-linux/chrome",
    ".cache/ms-playwright/chromium-*/chrome-mac*/Chromium.app/Contents/MacOS/Chromium",
)


def _ensure_browser_path() -> str | None:
    """Point kaleido at a browser, preferring one already on the machine.

    Without this the PDF section skips on any node that has no
    ``google-chrome`` on ``PATH`` -- which is every compute node here --
    and the one acceptance item that most needs a browser is the one that
    silently does not run. That is exactly the outcome the ``[SKIP]``
    banner exists to make visible, so the script resolves the browser
    itself rather than relying on the caller to export ``BROWSER_PATH``.

    An explicit ``BROWSER_PATH`` always wins; this only fills the gap.

    Returns:
        The browser path kaleido will use, or ``None`` when none was
        found.
    """
    explicit = os.environ.get("BROWSER_PATH")
    if explicit:
        return explicit
    for pattern in _PLAYWRIGHT_CHROME_GLOBS:
        # Sorted so a machine holding several vendored builds picks one
        # deterministically rather than by directory order.
        found = sorted(Path.home().glob(pattern))
        if found:
            os.environ["BROWSER_PATH"] = str(found[-1])
            return str(found[-1])
    on_path = next(
        (
            shutil.which(name)
            for name in ("google-chrome", "chromium", "chrome")
            if shutil.which(name)
        ),
        None,
    )
    if on_path:
        return on_path
    return str(Path.home() / ".cache/kaleido") if (
        Path.home() / ".cache/kaleido"
    ).exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)

    if not _run_is_readable(args.root):
        print(
            f"[SKIP] {args.root} is absent or mid-republish. Nothing was "
            f"checked -- this is NOT a pass. Re-run once the CLI has "
            f"republished."
        )
        return 0

    strict = args.root.resolve() == DEFAULT_ROOT.resolve()
    if not strict:
        print(
            "[note] --root is not the run these counts were measured on, "
            "so they are reported rather than asserted."
        )

    from phenotypic.gui._shared.tiles import crop_store_rgb
    from phenotypic.gui.results_viewer._filter_state import (
        METHOD_IS_ANY_OF,
        FilterSpec,
    )
    from phenotypic.gui.results_viewer._scatter_tab._callbacks import (
        section_values,
    )
    from phenotypic.gui.results_viewer._scatter_tab._facets import plan_facets
    from phenotypic.gui.results_viewer._scatter_tab._figure import (
        CUSTOMDATA_COL,
        build_scatter_figure,
    )
    from phenotypic.gui.results_viewer._scatter_tab._inspector import (
        index_frame,
        resolve_click,
    )
    from phenotypic.gui.results_viewer._scatter_tab._spec import (
        CURATION_PHANTOM_COL,
        FigureSpec,
        plottable,
    )
    from phenotypic.schema import BBOX, GENETIC, IMAGE, OBJECT, SHAPE
    from phenotypic.sdk_ import measurements_parquet_path

    mirror_path = measurements_parquet_path(args.root)
    if not mirror_path.is_file():
        print(f"[FAIL] no measurements mirror at {mirror_path}")
        return 2
    print(f"[note] reading {mirror_path}")

    mirror = pl.read_parquet(mirror_path)
    strain = str(GENETIC.STRAIN)
    image_col = str(IMAGE.IMAGE_NAME)
    label_col = str(OBJECT.LABEL)

    def expect(scope: str, key: str, actual: int) -> None:
        """Assert one count against the frame it was measured on.

        Args:
            scope: ``"mirror"`` or ``"plottable"``. Required, and named in
                the printed label, because the same key means different
                numbers on the two frames.
            key: Entry in that scope's expectation table.
            actual: The measured value.
        """
        want = _SCOPES[scope][key]
        if strict:
            check(f"{scope}.{key} == {want}", actual == want, f"got {actual}")
        else:
            print(f"[note] {scope}.{key} = {actual} (reference run: {want})")

    # -- 1. The mirror, whole -------------------------------------------
    expect("mirror", "rows", mirror.height)
    expect("mirror", "images", mirror[image_col].n_unique())
    expect("mirror", "null_strain_rows", mirror[strain].null_count())
    check(
        f"the phantom flag {CURATION_PHANTOM_COL!r} is present",
        CURATION_PHANTOM_COL in mirror.columns,
        "without it `plottable` is a no-op and the counts below prove "
        "nothing",
    )

    # -- 2. The plottable frame -----------------------------------------
    # The one legal order, copied from `prepare_frame`: stamp the index on
    # the WHOLE mirror first, so every carried index is master-anchored,
    # and only then drop phantoms and filter.
    stamped = index_frame(mirror)
    frame = plottable(stamped)
    expect("plottable", "rows", frame.height)
    expect("plottable", "images", frame[image_col].n_unique())
    expect("plottable", "null_strain_rows", frame[strain].null_count())

    phantoms = mirror.height - frame.height
    if strict:
        check(
            f"the phantom filter removes {EXPECTED_PHANTOMS} rows",
            phantoms == EXPECTED_PHANTOMS,
            f"got {phantoms}",
        )
        # Pinned as its own property because it is the kind of fact that
        # changes silently: the phantoms are not scattered one per image,
        # they account for four images ENTIRELY. If that ever stops being
        # true the two image counts above stop being 4 apart and this says
        # so in the terms that matter.
        lost_images = mirror[image_col].n_unique() - frame[image_col].n_unique()
        lost_nulls = mirror[strain].null_count() - frame[strain].null_count()
        check(
            "the phantoms account for 4 whole images and 1 null strain",
            (lost_images, lost_nulls) == (4, 1),
            f"{phantoms} phantom rows remove {lost_images} images and "
            f"{lost_nulls} null-strain rows",
        )

    # -- 3. Sections: a null is dropped, never paged --------------------
    sections = section_values(frame, strain)
    distinct_including_null = frame[strain].n_unique()
    nulls = frame[strain].null_count()
    expect("plottable", "sections", len(sections))
    # Asserted as the RULE, not as the number: a regression to 23 fails
    # here saying "the null became its own page", which is the thing that
    # would have gone wrong, rather than "22 != 23".
    check(
        "a null section value is dropped rather than becoming its own page",
        len(sections) == distinct_including_null - (1 if nulls else 0),
        f"{len(sections)} pages from {distinct_including_null} distinct "
        f"values, {nulls} null rows",
    )

    # -- 4. A facet grid plans and draws --------------------------------
    # Both from `MeasureShape`, which this run's pipeline configures.
    #
    # Asking the schema is necessary and NOT sufficient, which is the
    # lesson of the first run of this script. `SIZE.AREA` is a perfectly
    # real schema member -- `Size_Area` is even CLAUDE.md's worked example
    # of a category-prefixed column -- and it is absent from this mirror,
    # because the run's measurers are MeasureNeighborDist / MeasureShape /
    # MeasureIntensity / MeasureColor / MeasureTexture and none of them is
    # MeasureSize. The schema says a header is spellable; only the run
    # says it was measured. Hence the presence checks immediately below.
    x_col, y_col = str(SHAPE.PERIMETER), str(SHAPE.AREA)
    for column in (x_col, y_col):
        check(f"{column} is in the mirror", column in frame.columns)

    plottable_xy = frame.drop_nulls(subset=[x_col, y_col])
    # `index_frame` is idempotent, so filtering AFTER stamping is safe and
    # re-stamping the filtered frame would be the defect, not the fix.
    page = plottable_xy.filter(pl.col(strain).cast(pl.String) == sections[0])

    def _cardinality(column: str) -> int:
        """Distinct values, or 0 for a column polars cannot hash.

        Not an error: an unhashable column is simply not a facet
        candidate. Same lesson as commit ``8d09f076`` on the value-set
        path.
        """
        try:
            return frame[column].n_unique()
        except Exception:  # noqa: BLE001 - unhashable means "not a candidate"
            return 0

    facet_cols = [
        c for c in frame.columns if c != strain and 2 <= _cardinality(c) <= 4
    ]
    spec = FigureSpec(
        x_col=x_col,
        y_col=y_col,
        section_col=strain,
        col_col=facet_cols[0] if facet_cols else None,
    )
    plan = plan_facets(page, spec)
    check(
        "a facet grid plans at least one panel",
        len(plan.rows) * len(plan.cols) >= 1,
        f"{len(plan.rows)}x{len(plan.cols)} panels "
        f"(col_col={spec.col_col!r}), truncated={plan.truncated}",
    )

    figure = build_scatter_figure(page, spec, plan)
    points = sum(len(getattr(t, "customdata", None) or []) for t in figure.data)
    check(
        "the figure carries points, not just axes",
        points > 0,
        f"{points} points across {len(figure.data)} traces",
    )

    # -- 5. A click resolves to the colony it names ---------------------
    #
    # THE FILTER IS LOAD-BEARING, and this fixture proved it rather than
    # theory: the 121 phantoms occupy a CONTIGUOUS TAIL at mirror
    # positions 723..843, so the plottable rows are positions 0..722 and
    # every carried index equals its own row position. Over that frame a
    # master-anchored index and a positional one are observationally
    # identical, and the round trip passes against the very defect it
    # names -- which is the B1 defect Gate 0 found.
    #
    # So: filter first, through the REAL filter path, exactly as
    # `test_the_click_index_is_stamped_before_the_filter` does.
    #
    # The stale-fingerprint guard is deliberately NOT exercised: passing
    # one value for both sides is the tautology the plan warns about, and
    # `test_a_stale_fingerprint_is_refused_not_resolved` owns that case.
    keep = sections[: max(len(sections) // 3, 1)]
    payload = [{"column": strain, "method": METHOD_IS_ANY_OF, "values": keep}]
    filtered = plottable(FilterSpec.from_store(payload).apply_to(stamped))
    check(
        "the strain filter actually drops rows",
        0 < filtered.height < frame.height,
        f"{filtered.height} of {frame.height} rows kept by {len(keep)} of "
        f"{len(sections)} strains -- without a real drop the round trip "
        f"below proves nothing",
    )

    kept = filtered.drop_nulls(subset=[label_col])
    carried = kept[CUSTOMDATA_COL].to_list()
    # Asserted BEFORE the round trip that depends on it, so a fixture that
    # drifts back into the coincidence fails as a broken check rather than
    # passing vacuously.
    #
    # If this ever fires on the FILTERED frame it means the kept strains
    # happen to occupy a contiguous prefix. That is the guard working, not
    # a flake, and the fix is to choose a different subset -- never to
    # loosen the assertion, which is the only thing standing between this
    # script and a round trip that proves nothing.
    check(
        "the carried index is not degenerate (index != row position)",
        bool(carried) and carried != list(range(len(carried))),
        f"{len(carried)} rows, first carried index "
        f"{carried[0] if carried else 'n/a'}",
    )

    # The strong form, taken from the same precedent: not "an index
    # resolves somewhere" but "every index resolves to a colony THE
    # FILTER KEPT". An index stamped on the filtered frame resolves to a
    # colony the filter excluded, which is the direct observation of B1;
    # round-tripping one index against its own row cannot see it at all.
    resolved = 0
    wrong_strain: list[str] = []
    for index in carried:
        candidate = resolve_click(mirror, index, "fp", "fp")
        if candidate is None:
            continue
        resolved += 1
        match = mirror.filter(
            (pl.col(image_col).cast(pl.String) == candidate.stem)
            & (pl.col(label_col).cast(pl.Int64, strict=False) == candidate.label)
        )
        if not match.is_empty() and match[strain][0] not in keep:
            wrong_strain.append(f"{index}->{match[strain][0]}")
    check(
        "every carried index resolves to a colony the filter kept",
        resolved == len(carried) and not wrong_strain,
        f"{resolved}/{len(carried)} resolved"
        + (
            f"; stamped on the FILTERED frame: {wrong_strain[:3]}"
            if wrong_strain
            else ""
        ),
    )

    # One colony carried forward for the crop check below.
    probe = carried[len(carried) // 2] if carried else None
    colony = resolve_click(mirror, probe, "fp", "fp") if probe is not None else None
    row = (
        kept.filter(pl.col(CUSTOMDATA_COL) == probe).row(0, named=True)
        if probe is not None
        else None
    )

    # -- 6. A real crop is smooth, not mod-256 noise --------------------
    stores = sorted(args.root.glob("results/*/zarr/*.ome.zarr"))
    check("the run carries per-image stores", bool(stores), f"{len(stores)} found")
    if stores and colony is not None and row is not None:
        from PIL import Image as PILImage

        store = next(
            (s for s in stores if s.name.startswith(f"{colony.stem}.")), stores[0]
        )
        png = crop_store_rgb(
            store,
            "rgb",
            float(row[str(BBOX.CENTER_RR)]),
            float(row[str(BBOX.CENTER_CC)]),
            256,
            store.joinpath("zarr.json").stat().st_mtime_ns,
            contours=1,
        )
        crop = np.asarray(
            PILImage.open(io.BytesIO(png)).convert("L"), dtype=np.int16
        )
        # `astype(np.uint8)` on a uint16 store is a modular reduction, so a
        # smooth colony becomes a sawtooth in which adjacent pixels jump
        # the full range. Measured as a FRACTION of horizontal neighbours
        # rather than a count, so the threshold cannot drift when the crop
        # size changes -- which is how two earlier numbers on this branch
        # went wrong.
        fraction = float((np.abs(np.diff(crop, axis=1)) > 128).mean())
        check(
            "the crop is smooth, not mod-256 noise",
            fraction < 0.02,
            f"{fraction:.4%} of horizontal neighbours jump >128 grey "
            f"levels ({store.name})",
        )

    # -- 7. The exported PDF carries ink --------------------------------
    browser = _ensure_browser_path()
    if browser is None:
        print(
            "[skip] PDF export: no browser found for kaleido -- not on "
            "PATH, no ~/.cache/kaleido, and nothing vendored under "
            "~/.cache/ms-playwright. Fetch one with plotly_get_chrome or "
            "`uv run playwright install chromium`. This is the one "
            "acceptance item this script cannot answer without a browser."
        )
    else:
        print(f"[note] kaleido will render through {browser}")
        import kaleido
        from PIL import Image as PILImage
        from pypdf import PdfReader

        from phenotypic.gui.results_viewer._scatter_tab._pdf import (
            export_sections_pdf,
        )

        # Exported from the WHOLE plottable frame, not from one section's
        # page: handing it `page` would render page 2 empty and the
        # page-count check would pass while documenting nothing.
        two = sections[:2]
        pdf = export_sections_pdf(plottable_xy, spec, two)
        pages = len(PdfReader(io.BytesIO(pdf)).pages)
        check(
            "the export writes one page per section",
            pages == len(two),
            f"{pages} pages for {len(two)} sections",
        )

        # The axis ranges are pinned onto BOTH figures from the drawn
        # data, because an empty frame auto-ranges differently and the
        # moved gridlines would otherwise count as "difference". With them
        # pinned, the only thing that can differ between the two renders is
        # the marker layer.
        x_span = (page[x_col].min(), page[x_col].max())
        y_span = (page[y_col].min(), page[y_col].max())

        def _render(frame: pl.DataFrame) -> np.ndarray:
            """Render one page to a greyscale array.

            Size and axis ranges are set on the FIGURE rather than passed
            to kaleido -- that is how ``export_sections_pdf`` does it, and
            it is what makes two renders comparable pixel for pixel.

            Args:
                frame: Rows to draw. An empty frame gives the control.

            Returns:
                The rendered page as a 2-D uint8 array.
            """
            fig = build_scatter_figure(frame, spec, plan, for_export=True)
            fig.update_layout(width=800, height=600)
            fig.update_xaxes(range=list(x_span))
            fig.update_yaxes(range=list(y_span))
            buffer = (
                Path(os.environ.get("TMPDIR", "/tmp"))
                / f"_scatter_ink_{os.getpid()}.png"
            )
            kaleido.write_fig_sync(fig, buffer)
            arr = np.asarray(PILImage.open(buffer).convert("L"))
            buffer.unlink(missing_ok=True)
            return arr

        # COUNTING DIFFERING PIXELS, not ink above a threshold. Every
        # threshold tried on this branch was wrong in one direction or the
        # other, and both failures looked like a working test:
        #   `gray < 128` saw nothing -- the markers carry opacity 0.5, so
        #     50% navy over white is luminance ~149;
        #   `gray < 250` saw the paper -- 266,000 px of background swamping
        #     a ~200 px marker layer, and the drawn-minus-control delta
        #     came out NEGATIVE.
        # Two renders of the same figure differing only in their rows need
        # no threshold at all: the markers are the only thing that can
        # move.
        empty = page.clear()
        control = _render(empty)
        # The noise floor, measured rather than assumed. If the renderer
        # is deterministic this is 0 and "differs at all" becomes a real
        # assertion; if it ever stops being deterministic, the comparison
        # below still means something instead of silently weakening.
        noise = int((control != _render(empty)).sum())
        check(
            "two renders of the same page are pixel-identical",
            noise == 0,
            f"{noise} px differ between two renders of the control -- the "
            f"floor the signal below must clear",
        )
        signal = int((_render(page) != control).sum())
        check(
            "the drawn page differs from an identical page with no rows",
            signal > noise,
            f"{signal} px differ where the markers are, against a noise "
            f"floor of {noise}",
        )

    print(f"\n{_checks} checks, {len(_failures)} failed")
    for name in _failures:
        print(f"  - {name}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
