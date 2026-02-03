"""Manual smoke test harness for PipelineExplorer GUI."""

from __future__ import annotations


def main() -> None:
    try:
        import panel as pn
    except ImportError as exc:
        raise SystemExit(
            "Panel is required for the GUI harness. "
            "Install with: uv sync --group dev --group docs --extras gui"
        ) from exc

    from phenotypic.gui.explorer import PipelineExplorer

    pn.extension()
    explorer = PipelineExplorer()
    pn.serve(
        explorer.panel(),
        show=True,
        title="Pipeline Variant Explorer",
        port=0,
    )


if __name__ == "__main__":
    main()
