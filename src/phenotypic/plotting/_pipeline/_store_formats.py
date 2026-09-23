"""One deterministic serializer per store format (spec §2 "Serializers").

Plotting libraries are imported inside each function (lazy-import contract).
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Callable

from phenotypic.abc_.plotting import figure_backend_of


def _serialize_plotly_json(figure: Any, *, binding_id: str, page_key: str) -> bytes:
    # `to_json` drops trace uids by default, the one per-object random field.
    return figure.to_json().encode("utf-8")


def _serialize_png(figure: Any, *, binding_id: str, page_key: str) -> bytes:
    if figure_backend_of(figure) == "plotly":
        # Module attribute access, not a bound name, so a test that patches
        # `_backends.chrome_available` is honoured (see `_writer.py` imports).
        from . import _backends

        if not _backends.chrome_available():
            raise _backends.PlotBackendUnavailable(
                "Plotly PNG export needs Chrome (kaleido); install it with "
                "plotly_get_chrome"
            )
        import plotly.io as pio

        return pio.to_image(figure, format="png")
    buffer = BytesIO()
    figure.savefig(buffer, format="png")
    return buffer.getvalue()


_SERIALIZERS: dict[str, Callable[..., bytes]] = {
    "plotly-json": _serialize_plotly_json,
    "png": _serialize_png,
}


def serialize_store_format(
    fmt: str, figure: Any, *, binding_id: str, page_key: str
) -> bytes:
    """Serialize *figure* to one store format.

    Args:
        fmt: A name from ``STORE_FORMATS``.
        figure: A figure whose backend supports *fmt* (checked by the caller).
        binding_id: The plot binding id. Unused by the current formats; kept
            so a format that must salt generated ids has what it needs.
        page_key: The page key; same reason.

    Returns:
        The encoded bytes.

    Raises:
        KeyError: If *fmt* is not a store format.
        PlotBackendUnavailable: A Plotly PNG without Chrome.
    """
    return _SERIALIZERS[fmt](figure, binding_id=binding_id, page_key=page_key)


__all__ = ["serialize_store_format"]
