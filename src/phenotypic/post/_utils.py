from __future__ import annotations

import pandas as pd

# Schema-aware metadata prefixing lives in ``phenotypic.sdk_`` so the post
# package does not pull in the core image stack. Re-exported here for the post
# ops (and any legacy callers) that import it from this module.
from phenotypic.sdk_ import ensure_metadata_prefix

__all__ = ["ensure_metadata_prefix", "affix_preserving_na"]


def affix_preserving_na(
    col: "pd.Series", *, prefix: str = "", suffix: str = ""
) -> "pd.Series":
    """Wrap each value of *col* in *prefix*/*suffix*, leaving NA cells as NA.

    Shared by :class:`~phenotypic.post.PrependString` and
    :class:`~phenotypic.post.AppendString`, which differ only in which side
    they affix.

    ``.astype(str)`` renders NA as the literal string ``"nan"``, so a naive
    concatenation turns a missing value into a present-looking ``"WT-nan"``.
    That matters for an undetected colony carried by the CLI's ``--metadata``
    left join: every measurement column on such a row is null, and a
    stringified ``"nan"`` would read as real data. Masking the NA cells back
    out is a no-op when nothing is NA, so frames without missing values are
    unaffected.

    Args:
        col: The column to affix. Any dtype; cells are stringified.
        prefix: String concatenated before each present value.
        suffix: String concatenated after each present value.

    Returns:
        A new Series of affixed strings, NA-preserving.
    """
    return (prefix + col.astype(str) + suffix).mask(col.isna())
