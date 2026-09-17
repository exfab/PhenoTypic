from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd
from pydantic import PrivateAttr, field_validator, model_validator

from phenotypic.abc_._post_measurement import PostMeasurement

from phenotypic.sdk_ import (
    external_metadata_preserved_columns,
    normalize_metadata_columns,
)

from ._utils import coalesce_metadata_aliases, ensure_metadata_prefix


class JoinMetadata(PostMeasurement):
    """Join external metadata columns onto the measurement frame by key.

    Reads a metadata table from a ``.csv``/``.parquet`` path and left-joins its
    columns onto the measurement DataFrame on one or more shared key columns.
    Measurement frames carry only what the pipeline measured -- image name, grid
    position, object label and feature columns -- so experimental annotation
    (strain, medium, pH, timepoint) has no route into them. This operation is
    that route.

    Unlike :class:`MergeMetadata`, which concatenates columns that are *already*
    present, and :class:`ExpandMetadata`, which splits one delimited column that
    is already present, this brings in columns the frame does not have.

    Joined column names follow the same rule as the CLI's ``--metadata`` join,
    decided against the measurement frame at apply time: a table column that
    the measurement frame already carries (a shared key such as ``Grid_RowNum``
    or a raw ``plate``), or that is a known non-metadata schema header, keeps
    its name. Every other column is an annotation and takes the ``Metadata_``
    spelling, so a bare ``Strain`` arrives as ``Metadata_Strain``.

    Best For:
        - Grouping measurements by an experimental factor downstream --
          scorers, plots and QC checks that need ``Metadata_Strain`` or
          ``Metadata_pH`` on the same row as the measurement.
        - Plate layouts where the factor is a property of the *well*, not of
          the image, so it cannot be recovered from the filename by
          :class:`ExpandMetadata`.

    Consider Also:
        - :class:`MergeMetadata` to build a composite key from columns the
          frame already carries.
        - :class:`ExpandMetadata` when the factors are encoded in a delimited
          filename.

    Args:
        metadata: Path (``str`` or ``Path``) to a ``.csv`` or ``.parquet``
            table. A path is the only form that round-trips through
            ``pipeline.json``, mirroring
            :class:`~phenotypic.analysis.qc.ExpectedVsDetectedCount` -- an
            in-memory frame has no source to persist, so it is not accepted.
        on: Key columns to join on. Must be present in **both** the metadata
            table and the measurement frame. Names resolve literally first; a
            bare label falls back to its schema-prefixed spelling
            (``ImageName`` -> ``Metadata_ImageName``) only when the literal is
            absent, so a non-Metadata key such as ``Grid_RowNum`` is never
            rewritten.
        columns: Optional subset of metadata columns to bring in. ``None``
            (default) brings every column except the keys. Names resolve the
            same way as ``on``.
        strict: When ``True`` (default), raise if any measurement row finds no
            match in the metadata table. When ``False``, unmatched rows get
            ``NaN`` in the joined columns.

    Returns:
        pd.DataFrame: The measurement frame with the metadata columns appended.
        Row count and row order are preserved. Joined columns use the
        naming rule above.

    Raises:
        FileNotFoundError: If ``metadata`` does not exist (at construction, so
            the failure surfaces before the pipeline runs).
        ValueError: If ``on`` is empty, if ``metadata`` has an unsupported
            suffix, or if the metadata table has duplicate rows per key -- a
            duplicated key would silently multiply measurement rows, turning a
            join into a fan-out that inflates every downstream count. Also
            raised at run when a joined non-key column is already in the
            measurement frame, which pandas would otherwise split into
            ``_x``/``_y`` copies.
        KeyError: If a key or requested column is missing from the metadata
            table (at construction) or from the measurement frame (at run).

    Examples:
        Attach strain to each colony so a scorer can group replicates:

        >>> import pandas as pd, tempfile, os
        >>> from phenotypic.post import JoinMetadata
        >>> layout = pd.DataFrame({
        ...     "Metadata_ImageName": ["plate1", "plate1"],
        ...     "Grid_RowNum": [1, 3],
        ...     "Grid_ColNum": [1, 2],
        ...     "Metadata_Strain": ["WT", "mut"],
        ... })
        >>> path = os.path.join(tempfile.mkdtemp(), "layout.csv")
        >>> layout.to_csv(path, index=False)
        >>> measurements = pd.DataFrame({
        ...     "Metadata_ImageName": ["plate1", "plate1"],
        ...     "Grid_RowNum": [1, 3],
        ...     "Grid_ColNum": [1, 2],
        ...     "Object_Label": [1, 2],
        ... })
        >>> op = JoinMetadata(
        ...     metadata=path,
        ...     on=["Metadata_ImageName", "Grid_RowNum", "Grid_ColNum"],
        ... )
        >>> list(op.apply(measurements)["Metadata_Strain"])
        ['WT', 'mut']

        A bare annotation column is prefixed, while the shared grid keys keep
        their names:

        >>> pd.DataFrame({
        ...     "Metadata_ImageName": ["plate1", "plate1"],
        ...     "Grid_RowNum": [1, 3],
        ...     "Grid_ColNum": [1, 2],
        ...     "Medium": ["YPD", "SC"],
        ... }).to_csv(path, index=False)
        >>> out = JoinMetadata(
        ...     metadata=path,
        ...     on=["Metadata_ImageName", "Grid_RowNum", "Grid_ColNum"],
        ... ).apply(measurements)
        >>> [c for c in out.columns if c not in measurements.columns]
        ['Metadata_Medium']
    """

    metadata: Union[str, Path]
    on: List[str]
    columns: Optional[List[str]] = None
    strict: bool = True

    #: The resolved table, read once at construction so a bad path or a
    #: fan-out-inducing duplicate key fails here rather than inside a worker.
    _table: pd.DataFrame = PrivateAttr()

    @field_validator("on", "columns", mode="before")
    @classmethod
    def _as_list(cls, value: Any) -> Any:
        """Accept a single column name where a list is expected."""
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)

    @staticmethod
    def _resolve(name: str, available: "pd.Index") -> str:
        """Resolve one requested name against the columns actually present.

        Literal first, then the schema-prefixed spelling. Prefixing is only a
        *fallback*: ``ensure_metadata_prefix`` spares a name only when it is
        already in a Metadata category, so applying it unconditionally would
        rewrite a legitimate ``Grid_RowNum`` key to ``Metadata_Grid_RowNum`` and
        fail against a table that spells it correctly.
        """
        if name in available:
            return name
        prefixed = ensure_metadata_prefix(name)
        if prefixed in available:
            return prefixed
        raise KeyError(name)

    @field_validator("on")
    @classmethod
    def _require_keys(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("JoinMetadata requires at least one key column in 'on'")
        return value

    @model_validator(mode="after")
    def _load_table(self) -> "JoinMetadata":
        """Read and validate the metadata table at construction time."""
        path = Path(self.metadata)
        if not path.exists():
            raise FileNotFoundError(f"JoinMetadata: no metadata table at {path}")
        if path.suffix == ".parquet":
            table = pd.read_parquet(path)
        elif path.suffix in {".csv", ".tsv"}:
            table = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
        else:
            raise ValueError(
                f"JoinMetadata: unsupported metadata suffix {path.suffix!r}; "
                "use .csv, .tsv or .parquet"
            )

        keys, missing = [], []
        for key in self.on:
            try:
                keys.append(self._resolve(key, table.columns))
            except KeyError:
                missing.append(key)
        if missing:
            raise KeyError(
                f"JoinMetadata: key column(s) {missing} not in {path}. "
                f"Available: {list(table.columns)}"
            )
        object.__setattr__(self, "on", keys)

        if self.columns is None:
            wanted = [c for c in table.columns if c not in keys]
        else:
            wanted, absent = [], []
            for name in self.columns:
                try:
                    wanted.append(self._resolve(name, table.columns))
                except KeyError:
                    absent.append(name)
            if absent:
                raise KeyError(
                    f"JoinMetadata: column(s) {absent} not in {path}. "
                    f"Available: {list(table.columns)}"
                )
            object.__setattr__(self, "columns", wanted)

        table = table[[*keys, *wanted]].drop_duplicates()

        # A duplicated key turns a left join into a fan-out: one measurement row
        # becomes N, silently inflating every count and every group statistic
        # computed afterwards. Refuse rather than corrupt the frame.
        duplicated = table.duplicated(subset=self.on).sum()
        if duplicated:
            raise ValueError(
                f"JoinMetadata: {duplicated} duplicate row(s) per key {self.on} in "
                f"{path}. A duplicated key would multiply measurement rows; "
                "deduplicate the table or add keys until it is unique."
            )

        object.__setattr__(self, "_table", table)
        return self

    def _normalized_table(
        self, measurement_columns: "pd.Index"
    ) -> "tuple[pd.DataFrame, list[str]]":
        """Name the table's columns against the measurement frame.

        Columns the measurement frame already carries, and known non-metadata
        schema headers, keep their names; the rest take the ``Metadata_``
        spelling. The rule is shared with the CLI ``--metadata`` join through
        :func:`~phenotypic.sdk_.external_metadata_preserved_columns`.

        Args:
            measurement_columns: Columns of the frame being joined onto.

        Returns:
            The renamed table, keys first, and the renamed key names.
        """
        table = self._table
        preserved = external_metadata_preserved_columns(
            measurement_columns, table.columns
        )
        kept = [c for c in table.columns if c in preserved]
        renamed = normalize_metadata_columns(table.drop(columns=kept))
        table = pd.concat([table[kept], renamed], axis=1)

        keys = list(
            dict.fromkeys(
                key if key in preserved else ensure_metadata_prefix(key)
                for key in self.on
            )
        )
        rest = [c for c in table.columns if c not in keys]
        return table[[*keys, *rest]], keys

    def _operate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Left-join the metadata columns onto ``df``.

        Args:
            df: Measurement DataFrame carrying every column named in ``on``.

        Returns:
            The frame with the metadata columns appended, same rows, same order.

        Raises:
            KeyError: If a key column is absent from ``df``.
            ValueError: If ``strict`` and some rows find no match.
        """
        table, keys = self._normalized_table(df.columns)
        result = coalesce_metadata_aliases(df, keys)
        missing = [key for key in keys if key not in result.columns]
        if missing:
            raise KeyError(
                f"JoinMetadata: key column(s) {missing} not in the measurement "
                f"frame. Available: {list(df.columns)}"
            )

        clashing = [
            c for c in table.columns if c not in keys and c in result.columns
        ]
        if clashing:
            raise ValueError(
                f"JoinMetadata: column(s) {clashing} are already in the "
                "measurement frame. Add them to 'on' to join on them, or leave "
                "them out of 'columns'."
            )

        # Align key dtypes before merging: a CSV-read int64 key against a frame
        # whose key is object/float matches nothing, and pandas reports that as
        # an all-NaN join rather than an error.
        for key in keys:
            if result[key].dtype != table[key].dtype:
                try:
                    table = table.assign(**{key: table[key].astype(result[key].dtype)})
                except (TypeError, ValueError):
                    table = table.assign(**{key: table[key].astype(str)})
                    result = result.assign(**{key: result[key].astype(str)})

        merged = result.merge(table, on=keys, how="left", sort=False)
        merged.index = result.index

        if self.strict:
            joined = [c for c in table.columns if c not in keys]
            if joined:
                unmatched = int(merged[joined[0]].isna().sum() - result[keys[0]].isna().sum())
                if unmatched > 0:
                    raise ValueError(
                        f"JoinMetadata: {unmatched} measurement row(s) matched no "
                        f"metadata row on {keys}. Pass strict=False to accept "
                        "NaN for unmatched rows."
                    )
        return merged


JoinMetadata.apply.__doc__ = JoinMetadata._operate.__doc__
