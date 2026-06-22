"""Single-variant benchmark runner for the polars-vs-duckdb+pandas diagnostic.

Run ONE benchmark variant exactly once in a fresh process and emit a JSON
record (stage timings + peak RSS) on stdout. The orchestrator
(``run_diagnostic.py``) invokes this once per (variant, repeat) so that:

* peak RSS (``ru_maxrss``) is a clean high-watermark for that variant alone,
* warm interpreter / import state never biases the measured work,
* a crash in one engine never poisons the others.

Timing covers only the measured operation -- imports and frame construction
happen *before* the ``perf_counter`` window. Thread parallelism is capped
identically across engines (``--threads``) so the comparison reflects
algorithmic/library throughput, not who grabbed more cores.

This file is a standalone diagnostic; it imports nothing from ``phenotypic``
except the real ``duckdb_aggregate`` used by the production forward path, so
the "polars" numbers reflect the code actually shipping in the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Thread capping -- MUST run before polars / pyarrow import.
# ---------------------------------------------------------------------------
def _cap_threads(n: int) -> None:
    os.environ.setdefault("POLARS_MAX_THREADS", str(n))
    os.environ.setdefault("RAYON_NUM_THREADS", str(n))
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n))
    os.environ.setdefault("MKL_NUM_THREADS", str(n))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(n))


def _peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# ---------------------------------------------------------------------------
# Corpus discovery
# ---------------------------------------------------------------------------
def _discover_corpus(corpus_dir: Path) -> tuple[list[Path], dict[Path, str]]:
    """Return (file_paths, path->dataset) for per-image parquets under corpus_dir.

    Layout mirrors a real run: ``<corpus>/<dataset>/measurements/<stem>.parquet``.
    Files beginning with ``_`` (e.g. ``_dataset_aggregated.parquet``) are skipped,
    exactly like the production aggregator.
    """
    paths: list[Path] = []
    mapping: dict[Path, str] = {}
    for ds_dir in sorted(corpus_dir.iterdir()):
        meas = ds_dir / "measurements"
        if not meas.is_dir():
            continue
        for pq in sorted(meas.glob("*.parquet")):
            if pq.name.startswith("_"):
                continue
            paths.append(pq)
            mapping[pq] = ds_dir.name
    return paths, mapping


# ---------------------------------------------------------------------------
# Concat strategies -> return a frame (polars or pandas)
# ---------------------------------------------------------------------------
def _concat_duckdb_polars(paths, mapping, threads):
    """Former production forward path: DuckDB UNION + dataset JOIN -> arrow -> polars.

    Self-contained (replicates the now-removed ``duckdb_aggregate``) so this
    historical diagnostic still runs after the repo switched to the polars-native
    aggregator. Requires ``duckdb`` to be installed (no longer a project dep).
    """
    import polars as pl
    return pl.from_arrow(_duckdb_arrow(paths, mapping, threads))


def _duckdb_arrow(paths, mapping, threads):
    """Run the EXACT SQL the repo's ``duckdb_aggregate`` uses, return an Arrow table.

    Mirrors ``phenotypic._cli._cli_duckdb_agg.duckdb_aggregate`` (UNION ALL BY
    NAME read with ``filename=true`` plus the ``Metadata_Dataset`` mapping
    LEFT JOIN) so the polars and pandas paths differ *only* in how the Arrow
    result is materialized (``pl.from_arrow`` vs ``Table.to_pandas``).
    """
    import duckdb
    conn = duckdb.connect()
    conn.execute(f"SET threads = {threads}")
    conn.execute("SET preserve_insertion_order = false")
    pq_list = ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths)
    base_query = (
        f"SELECT * FROM read_parquet([{pq_list}], union_by_name=true, filename=true)"
    )
    conn.execute("CREATE TEMP TABLE _ds_map(path VARCHAR, dataset VARCHAR)")
    conn.executemany("INSERT INTO _ds_map VALUES (?, ?)",
                     [(str(p), name) for p, name in mapping.items()])
    has_col = conn.execute(
        "SELECT COUNT(*) FROM parquet_schema(?) WHERE name = 'Metadata_Dataset'",
        [str(paths[0])],
    ).fetchone()[0] > 0
    if has_col:
        query = base_query
    else:
        query = (
            'SELECT t.*, m.dataset AS "Metadata_Dataset" '
            f"FROM ({base_query}) AS t LEFT JOIN _ds_map m ON t.filename = m.path"
        )
    # fetch_arrow_table() materializes a pyarrow.Table (the repo's .arrow()
    # returns a RecordBatchReader, which pl.from_arrow accepts but pandas does
    # not). Both pl.from_arrow and Table.to_pandas accept a Table, so this
    # keeps the polars/pandas comparison materializing from identical bytes.
    table = conn.execute(query).fetch_arrow_table()
    conn.close()
    return table


def _concat_duckdb_pandas(paths, mapping, threads):
    """Proposed: identical DuckDB SQL -> arrow -> to_pandas (no polars)."""
    return _duckdb_arrow(paths, mapping, threads).to_pandas()


def _concat_polars_native(paths, mapping, threads):
    """Optimal pure-polars equivalent of duckdb_aggregate (same output columns).

    Single multithreaded ``pl.read_parquet`` over all files with
    ``include_file_paths``, then vectorized derivation of ``Metadata_Dataset``
    (from the ``<dataset>/measurements/<file>`` path) and ``Metadata_ImageFile``
    -- matching the columns the repo's DuckDB path produces, with no duckdb.
    """
    import polars as pl
    df = pl.read_parquet([str(p) for p in paths], include_file_paths="filepath")
    df = df.with_columns(
        pl.col("filepath").str.extract(r"([^/\\]+)[/\\]measurements[/\\][^/\\]+$", 1)
          .alias("Metadata_Dataset"),
        pl.col("filepath").str.extract(r"([^/\\]+)\.[^.]+$", 1)
          .alias("Metadata_ImageFile"),
    ).drop("filepath")
    # Reading 7.9k files yields a 7.9k-chunk frame; rechunk to one contiguous
    # block so downstream writes are not penalized for fragmentation (matches
    # the contiguous frame the repo's duckdb_aggregate -> pl.from_arrow yields).
    return df.rechunk()


def _concat_polars_read(paths, mapping, threads):
    """Pure-polars native multi-file read (no duckdb)."""
    import polars as pl
    return pl.read_parquet([str(p) for p in paths])


def _concat_polars_scan(paths, mapping, threads):
    """Pure-polars lazy scan + concat + collect (chunk-writer style)."""
    import polars as pl
    lfs = [pl.scan_parquet(str(p)) for p in paths]
    return pl.concat(lfs, how="diagonal_relaxed").collect()


def _concat_pyarrow_pandas(paths, mapping, threads):
    """Pure pyarrow dataset -> pandas (no duckdb, no polars)."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    pa.set_cpu_count(threads)
    dataset = ds.dataset([str(p) for p in paths], format="parquet")
    table = dataset.to_table()
    return table.to_pandas()


_CONCAT = {
    "concat_duckdb_polars": _concat_duckdb_polars,
    "concat_duckdb_pandas": _concat_duckdb_pandas,
    "concat_polars_native": _concat_polars_native,
    "concat_polars_read": _concat_polars_read,
    "concat_polars_scan": _concat_polars_scan,
    "concat_pyarrow_pandas": _concat_pyarrow_pandas,
}


# ---------------------------------------------------------------------------
# Macro end-to-end compile pipelines
# ---------------------------------------------------------------------------
def _macro_polars(paths, mapping, out_dir, threads, stages):
    import polars as pl

    t = time.perf_counter
    s = t()
    df = _concat_duckdb_polars(paths, mapping, threads)
    stages["concat"] = t() - s

    s = t()
    if "Metadata_ImageFile" not in df.columns and "filename" in df.columns:
        df = df.with_columns(
            pl.col("filename").str.extract(r"([^/\\]+)\.[^.]+$", 1).alias("Metadata_ImageFile")
        )
    if "filename" in df.columns:
        df = df.drop("filename")
    stages["derive"] = t() - s

    s = t()
    df.write_parquet(str(out_dir / "master.parquet"), compression="zstd", compression_level=3)
    stages["write_parquet"] = t() - s

    s = t()
    df.write_csv(str(out_dir / "master.csv"))
    stages["write_csv"] = t() - s

    s = t()
    _split_polars(df, out_dir, threads)
    stages["split"] = t() - s
    return df.height, df.width


def _macro_pandas(paths, mapping, out_dir, threads, stages):
    import pandas as pd

    t = time.perf_counter
    s = t()
    df = _concat_duckdb_pandas(paths, mapping, threads)
    stages["concat"] = t() - s

    s = t()
    if "Metadata_ImageFile" not in df.columns and "filename" in df.columns:
        df["Metadata_ImageFile"] = (
            df["filename"].str.extract(r"([^/\\]+)\.[^.]+$", expand=False)
        )
    if "filename" in df.columns:
        df = df.drop(columns=["filename"])
    stages["derive"] = t() - s

    s = t()
    df.to_parquet(str(out_dir / "master.parquet"), engine="pyarrow",
                  compression="zstd", index=False)
    stages["write_parquet"] = t() - s

    s = t()
    df.to_csv(str(out_dir / "master.csv"), index=False)
    stages["write_csv"] = t() - s

    s = t()
    _split_pandas(df, out_dir, threads)
    stages["split"] = t() - s
    return len(df), df.shape[1]


def _macro_pandas_duckwrite(paths, mapping, out_dir, threads, stages):
    """duckdb+pandas, but master writes go through DuckDB COPY (its fast writer)."""
    import duckdb
    import pandas as pd

    t = time.perf_counter
    s = t()
    df = _concat_duckdb_pandas(paths, mapping, threads)
    stages["concat"] = t() - s

    s = t()
    if "Metadata_ImageFile" not in df.columns and "filename" in df.columns:
        df["Metadata_ImageFile"] = (
            df["filename"].str.extract(r"([^/\\]+)\.[^.]+$", expand=False)
        )
    if "filename" in df.columns:
        df = df.drop(columns=["filename"])
    stages["derive"] = t() - s

    conn = duckdb.connect()
    conn.execute(f"SET threads = {threads}")
    conn.register("master", df)

    s = t()
    conn.execute(
        f"COPY master TO '{out_dir / 'master.parquet'}' "
        "(FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 3)"
    )
    stages["write_parquet"] = t() - s

    s = t()
    conn.execute(f"COPY master TO '{out_dir / 'master.csv'}' (FORMAT csv, HEADER)")
    stages["write_csv"] = t() - s

    s = t()
    _split_duckdb(conn, df, out_dir)
    stages["split"] = t() - s
    conn.close()
    return len(df), df.shape[1]


# ---------------------------------------------------------------------------
# Per-feature split implementations (real feature columns hard-coded from schema)
# ---------------------------------------------------------------------------
# In this dataset the only producer-owned feature family is SymmetricZones.
# split_measurements keeps every non-feature column as context + that family's
# columns. We approximate the same shape: context cols + SymmetricZones_* cols.
def _feature_cols(all_cols):
    feature = [c for c in all_cols if c.startswith("SymmetricZones_")]
    context = [c for c in all_cols if not c.startswith("SymmetricZones_")]
    return context, feature


def _split_polars(df, out_dir, threads):
    context, feature = _feature_cols(df.columns)
    if not feature:
        return
    subset = df.select(context + feature)
    subset.write_parquet(str(out_dir / "feature.parquet"), compression="zstd",
                         compression_level=3)
    subset.write_csv(str(out_dir / "feature.csv"))


def _split_pandas(df, out_dir, threads):
    context, feature = _feature_cols(list(df.columns))
    if not feature:
        return
    subset = df.loc[:, context + feature]
    subset.to_parquet(str(out_dir / "feature.parquet"), engine="pyarrow",
                      compression="zstd", index=False)
    subset.to_csv(str(out_dir / "feature.csv"), index=False)


def _split_duckdb(conn, df, out_dir):
    context, feature = _feature_cols(list(df.columns))
    if not feature:
        return
    cols = ", ".join('"' + c + '"' for c in context + feature)
    conn.execute(
        f"COPY (SELECT {cols} FROM master) TO '{out_dir / 'feature.parquet'}' "
        "(FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 3)"
    )
    conn.execute(
        f"COPY (SELECT {cols} FROM master) TO '{out_dir / 'feature.csv'}' (FORMAT csv, HEADER)"
    )


def _macro_polars_native(paths, mapping, out_dir, threads, stages):
    """Pure-polars end-to-end: no duckdb anywhere (concat via pl.read_parquet)."""
    import polars as pl

    t = time.perf_counter
    s = t()
    df = _concat_polars_native(paths, mapping, threads)
    stages["concat"] = t() - s
    stages["derive"] = 0.0  # Metadata_* derived inside concat

    s = t()
    df.write_parquet(str(out_dir / "master.parquet"), compression="zstd", compression_level=3)
    stages["write_parquet"] = t() - s

    s = t()
    df.write_csv(str(out_dir / "master.csv"))
    stages["write_csv"] = t() - s

    s = t()
    _split_polars(df, out_dir, threads)
    stages["split"] = t() - s
    return df.height, df.width


_MACRO = {
    "macro_polars": _macro_polars,
    "macro_polars_native": _macro_polars_native,
    "macro_pandas": _macro_pandas,
    "macro_pandas_duckwrite": _macro_pandas_duckwrite,
}


# ---------------------------------------------------------------------------
# Micro write / convert / split shootouts (operate on the real master parquet)
# ---------------------------------------------------------------------------
def _run_micro(variant, master_pq, out_dir, threads, stages):
    t = time.perf_counter

    if variant == "wpq_polars":
        import polars as pl
        df = pl.read_parquet(master_pq)
        s = t()
        df.write_parquet(str(out_dir / "m.parquet"), compression="zstd", compression_level=3)
        stages["op"] = t() - s

    elif variant == "wpq_pandas_pyarrow":
        import pandas as pd
        df = pd.read_parquet(master_pq)
        s = t()
        df.to_parquet(str(out_dir / "m.parquet"), engine="pyarrow",
                      compression="zstd", index=False)
        stages["op"] = t() - s

    elif variant == "wpq_duckdb":
        import duckdb
        conn = duckdb.connect()
        conn.execute(f"SET threads = {threads}")
        conn.execute(f"CREATE TABLE m AS SELECT * FROM read_parquet('{master_pq}')")
        s = t()
        conn.execute(
            f"COPY m TO '{out_dir / 'm.parquet'}' "
            "(FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 3)"
        )
        stages["op"] = t() - s
        conn.close()

    elif variant == "wcsv_polars":
        import polars as pl
        df = pl.read_parquet(master_pq)
        s = t()
        df.write_csv(str(out_dir / "m.csv"))
        stages["op"] = t() - s

    elif variant == "wcsv_pandas":
        import pandas as pd
        df = pd.read_parquet(master_pq)
        s = t()
        df.to_csv(str(out_dir / "m.csv"), index=False)
        stages["op"] = t() - s

    elif variant == "wcsv_duckdb":
        import duckdb
        conn = duckdb.connect()
        conn.execute(f"SET threads = {threads}")
        conn.execute(f"CREATE TABLE m AS SELECT * FROM read_parquet('{master_pq}')")
        s = t()
        conn.execute(f"COPY m TO '{out_dir / 'm.csv'}' (FORMAT csv, HEADER)")
        stages["op"] = t() - s
        conn.close()

    elif variant == "conv_to_pandas":
        import polars as pl
        df = pl.read_parquet(master_pq)
        s = t()
        _ = df.to_pandas()
        stages["op"] = t() - s

    elif variant == "conv_from_pandas":
        import polars as pl
        import pandas as pd
        pdf = pd.read_parquet(master_pq)
        s = t()
        _ = pl.from_pandas(pdf)
        stages["op"] = t() - s

    elif variant == "split_polars":
        import polars as pl
        df = pl.read_parquet(master_pq)
        s = t()
        _split_polars(df, out_dir, threads)
        stages["op"] = t() - s

    elif variant == "split_pandas":
        import pandas as pd
        df = pd.read_parquet(master_pq)
        s = t()
        _split_pandas(df, out_dir, threads)
        stages["op"] = t() - s

    else:
        raise SystemExit(f"unknown micro variant: {variant}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--family", required=True, choices=["concat", "macro", "micro"])
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--master-parquet", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()

    _cap_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    stages: dict[str, float] = {}
    rows = cols = None
    total_s = time.perf_counter()

    if args.family == "concat":
        paths, mapping = _discover_corpus(args.corpus)
        fn = _CONCAT[args.variant]
        s = time.perf_counter()
        frame = fn(paths, mapping, args.threads)
        stages["op"] = time.perf_counter() - s
        # shape works for both polars (.shape) and pandas (.shape)
        rows, cols = (frame.shape[0], frame.shape[1]) if frame is not None else (0, 0)

    elif args.family == "macro":
        paths, mapping = _discover_corpus(args.corpus)
        fn = _MACRO[args.variant]
        rows, cols = fn(paths, mapping, args.out, args.threads, stages)

    else:  # micro
        _run_micro(args.variant, str(args.master_parquet), args.out, args.threads, stages)

    total = time.perf_counter() - total_s

    record = {
        "variant": args.variant,
        "family": args.family,
        "threads": args.threads,
        "stages": stages,
        "wall_total_s": total,
        "peak_rss_mb": _peak_rss_mb(),
        "rows": rows,
        "cols": cols,
    }
    print(json.dumps(record))


if __name__ == "__main__":
    main()
