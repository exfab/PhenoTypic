# C8: DVC timing on a bounded subset

**Job:** Slurm `32059702`, `intel` node `i29`, 4 CPU / 16 GB, 2026-11-12, 40 min 48 s,
`COMPLETED 0:0`. Script `docs/superpowers/plans/2026-11-12-dvc-embeddings-refactor/c8_dvc_timing.sbatch`
(commit `22109e7fb`). Log `/bigdata/exfab/anguy344/slurm_logs/c8-dvc-timing-32059702.log`.
DVC 3.67.1 via `uvx`, default `cache.type` (`reflink,copy`, which is copy on GPFS).
One configuration on one cold node.

**Subset:** dataset `7-24-26_redo_full` of the migrated maresca copy
(`2026-08-11-ome-zarr`), every 22nd store: 303 of 6,657 stores, 17,101 files,
60.16 GB, hard-linked with `cp -al`, `hdf/` excluded. The scratch repo was removed
afterwards.

| Step | Measured (303 stores) | Extrapolated (6,657 stores, ×21.97 by store count) |
|---|---|---|
| `dvc add out/results` | 1,909.7 s | ≈ 11.7 h |
| `dvc status`, cold | 13.7 s | ≈ 5.0 min |
| `dvc status`, warm | 6.1 s | ≈ 2.2 min |
| `dvc status`, after one modified file | 6.1 s | ≈ 2.2 min |
| `dvc commit` after one modified file | 10.4 s | ≈ 3.8 min |
| Cache size | 60.12 GB | ≈ 1.32 TB |

Linear extrapolation by store count; files per store and bytes per store are taken
as uniform across the dataset. `status`/`commit` likely scale sub-linearly (they
re-hash only changed files), so their extrapolations are upper bounds.

**Recommendation (plan P2 T6 rule):** the extrapolated `dvc add` exceeds an hour,
so the how-to page recommends `cache: false` for `<out>/results` on trees of this
size. That keeps `persist: true` and DVC's change detection (the out is still
hashed into `dvc.lock`) while not duplicating ~1.3 TB of stores into the cache.
`deliverables/` and `.phenotypic/` stay `cache: true`.

Not measured: how the 1,909.7 s `add` splits between hashing and copying into the
cache. `cache: false` saves only the copy, so the first hash of a full tree still
costs part of that time. The page should say so rather than promise 11.7 h → 0.
