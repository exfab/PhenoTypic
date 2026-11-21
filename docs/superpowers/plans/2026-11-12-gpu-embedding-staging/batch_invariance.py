"""#282 P2 Task 4: is a stored DINOv2 vector independent of its batch on one GPU model?

Spec goal 3 as restated (S1): within one GPU model and one physical batch size
``m``, a region's stored float16 vector must not depend on its batch
neighbours, its position in the batch, its shard, or the process. Across GPU
models, and between ``m`` and ``m/2`` (what an OOM halving does), the distance
is measured and reported, not required to be zero.

This drives the shipped code (``MeasureDinoEmbedding.prepare_regions`` /
``vectors_from``, ``BatchEncoder``, ``configure_determinism``,
``physical_batch_for``), so it lives beside the plan, not under
``logic_validation_scripts/``. Run it from a detached worktree at a fixed
commit (``submit_batch_invariance.sh``), never from a tree being edited.

Inputs: the three corrected Salt_081 stores spec §2 measured
(``gate-results/gpu-baseline/measure_embedding_baseline.py:25``), read only;
regions are the control pipeline's colonies (``objects``) and the approved
8 x 12 rig1 grid (``wells``).

Rows (DINOv2 only, U11): base float32 ``crop_cls``; base float32
``masked_dense``; large float32 ``crop_cls``; giant bfloat16 ``crop_cls``.

For each row, at the ``m`` the physical-batch rule gives on this GPU:

- ``reference``: every image submitted as one ticket per (region, image), in
  order, the way Stage 4 submits.
- Per-row cases, where each crop (or ``masked_dense`` tile) is its own
  ticket so its batch and position are exactly known: ``reversed``,
  ``shuffled`` (seeded), ``interleaved`` (neighbours from the other plates),
  ``position`` (every row once at position 0 and once at position ``m - 1``
  of a full batch; every other row in those batches is checked too),
  ``pad_1`` / ``pad_m_minus_1`` (the row's last batch zero-padded with 1, or
  with ``m - 1``, zeros).
- Ticket cases: ``shards`` (the images split across two ``BatchEncoder``
  instances on one model) and ``in_flight_1`` (``max_in_flight_bytes=1``,
  so every submission flushes every partial queue).

A case is invariant when every stored float16 vector is bit-identical to the
reference. For crops that is checked on every occurrence of every row; for
``masked_dense`` the stored vectors pool many tiles, so each full pass is
pooled and compared (``position`` gives two passes: the position-0 and the
position-``m - 1`` occurrences), and every tile occurrence's float32 patch
grid is compared bit for bit as well (information).

Also per row: ``half_m`` (the reference at ``m // 2``; max cosine distance,
information), ``f16_max_abs`` / ``f16_overflow`` (> 65504), and every warning
raised while it ran. A determinism warning (``use_deterministic_algorithms(
True, warn_only=True)``) fails the row. Warnings are recorded rather than
raised, so the row still finishes and every warning is listed.

Usage::

    batch_invariance.py OUT.json [--images N] [--rows a,b] [--repeat-of PREV.json]
    batch_invariance.py --compare OUT.json A.json B.json [C.json ...]

``--repeat-of`` runs the reference only and compares it bit for bit with an
earlier process's vectors (``PREV.npz``, written beside ``PREV.json``).
``--compare`` reads finished runs and writes the cross-GPU distances.
Exit status: 0 when every measured row is invariant (and repeat-identical)
with no determinism warning, 1 otherwise, 2 on a usage error.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import os
import socket
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Hashable

import numpy as np

STORES_DIR = Path(
    "/bigdata/exfab/anguy344/projects/ucr_012_i_d_RhodoHeavyMetals/data/timeseries/Salt_081"
)
CONTROL_PIPE = Path(
    "/bigdata/exfab/anguy344/phenotypic-example-data/control-v1/pipeline.json.pht-pipe"
)
WELLS = dict(
    row_edges=[393, 796, 1199, 1602, 2005, 2408, 2811, 3214, 3617],
    col_edges=[574, 977, 1380, 1783, 2186, 2589, 2992, 3395, 3798, 4201, 4604, 5007, 5410],
)
ROWS: dict[str, dict[str, str]] = {
    "base_crop_cls": {"size": "base", "compute_dtype": "float32", "pooling": "crop_cls"},
    "base_masked_dense": {"size": "base", "compute_dtype": "float32", "pooling": "masked_dense"},
    "large_fp32": {"size": "large", "compute_dtype": "float32", "pooling": "crop_cls"},
    "giant_bf16": {"size": "giant", "compute_dtype": "bfloat16", "pooling": "crop_cls"},
}
REGIONS = ("objects", "wells")
F16_MAX = 65504.0
SHUFFLE_SEED = 20261120

#: A row of the encoder: its unit (one image's crops for a region, or one
#: image's tiles) and its index there.
Ref = tuple[Hashable, int]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def load_images(n_images: int) -> list[dict[str, Any]]:
    """The last *n_images* of the three stores: detected (objects) and plain (wells)."""
    from phenotypic import GridImage, Image, ImagePipeline

    stores = sorted(STORES_DIR.glob("*.ome.zarr"))[-3:][:n_images]
    pipe = ImagePipeline.from_json(CONTROL_PIPE)
    images = []
    for store in stores:
        t0 = time.perf_counter()
        detected = pipe.apply(GridImage.imread(store, nrows=8, ncols=12))
        images.append({"store": store.name, "objects": detected, "wells": Image.imread(store),
                       "detect_s": round(time.perf_counter() - t0, 2)})
        print(f"loaded {store.name}: {int(detected.num_objects)} colonies", flush=True)
    return images


def make_embedder(row: str, region: str) -> Any:
    from phenotypic.grid import ManualGridFinder
    from phenotypic.measure import MeasureDinoEmbedding

    spec = ROWS[row]
    kw = dict(size=spec["size"], version=2, pooling=spec["pooling"],
              compute_dtype=spec["compute_dtype"], device="cuda")
    if region == "objects":
        return MeasureDinoEmbedding(regions="objects", **kw)
    return MeasureDinoEmbedding(regions="grid", grid=ManualGridFinder(**WELLS), **kw)


class RowSet:
    """Every row a backbone runs for one table row, grouped by queue geometry.

    One unit per (region, image): its crops, or its ``masked_dense`` tiles.
    The two regions read different images (the control pipeline crops the
    plate before detecting colonies), so their tiles are not shared.
    """

    def __init__(self, embedders: dict[str, Any], images: list[dict[str, Any]], processor: Any):
        from phenotypic.abc_ import BackboneInputs

        self.BackboneInputs = BackboneInputs
        self.embedders = embedders
        self.prepared: dict[tuple[str, int], Any] = {}
        self.unit_of: dict[tuple[str, int], Hashable] = {}
        self.unit_inputs: dict[Hashable, Any] = {}
        self.unit_rows: dict[Hashable, list[Ref]] = {}
        self.pixels: dict[Ref, np.ndarray] = {}
        self.group_of: dict[Ref, tuple] = {}
        self.image_of: dict[Hashable, int] = {}
        for img, image in enumerate(images):
            for region in REGIONS:
                prep = embedders[region].prepare_regions(image[region], processor=processor)
                self.prepared[(region, img)] = prep
                self.kind = prep.inputs.kind
                self.pooling = prep.inputs.pooling
                unit: Hashable = (region, img)
                if self.kind == "crops":
                    arrays = list(prep.inputs.arrays[0])
                else:
                    arrays = list(prep.inputs.arrays)
                self.unit_of[(region, img)] = unit
                self.unit_inputs[unit] = prep.inputs
                self.image_of[unit] = img
                self.unit_rows[unit] = [(unit, i) for i in range(len(arrays))]
                for i, px in enumerate(arrays):
                    self.pixels[(unit, i)] = px
                    self.group_of[(unit, i)] = (px.shape, self.pooling)
        self.units = list(self.unit_rows)
        self.all_rows = [r for u in self.units for r in self.unit_rows[u]]
        groups: dict[tuple, list[Ref]] = defaultdict(list)
        for r in self.all_rows:
            groups[self.group_of[r]].append(r)
        self.groups = dict(groups)

    def single(self, ref: Ref) -> Any:
        px = self.pixels[ref]
        if self.kind == "crops":
            return self.BackboneInputs(kind="crops", arrays=(px[None],), pooling=self.pooling)
        return self.BackboneInputs(kind="tiles", arrays=(px,), pooling="patches")

    def single_output(self, outputs: list[np.ndarray]) -> np.ndarray:
        return outputs[0][0] if self.kind == "crops" else outputs[0]

    def unit_outputs(self, unit: Hashable, outputs: list[np.ndarray]) -> dict[Ref, np.ndarray]:
        rows = self.unit_rows[unit]
        if self.kind == "crops":
            return {r: outputs[0][i] for i, r in enumerate(rows)} if rows else {}
        return {r: outputs[i] for i, r in enumerate(rows)}

    def stored(self, pass_: dict[Ref, np.ndarray]) -> dict[tuple[str, int], np.ndarray]:
        """The float32 vectors each (region, image) would store, pooled from one pass."""
        vectors = {}
        for (region, img), prep in self.prepared.items():
            rows = self.unit_rows[self.unit_of[(region, img)]]
            outs = [pass_[r] for r in rows]
            if prep.n_regions == 0:
                vectors[(region, img)] = np.zeros((0, 0), dtype=np.float32)
                continue
            outputs = [np.stack(outs)] if self.kind == "crops" else outs
            vectors[(region, img)] = np.asarray(
                self.embedders[region].vectors_from(prep, outputs), dtype=np.float32
            )
        return vectors


# --------------------------------------------------------------------------- #
# Running sequences
# --------------------------------------------------------------------------- #


def run_sequence(enc: Any, rows: RowSet, seq: list[tuple[Ref, str]],
                 on_out: Callable[[Ref, str, np.ndarray], None]) -> None:
    """Submit each row of *seq* as its own ticket, in order; route every output to *on_out*."""
    def drain() -> None:
        for done in enc.drain():
            _, j = done.ticket
            ref, label = seq[j]
            on_out(ref, label, rows.single_output(done.outputs))

    for j, (ref, _) in enumerate(seq):
        enc.submit(("occ", j), rows.single(ref))
        drain()
    enc.flush()
    drain()


def run_units(enc: Any, rows: RowSet, units: list[Hashable]) -> dict[Ref, np.ndarray]:
    """Submit one ticket per unit, as Stage 4 does; the pass it gives."""
    pass_: dict[Ref, np.ndarray] = {}
    for unit in units:
        enc.submit(unit, rows.unit_inputs[unit])
        for done in enc.drain():
            pass_.update(rows.unit_outputs(done.ticket, done.outputs))
    enc.flush()
    for done in enc.drain():
        pass_.update(rows.unit_outputs(done.ticket, done.outputs))
    return pass_


def cyc(group: list[Ref], start: int, count: int) -> list[Ref]:
    return [group[(start + j) % len(group)] for j in range(count)]


def interleaved(rows: RowSet, group: list[Ref]) -> list[Ref]:
    by_image: dict[int, list[Ref]] = defaultdict(list)
    for r in group:
        by_image[rows.image_of[r[0]]].append(r)
    lists = [by_image[k] for k in sorted(by_image)]
    return [r for tup in itertools.zip_longest(*lists) for r in tup if r is not None]


def position_sequence(group: list[Ref], m: int) -> list[tuple[Ref, str]]:
    """``len(group)`` full batches: row k at position 0, row k + n//2 at position m - 1."""
    n, seq = len(group), []
    for k in range(n):
        if m == 1:
            seq.append((group[k], "pos0"))
            continue
        seq.append((group[k], "pos0"))
        seq.extend((r, "mid") for r in cyc(group, k + 1, m - 2))
        seq.append((group[(k + n // 2) % n], "posm1"))
    return seq


def padded_sequence(group: list[Ref], m: int, last_real: int) -> list[tuple[Ref, str]]:
    """*group*, then cyclic filler, so the last batch holds *last_real* rows; label ``last``."""
    filler = (last_real - len(group)) % m
    seq = list(group) + cyc(group, 0, filler)
    last = {r: j for j, r in enumerate(seq)}
    return [(r, "last" if last[r] == j else "mid") for j, r in enumerate(seq)]


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #


def f16_bits(v: np.ndarray) -> np.ndarray:
    return np.asarray(v, dtype=np.float32).astype(np.float16).view(np.uint16)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.sum(a * b, axis=1)


def compare_stored(ref: dict, other: dict) -> dict[str, Any]:
    """Stored float16 identity of two passes' pooled vectors, every (region, image).

    ``rows_by_region`` counts the vectors compared per region, so
    :func:`stored_invariant` can refuse a comparison that compared none.
    """
    mismatch_rows, total, max_abs, cos = 0, 0, 0.0, [0.0]
    rows_by_region = dict.fromkeys(REGIONS, 0)
    for key, a in ref.items():
        b = other[key]
        if a.size == 0:
            continue
        diff_rows = np.any(f16_bits(a) != f16_bits(b), axis=1)
        mismatch_rows += int(diff_rows.sum())
        total += len(a)
        rows_by_region[key[0]] += len(a)
        max_abs = max(max_abs, float(np.max(np.abs(a - b))))
        d = cosine_distance(a.astype(np.float16), b.astype(np.float16))
        d[~diff_rows] = 0.0          # identical stored bits: no rounding residue
        cos.extend(d.tolist())
    return {"rows": total, "rows_by_region": rows_by_region, "f16_mismatch_rows": mismatch_rows,
            "f32_max_abs": max_abs, "f16_max_cos": float(np.max(cos))}


def stored_invariant(stats: dict[str, Any]) -> bool:
    """No stored vector differs, and every region compared at least one (#282 review M4).

    A detection with no colonies would otherwise make the ``objects`` half
    pass vacuously.
    """
    return stats["f16_mismatch_rows"] == 0 and all(n > 0 for n in stats["rows_by_region"].values())


class OccurrenceCheck:
    """Compares each occurrence's raw output with the reference row's, as it arrives."""

    def __init__(self, rows: RowSet, ref_out: dict[Ref, np.ndarray], keep: set[str]):
        self.rows, self.ref_out, self.keep = rows, ref_out, keep
        self.passes: dict[str, dict[Ref, np.ndarray]] = {k: {} for k in keep}
        self.occurrences = self.f32_mismatch = self.f16_mismatch = 0
        self.f32_max_abs = 0.0

    def __call__(self, ref: Ref, label: str, out: np.ndarray) -> None:
        expected = self.ref_out[ref]
        self.occurrences += 1
        if not np.array_equal(out.view(np.uint32), expected.view(np.uint32)):
            self.f32_mismatch += 1
            self.f32_max_abs = max(self.f32_max_abs, float(np.max(np.abs(out - expected))))
        if self.rows.kind == "crops" and not np.array_equal(f16_bits(out), f16_bits(expected)):
            self.f16_mismatch += 1          # a crop's stored vector is its output, cast
        if label in self.keep:
            self.passes[label][ref] = out

    def summary(self) -> dict[str, Any]:
        return {"occurrences": self.occurrences, "f32_bit_mismatch_occurrences": self.f32_mismatch,
                "f32_max_abs": self.f32_max_abs,
                **({"f16_mismatch_occurrences": self.f16_mismatch} if self.rows.kind == "crops" else {})}


# --------------------------------------------------------------------------- #
# One row
# --------------------------------------------------------------------------- #


def m_for(model: Any, embedder: Any) -> tuple[int, dict[str, Any]]:
    """``m`` and its inputs, from the shipped producer Stage 4 also calls (#282 review I3)."""
    import torch

    from phenotypic.detect.nn._helper._batch_encoder import (
        physical_batch_for,
        physical_batch_inputs,
    )

    total = int(torch.cuda.get_device_properties(0).total_memory)
    kw = dict(crop_px=embedder.crop_px, compute_dtype=embedder.compute_dtype)
    m = physical_batch_for(model, batch_size=embedder.batch_size, gpu_total_bytes=total, **kw)
    return m, {"batch_size": embedder.batch_size, "gpu_total_bytes": total,
               **physical_batch_inputs(model, **kw)}


def measure_row(row: str, images: list[dict[str, Any]], reference_only: bool) -> tuple[dict, dict]:
    from phenotypic.detect.nn._helper._batch_encoder import BatchEncoder, configure_determinism
    from phenotypic.detect.nn._helper._dino_support import load_dino_backbone

    spec = ROWS[row]
    result: dict[str, Any] = {**spec}
    t_row = time.perf_counter()
    result["determinism"] = configure_determinism(device="cuda")
    t0 = time.perf_counter()
    model, processor = load_dino_backbone(2, spec["size"], "cuda", dtype=spec["compute_dtype"])
    result["load_s"] = round(time.perf_counter() - t0, 2)
    embedders = {region: make_embedder(row, region) for region in REGIONS}
    t0 = time.perf_counter()
    rows = RowSet(embedders, images, processor)
    result["prepare_s"] = round(time.perf_counter() - t0, 2)
    m, m_inputs = m_for(model, embedders["objects"])
    result.update(m=m, m_inputs=m_inputs, kind=rows.kind, encoder_rows=len(rows.all_rows),
                  geometries={str(k[0]): len(v) for k, v in rows.groups.items()},
                  regions={f"{r}/{i}": p.n_regions for (r, i), p in rows.prepared.items()})

    def encoder(physical_batch: int = m, **kw: Any) -> Any:
        return BatchEncoder(model, processor, device="cuda", compute_dtype=spec["compute_dtype"],
                            physical_batch=physical_batch, **kw)

    cases: dict[str, Any] = {}
    t0 = time.perf_counter()
    enc = encoder()
    ref_out = run_units(enc, rows, rows.units)
    assert not enc.halved, "the reference halved m (out of memory)"
    ref = rows.stored(ref_out)
    cases["reference"] = {"seconds": round(time.perf_counter() - t0, 2)}
    all_ref = np.concatenate([v for v in ref.values() if v.size], axis=0)
    result["f16_max_abs"] = float(np.max(np.abs(all_ref)))
    result["f16_overflow"] = bool(np.any(np.abs(all_ref) > F16_MAX))
    stored16 = {f"{row}/{region}/{img}": v.astype(np.float16) for (region, img), v in ref.items()}
    if reference_only:
        result["cases"] = cases
        result["row_s"] = round(time.perf_counter() - t_row, 2)
        return result, stored16

    rng = np.random.default_rng(SHUFFLE_SEED)
    per_row_cases: dict[str, Callable[[list[Ref]], list[tuple[Ref, str]]]] = {
        "reversed": lambda g: [(r, "pass") for r in reversed(g)],
        "shuffled": lambda g: [(g[i], "pass") for i in rng.permutation(len(g))],
        "interleaved": lambda g: [(r, "pass") for r in interleaved(rows, g)],
        "position": lambda g: position_sequence(g, m),
        "pad_1": lambda g: padded_sequence(g, m, m - 1 if m > 1 else 0),
        "pad_m_minus_1": lambda g: padded_sequence(g, m, 1),
    }
    keeps = {"reversed": {"pass"}, "shuffled": {"pass"}, "interleaved": {"pass"},
             "position": {"pos0", "posm1"}, "pad_1": {"last"}, "pad_m_minus_1": {"last"}}
    for name, build in per_row_cases.items():
        t0 = time.perf_counter()
        seq = [item for group in rows.groups.values() for item in build(group)]
        check = OccurrenceCheck(rows, ref_out, keeps[name])
        enc = encoder()
        run_sequence(enc, rows, seq, check)
        record: dict[str, Any] = {"halved": enc.halved, **check.summary()}
        passes = {}
        for label, pass_ in check.passes.items():
            missing = len(rows.all_rows) - len(pass_)
            assert missing == 0, f"{name}/{label}: {missing} rows have no occurrence"
            passes[label] = compare_stored(ref, rows.stored(pass_))
        record["passes"] = passes
        f16_ok = bool(passes) and all(stored_invariant(p) for p in passes.values())
        if rows.kind == "crops":
            f16_ok = f16_ok and check.f16_mismatch == 0
        record["invariant"] = bool(f16_ok and not record["halved"])
        record["seconds"] = round(time.perf_counter() - t0, 2)
        cases[name] = record
        del check, passes
        gc.collect()
        print(f"  {row} {name}: invariant={record['invariant']} {record['seconds']}s", flush=True)

    # Ticket cases: two shards on one model; every submission flushing every queue.
    t0 = time.perf_counter()
    images_of = sorted({rows.image_of[u] for u in rows.units})
    if len(images_of) > 1:
        shard_a = [u for u in rows.units if rows.image_of[u] % 2 == 0]
        shard_b = [u for u in rows.units if rows.image_of[u] % 2 == 1]
    else:
        half = len(rows.units) // 2 or 1
        shard_a, shard_b = rows.units[:half], rows.units[half:]
    enc_a, enc_b = encoder(), encoder()
    shard_pass = {**run_units(enc_a, rows, shard_a), **run_units(enc_b, rows, shard_b)}
    stats = compare_stored(ref, rows.stored(shard_pass))
    halved = enc_a.halved or enc_b.halved
    cases["shards"] = {**stats, "halved": halved, "shard_units": [len(shard_a), len(shard_b)],
                       "invariant": stored_invariant(stats) and not halved,
                       "seconds": round(time.perf_counter() - t0, 2)}
    del shard_pass, enc_a, enc_b

    t0 = time.perf_counter()
    enc = encoder(max_in_flight_bytes=1)
    stats = compare_stored(ref, rows.stored(run_units(enc, rows, rows.units)))
    cases["in_flight_1"] = {**stats, "halved": enc.halved,
                            "invariant": stored_invariant(stats) and not enc.halved,
                            "seconds": round(time.perf_counter() - t0, 2)}

    if m >= 2:
        t0 = time.perf_counter()
        enc = encoder(physical_batch=m // 2)
        stats = compare_stored(ref, rows.stored(run_units(enc, rows, rows.units)))
        result["half_m"] = {"m": m // 2, **stats, "identical": stats["f16_mismatch_rows"] == 0,
                            "seconds": round(time.perf_counter() - t0, 2)}
        result["half_m_max_cos"] = stats["f16_max_cos"]

    result["cases"] = cases
    result["invariant"] = all(c["invariant"] for k, c in cases.items() if k != "reference")
    result["row_s"] = round(time.perf_counter() - t_row, 2)
    return result, stored16


def is_determinism_warning(message: str) -> bool:
    text = message.lower()
    return "deterministic" in text or "determinism" in text


def run_row_recording_warnings(row: str, images: list, reference_only: bool) -> tuple[dict, dict]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result, stored16 = measure_row(row, images, reference_only)
        except Exception as exc:  # one row's failure is recorded, the others still run
            import traceback

            result, stored16 = {**ROWS[row], "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc()}, {}
    import torch

    gc.collect()                    # the row's model is unreachable once measure_row returns
    torch.cuda.empty_cache()
    messages = sorted({f"{w.category.__name__}: {w.message}" for w in caught})
    result["determinism_warnings"] = [m for m in messages if is_determinism_warning(m)]
    result["other_warnings"] = [m[:400] for m in messages if not is_determinism_warning(m)]
    return result, stored16


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def environment() -> dict[str, Any]:
    import torch

    major, minor = torch.cuda.get_device_capability(0)
    tree = Path(__file__).resolve().parents[4]
    commit = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    import phenotypic

    return {"gpu": torch.cuda.get_device_name(0), "capability": f"{major}.{minor}",
            "torch": torch.__version__, "cuda": torch.version.cuda, "node": socket.gethostname(),
            "job": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "commit": commit, "dirty": bool(dirty), "phenotypic": phenotypic.__file__,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def measure(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        print("CUDA is not available", file=sys.stderr)
        return 2
    out = Path(args.out)
    result: dict[str, Any] = {**environment(), "images": [], "models": {}}
    rows = args.rows.split(",") if args.rows else list(ROWS)
    previous = None
    if args.repeat_of:
        prev_json = json.loads(Path(args.repeat_of).read_text())
        previous = dict(np.load(Path(args.repeat_of).with_suffix(".npz")))
        result["repeat_of"] = {"file": args.repeat_of, "job": prev_json.get("job"),
                               "node": prev_json.get("node"), "gpu": prev_json.get("gpu")}
        if prev_json.get("gpu") != result["gpu"]:
            print(f"repeat on {result['gpu']!r}, not {prev_json.get('gpu')!r}", file=sys.stderr)
            return 2
        rows = [r for r in rows if r in prev_json["models"] and "error" not in prev_json["models"][r]]
        if not rows:  # #282 review M4: an empty repeat must not print PASS
            print(f"no row of {args.repeat_of} finished; there is nothing to repeat", file=sys.stderr)
            return 1
    t0 = time.perf_counter()
    images = load_images(args.images)
    result["images"] = [{"store": i["store"], "detect_s": i["detect_s"]} for i in images]
    result["images_s"] = round(time.perf_counter() - t0, 2)
    stored_all: dict[str, np.ndarray] = {}
    for row in rows:
        print(f"row {row}", flush=True)
        record, stored16 = run_row_recording_warnings(row, images, reference_only=previous is not None)
        if previous is not None and stored16:
            mismatched = [k for k, v in stored16.items()
                          if k not in previous or not np.array_equal(v.view(np.uint16),
                                                                     previous[k].view(np.uint16))]
            record["repeat_identical"] = not mismatched
            record["repeat_mismatched"] = mismatched
        stored_all.update(stored16)
        result["models"][row] = record
        print(json.dumps({k: v for k, v in record.items() if k not in ("traceback", "cases")}),
              flush=True)
        np.savez(out.with_suffix(".npz"), **stored_all)
        out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    result["total_s"] = round(time.perf_counter() - t0, 2)
    out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")

    ok = verdict(result["models"], repeat=previous is not None)
    print(f"VERDICT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


def verdict(models: dict[str, dict[str, Any]], *, repeat: bool) -> bool:
    """PASS: at least one row ran, and every row is invariant (or repeat-identical) with no
    error and no determinism warning. No row is a FAIL, never a vacuous PASS (#282 review M4)."""
    if not models:
        return False
    for record in models.values():
        if "error" in record or record["determinism_warnings"]:
            return False
        if not repeat and not record["invariant"]:
            return False
        if repeat and not record.get("repeat_identical", False):
            return False
    return True


def compare(out: Path, runs: list[Path]) -> int:
    """Cross-GPU distance per row: median and max cosine distance of the stored float16 vectors."""
    loaded = []
    for path in runs:
        if not path.exists() or not path.with_suffix(".npz").exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        loaded.append((json.loads(path.read_text()), dict(np.load(path.with_suffix(".npz")))))
    table: dict[str, Any] = {"runs": [{"file": str(p), "gpu": j["gpu"], "job": j["job"]}
                                      for p, (j, _) in zip(runs, loaded)], "rows": {}}
    for row in ROWS:
        pairs = {}
        for (ja, va), (jb, vb) in itertools.combinations(loaded, 2):
            keys = sorted(k for k in va if k.startswith(row + "/") and k in vb and va[k].size)
            if not keys:
                continue
            a = np.concatenate([va[k] for k in keys]).astype(np.float32)
            b = np.concatenate([vb[k] for k in keys]).astype(np.float32)
            same = np.all(a.astype(np.float16).view(np.uint16) == b.astype(np.float16).view(np.uint16), axis=1)
            cos = np.where(same, 0.0, cosine_distance(a, b))
            pairs[f"{ja['gpu']} vs {jb['gpu']}"] = {
                "rows": len(a), "median_cos": float(np.median(cos)), "max_cos": float(np.max(cos)),
                "bit_identical_rows": int(np.sum(same)),
                "max_abs": float(np.max(np.abs(a - b)))}
        table["rows"][row] = pairs
    out.write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps(table, indent=1))
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("out")
    parser.add_argument("runs", nargs="*", type=Path)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--repeat-of")
    parser.add_argument("--images", type=int, default=3)
    parser.add_argument("--rows", default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    ns = parse_args(sys.argv[1:])
    if ns.compare:
        sys.exit(compare(Path(ns.out), ns.runs))
    sys.exit(measure(ns))
