# Finding the colour chips automatically: experiment summary and recommendation

**Project:** SnP-ColorCorrection · plate-imaging rig, *Rhodotorula* screening
**Question:** can the ColorChecker patches be located and measured automatically,
reliably enough to run a per-plate colour correction over a whole screening batch?
**Answer:** yes. Two configurations reach 100 % correct tile placement and labelling
on a 20-plate test set spanning eight runs and six months, and the remaining risk is
not localisation — it is knowing when something is sitting on top of the card.

This is a summary. The full method log, including the derivations and the dead ends,
is in `chip_detection_methods.md`.

---

## 1. Why this was needed

A per-plate fit is the right correction for this rig, and a per-plate fit needs the
24 patch colours measured from *that plate's own image*. Doing that by hand does not
scale to a screening batch. The existing pipeline used fixed pixel coordinates, which
produced a visible smear on the corrected images whenever the plate or the card had
shifted — the tile boxes were partly measuring the wrong thing, and the reflect-padding
used to rebuild clipped tiles turned that into a blur.

So the problem is not "find a colour chart in a photograph". It is narrower and easier:
the card is bolted to the rig and moves by a few pixels between sessions. The task is
to **refine a stored layout**, not to search blindly.

## 2. What the rig actually looks like

Each frame carries a **2 x 6 half-card at each edge** — twelve patches on the left, twelve
on the right, together making up the 24-patch chart. Measured geometry: the card occupies
rows 950–3100 of the frame, within 340 px of each edge; tiles are about 195 x 65 px on a
255.5 px pitch running down the frame.

The mapping from rig position to chart position is fixed:

| band | column | chart row | contents |
|---|---|---|---|
| left | inner | 1 | dark skin … bluish green |
| left | outer | 2 | orange … yellow green |
| right | outer | 3 | blue, green, red, yellow, magenta, cyan |
| right | inner | 4 | white → black neutrals |

Top-to-bottom in the frame is chart columns A → F.

Two consequences matter for detection. The **outer** column on each side is clipped by
the frame border, so part of every one of those tiles is missing — which is why methods
that track a patch's visible centre drift on that column. And the **inner** columns are
*not* obstructed: the plate sits inboard of them. An earlier read of this was wrong; the
plate never overlaps the patch bodies, it only approaches them. The wall between plate
and card does throw refracted ghost copies of the neutral column into the empty strip at
right-band x ≈ 100–150, outside the card, which is a trap for blob-finding methods.

![rig geometry]({{artifact:art_8b93df02-30ea-47bc-873c-d074abd6c90d}})

## 3. How much does it actually move?

Small, and structured:

- Within one imaging session (Sept 2025): **≤ 4.4 px**.
- Across five months (Sept 2025 → Feb 2026): **dy −9.7, dx +8.0** on the left band,
  **dy −3.3, dx +8.4** on the right. Rotation **≤ 0.08°** throughout.
- Across the 20-plate test set (Feb–Aug 2026), measured with the recommended method:
  left band dx **+10 to +14**, dy **−14 to −7**; right band dx **+0.5 to +6**, dy **−6 to +5**.

So a detector needs a capture range of a few tens of pixels, not hundreds, and rotation
can be ignored — though it is worth *measuring* rotation as a fault signal.

## 4. Methods tested

All six work on the same stored layout and differ only in how they refine it.

| method | how it finds the card | needs a reference image? |
|---|---|---|
| `M0_frozen` | uses the stored layout unchanged | no |
| `M1_translation` | cross-correlates the band against a stored reference band | **yes** |
| `M1_euclidean` | ECC registration against a stored reference band, allows rotation | **yes** |
| `M2_profile` | finds each column's own bright/dark run down the band and re-snaps to it | no |
| `M2_rigid` | measures the shift on the unclipped inner column, applies it rigidly to both | no |
| `M3_blob` | segments patch-like blobs and fits the lattice to them | no |

A note on the word *reference*, because it is overloaded here. The `M1_*` methods need a
reference **image** — the band cropped from a stored calibration frame. Separately, every
configuration needs reference **colour values** — the 24 chart patches — for the identity
step and for the correction itself. "Reference-free" means the first, never the second.

`colour-checker-detection` 0.2.3 was tried as an off-the-shelf alternative and found
**zero** charts in all 28 parameter configurations, on plate frames and on an unobstructed
control capture alike. It expects one contiguous 4 x 6 chart; a card split into two
half-cards at opposite frame edges is outside what it does.

## 5. How they were tested

- **Capture range**: synthetic shifts applied to real bands, swept ±60 px in each axis,
  8 bands; a method "holds" where its worst-case error stays under 3 px.
- **Field test**: 20 plate frames from 8 runs, Feb–Aug 2026, 40 bands, 480 tiles.
- **Downstream**: the full correction refitted from each method's measurements.

---

## 6. Outcomes

### 6.1 Capture range

| method | dy holds | dx holds | s/band |
|---|---|---|---|
| `M0_frozen` | ±3 | ±3 | 0.000 |
| `M1_translation` | ±60 (sweep limit) | −6 … +3 | 0.012 |
| `M1_euclidean` | ±60 (sweep limit) | ±40 | 0.059 |
| `M2_profile` | ±30 | −10 … +6 | 0.265 |
| `M2_rigid` | ±45 | **±30** | 0.361 |
| `M3_blob` | ±45 | −10 … +6 | 1.475 |
| `M23_cascade` | ±45 | ±6 | 1.835 |

Vertical displacement is easy — everything except the frozen layout handles far more than
the rig produces. Horizontal is where methods separate, because the clipped outer column
gives a misleading signal: as the card moves, that tile's visible extent changes and its
apparent centre travels only half as far. `M2_profile` re-snaps each column independently
and so inherits that error; `M2_rigid` measures the shift on the *unclipped* column only
and applies it rigidly, which is what buys it ±30 px with no reference image.

Cascading blob-finding into rigid refinement (`M23_cascade`) was tried and is **worse than
`M2_rigid` alone** (dx ±6) at five times the cost. Rejected.

![capture range]({{artifact:art_d8f52a07-99cc-4c23-a6ba-11905b50ce86}})

### 6.2 The win condition: does a box include plate-veiled pixels?

The criterion that matters operationally is not clearance from some fixed line — the plate
shifts by a few pixels from frame to frame, so there is no fixed line. It is whether a
measurement box includes the hazy region where the plate veils the patch.

Measured per frame, against each tile's own interior colour: **plate-side contamination is
0.000 for every method, on all 720 left-band box-tiles.** No method reaches the veil.

The only left-band contamination is on the *opposite* edge, the dark card border, and only
for methods that fail to follow the card's horizontal shift:

| method | card-border side, mean / max | plate side, max |
|---|---|---|
| `M2_rigid` | 0.0000 / 0.0000 | 0.000 |
| `M1_euclidean` | 0.0001 / 0.0055 | 0.000 |
| `M1_translation` | 0.0001 / 0.0055 | 0.000 |
| `M2_profile` | 0.0006 / 0.0070 | 0.000 |
| `M0_frozen` | 0.0553 / 0.3475 | 0.000 |
| `M3_blob` | 0.0851 / 0.6104 | 0.000 |

The right band's neutral column is tighter, because of the refracted ghosts in the adjacent
strip. Over the 19 unoccluded frames: `M2_rigid` 0.6 % mean / 9.5 % max — the best of the
six; `M3_blob` 2.4 % / 49 % — the worst.

![veil]({{artifact:art_a1be2d3c-4c2e-46c6-aa84-32d00401692d}})

### 6.3 Identity: which patch is which

Localisation never reads patch colour, so a rotated or re-mounted card is still *found*.
Deciding *which* chart patch sits in each cell is a separate step, and it is solved rather
than assumed: `assign_identity` enumerates the eight discrete ways a 2 x 6 half-card can sit
— which pair of chart rows, in which order, plus the 180° flip that reverses A–F — and
scores each directly.

Two details earned their place:

**The reference must be illuminant-adapted.** The X-Rite values are defined under D50. Pushing
them through a D65-assuming conversion displaces the saturated patches by up to **6.4 Lab
units** (A3 6.38, B2 5.60, F3 4.83) while moving the neutrals by ≤ 0.27.
`reference_linear_rgb()` reads the chart from colour-science and Bradford-adapts D50 → D65.

To be exact about what that bought: the displacement is measured, the consequence is not.
Swapping the un-adapted reference for the adapted one changed **no labels** in the A/B on
this rig — both gave left 12/12 and right 8/12. Adapt anyway, because the un-adapted
comparison is wrong on principle and the error lands precisely where patches sit closest
together, but the correctness gain here came from the scoring change below, not from the
adaptation.

**Score hypotheses directly, do not solve a free assignment.** A free permutation lets labels
wander onto whichever reference happens to fit. On the occluded band it scored 8/12 and
chose the wrong chart rows; direct scoring — "under this hypothesis, cell *k* must be patch
`ids[k]`" — took the same data to 12/12.

Result over all 40 bands: **480/480 tiles correct upright, 480/480 with the tile order
reversed** to simulate a rotated card. One unanimous hypothesis per side — `rows(2,1)` left,
`rows(4,3)` right, and the matching flipped variants when rotated — with the margin to the
runner-up never below **0.365** (left 0.365–0.409, right 0.367–0.484).

One limitation, stated plainly because it is easy to miss. The option to let only *clean*
tiles vote — intended for occluded bands — **fails on the one occluded band in the set**:
restricted to the six clean tiles it picks the wrong hypothesis and gets 0/12, because six
tiles from a single chart row do not constrain the orientation. Its margin collapses to
0.044, and that low margin is the only signal that catches it. So `margin < 0.1` must be
treated as "orientation undetermined" and fall back to the expected layout. With all twelve
tiles voting, the same band labels 12/12 correctly — which is what the 480/480 validation
exercises. Pass the clean mask only when most tiles are clean.

![overlays]({{artifact:art_1269f79e-1190-47e3-8526-9ea6764583c0}})

### 6.4 Detection is not the accuracy bottleneck

Refitting the correction from each method's measurements, on four calibration frames:
mean ΔE2000 after correction is **5.33–5.38** across the five detectors — a spread of less
than **0.05**, against a before-correction error around 20. Any of them is good enough for
the fit. The choice between them is about **robustness and fault reporting**, not accuracy.

![downstream]({{artifact:art_0e23065d-8a40-4869-9f26-2031a6466457}})

### 6.5 The one real failure: something on the card

In `CadmiumArrayRun`, a loose barcode sticker lies across the right band's neutral column.
**No method resists it** — the patches are physically covered, so every box measures the
sticker. The methods differ only in whether they *report* it.

Running the recommended method over all 40 bands with the impurity check, six tiles exceed
the 5 % limit, across two frames of that run:

| frame | tile | impurity |
|---|---|---|
| d000375_300_020 | F4 | 0.943 |
| d000375_300_020 | C4 | 0.700 |
| d000375_300_020 | E4 | 0.381 |
| d000375_300_020 | D4 | 0.368 |
| d000375_300_020 | B4 | 0.344 |
| d000378_300_100 | F4 | 0.087 |

`d000377_300_060` is genuinely clean (max 2.7 %), and the left band is 0.000 on all 240
tiles. The 8.7 % tile in `d000378` is useful: it is a mild, marginal case against which the
5 % limit can be judged, which the 94 % tile cannot provide.

The uncomfortable finding here concerns `M2_rigid`: on the occluded band its own confidence
reads **35.1**, inside — and near the top of — the clean range **25.4–36.0**. A reference-free
pipeline trusting that number alone would accept the frame. This is why the gate does not
rest on any single method's confidence.

### 6.6 Quality gate

Five signals: displacement from the stored layout, disagreement between methods, ECC
correlation (reference mode only), identity agreement, and **tile impurity** — the fraction
of pixels in a box that differ from that box's own median colour, which is what catches an
object lying across the card.

**Correction, from a defect found by inspecting the overlay figure.** The gate originally
tested only the *mean* impurity over a band's twelve tiles. A defect confined to two or
three tiles is diluted by the nine clean ones and passes: on the Sept-2025 calibration band
`d000220_300_038` right, tiles C3 (14.4 %), B3 and F4 all exceed the per-tile limit while
the twelve-tile mean is 2.8 % — and the band passed. The barcode sticker was caught only
because it covered five of six tiles and dragged the mean up. The gate now tests the
**worst tile** as well as the mean, and names the offending tiles in the flag.

The per-tile limit has to be mode-aware. Over 48 bands, worst-tile impurity on the 38 bands
with no known defect reaches 4.5 % reference-free but 5.7 % in reference mode — a cluster of
5.1–5.7 % readings on tile C4 across nine frames, which is `M1_euclidean`'s box placement
catching a patch edge, not an occluder. The mildest genuine defect is 6.3 %. So reference-free
separates the two populations by a factor of 1.4 at a 5 % limit, while reference mode needs
6 % and separates them by 0.6 points. One more reason to prefer the reference-free panel.

**Second correction, from the same source.** Testing the worst tile made the gate reject
`d000378` on a single tile at 8.7 % impurity — and that rejection was wrong. Impurity counts
contaminated *pixels*; it does not say whether they moved the *answer*. A per-channel median
already absorbs a minority of outliers, and on that tile it did: the median shifts by
**0.57 ΔE** and lands 3.6 ΔE from the clean-frame consensus, against a 2.8 ΔE spread among
the clean frames themselves. Detectable, not harmful. The same statistic on the barcode
sticker moves **2.5–18 ΔE**.

The gate is therefore restructured around *harm* rather than *presence*:

| signal | what it answers | verdict |
|---|---|---|
| `tile_impurity` (per tile) | is something there? | **warning** — reported and named, band still passes |
| `tile_impurity` (12-tile mean) | is most of the card covered? | reject |
| `tile_robust_shift` | did it move the measured colour? | reject above 1.5 ΔE |
| `tile_clipped` | did the sensor record this channel at all? | reject above 20 % of pixels |

`tile_clipped` is a third, independent fault. A channel pinned at the sensor floor is not an
outlier, so neither impurity nor the robust shift sees it, and no estimator can recover a
value that was never recorded.

With that structure the pipeline passes **42 of 48 bands in both modes**. One rejection is
the barcode sticker; five are Sept-2025 calibration bands with a clipped channel (6.7).
`d000378` now passes with a warning. The occlusion failure:

- reference mode: methods disagree by 10.4 px · ECC 0.82 · impurity 18.9 % — three signals
- reference-free mode: methods disagree by 22.2 px · impurity 22.9 % — **two signals**

The reference-free path used to catch it on three signals, identity included. Routing the
gate through the improved identity stage removed that one: direct hypothesis scoring now
labels the occluded band 12/12 correctly, so identity no longer objects to it. The better
identity method is the *less* diagnostic one here. Nothing is misgated as a result — the
band is still caught, and every clean band still passes — but the reference-free path now
has two independent objections rather than three, which is worth knowing before leaning on
it unsupervised.

The earlier version of the gate polled `M2_profile` and `M3_blob`, whose column estimates
are erratic, and flagged 16 of 40 bands — of which only one, the occluded band, was
genuinely faulty. Fifteen false alarms in forty is useless in production.

A note on the units these thresholds are expressed in: the impurity figure is a ΔE
computed by handing **linear** band values to a conversion that expects sRGB. It is a
consistent internal contrast measure, not calibrated CIE ΔE2000. Re-running the check with
a properly encoded band changes no gate decision on the test set — the same five tiles
exceed 5 % on the occluded band, none elsewhere — and returns slightly *smaller* numbers,
so the thresholds as set are the conservative side of the choice. The figures should not be
quoted as perceptual ΔE2000.

![qc gate]({{artifact:art_45abb2ae-fdee-4b6e-9553-0b38eeecef03}})

---

### 6.7 The Sept-2025 calibration frames are black-clipped

Found by following up an annotation on the overlay figure. In all three Sept-2025
calibration frames, some patches have a colour channel floored at exactly zero over much of
the tile: 20 (tile, channel) pairs across 16 tiles read >20 % of pixels at zero while the
reference value for that channel is appreciably positive. The worst cases are B3's red
channel at 67–75 % of pixels floored against a reference of 0.046, and C3's blue and green
at 43–64 % against 0.041 and 0.036. Those tiles report a colour the sensor did not record.

Two things make this matter more than a data-quality footnote:

- **`d000220_300_021` is the stored registration reference frame**, and with the corrected
  gate it now fails QC on *both* of its bands. The frame the `M1_*` methods align to is the
  worst frame in the set.
- **The 20 plate frames are clean** — zero channel-tiles floored above 20 %, and zero
  ceiling-clipped. So this is a property of the Sept-2025 session's exposure, not of the rig
  or the pipeline, and it does not affect any plate measurement.

Detection is unaffected either way: those bands are still located correctly and labelled
12/12. But a calibration frame whose saturated patches are clipped cannot anchor a colour
fit, and the reference frame should be re-shot or replaced with `d000320_300_003`, which is
clean on both bands.

## 7. Claims made during this work and then withdrawn

Recorded because the corrected versions are the ones to act on.

1. **"The wall hazes the patches nearest the plate."** Withdrawn. The apparent gain
   difference between chart rows is confounded with patch content — the affected rows are
   the neutrals and muted naturals, the others are the saturated patches. An unobstructed
   control capture of the same chart shows the same divergence (0.238, against 0.227–0.243
   on the plate frames). The proposal to fit on rows 2–3 only is withdrawn with it; **fit all
   24 patches**.
2. **"Method X crosses the plate boundary."** Withdrawn. Clearance had been measured against
   a boundary frozen from the 2025 frame, but that edge moves with the card. Re-derived per
   frame, no method reaches the plate-side edge. `M3_blob` does cross on the *card-border*
   side (−1.4 px), which is a different and less serious fault.
3. **"52 % of this tile is veiled."** Withdrawn. The veil metric had used the outer 30 % of
   each tile as its clean reference, but on this card that region is the dark border
   (L\* ≈ 25 against a patch body of L\* ≈ 49), so a clean tile looked half-covered and the
   ranking inverted to favour the frozen layout. With the tile's *interior* as reference the
   same tile is flat at 0.5–0.8 ΔE and the real departure begins at band x ≈ 215.
4. **A veil-aware placement rule** was proposed on the strength of that broken metric.
   Dropped.
5. **`M3_blob` looked competitive** until its per-frame behaviour was examined. The mechanism
   has now been measured on all 20 frames rather than inferred from one, and it separates
   without overlap: on the **9 frames where the segmenter finds zero blobs in the unclipped
   inner column**, its left-band shift reads −1.6 to −2.5 px against a true +12 to +14, an
   error of **−14.1 to −15.8 px**; on the 11 frames where it finds one inner blob the error
   is −6.5 to +0.9 px. It never finds more than one blob in that column on any frame. So it
   is not a marginal method — it is systematically wrong by most of a tile whenever the one
   informative column drops out, which is nearly half the time. Unfit for production.

---

## 8. Recommendation

**Use `M2_rigid` reference-free, with `M2_profile` as a second opinion, `assign_identity`
for labelling, and the impurity gate as the alarm.**

```python
rec = qc_record(band.rgb, band.lab, prior, side, mode="reference_free")
```

Why this over the alternatives:

- It needs no stored reference image, so it survives a rig teardown, a re-mounted card, a
  changed render crop or a different frame size — all of which bias the `M1_*` methods
  without necessarily failing loudly.
- It has the widest reference-free horizontal capture range (±30 px) against a rig that
  moves ≤ 14 px, so there is real headroom.
- It has the cleanest contamination record of the six methods on both bands.
- It is the only path on which a **physically rotated card** is recoverable end to end:
  a stored reference image would no longer match, but `M2_rigid` still finds the lattice and
  the identity step recovers the labelling.

**If a reference band is available and the rig is known not to have been touched**, use
`M1_euclidean` as primary with `M2_rigid` as cross-check and `mode="reference"`. It is six
times faster (0.06 s vs 0.36 s per band), it allows rotation, and ECC correlation is a
genuinely independent fault signal that reference-free mode does not have. This is a
speed-and-diagnostics choice, not an accuracy one.

**Do not use** `M0_frozen` (the original fixed-coordinate approach — it is the source of the
smear), `M3_blob`, or `M23_cascade`.

**Regardless of method:**

- Feed the measurement step **linear RGB** (RawTherapee 16-bit / 65535). Everything
  downstream assumes it; handing it sRGB breaks identity and every ΔE silently.
- Fit the correction on **all 24 patches**. If you ever fit fewer, drop to degree 2 —
  a degree-3 polynomial has 13 terms and interpolates a 12-patch subset exactly.
- Never trust a single method's confidence. The gate's value is that its signals fail
  independently.

## 9. Open items

1. **Occlusion recovery is unfinished.** The sticker leaves 6–17 px of clean width on some
   neutral tiles, and the gutters between tiles give a non-circular way to find it (they
   should read dark card, L\* ≈ 21, but read L\* ≈ 90 across x 157–187). Whether a 17-px-wide
   sample is worth measuring, versus dropping the tile, has not been tested. Compare against
   `d000377_300_060`, the genuinely clean frame in that run.
2. **The 5 % impurity limit rests on two positives**, one severe and one marginal. It should
   see more real occlusion events before it is treated as settled.
3. **`M2_rigid` takes its horizontal shift from one column.** That is exactly why it works,
   but it is a single point of failure; a second anchor is worth testing.
4. **The right band's inner window is too narrow** — 170–202 against a patch whose measured
   extent is ≈ 165–203 (the brightness profile through A4 falls from L\* 88 at x=199 to 63 at
   x=206 and 35 at x=213, and the patch-bearing zone is 152–207). It measures fewer pixels
   than are available, but the headroom is a few px per side, not the ~13 px an earlier
   note claimed — widening past ~203 would pull in border.
5. **A batch runner** — walk a directory, emit measurements and a QC report, exit non-zero on
   flagged frames — is not written yet.
6. **Rig-side:** pin RawTherapee's white balance (`Setting=Custom`) to remove per-frame
   auto-white-balance variation from the inputs.
7. **Re-shoot or replace the registration reference frame.** `d000220_300_021` is
   black-clipped on both bands (6.7). `d000320_300_003` is clean and is the better anchor.
8. **`tile_robust_shift`'s 1.5 ΔE limit rests on one positive and one negative.** The
   separation is wide — 0.57 ΔE absorbed against 2.5–18 ΔE harmful — but it is two events.
9. **The reference-free gate is down to two independent signals** on the one occlusion in
   the set, because the improved identity stage no longer objects to it. A third
   reference-free signal — the gutter check below is the obvious candidate — would restore
   the redundancy.
10. **A gutter check is the most promising unused signal.** The strips between tiles should
   read dark card; on the occluded band they read L\* ≈ 90 across x 157–187. It is
   independent of everything the gate currently measures, and it detects an occluder
   precisely because the occluder does not respect tile boundaries.

---

## Files

| file | what it holds |
|---|---|
| `chipdetect.py` | all six methods, identity, measurement, QC gate |
| `chiprig.py` | frame loading and band extraction |
| `rig_prior.json` | the stored layout |
| `chip_detection_methods.md` | full method log and derivations |
| `overlay_boxes.csv` | 480 tiles: box, measured colour, assigned patch, impurity |
| `capture_range.csv`, `capture_range_reffree.csv` | shift sweeps |
| `veil_left_by_side.csv`, `veil_right_neutral.csv` | contamination per box |
| `subset_qc_tuned.csv` | gate decisions on 40 bands |
| `identity_validation.csv` | identity, upright and rotated |
| `downstream_rpcc.csv` | correction error per method |
