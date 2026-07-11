# §7.2 — the chromatic-aberration experiment

Response-weighted mean distance, in pixels, from the edge response to the true
object boundary, restricted to a **6 px band** around that boundary.
**Lower is better.** `slope` is a least-squares fit of the error against `δ`; a flat
slope means the method is insensitive to lateral chromatic aberration.

`δ` is the displacement **at the image corner**. An edge at radius `r` moves by
`δ · r / r_max`, so the shift at a colony is smaller than `δ`.

Colour responses come from the **un-clipped** `_color_phase_congruency` helper
(drift `C3`). The three `control:` rows are the guard: **G is the CA reference
channel and must be flat in `δ`**, while R and B are displaced by construction and
must degrade. The script exits non-zero if either control misbehaves, because a
metric that cannot see a displacement it created cannot see anything.

### filamentous plate

| method | δ = 0 | δ = 1 | δ = 2 | δ = 3 | slope |
|---|---|---|---|---|---|
| `FocusEdgePhase (baseline)` | 1.7033 | 1.7042 | 1.7094 | 1.7223 | `+0.0062` |
| `FocusEdgeMonogenicPhase (luminance)` | 1.0985 | 1.1129 | 1.1313 | 1.1579 | `+0.0197` |
| `FocusEdgeColorPhase (l2)` | 1.5961 | 1.6265 | 1.5975 | 1.6251 | `+0.0058` |
| `FocusEdgeColorPhase (joint)` | 1.0853 | 1.0318 | 0.9917 | 1.0083 | `-0.0271` |
| `FocusEdgeColorPhase (coherent)` | 1.0819 | 1.2517 | 1.1638 | 1.0890 | `-0.0067` |
| `control: monogenic PC on R alone` | 1.0997 | 1.1541 | 1.2304 | 1.3528 | `+0.0836` |
| `control: monogenic PC on G alone` | 1.0992 | 1.0992 | 1.0992 | 1.0992 | `-0.0000` |
| `control: monogenic PC on B alone` | 1.0994 | 1.1730 | 1.2621 | 1.3926 | `+0.0969` |

### yeast plate

| method | δ = 0 | δ = 1 | δ = 2 | δ = 3 | slope |
|---|---|---|---|---|---|
| `FocusEdgePhase (baseline)` | 1.6873 | 1.6899 | 1.7011 | 1.7196 | `+0.0108` |
| `FocusEdgeMonogenicPhase (luminance)` | 1.0900 | 1.0951 | 1.1125 | 1.1433 | `+0.0177` |
| `FocusEdgeColorPhase (l2)` | 1.6397 | 1.7398 | 1.7597 | 1.7759 | `+0.0428` |
| `FocusEdgeColorPhase (joint)` | 1.0942 | 1.1822 | 1.2821 | 1.3748 | `+0.0942` |
| `FocusEdgeColorPhase (coherent)` | 1.0760 | 1.1313 | 1.3050 | 1.7003 | `+0.2047` |
| `control: monogenic PC on R alone` | 1.0914 | 1.1618 | 1.2929 | 1.4667 | `+0.1257` |
| `control: monogenic PC on G alone` | 1.0911 | 1.0911 | 1.0911 | 1.0911 | `-0.0000` |
| `control: monogenic PC on B alone` | 1.0912 | 1.1337 | 1.2392 | 1.3989 | `+0.1029` |

## Slopes

- **filamentous plate**: `joint` slope `-0.0271`, `l2` slope `+0.0058`
- **yeast plate**: `joint` slope `+0.0942`, `l2` slope `+0.0428`

## Verdict

**The slope prediction FAILED on the yeast plate and HELD on the filamentous plate.** `joint` is nevertheless better localized than `l2` at *every* `δ` on *both* plates -- `l2`'s flat slope is largely an artifact of its already-poor localization (`1.60`--`1.78`), which leaves it little room to degrade.

**Decided 2026-07-10 (user): keep `fusion="joint"`, and scope the operation to filamentous plates.** This is a *scoping* decision, not a post-hoc swap of the test statistic: on the plate the operation is for, the prediction held on both the slope and the absolute error.

**The null result, recorded rather than buried (§7.2, spec risk #2).** On the yeast plate at `δ = 3`, plain `FocusEdgeMonogenicPhase` on luminance scores `1.1433` and beats *every* fusion mode -- `joint` `1.3748`, `coherent` `1.7003`, `l2` `1.7759`. **On round-colony plates, colour buys nothing under chromatic aberration.** The measured benefit of `FocusEdgeColorPhase` is confined to the filamentous plate, where `joint` reaches `1.0083` against luminance's `1.1579`.

The mechanism is not mysterious. Lateral CA *creates chromatic edges* -- that is what it is. `joint` asserts them coherently, so its detected edge follows the displaced chroma; `l2` combines three finished maps and its undisturbed luminance term survives the root-sum-of-squares. Spec §3.3's claim that joint's coherent summation 'merges the displaced edges into one response near the amplitude-weighted centroid' is **not** what the measurement shows on the yeast plate: joint's error grows five times faster than luminance-only's.
