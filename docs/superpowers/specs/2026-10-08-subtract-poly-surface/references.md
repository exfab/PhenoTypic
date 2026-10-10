# SubtractPolySurface — references

**Date:** 2026-10-08
**Status:** reference record for [`design.md`](./design.md); deviations are catalogued in
[`drift-register.md`](./drift-register.md).

## 1. Where the sources live, and why not here

Every source below is on disk at

```
/bigdata/exfab/anguy344/reference-sources/2026-10-08-subtract-poly-surface/
├── PROVENANCE.md        # URL, retrieval date, licence, integrity per item
├── SHA256SUMS
├── gwyddion-2.71/       # release tarball + .sig, and src/ (6 files extracted verbatim)
├── gwyddion-user-guide/ # leveling-and-background.html, scan-line-defects.html
└── astropy-7.1.0/       # stats/sigma_clipping.py, modeling/fitting.py, LICENSE.rst
```

That directory is **outside the repository on purpose.** Gwyddion is GPL-2.0-or-later and its user
guide is GFDL; PhenoTypic is Apache-2.0. The repository's precedent for copyleft oracles —
[`2026-07-13-fungi-detection-method-ports/refs/nfa/ATTRIBUTION.md`](../2026-07-13-fungi-detection-method-ports/refs/nfa/ATTRIBUTION.md)
— is that the copyleft source is never vendored, imported, linked, copied, or structurally
transcribed, and that production code is written **clean-room** from a source-free behavioural
contract plus numeric fixtures. This port follows that precedent:

- `design.md` §4 is the behavioural contract. It states conventions, defaults and formulas
  (facts about behaviour), never code.
- The implementer receives `design.md`, `drift-register.md`, and the oracle's numeric fixture.
  **The implementer does not receive, and must not open, anything under `gwyddion-2.71/` or
  `gwyddion-user-guide/`.**
- Citations below are `path:line` against the out-of-repo copies, for reviewers and the oracle.

astropy is BSD-3-Clause and could be vendored, but it is a *design* reference (the robust loop's
structure and defaults), not a port target; it stays beside the others so all citations resolve
against one tree.

## 2. Integrity

| Item | Integrity |
|---|---|
| `gwyddion-2.71.tar.xz` | sha256 `2df721befccbe4d5ee2ba564b32e69341f8ce1de637e2045838a09a2d46b5dba`. Detached signature by EdDSA key `77817A9141442926DDE3E5E56D4182E78232D84B` **could not be verified**: the key is absent from `keys.openpgp.org` and `keyserver.ubuntu.com`. The source is only read and (in the oracle task) compiled inside a Slurm job; an unverified signature is recorded, not ignored. |
| everything else | `SHA256SUMS` in the reference directory |

Gwyddion **2.71** is the current stable release (download page, 2026-10-08: "stable version
Gwyddion 2.71 … released 15 April 2026"). The 3.x series is the unstable branch and is not used.

## 3. Literature

1. **Nečas, D., & Klapetek, P. (2012).** Gwyddion: an open-source software for SPM data analysis.
   *Central European Journal of Physics* 10(1), 181–188. doi:10.2478/s11534-011-0096-2.
   The citation for Gwyddion itself. **Not on disk:** the publisher returned a bot challenge
   (HTTP 202) on 2026-10-08. The paper is a software overview; the leveling *behaviour* this spec
   relies on is taken from the source (§4), not from the paper.
2. **Gwyddion user guide**, chapters *Leveling and Background Subtraction* and *Scan Line
   Artefacts* (GFDL; on disk). Describes Plane Level ("computed from all the image points and is
   subtracted from the data"), Polynomial Background (independent vs. limited-total degree), and
   Align Rows (Median default; Polynomial with degree 0 = mean, 1 = tilt, 2 = bow). **The guide
   does not state where the output level ends up for any of these tools** — that is established
   from the source in §4.
3. **astropy** `astropy.stats.sigma_clip` / `SigmaClip` and
   `astropy.modeling.fitting.FittingWithOutlierRemoval` (BSD-3-Clause, v7.1.0; on disk). The
   structural reference for the robust fit (`fit="robust"`), which Gwyddion does not have.
4. **Rousseeuw, P. J., & Croux, C. (1993).** Alternatives to the median absolute deviation.
   *Journal of the American Statistical Association* 88(424), 1273–1283.
   doi:10.1080/01621459.1993.10476408. The MAD's consistency constant (≈1.4826 for a normal
   distribution), which `scipy.stats.median_abs_deviation(scale="normal")` applies. Not on disk;
   cited for the constant only.

## 4. Gwyddion 2.71 behaviour, cited

Paths are relative to `gwyddion-2.71/src/`.

### 4.1 Plane Level (`level` function of `modules/process/level.c`)

| Fact | Citation |
|---|---|
| Registered as `/_Level/Plane _Level`, "Level data by mean plane subtraction" | `modules/process/level.c:60-66` |
| Masking default is *exclude*; it only applies when a mask exists | `modules/process/level.c:108` |
| With no mask, the plane is the full-field least-squares fit `z ≈ a + bx·j + by·i` in **pixel-index** coordinates (`j` column, `i` row) | `modules/process/level.c:165`; `libprocess/level.c:34` (documented relation), `:76-83` (closed-form slopes) |
| The fitted constant `a` is **discarded** and replaced by `c = -½(bx·xres + by·yres)` | `modules/process/level.c:174` |
| `plane_level` subtracts `c + bx·j + by·i` from every pixel | `modules/process/level.c:175`; `libprocess/level.c:484-488` |
| ⇒ output `= z − bx(j − xres/2) − by(i − yres/2)`: tilt removed about the pivot `(xres/2, yres/2)`, so `mean(out) = mean(z) + ½(bx + by)` | derived; verified by `level_conventions.py` L1–L2 |

### 4.2 Zero Mean Value (`zero_mean`, same module)

| Fact | Citation |
|---|---|
| Subtracts the plain mean (masked mean if a mask exists) | `modules/process/level.c:185-190` |

### 4.3 Polynomial Background (`modules/process/polylevel.c`)

| Fact | Citation |
|---|---|
| `MAX_DEGREE = 11` | `polylevel.c:33` |
| Defaults: `col_degree = row_degree = 3`, `max_degree = 3`, `same_degree = TRUE`, `independent = TRUE`, masking *ignore* | `polylevel.c:141-150` |
| Independent mode: term set `{x^p · y^q : 0 ≤ p ≤ col_degree, 0 ≤ q ≤ row_degree}` (tensor product) | `polylevel.c:218-227` |
| Total-degree mode: term set `{x^p · y^q : p + q ≤ max_degree}` | `polylevel.c:228-237` |
| Fits the term set and subtracts the **whole** polynomial, constant included | `polylevel.c:239-241` |
| Coordinates are normalized to `[-1, 1]`: `y = 2i/(height−1) − 1`, `x = 2j/(width−1) − 1` | `libprocess/level.c:1623, 1629` (fit); `:1748, 1754` (subtract) |
| Monomial basis; normal equations accumulated and solved by Cholesky | `libprocess/level.c:1574-1661` |
| A failed Cholesky decomposition **silently clears the coefficients to zero** (nothing is subtracted) | `libprocess/level.c:1656-1657` |

### 4.4 Align Rows (`modules/process/linematch.c`, `libprocess/correct.c`)

| Fact | Citation |
|---|---|
| Method default **Median**; direction default **horizontal**; polynomial `max_degree` default **1**, range 0–`MAX_DEGREE` (= 5); masking *ignore*; trim fraction 0.05 | `linematch.c:44, 168-176` |
| Vertical direction is handled by transposing (`flip_xy`), leveling rows, and transposing back | `linematch.c:362-371, 400-404` |
| Polynomial method, degree 0 → trimmed mean with fraction 0 (= row mean); degree ≥ 1 → `gwy_data_field_row_level_poly` | `linematch.c:375-379` |
| `row_level_poly`: per-row least squares on the **centred pixel** coordinate `x = j − (xres−1)/2`, normal equations by Cholesky | `correct.c:1814, 1836-1862` |
| The global mean of the input field is computed once, before leveling … | `correct.c:1815` |
| … and added back: each row's fitted constant has `avg` subtracted from it, so each row is leveled to the global mean | `correct.c:1867-1879` |
| A row with `≤ degree` usable points gets **zeroed** coefficients, after which the `avg` adjustment still applies — the row is shifted by `+avg` | `correct.c:1855-1867` |
| Row-shift methods (trimmed mean, Median) re-centre the shifts to zero mean (`zero_level_row_shifts`) | `correct.c:1565-1568, 1669` |

### 4.5 astropy robust fitting

Paths relative to `astropy-7.1.0/`.

| Fact | Citation |
|---|---|
| `SigmaClip` defaults: `sigma=3.0`, `maxiters=5`, `cenfunc="median"`, `stdfunc="std"` (`"mad_std"` available) | `astropy_sigma_clipping.py:175-182, 99-107` |
| `FittingWithOutlierRemoval(fitter, outlier_func, niter=3, …)` | `astropy_fitting.py:864, 899` |
| Initial fit on all points, before any clipping | `astropy_fitting.py:1006` |
| Up to `niter` rounds: clip the residuals **of the already-masked data** (a clipped point stays clipped), add the model back, refit on the unmasked points | `astropy_fitting.py:1017, 1027, 1075, 1085` |
| Stops early when the number of masked points did not change | `astropy_fitting.py:1105-1108` |
