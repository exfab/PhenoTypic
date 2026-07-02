# Branch Phenotypes for Filamentous Fungi — Comprehensive Catalog & PhenoTypic Schema Map

**Date:** 2026-06-25
**Scope:** Every branch / branching-morphology phenotype worth computing for filamentous
fungi, spanning *morphology + topology + growth dynamics + physiology/pigment*, collected
exhaustively from mycology and four cross-domain analog fields (neuroscience, plant root
architecture, vascular/angiogenesis, geomorphology + growth physics). Collection was
deliberately **not** filtered by feasibility; instead every metric is tagged with its
resolution/separability assumptions, and a separate **Fit** column scores it against
PhenoTypic's real operating point (poor resolution, dense, overlapping/crossing hyphae,
and an in-progress branch-retracing module).

> Built from 7 parallel literature agents (web + scite + PubMed/PMC + bioRxiv). Citations
> were DOI-verified by the agents; a **Verification caveats** section at the end carries
> forward every confidence flag they raised. `bio_desc` fields in the schema map are left
> **empty for human authoring**, per the repo's biological-claim guardrail.

---

## 0. Framing: the information-content boundary

A "branch phenotype" is gated by one question — **can segmentation resolve individual
hyphae/branches?** This is not a method preference; it is an information boundary that
partitions essentially every metric into two regimes, with two orthogonal overlays:

| Code | Regime | What you have | Metric character |
|------|--------|---------------|------------------|
| **A** | Resolved branches | A skeleton + topological **graph** (nodes = tips/junctions, edges = segments) | Per-branch geometry + network topology |
| **B** | Whole-mycelium silhouette | A binary **mask** or grayscale field; individual branches *not* separable | Global descriptors that *infer* branchiness |
| **Dyn** | Dynamics overlay | Registered **time-lapse** | Rates of the A/B quantities |
| **Phys** | Physiology overlay | **Color/RGB** (or oblique lighting) | Pigment/zonation tied to branch structure |

**Why this matters for PhenoTypic.** The existing `SHAPE`/`SIZE`/`TEXTURE` schema *is* the
Regime-B toolkit applied to compact yeast/bacterial colonies. Filamentous fungi on a plate
are still Regime B — but with vastly more boundary complexity — so the existing descriptors
are a starting point that mainly needs **fractal-dimension + lacunarity + Sholl-type radial
+ Minkowski** extensions. Regime A is a genuinely new architectural layer: it needs a
skeleton/graph accessor and depends on the in-progress retracing module
(`sdk_/branch_pathfinding/`, a multi-source Dijkstra / least-cost-path fragment-reconnection
engine fronted by `FilamentousFungiDetector`). That module's accuracy ceiling is exactly
the crossing/overlap case — which is precisely why the **Fit** column below weights
robustness to overlap so heavily.

### Fit legend (robustness to PhenoTypic's real operating point)

| Fit | Meaning |
|-----|---------|
| 🟢 **B-robust** | Computable from a binary mask/grayscale; tolerates dense, overlapping, low-res hyphae; mostly extends existing shape/texture. **Ship-now candidates.** |
| 🟡 **A-feasible** | Needs a skeleton/partial graph; reachable through the current retracing module but accuracy-bounded; degrades as overlap rises. **Gated on retracing maturity.** |
| 🔴 **A-fragile** | Needs clean per-branch separation (microscopy / high-res / strong instance segmentation); fails on dense overlapping low-res mycelium. **Aspirational.** |
| ⏱ **Dyn** | Additionally needs registered time-lapse (orthogonal to the spatial fit). |
| 🎨 **Phys** | Additionally needs color/RGB; spatially usually B-robust. |

> Rule of thumb that fell straight out of every analog field: **mask-level metrics survive
> overlap because they never separate branches; skeleton-graph metrics that count tips,
> junctions, angles, and loops fail first.** Three fields (neuroscience, RSA, vascular)
> independently concluded that when structures can't be separated, you fall back to
> fractal dimension / lacunarity / density / Sholl-radial / convex-hull descriptors — the
> 🟢 rows below.

---

## 1. The decision spine

```
Can you segment individual hyphae into a reliable skeleton + graph?
│
├── NO  → Regime B. Use mask/grayscale descriptors:
│         size/shape · fractal dimension (mass + surface) · lacunarity ·
│         multifractal · Minkowski/Euler · Sholl radial profile ·
│         GLCM/Fourier texture & anisotropy · margin complexity · convex-hull occupancy
│         (+ Phys: pigment/zonation/sectoring · Dyn: radial growth rate, FD-over-time)
│
└── YES (or partially, via retracing) → Regime A. Add network + per-branch metrics:
          tip/node counts · branching frequency · branch order (Strahler/centrifugal) ·
          HGU · branch angle · internode distance · tortuosity · width/taper ·
          anastomosis/loop count · weighted-graph transport · TMD persistence
          (+ Dyn: tip-extension & branching rates, anastomosis rate, autotropism)
```

---

## 2. Master catalog

Deduped across all seven sources. `Prov.` = provenance fields that use the metric
(**My**=mycology, **Ne**=neuroscience, **Ro**=roots/RSA, **Va**=vascular, **Ge**=geomorph/physics).
Definitions are condensed; see §6 References for sources keyed `Author Year`.

### Family A — Whole-silhouette size & shape (Regime B) — *largely already in PhenoTypic*

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| Projected area | Foreground pixel count × pixel area | mask | very high | My,Ro | 🟢 | Paul & Thomas 1998 |
| Perimeter | Chain-coded contour length | mask | med (scale-sensitive) | My,Ro | 🟢 | Paul & Thomas 1998 |
| Equivalent diameter | √(4A/π) | mask | high | My | 🟢 | Papagianni & Mattey 2006 |
| Circularity / form factor | 4πA/P² (→0 as margin filaments) | mask | med | My | 🟢 | Paul & Thomas 1998 |
| Solidity / convexity / fullness | Area ÷ convex-hull area (compact↔diffuse) | mask | high | My,Ro,Ne | 🟢 | Tucker & Thomas 1992 |
| Convex-hull area / territory | Area of enclosing convex polygon | mask | high | Ro,Ne | 🟢 | Galkovskyi 2012 |
| Hull occupancy / coverage | Network area ÷ hull area (space-filling within reach) | mask | high | Ne,Ro | 🟢 | Sholl/SNT |
| Eccentricity / aspect ratio | Best-fit-ellipse foci ratio / axis ratio | mask | high | My,Ro | 🟢 | Paul & Thomas 1998 |
| Major/minor ellipse axes | Best-fit ellipse axis lengths | mask | high | Ro | 🟢 | Galkovskyi 2012 |
| Width / depth / width:depth | Bounding-box extents & ratio (spread anisotropy) | mask | high | Ro,Ne | 🟢 | Galkovskyi 2012 |
| Centroid / radius of gyration | Mass center; √(Σr²/N) spread about it | mask | high | My,Ro,Ge | 🟢 | Cox & Thomas (in Paul & Thomas 1998) |
| Zernike shape moments | Orthogonal rotation-invariant shape signature | mask | high | My | 🟢 | CellProfiler |
| Number of enclosed holes / mean hole size | Count/area of background regions fully enclosed | mask | med (small holes lost at low res) | Ro,Ge | 🟢 | Seethepalli 2021 |

### Family B — Space-filling & heterogeneity (Regime B) — *the highest-value new additions*

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| Mass fractal dimension D_BM (box-counting) | Slope of log N(ε) vs log(1/ε) over the **filled** mask; space-filling index ~1.2–2.0 | mask | high (integrates many scales) | My,Ne,Va,Ge,Ro | 🟢 | Obert 1990; Papagianni & Mattey 2006 |
| Surface/border fractal dimension D_BS | Same on the **outline** only; margin/foraging-front complexity | mask (outline) | high | My,Ge | 🟢 | Papagianni & Mattey 2006 |
| Mass-vs-surface FD divergence | D_BM≈D_BS → true mass fractal (dispersed); divergence → surface fractal (pelleted core) | mask | med | My | 🟢 | Papagianni & Mattey 2006 |
| Lacunarity Λ(r) (gliding-box) | Var/mean² of box occupancy vs scale; gappiness — separates equal-FD textures | mask | very high | My,Ne,Va,Ge,Ro | 🟢 | Plotnick 1996; Allain & Cloitre 1991 |
| Multifractal spectrum f(α), Δα | Generalized dims D_q; width Δα = density heterogeneity across scales | mask/gray | med (needs ≥1 decade of scale) | Ne,Va,Ge | 🟢 | Halsey 1986; Stošić 2006 |
| Minkowski functionals (area, perimeter, Euler χ) | 3 additive morphological numbers vs threshold; **χ indexes loops/anastomosis** | mask | very high | Ge | 🟢 | Mecke 2000 |
| Minkowski–Bouligand dimension | Dilation ("sausage") fractal dim; tolerant of thin-line breaks | mask | high | Ge | 🟢 | Florindo & Bruno 2018 |
| Mass–radius scaling M(R)∝R^D | Radial mass-fractal dim from colony center | mask | high | Ge | 🟢 | Witten & Sander 1981 |
| Perimeter–area fractal dim | D from log P vs log A; margin ruggedness | mask (contour) | high | Ge | 🟢 | Mandelbrot 1983 |
| Two-point density correlation C(r) | r^−(d−D) decay → characteristic branch spacing / mesh correlation length | mask/gray | med | Ge | 🟢 | Witten & Sander 1981 |
| DLA/DBM regime fit (effective η, D_f) | Place D on the Laplacian-growth family: η≈1/D≈1.71 → diffusion-limited foraging; D→2 → compact | mask | high | Ge | 🟢 | Niemeyer 1984; Ben-Jacob & Garik 1990 |
| Dense-branching-morphology signature | Branch-width ≈ inter-branch gap + exponential branch-length dist → random tip-splitting morphotype | mask | high | Ge | 🟢 | Ben-Jacob & Garik 1990 |

### Family C — Radial / Sholl-type density profiles (Regime B, soma = inoculum/centroid)

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| Sholl intersection profile N(r) | Count of mask crossings of concentric rings centered on the origin | mask | med (miscounts where strands touch in a ring) | Ne | 🟢 | Sholl 1953; Ferreira 2014 |
| Critical radius r_c | Radius of peak N(r) — active-growth annulus | mask | med | Ne | 🟢 | Ferreira 2014 |
| Max intersections N_m | Peak hyphal-ring count | mask | med | Ne | 🟢 | Sholl 1953 |
| Enclosing radius | Largest radius with non-zero intersection (extent) | mask | high | Ne | 🟢 | Ferreira 2014 |
| Sholl regression coefficient k | Slope of semi-log/log-log N(r) decay with radius | mask | med | Ne | 🟢 | Sholl 1953 |
| Sholl polynomial decomposition | Polynomial fit → smoothed radial-density model + decay terms | mask | med | Ne | 🟢 | Ferreira 2014 |
| Ramification index | N_m ÷ number of primary hyphae at origin | mask + stem count | med (needs primary count) | Ne | 🟡 | Ferreira 2014 |
| Radial mass distribution / density falloff | Occupied-pixel fraction in concentric annuli vs radius (core→hairy profile) | mask | high | My,Ro | 🟢 | Tucker & Thomas 1992 |
| Core/annulus split & relative annular diameter (RAD) | (d_total−d_core)/d_total — fraction of radius that is loose outer zone | mask/gray | med | My | 🟢 | Veiter 2018 |
| Width function W(x) | Tip/segment count vs topological distance from origin (growth-front shells) | skeleton (rooted) | high cost | Ge,Ne | 🔴 | Rodríguez-Iturbe & Rinaldo 1997 |
| Bushiness | max ÷ median strand-crossing count over scan lines | mask | med | Ro | 🟢 | Galkovskyi 2012 |

### Family D — Texture & anisotropy (Regime B, grayscale — no segmentation needed)

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| GLCM / Haralick features | Contrast, homogeneity, energy, entropy, correlation of grey-level co-occurrence | gray | very high | My | 🟢 | Haralick 1973 |
| Directional GLCM anisotropy | Angular dependence (0/45/90/135°) of Haralick features | gray | high | My | 🟢 | Haralick 1973 |
| FFT / Fourier orientation & power spectrum | Angular/radial integration of 2-D power spectrum → preferred orientation + coarseness | gray | very high | My | 🟢 | OrientationJ |
| Gabor filter-bank energy | Multi-scale/orientation Gabor responses → branch directionality & fineness | gray | high | My | 🟢 | std texture |
| Local-orientation angle histogram | Dominant orientation per window → shallow/medium/steep angle frequencies | mask/gray | high | Ro,Ge | 🟢 | Seethepalli 2021 |
| Growth-direction order parameter / anisotropy | Variance/concentration of local growth directions (structure tensor) | mask/gray | high | Ge | 🟢 | Ben-Jacob 1997 |
| Angular power spectrum of boundary | FFT of branch-orientation distribution → n-fold symmetry / preferred angle | mask (boundary) | high | Ge | 🟢 | Ben-Jacob & Garik 1990 |

### Family E — Margin / boundary complexity & macromorphotype (Regime B)

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| Macroscopic margin type | Categorical: entire / undulate / lobate / filamentous / rhizoid | mask/RGB | med (rhizoid vs filamentous needs edge detail) | My | 🟢 | morphotyping; CNN classifiers |
| Convex-perimeter deficit / convexity (perimeter) | Convex-perimeter ÷ perimeter (concavity of margin) | mask | med | Ro | 🟢 | Paul & Thomas 1998 |
| Perimeter-to-area ratio | Boundary length per unit area (rises with filamentous spread) | mask | med | Ro | 🟢 | Paul & Thomas 1998 |
| Roughness / hairy-length | Length of filaments protruding beyond a compact core | mask (hi-res) | low (needs filaments resolved) | My | 🟡 | Tucker & Thomas 1992 |
| Pellet/clump/dispersed classification | Rule/ML label from area+compactness+roughness+circularity | mask | med (clump↔pellet ambiguity) | My | 🟢 | Tucker & Thomas 1992; Wucherpfennig 2019 |
| Smooth-vs-hairy pellet class | Threshold on roughness/RAD/compactness | mask | med | My | 🟢 | Tucker & Thomas 1992 |
| Elevation / surface texture | Flat/raised/umbonate; smooth/fuzzy/cottony | RGB/oblique/3-D | med–low (needs shading/3-D) | My | 🎨 | morphotyping |

### Family F — Network topology (Regime A) — *needs skeleton + graph*

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| Total hyphal length L_t | Σ of skeleton segment lengths | skeleton | med (robust if segmentation closes) | My,Ne,Ro,Va,Ge | 🟡 | Arganda-Carreras 2010 |
| Tip / apex count | Degree-1 skeleton nodes | graph | low (crossings hide/fuse tips) | My,Ne,Ro,Va,Ge | 🔴 | Trinci 1974; Arganda-Carreras 2010 |
| Branch-point / node count | Degree-≥3 nodes (triple vs quadruple) | graph | low (overlap → false nodes) | My,Ne,Ro,Va,Ge | 🔴 | Arganda-Carreras 2010 |
| Branching frequency φ | Branch points per unit length | graph | low | My,Ro,Va | 🔴 | Riquelme 2019 |
| Junction / branch density | Branch points per unit area | graph | med (global normalization helps) | My,Va,Ge | 🟡 | Barry 2015; Zudaire 2011 |
| Hyphal Growth Unit (HGU) | L_t ÷ tip count (mean length per tip) | graph | low–med | My | 🟡 | Trinci 1974 |
| Branch order (centrifugal) | Order incremented at each bifurcation from main axis outward | graph (tree) | low (needs clean parent/child) | My,Ne,Ro | 🔴 | Scorcioni 2008 |
| Horton–Strahler order | Terminal=1; two order-k meet → k+1 | graph (tree) | low (loops from anastomosis break it) | My,Ne,Ro,Va,Ge | 🔴 | Strahler 1957 |
| Shreve magnitude | # upstream source links per segment | graph (tree) | low | Ro,Ge | 🔴 | Shreve 1966 |
| Bifurcation ratio R_b | N_ω/N_(ω+1) across orders (Horton's law) | graph (tree) | low | Ne,Ro,Va,Ge | 🔴 | Horton 1945 |
| Stream-length ratio R_L | Mean-length ratio across orders | graph (tree) | low | Ge | 🔴 | Horton 1945 |
| Tokunaga (a,c) self-similarity | Side-tributary matrix T_k=a·c^(k−1); compact branching-style fingerprint | graph (tree) | low–med (global a,c fit tolerant) | Ge | 🔴 | Peckham 1995 |
| Topological index TI | Slope log(altitude) vs log(magnitude); herringbone↔dichotomous | graph (tree) | low | Ro | 🔴 | Fitter/Delory 2016 |
| Magnitude / altitude / total exterior path length | # tips / longest base-tip link path / Σ path-to-tips | graph (tree) | low | Ro | 🔴 | Fitter/Delory 2016 |
| #internal vs #external links; mean link lengths | Segment partition (bif→bif vs bif→tip) and metric lengths | graph | low | Ro,Ne | 🔴 | Delory 2016; Scorcioni 2008 |
| Anastomosis / hyphal fusion count & rate | Loop-creating tip-to-tip/tip-to-side reconnections | graph (+time) | low (true fusion vs crossing; time-lapse disambiguates) | My | 🔴⏱ | Dikec 2020 |
| Network loop / mesh count (1st Betti) | Independent cycles E−V+C | graph | low (spurious crossings add false loops) | My,Va,Ge | 🔴 | Dirnberger 2015; Carpentier 2020 |
| Euler number χ | objects − holes (connectivity) | mask | med (computed on mask) | My,Ge | 🟢 | scikit-image |
| Connectivity indices (α, β, γ / meshedness) | Edge-to-node / edge-to-max ratios | graph | low | My | 🔴 | Dirnberger 2015 |

### Family G — Per-branch geometry (Regime A)

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| Branch / segment length distribution | Per inter-node arc length (mean, max, histogram) | graph | med (once skeleton correct) | My,Ne,Va | 🟡 | Arganda-Carreras 2010 |
| Internode / inter-branch distance | Arc length between consecutive branch points | graph | low | My,Ne,Ro | 🔴 | Barry 2015 |
| Branch / bifurcation angle (local & remote) | Angle between parent & daughter at node (initial vs remote points) | graph (geom) | low (angle collapses near crossings) | My,Ne,Ro,Va,Ge | 🔴 | Scorcioni 2008; Lamour 2022 |
| Bifurcation tilt & torque | Parent–daughter-plane angle; twist between successive bifurcations | graph (3-D) | very low (3-D only) | Ne | 🔴 | Scorcioni 2008 |
| Partition asymmetry | Per-node \|n_L−n_R\|/(n_L+n_R−2) over subtree tips | graph (tree) | low | Ne,Ro | 🔴 | van Pelt 1992; Scorcioni 2008 |
| Caulescence / tree asymmetry | Degree of a dominant main path | graph (tree) | low | Ne | 🔴 | Torben-Nielsen 2014 |
| Tortuosity / contraction | Path length ÷ Euclidean (≥1) per branch | graph | med (per-branch robust once traced) | My,Ne,Ro,Va | 🟡 | Nunez-Iglesias 2018 |
| Tortuosity variants (DM/ICM/SOAM/∫κ²) | Distance-metric / inflection-count / sum-of-angles / curvature-integral | graph | med–high | Va | 🟡 | Bullitt 2003; Hart 1999 |
| Hyphal width / diameter | Distance-transform radius along skeleton | mask/gray | low (few-px width) | My,Ro,Va | 🔴 | Nunez-Iglesias 2018 |
| Diameter classes (fine vs cords/rhizomorphs) | Length/area binned by width thresholds | mask + DT | med | Ro | 🟡 | Seethepalli 2021 |
| Taper rate | Diameter gradient tip→base | mask/gray | low | My,Ne | 🔴 | Barry 2015 |
| Rall / Murray ratio & junction exponent | (d₁ʳ+d₂ʳ)/d₀ʳ; x solving d₀ˣ=d₁ˣ+d₂ˣ (optimum ~3) | graph + diam. | very low (diameter-cubed noise) | Ne,Va | 🔴 | Patton 2006 |
| Surface area / biovolume | Σ frustum surface/volume of segments | graph + radii | low | My,Ne,Va,Ro | 🔴 | Scorcioni 2008 |

### Family H — Weighted-graph / transport & TDA fingerprint (Regime A + Phys)

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| Edge (cord) conductance | Edge weight ∝ diameter⁴ (Poiseuille) | graph + width | low | My | 🔴 | Fricker 2007 |
| Betweenness centrality | Fraction of weighted shortest paths through a node/edge | weighted graph | med (stable summary on imperfect graph) | My | 🟡 | Fricker 2007; Bebber 2007 |
| Transport efficiency / robustness | Flux vs cost; connected-core fraction after link removal | weighted graph | med | My | 🟡 | Heaton 2012 |
| Mesoscale community structure | Multiscale community signature clustering networks by phenotype | weighted graph | med | My | 🟡 | Lee 2017 |
| Network "traits" trade-off (cost–efficiency–resilience) | Composite ecological-strategy descriptor | weighted graph | med | My | 🟡 | Aguilar-Trigueros 2022 |
| **TMD persistence barcode / image** | Track branch birth(tip)–death(bifurcation) along radial/path function → barcode; **order-invariant, degrades gracefully** | graph (rooted tree) | med (robust *summary* once traced) | Ne | 🟡 | Kanari 2018 |

### Family I — Growth dynamics (Dyn overlay)

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| Colony radial growth rate Kr/Km | dR/dt of colony edge (linear after lag) | time-lapse mask | very high (edge only) | My | 🟢⏱ | Trinci 1971 |
| Radial expansion velocity / colonization rate | dArea/dt of colonized region | time-lapse mask | very high | My | 🟢⏱ | Camy 2023 |
| Specific (biomass) growth rate µ | d ln(L_t or biomass)/dt | time-lapse mask | high | My | 🟡⏱ | Trinci 1969 |
| Peripheral growth-zone width w | w = Kr/µ (derived) | radius+µ series | high | My | 🟢⏱ | Trinci 1971 |
| Mass-FD trajectory FD(t) | Box-counting FD tracked over growth (rising = densifying) | time-lapse mask | high | My,Ne | 🟢⏱ | Lecault 2009 |
| Tip extension (elongation) rate | d(tip position)/dt along axis | time-lapse + skeleton | low (per-tip tracking) | My,Ne,Va | 🔴⏱ | Spitzer 2017 |
| Branching rate / new-branch emergence | New branch points per unit time | time-lapse + skeleton | low | My | 🔴⏱ | Dikec 2020 |
| Specific branching frequency | New tips per unit length per time | time-lapse + skeleton | low | My | 🔴⏱ | Katz 1972 |
| Apical vs lateral branching kinetics | Separate dichotomous vs subapical branch rates | time-lapse + skeleton | low | My | 🔴⏱ | Katz 1972 |
| First-branch lag | Time germination→first branch | time-lapse | low | My | 🔴⏱ | Spitzer 2017 |
| Anastomosis rate | Fusion events per unit time | time-lapse + graph | low | My | 🔴⏱ | Dikec 2020 |
| Negative autotropism (tip avoidance) | Tip-trajectory deflection away from neighbors | time-lapse + skeleton | low | My | 🔴⏱ | Hutchinson/Meškauskas 2004 |
| Chemotropism / nutrient-gradient bias | Tip reorientation index toward a gradient | time-lapse + skeleton (microfluidic) | low | My | 🔴⏱ | Moore 2024; Nordzieke 2020 |
| Growth-front anisotropy (colony) | Ellipticity/eccentricity of outline over time | time-lapse mask | high | My,Ge | 🟢⏱ | Held 2019 |
| Space-filling / coverage rate | d(coverage fraction)/dt | time-lapse mask | med | My | 🟡⏱ | Ritz & Crawford 1995 |

### Family J — Physiology / pigment tied to branch structure (Phys overlay)

| Phenotype | Definition | Min input | Robustness | Prov. | Fit | Key source |
|---|---|---|---|---|---|---|
| Pigmentation intensity (mean color) | Mean R/G/B or CIELAB L*a*b* over colony | RGB | very high | My | 🎨 | Hernández-Lauzardo 2018 |
| Pigment radial gradient | Color/intensity vs radius | RGB/gray | very high | My | 🎨 | Hernández-Lauzardo 2018 |
| Concentric ring / zonation strength | Periodicity/amplitude of radial-transect intensity (FFT) | gray/RGB series | high | My | 🎨 | Cesbron 2024 |
| Circadian conidiation banding period | Band spacing ÷ growth rate → period | gray time-series | high | My | 🎨⏱ | Dunlap & Loros 2007 |
| Sectoring | Angular sectors of distinct color/morphology; fraction sectored | RGB | high | My | 🎨 | Vicente 2025 |
| Aerial vs substrate mycelium | Brightness/texture contrast; fraction aerial | gray/RGB (oblique) | med | My | 🎨 | Yagüe-type pattern analysis |
| Exudate / guttation droplets | Count/area/distribution of surface droplets (specular) | RGB (specular) | med (lighting-dependent) | My | 🎨 | Hernández-Lauzardo 2018 |
| Sporulation density zones | Local conidia density via gray-value↔count calibration | gray (calibrated) | med | My | 🎨 | Linde 2021 |
| Melanization | Darkening (low L*) of zones over time | RGB/gray | high | My | 🎨 | Zapanta et al. 2025 (PUPMCR) |
<!-- FACT-CHECK: [CORRECTED] Was: "Lyra 2025 (PUPMCR)". PUPMCR is authored by Zapanta, N.R., Santos, R.H., et al. (Deocaris group), not "Lyra". No "Lyra 2025" fungal melanization paper was findable; "Lyra 2025" appears to be a misattribution. Source: Zapanta et al. (2025), Biology Methods and Protocols 10(1):bpaf004, https://doi.org/10.1093/biomethods/bpaf004 -->
| Colony color heterogeneity | Variance/entropy of color across colony | RGB | very high | My | 🎨 | Vicente 2025 |
| Color→biomass/toxin proxy | Regression of RGB features onto biomass/DON | RGB | high | My | 🎨 | Belmonte 2019 |

---

## 3. Cross-domain transfer — what each field contributes & the dense/overlap toolkit

### 3.1 Per-field signature contribution

| Field | Strongest transfer | Fungal use |
|---|---|---|
| **Neuroscience** | Sholl radial analysis + **L-Measure metric vocabulary** + **TMD persistence barcodes** + flood-filling/U-Net dense tracing | Richest branch-tree vocabulary; Sholl → radial hyphal-density rings; TMD → order-invariant network fingerprint that degrades gracefully |
| **Plant RSA** | Topological indices (magnitude/altitude/TI), interactive deep-seg (**RootNav 2 / RootPainter**), **A\*** path recovery through crossings, occlusion is *the* problem | RSA already solves "2D projection of an overlapping branched net"; its A\* tracing **is the same algorithm PhenoTypic's retracing uses** |
| **Vascular** | **Frangi/Sato vesselness** (origin of curvilinear enhancement), tortuosity formalisms, **degree-4 node = crossing vs branch** classification, density-first OCT-A metrics | Vesselness as a cost-surface term; degree-classification to separate true branch vs overlap/anastomosis; density/FD/lacunarity fallback |
| **Geomorph + physics** | Strahler/Shreve/**Tokunaga** ordering theory; **DLA/DBM regime classification**; Minkowski functionals; lacunarity/multifractal | Whole-pattern process descriptors readable from a coarse mask; **already the operative theory of microbial colony fractal growth** (Matsushita 1990, D≈1.73) |
| **Astronomy** (see [companion file](astronomy-thin-structures.md)) | **Noise-beating extraction** of thin faint filaments/streaks: Steger subpixel curvilinear detector, **Rolling Hough Transform** orientation, **DisPerSE/DRUID persistence**, starlet/curvelet denoise, matched/Radon line filters | The only analog built *around* low-SNR noise; supplies the front-end (denoise → enhance → confidence-ranked trace) the others assume away — closest match to faint dense overlapping hyphae on textured agar |

### 3.2 The dense / overlapping / low-res / crossing toolkit (the part you actually need)

Every field converged on the same families of answers. In rough order of value to PhenoTypic:

1. **Density / fractal / lacunarity / Minkowski fallback (no tracing needed).** When structures
   can't be separated, all four analog fields abandon graph metrics and use mask-level
   descriptors. These are the 🟢 rows of Families B–D and are the safe backbone for confluent
   mycelium. *(Stošić 2006; Zahid 2016; Plotnick 1996; Mecke 2000.)*
2. **Hessian vesselness pre-filter (Frangi/Sato).** Multiscale tube-enhancement that recovers
   faint/thin hyphae before thresholding — the single most portable preprocessing step, and a
   natural new term in the existing composite **cost surface**. *(Frangi 1998; Sato 1998.)*
3. **A\* / least-cost-path tracing through crossings.** RootNav and SNT trace each filament as a
   minimal-cost path so a crossing is *traversed*, not mis-read as a branch. **This is exactly
   what `branch_pathfinding` already does** (multi-source Dijkstra). The transfer is the
   *refinements*: continuity-of-direction edge pairing at junctions, gray-weighted distance trees
   (APP2), hierarchical spur pruning. *(Pound 2013; Longair 2011; Xiao & Peng 2013.)*
4. **Graph degree-classification at junctions.** Classify nodes by degree and resolve degree-4
   nodes as X-crossings (continue each strand by smallest turning angle) vs true branches vs
   anastomoses. Directly addresses the retracing module's accuracy ceiling. *(Dirnberger 2015;
   vascular A/V topology.)*
5. **Deep-learning segmentation, ideally interactive.** U-Net / flood-filling for dense fields;
   **RootPainter's corrective-annotation loop** reaches a usable segmenter on occlusion-heavy
   images without a big pre-labeled set — a realistic path given fungal edge-precision is the
   documented bottleneck (MyceliumSeg: ~84% region vs ~28% *edge* accuracy). *(Januszewski 2018;
   Smith 2022; Wang 2025.)*
6. **Add a time axis to disambiguate.** Whole-field time-lapse turns ambiguous static crossings
   into resolvable events (a true fusion persists as a load-bearing loop; a crossing does not).
   *(Dikec 2020.)*
7. **3-D to physically remove occlusion** (confocal/light-sheet ↔ RSA's X-ray-CT + RSAtrace3D) —
   trades throughput for separability. *(Teramoto 2020.)*
8. **Topology-aware validation (DIADEM-style).** Score automated traces against ground truth with
   a bifurcation/termination-matching metric rather than per-metric agreement — the right QC for a
   maturing retracing module. *(Gillette 2011.)*

> **Concrete takeaway for `branch_pathfinding`:** you already have the right backbone (multi-source
> Dijkstra least-cost path + structural-quality path filtering). The literature says the accuracy
> gains at crossings come from (a) a **vesselness term** in the cost surface, (b) **degree-4 node
> resolution by direction continuity**, (c) **interactive correction** at ambiguous junctions, and
> (d) a **DIADEM-style validation** metric — not from a different tracing paradigm.

### 3.3 Astronomy — thin elongated structures in low-SNR noise (added 2026-06-25)

Full table + references in the [companion file](astronomy-thin-structures.md). Astronomy is the
one analog field whose entire methodology is built *around* the noise problem the others assume
away — extracting sub-pixel-faint filaments and streaks from low-SNR frames. Its core principle —
**integration beats thresholding** (accumulate faint signal along a line, across scales, or over
topology *before* committing to a detection, so SNR grows as √(pixels integrated)) — is exactly
what a poorly-resolved, dense, overlapping hyphal field needs. Best transfers, ranked:

1. **Steger unbiased curvilinear detector** *(Steger 1998)* — Hessian-of-Gaussian line model giving
   **subpixel centerline + width** at poor resolution; density-tolerant and already standard in
   biomedical vessel/neurite tracing. **Top pick for tracing faint hyphae.**
2. **Rolling Hough Transform** *(Clark 2014)* — per-pixel local Hough → noise-robust
   **orientation/anisotropy field**; asks "is there local linear coherence?" not "is this pixel
   bright?", so faint coherent hyphae survive while agar texture averages out. **Top pick for
   orientation in noise** (feeds Family-D anisotropy and the cost surface's orientation term).
3. **Multiscale Hessian ridge (NEXUS+ / Frangi-style)** *(Cautun 2013; Frangi 1998)* — variable-width
   tube enhancement across a scale ladder before segmentation (same eigenvalue idea already used in
   the vascular analog and proposed for the cost surface).
4. **DisPerSE persistence + DRUID topological deblending** *(Sousbie 2011; Whitehead 2024)* — a
   **confidence-ranked skeleton**: filaments are ridge lines pruned by *persistence* (statistical
   significance), so it won't hallucinate branches from texture, and the same idea **deblends dense
   overlapping sources**. Ties directly to the neuroscience TMD persistence barcode — persistence is
   the recurring noise-robust primitive across both fields.
5. **Starlet/curvelet denoise + matched/Radon line filtering** *(Starck 2007, 2002; Vio & Andreani
   2016; Nir 2018)* — an SNR-lifting front-end: per-scale wavelet denoise (curvelets matched to thin
   curves) then a matched/Radon filter that detects line-shaped signal invisible per-pixel.
6. **U-Net with Dice/Combo loss** *(Jeong 2024)* — the learned route once labels exist; the
   class-imbalance-aware loss (thin positives vs background) transfers directly to hyphal
   segmentation, and the learned-mask → Hough-cleanup pattern is a strong template.

*Lower-fit (assumption mismatch):* Bisous, SCMS, MST, and the tidal/velocity-web classifiers assume
**sparse, well-separated tracers** — the opposite of dense hyphae — so they apply only to a sparse
high-confidence detection cloud, not the raw field.

---

## 4. PhenoTypic schema map (draft `MeasurementInfo` entries)

Drafts only — `bio_desc=""` and `image=None` are intentional (biological claims are
human-authored per the repo guardrail). `desc` carries the technical/algorithm description.
Tiers follow the existing convention (1 = Direct phenotype, 2 = Descriptive trait,
3 = Discriminative feature). Proposed structural placement noted per block.

### 4.1 New category `COMPLEXITY` (Regime B, 🟢 — extends `SHAPE`/`TEXTURE`)

```python
# src/phenotypic/schema/_complexity.py
class COMPLEXITY(PrimaryMeasure):
    """Space-filling and heterogeneity descriptors of the whole-mycelium silhouette.

    Regime-B branchiness proxies computed from the binary mask / grayscale without
    resolving individual hyphae. Robust to dense, overlapping, low-resolution growth.
    """
    @classmethod
    def category(cls): return "Complexity"
    @classmethod
    def tier(cls): return 2

    FRACTAL_DIMENSION_MASS = Entry(
        "FractalDimensionMass",
        "Box-counting fractal dimension of the filled colony mask: slope of "
        "log(occupied box count) vs log(1/box size). Indexes interior space-filling "
        "/ branching density on a ~1.2-2.0 scale.",
        tier=2)
    FRACTAL_DIMENSION_SURFACE = Entry(
        "FractalDimensionSurface",
        "Box-counting fractal dimension of the colony outline only. Indexes margin / "
        "foraging-front ruggedness; divergence from the mass dimension distinguishes a "
        "pelleted (filled-core) form from a dispersed (mass-fractal) form.",
        tier=2)
    LACUNARITY = Entry(
        "Lacunarity",
        "Gliding-box variance/mean^2 of mask occupancy versus box size. Quantifies "
        "gappiness/heterogeneity of the inter-hyphal void distribution; separates "
        "patterns that share a fractal dimension but differ in clumping.",
        tier=2)
    MULTIFRACTAL_WIDTH = Entry(
        "MultifractalWidth",
        "Width (alpha_max - alpha_min) of the multifractal singularity spectrum f(alpha). "
        "Measures how much the local space-filling exponent varies across the colony.",
        tier=3)
    EULER_NUMBER = Entry(
        "EulerNumber",
        "Topological connectivity number (objects minus enclosed holes) of the binary "
        "mask; tracks loop/anastomosis formation at the silhouette level.",
        tier=2)
    PERIMETER_AREA_DIMENSION = Entry(
        "PerimeterAreaDimension",
        "Fractal dimension from the log-perimeter vs log-area scaling of the contour; "
        "boundary roughness (smooth/compact vs fjord-like/ramified margin).",
        tier=3)
    LAPLACIAN_GROWTH_EXPONENT = Entry(
        "LaplacianGrowthExponent",
        "Effective dielectric-breakdown exponent eta obtained by placing the measured "
        "mass fractal dimension on the Laplacian-growth (DLA/DBM) family; locates the "
        "colony on the ramified(diffusion-limited)<->compact(supply-rich) morphotype axis.",
        tier=3)
```

### 4.2 New category `RADIAL_PROFILE` (Regime B, 🟢 — or fold into existing `_radial_expansion.py`)

```python
# Sholl-style radial density profile centered on inoculum/centroid.
class RADIAL_PROFILE(PrimaryMeasure):
    @classmethod
    def category(cls): return "Radial"
    @classmethod
    def tier(cls): return 2

    SHOLL_MAX_INTERSECTIONS = Entry(
        "ShollMaxIntersections",
        "Maximum number of mask crossings over concentric rings centered on the colony "
        "origin (peak hyphal-density ring count).", tier=2)
    SHOLL_CRITICAL_RADIUS = Entry(
        "ShollCriticalRadius",
        "Radius at which the Sholl intersection count is maximal (active-growth annulus).",
        tier=2)
    SHOLL_ENCLOSING_RADIUS = Entry(
        "ShollEnclosingRadius",
        "Largest radius with a non-zero ring intersection (radial extent of growth).",
        tier=1)
    SHOLL_REGRESSION_COEFF = Entry(
        "ShollRegressionCoefficient",
        "Slope of the semi-log Sholl profile: rate of hyphal-density decline with radius.",
        tier=3)
    RADIAL_DENSITY_FALLOFF = Entry(
        "RadialDensityFalloff",
        "Occupied-pixel fraction per concentric annulus vs radius (dense core -> sparse "
        "margin profile).", tier=2)
    RELATIVE_ANNULAR_DIAMETER = Entry(
        "RelativeAnnularDiameter",
        "(d_total - d_core) / d_total: fraction of the colony radius occupied by the loose "
        "outer (hairy) zone versus the compact core.", tier=3)
```

### 4.3 Extend `TEXTURE` (Regime B, 🟢) — anisotropy members

```python
# Add to existing src/phenotypic/schema/_texture.py
ORIENTATION_ANISOTROPY = Entry(
    "OrientationAnisotropy",
    "Concentration/variance of the local dominant-orientation field (structure tensor or "
    "directional GLCM); degree of preferred hyphal growth direction vs isotropic spread.",
    tier=3)
FOURIER_DIRECTIONALITY = Entry(
    "FourierDirectionality",
    "Angular distribution peak of the 2-D Fourier power spectrum; preferred branch "
    "orientation and n-fold symmetry without segmentation.", tier=3)
```

### 4.4 New category `BRANCH` (Regime A, 🟡 retracing-gated) — *requires the skeleton/graph accessor*

```python
# src/phenotypic/schema/_branch.py  — gated on branch_pathfinding maturity.
# Members carry derivation/robustness intent; many are A-fragile and should ship
# only once the retracing accuracy + a DIADEM-style validation gate are in place.
class BRANCH(PrimaryMeasure):
    @classmethod
    def category(cls): return "Branch"
    @classmethod
    def tier(cls): return 3   # discriminative by default; counts/lengths override to 1-2

    TOTAL_HYPHAL_LENGTH = Entry(
        "TotalHyphalLength",
        "Sum of all skeleton segment lengths in the reconnected hyphal network.", tier=1)
    TIP_COUNT = Entry(
        "TipCount",
        "Number of degree-1 skeleton nodes (hyphal apices).", tier=1)
    BRANCH_POINT_COUNT = Entry(
        "BranchPointCount",
        "Number of degree->=3 skeleton nodes (hyphal branch points), with degree-4 nodes "
        "flagged as candidate crossings/anastomoses rather than branches.", tier=1)
    BRANCHING_FREQUENCY = Entry(
        "BranchingFrequency",
        "Branch points per unit hyphal length.", tier=2)
    HYPHAL_GROWTH_UNIT = Entry(
        "HyphalGrowthUnit",
        "Total hyphal length divided by tip count (mean length supporting one tip).",
        tier=2)
    MEAN_BRANCH_ANGLE = Entry(
        "MeanBranchAngle",
        "Mean angle between parent hypha and daughter branch over resolved bifurcations.",
        tier=2)
    MEAN_INTERNODE_DISTANCE = Entry(
        "MeanInternodeDistance",
        "Mean skeleton arc length between consecutive branch points along a hypha.", tier=2)
    MEAN_TORTUOSITY = Entry(
        "MeanTortuosity",
        "Mean per-branch geodesic length divided by Euclidean endpoint distance (>=1).",
        tier=2)
    STRAHLER_MAX_ORDER = Entry(
        "StrahlerMaxOrder",
        "Maximum Horton-Strahler order of the hyphal tree (terminal segments = order 1).",
        tier=3)
    BIFURCATION_RATIO = Entry(
        "BifurcationRatio",
        "Ratio of segment counts between successive Strahler orders (Horton's law).",
        tier=3)
    ANASTOMOSIS_LOOP_COUNT = Entry(
        "AnastomosisLoopCount",
        "First Betti number (independent cycles, E - V + C) of the network graph; "
        "loop-forming hyphal fusions.", tier=3)
    PARTITION_ASYMMETRY = Entry(
        "PartitionAsymmetry",
        "Mean per-bifurcation tip-count imbalance |nL - nR|/(nL + nR - 2) over the tree.",
        tier=3)
    TMD_PERSISTENCE = Entry(
        "TmdPersistenceImage",
        "Vectorized topological-morphology-descriptor persistence image: branch "
        "birth(tip)-death(bifurcation) along radial distance; order-invariant network "
        "fingerprint that degrades gracefully under tracing error.", tier=3)
```

> **Dynamics & pigment** map onto existing surfaces rather than new enums: colony radial
> growth rate / FD-over-time fit the growth-model schema (`_log_growth_model.py`,
> `_radial_expansion.py`); pigment intensity, radial gradient, sectoring, zonation, and
> color heterogeneity fit the existing color categories (`_color_lab.py`, `_color_hsv.py`,
> `_color_composition.py`). These need acquisition support (time-lapse / RGB) more than new
> schema. None of the pigment members should be authored with `bio_desc` — leave for a human.

---

## 5. Recommended phased rollout

| Phase | Adds | Gating requirement | Fit |
|---|---|---|---|
| **1 — Regime B (ship now)** | `COMPLEXITY` (fractal mass+surface, lacunarity, Euler, multifractal, Laplacian-η), `RADIAL_PROFILE` (Sholl + radial falloff + RAD), `TEXTURE` anisotropy, margin/macromorphotype | Mask + grayscale only; you already have these inputs | 🟢 |
| **2 — Regime A (retracing-gated)** | `BRANCH` topology + per-branch geometry + TMD | `branch_pathfinding` accuracy ↑ via vesselness cost term + degree-4 junction resolution + interactive correction + **DIADEM-style validation gate** | 🟡→🔴 |
| **3 — Dyn / Phys overlays** | Growth-rate & FD-over-time (growth-model schema); pigment/zonation/sectoring (color schema) | Time-lapse acquisition; RGB acquisition | 🟢⏱ / 🎨 |

**Engineering corollaries for the retracing module** (`sdk_/branch_pathfinding/`):
1. Add a **Frangi/Sato vesselness** channel to `assemble_composite_cost` (you already fuse
   phase-congruency / orientation-coherence / anisotropy / local-MAD).
2. After `extract_fragment_paths`, add **degree-classification** of graph nodes and resolve
   degree-4 nodes by direction continuity → separates *branch* vs *crossing* vs *anastomosis*
   (the documented accuracy ceiling).
3. Add a **DIADEM-style topology-similarity** diagnostic to `_path_quality` / `_diagnostics`
   for validation against hand-traced ground truth.
4. Consider a **RootPainter-style interactive correction** affordance in the GUI for the
   ambiguous-junction minority — every analog field treats human-in-the-loop as the accuracy
   ceiling for trustworthy topology.
5. **(Astronomy front-end)** Prepend a **starlet/curvelet denoise** stage (strip agar texture while
   keeping thin hyphae) and a **matched/Radon line filter** to lift faint hyphae above the per-pixel
   noise floor *before* the cost surface is assembled.
6. **(Astronomy tracer)** Evaluate the **Steger unbiased curvilinear detector** for subpixel
   centerline + width — a principled alternative/complement to skeleton-from-mask that degrades more
   gracefully at low resolution — and add a **Rolling Hough Transform** orientation channel to the
   cost surface (noise-robust replacement/augmentation of the current orientation-coherence term).
7. **(Astronomy topology)** Use **persistence (DisPerSE/DRUID-style) pruning** to rank candidate
   branches by statistical significance rather than a raw cost threshold — the same persistence
   primitive as the neuroscience TMD barcode, and a natural fit for `_path_quality` filtering.

---

## 6. References (consolidated, DOI-verified by source agents)

*Mycology — morphology, networks, dynamics, pigment*
- Aguilar-Trigueros, C. A., et al. (2022). Network traits predict ecological strategies in fungi. *ISME Communications, 2*, 2. https://doi.org/10.1038/s43705-021-00085-1
- Arganda-Carreras, I., et al. (2010). 3D reconstruction of histological sections: Application to mammary gland tissue [designated canonical citation for the AnalyzeSkeleton Fiji plugin]. *Microscopy Research and Technique, 73*(11), 1019–1029. https://doi.org/10.1002/jemt.20829
<!-- FACT-CHECK: [CONFIRMED] DOI and bibliographic details correct per ImageJ plugin documentation (imagej.net/plugins/analyze-skeleton/). The paper subject is histological 3D-reconstruction (not a skeleton-analysis paper per se); the plugin developers explicitly designated this as the citation for the software. Source: PubMed PMID 20232465; imagejdocu.list.lu -->
- Barry, D. J., & Williams, G. A. (2015). AnaMorf. *Biotechnology Progress, 31*(3), 849–852. https://doi.org/10.1002/btpr.2087
- Bebber, D. P., et al. (2007). Biological solutions to transport network design. *Proc. R. Soc. B, 274*(1623), 2307–2315. https://doi.org/10.1098/rspb.2007.1093
- Belmonte, R. C., et al. (2019). RGB imaging of *Fusarium graminearum*. *Methods and Protocols, 2*(1), 25. https://doi.org/10.3390/mps2010025
- Cairns, T. C., et al. (2019). Image-analysis pipeline for fungal morphology (aplD). *Biotechnology for Biofuels, 12*, 149. https://doi.org/10.1186/s13068-019-1473-0
- Camy, C., et al. (2023). Fungal drops. *microLife, 4*, uqad042. https://doi.org/10.1093/femsml/uqad042
- Cesbron, F., et al. (2024). Rhythmidia. *PLOS Computational Biology, 20*(7), e1012167. https://doi.org/10.1371/journal.pcbi.1012167
- Dikec, J., et al. (2020). Hyphal network whole-field imaging (*Podospora*). *Scientific Reports, 10*, 3131. https://doi.org/10.1038/s41598-020-57808-y
- Dirnberger, M., Kehl, T., & Neumann, A. (2015). NEFI: Network Extraction From Images. *Scientific Reports, 5*, 15669. https://doi.org/10.1038/srep15669
- Dunlap, J. C., & Loros, J. J. (2007). Rhythmic conidiation in *Neurospora crassa*. *Methods Mol. Biol., 362*, 43–58. https://doi.org/10.1007/978-1-59745-257-1_3
- Fricker, M. D., et al. (2007). Network organisation of mycelial fungi. *PNAS, 104*(5), 1750–1757. https://doi.org/10.1073/pnas.0703255104
- Heaton, L., et al. (2012). Analysis of fungal networks. *J. R. Soc. Interface, 9*(72), 1–18. https://doi.org/10.1098/rsif.2011.0735
- Held, M., et al. (2019). Fungal space searching in microenvironments. *PNAS, 116*(27), 13543–13552. https://doi.org/10.1073/pnas.1816423116
- Hernández-Lauzardo, A. N., et al. (2018). Colors vs size in *Fusarium graminearum*. *Foods, 7*(7), 100. https://doi.org/10.3390/foods7070100
- Katz, D., Goldstein, D., & Rosenberger, R. F. (1972). Model for branch initiation in *Aspergillus nidulans*. *J. Bacteriol., 109*(3), 1097–1100. https://doi.org/10.1128/jb.109.3.1097-1100.1972
- Lamour, C., et al. (2022). Angular branching optimisation in *Podospora anserina*. *Scientific Reports, 12*, 12351. https://doi.org/10.1038/s41598-022-16245-9
- Lecault, V., Patel, N., & Thibault, J. (2009). Branching frequency via fractal analysis. *Biotechnology and Bioengineering, 104*(4), 762–770. https://doi.org/10.1002/bit.24709
- Lee, S. H., Fricker, M. D., & Porter, M. A. (2017). Mesoscale analyses of fungal networks. *J. Complex Networks, 5*(1), 145–159. https://doi.org/10.1093/comnet/cnw010
- Linde, T., et al. (2021). Conidia counting + gray-value correlation. *MethodsX, 8*, 101218. https://doi.org/10.1016/j.mex.2021.101218
- Meškauskas, A., Fricker, M. D., & Moore, D. (2004). Neighbour-Sensing model. *Mycological Research, 108*(11), 1241–1256. https://doi.org/10.1017/S0953756204001261
- Moore, R. T., et al. (2024). Chemotropism in *Aspergillus nidulans*. *PLOS Biology, 22*(7), e3002726. https://doi.org/10.1371/journal.pbio.3002726
- Nordzieke, D. E., et al. (2020). 3D-printed fungal chemotropism device. *Frontiers in Microbiology, 11*, 584525. https://doi.org/10.3389/fmicb.2020.584525
- Nunez-Iglesias, J., et al. (2018). skan. *PeerJ, 6*, e4312. https://doi.org/10.7717/peerj.4312
- Obert, M., Pfeifer, P., & Porstendörfer, M. (1990). Microbial growth patterns described by fractal geometry. *J. Bacteriol., 172*(3), 1180–1185. https://doi.org/10.1128/jb.172.3.1180-1185.1990
- Papagianni, M. (2006). Fractal nature of mycelial aggregation in *A. niger*. *Microbial Cell Factories, 5*, 5. https://doi.org/10.1186/1475-2859-5-5
<!-- FACT-CHECK: [CORRECTED] Was: "Papagianni, M., & Mattey, M.". This paper (DOI 10.1186/1475-2859-5-5) is sole-authored by Papagianni. The co-authored Papagianni & Mattey 2006 paper is a separate article (DOI 10.1186/1475-2859-5-3) on morphological development in citric acid fermentation. Source: PMC1382250; PubMed PMID 16472407. -->
- Paul, G. C., & Thomas, C. R. (1998). Image analysis of filamentous micro-organism morphology. *Microbiology, 144*(4), 817–827. https://doi.org/10.1099/00221287-144-4-817
- Riquelme, M., et al. (2019). Hyphal branching in filamentous fungi. *Developmental Biology, 451*(1), 39–48. https://doi.org/10.1016/j.ydbio.2018.12.008
- Ritz, K., & Crawford, J. (1995). Image analysis of space-filling by fungal networks. *Biotechnology Techniques, 9*(7), 461–466. https://doi.org/10.1007/BF00158947
- Spitzer, M., et al. (2017). HyphaTracker. *Scientific Reports, 7*, 16523. https://doi.org/10.1038/s41598-017-19103-1
- Trinci, A. P. J. (1969). Kinetic study of growth of *A. nidulans*. *J. Gen. Microbiol., 57*(1), 11–24. https://doi.org/10.1099/00221287-57-1-11
- Trinci, A. P. J. (1971). Peripheral growth zone and radial growth rate. *J. Gen. Microbiol., 67*(3), 325–344. https://doi.org/10.1099/00221287-67-3-325
- Trinci, A. P. J. (1974). Hyphal extension and branch initiation. *J. Gen. Microbiol., 81*(1), 225–236. https://doi.org/10.1099/00221287-81-1-225
- van Dissel, D., et al. (2017). SParticle. *Antonie van Leeuwenhoek, 111*(2), 171–182. https://doi.org/10.1007/s10482-017-0939-y
- Veiter, L., Rajamanickam, V., & Herwig, C. (2018). Filamentous fungal pellet morphology vs productivity. *Appl. Microbiol. Biotechnol., 102*(7), 2997–3006. https://doi.org/10.1007/s00253-018-8818-7
- Vicente, A., et al. (2025). FungID. *Pathogens, 14*(3), 242. https://doi.org/10.3390/pathogens14030242
- Vidal-Diez de Ulzurrun, G., et al. (2019). Fungal Feature Tracker (FFT). *PLOS Computational Biology, 15*(10), e1007428. https://doi.org/10.1371/journal.pcbi.1007428
<!-- FACT-CHECK: [REPLACED] Was: journal "Fungal Genetics and Biology", DOI 10.1016/j.fgb.2018.11.005. That DOI resolves to an entirely unrelated paper: Zhang et al. (2019) "Reference genes for accurate normalization of gene expression in wood-decomposing fungi," FGB 123:33–40. The correct FFT citation is PLOS Computational Biology 15(10):e1007428. Source: journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007428; PMC6822706. -->
- Wang, et al. (2025). MyceliumSeg dataset. *Scientific Data, 12*. https://doi.org/10.1038/s41597-025-06265-1
- Zapanta, N. R., et al. (2025). PUPMCR: an R package for image-based identification of color based on Rayner's (1970) terminology and known fungal pigments. *Biology Methods and Protocols, 10*(1), bpaf004. https://doi.org/10.1093/biomethods/bpaf004
<!-- FACT-CHECK: [CONFIRMED] Full citation added; DOI verified via Oxford Academic (academic.oup.com/biomethods/article/10/1/bpaf004/7952017). PMC identifier: PMC11825390. Previously only PMC11825390 was pinned; DOI was unresolved. -->

*Neuroscience*
- Brown, K. M., et al. (2011). DIADEM data sets. *Neuroinformatics, 9*(2–3), 143–157. https://doi.org/10.1007/s12021-011-9118-x
- Caserta, F., et al. (1995). Fractal dimension of neurons. *J. Neurosci. Methods, 56*(2), 133–144. https://doi.org/10.1016/0165-0270(94)00115-W
- Feng, L., Zhao, T., & Kim, J. (2015). neuTube 1.0. *eNeuro, 2*(1). https://doi.org/10.1523/ENEURO.0049-14.2014
- Ferreira, T. A., et al. (2014). Sholl analysis / morphometry from bitmap images. *J. Neurosci. Methods, 226*, 19–28. https://doi.org/10.1016/j.jneumeth.2014.01.016 (and *Nat. Methods, 11*, 982–984. https://doi.org/10.1038/nmeth.3125)
- Gillette, T. A., Brown, K. M., & Ascoli, G. A. (2011). The DIADEM metric. *Neuroinformatics, 9*(2–3), 233–245. https://doi.org/10.1007/s12021-011-9117-y
- Januszewski, M., et al. (2018). Flood-filling networks. *Nature Methods, 15*(8), 605–610. https://doi.org/10.1038/s41592-018-0049-4
- Jelinek, H. F., & Fernández, E. (1998). Neurons and fractals. *J. Neurosci. Methods, 81*(1–2), 9–18. https://doi.org/10.1016/S0165-0270(98)00021-1
- Kanari, L., et al. (2018). Topological representation of branching morphologies (TMD). *Neuroinformatics, 16*(1), 3–13. https://doi.org/10.1007/s12021-017-9341-1
- Longair, M. H., Baker, D. A., & Armstrong, J. D. (2011). Simple Neurite Tracer. *Bioinformatics, 27*(17), 2453–2454. https://doi.org/10.1093/bioinformatics/btr390
- Meijering, E., et al. (2004). NeuronJ. *Cytometry A, 58A*(2), 167–176. https://doi.org/10.1002/cyto.a.20022
- Peng, H., et al. (2010). V3D / Vaa3D. *Nature Biotechnology, 28*(4), 348–353. https://doi.org/10.1038/nbt.1612
- Scorcioni, R., Polavaram, S., & Ascoli, G. A. (2008). L-Measure. *Nature Protocols, 3*(5), 866–876. https://doi.org/10.1038/nprot.2008.51
- Sholl, D. A. (1953). Dendritic organization in neurons. *Journal of Anatomy, 87*(4), 387–406. [PMC1244622 — pre-DOI]
- Smith, T. G., Lange, G. D., & Marks, W. B. (1996). Fractal methods in cellular morphology. *J. Neurosci. Methods, 69*(2), 123–136. https://doi.org/10.1016/S0165-0270(96)00080-5
- Torben-Nielsen, B. (2014). btmorph. *Neuroinformatics, 12*(4), 619–622. https://doi.org/10.1007/s12021-014-9232-7
- Xiao, H., & Peng, H. (2013). APP2. *Bioinformatics, 29*(11), 1448–1454. https://doi.org/10.1093/bioinformatics/btt170
- Zhou, Z., et al. (2018). DeepNeuron. *Brain Informatics, 5*(2), 3. https://doi.org/10.1186/s40708-018-0081-2

*Plant root architecture*
- Armengaud, P., et al. (2009). EZ-Rhizo. *The Plant Journal, 57*(5), 945–956. https://doi.org/10.1111/j.1365-313X.2009.03829.x
- Bouda, M., Caplan, J. S., & Saiers, J. E. (2016). Box-counting dimension of root systems. *Frontiers in Plant Science, 7*, 149. https://doi.org/10.3389/fpls.2016.00149
- Das, A., et al. (2015). DIRT. *Plant Methods, 11*, 51. https://doi.org/10.1186/s13007-015-0093-3
- Delory, B. M., et al. (2016). archiDART. *Plant and Soil, 398*, 351–365. https://doi.org/10.1007/s11104-015-2673-4
- Galkovskyi, T., et al. (2012). GiA Roots. *BMC Plant Biology, 12*, 116. https://doi.org/10.1186/1471-2229-12-116
- Liu, S., et al. (2021). DIRT/3D. *Plant Physiology, 187*(2), 739–757. https://doi.org/10.1093/plphys/kiab311
- Lobet, G., Pagès, L., & Draye, X. (2011). SmartRoot. *Plant Physiology, 157*(1), 29–39. https://doi.org/10.1104/pp.111.179895
- Pound, M. P., et al. (2013). RootNav. *Plant Physiology, 162*(4), 1802–1814. https://doi.org/10.1104/pp.113.221531
- Seethepalli, A., et al. (2021). RhizoVision Explorer. *AoB PLANTS, 13*(6), plab056. https://doi.org/10.1093/aobpla/plab056
- Smith, A. G., et al. (2022). RootPainter. *New Phytologist, 236*(2), 774–791. https://doi.org/10.1111/nph.18387
- Teramoto, S., et al. (2020). RSAtrace3D. *BMC Plant Biology, 21*, 398. https://doi.org/10.1186/s12870-021-03161-9
- Wang, T., et al. (2019). SegRoot. *Computers and Electronics in Agriculture, 162*, 845–854. https://doi.org/10.1016/j.compag.2019.05.017
- Yasrab, R., et al. (2019). RootNav 2.0. *GigaScience, 8*(11), giz123. https://doi.org/10.1093/gigascience/giz123

*Vascular / angiogenesis*
- Bullitt, E., et al. (2003). Tortuosity from MRA. *IEEE TMI, 22*(9), 1163–1171. https://doi.org/10.1109/TMI.2003.816964
- Carpentier, G., et al. (2020). Angiogenesis Analyzer for ImageJ. *Scientific Reports, 10*, 11568. https://doi.org/10.1038/s41598-020-67289-8
- Chaudhuri, S., et al. (1989). Matched filters for retinal vessels. *IEEE TMI, 8*(3), 263–269. https://doi.org/10.1109/42.34715
- Corliss, B. A., et al. (2020). REAVER. *Microcirculation, 27*(5), e12618. https://doi.org/10.1111/micc.12618
- Frangi, A. F., et al. (1998). Multiscale vessel enhancement filtering. *MICCAI 1998*, LNCS 1496, 130–137. https://doi.org/10.1007/BFb0056195
- Hart, W. E., et al. (1999). Retinal vascular tortuosity. *Int. J. Med. Informatics, 53*(2–3), 239–252. https://doi.org/10.1016/S1386-5056(98)00163-4
- Masters, B. R. (2004). Fractal analysis of the retinal vascular tree. *Annu. Rev. Biomed. Eng., 6*, 427–452. https://doi.org/10.1146/annurev.bioeng.6.040803.140100
- Niemistö, A., et al. (2005). AngioQuant. *IEEE TMI, 24*(4), 549–553. https://doi.org/10.1109/TMI.2004.837339
- Patton, N., et al. (2006). Retinal vascular image analysis. *Journal of Anatomy, 206*(4), 319–348. https://doi.org/10.1111/j.1469-7580.2005.00395.x
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net. *MICCAI 2015*, LNCS 9351, 234–241. https://doi.org/10.1007/978-3-319-24574-4_28
- Sato, Y., et al. (1998). 3-D multi-scale line filter. *Medical Image Analysis, 2*(2), 143–168. https://doi.org/10.1016/S1361-8415(98)80009-1
- Seaman, M. E., Peirce, S. M., & Kelly, K. (2011). RAVE. *PLoS ONE, 6*(6), e20807. https://doi.org/10.1371/journal.pone.0020807
- Stošić, T., & Stošić, B. D. (2006). Multifractal analysis of retinal vessels. *IEEE TMI, 25*(8), 1101–1107. https://doi.org/10.1109/TMI.2006.879316
- Vickerman, M. B., et al. (2009). VESGEN 2D. *The Anatomical Record, 292*(3), 320–332. https://doi.org/10.1002/ar.20887
- Zahid, S., et al. (2016). Fractal dimensional analysis of OCT-A. *IOVS, 57*(11), 4940–4947. https://doi.org/10.1167/iovs.16-19656
- Zudaire, E., et al. (2011). AngioTool. *PLoS ONE, 6*(11), e27385. https://doi.org/10.1371/journal.pone.0027385

*Geomorphology + growth physics*
- Allain, C., & Cloitre, M. (1991). Lacunarity of fractal sets. *Physical Review A, 44*(6), 3552–3558. https://doi.org/10.1103/PhysRevA.44.3552
- Ben-Jacob, E. (1997). From snowflake formation to bacterial colonies II. *Contemporary Physics, 38*(3), 205–241. https://doi.org/10.1080/00018739700101498
- Ben-Jacob, E., & Garik, P. (1990). Patterns in non-equilibrium growth. *Nature, 343*(6258), 523–530. https://doi.org/10.1038/343523a0
- Ben-Jacob, E., Cohen, I., & Gutnick, D. L. (1998). Cooperative organization of bacterial colonies. *Annu. Rev. Microbiol., 52*, 779–806. https://doi.org/10.1146/annurev.micro.52.1.779
- Florindo, J. B., & Bruno, O. M. (2018). Bouligand–Minkowski fractal descriptors. *Information Sciences, 459*, 36–52. https://doi.org/10.1016/j.ins.2018.06.025
- Halsey, T. C., et al. (1986). Fractal measures and singularities. *Physical Review A, 33*(2), 1141–1151. https://doi.org/10.1103/PhysRevA.33.1141
- Horton, R. E. (1945). Erosional development of streams. *GSA Bulletin, 56*(3), 275–370. https://doi.org/10.1130/0016-7606(1945)56[275:EDOSAT]2.0.CO;2
- La Barbera, P., & Rosso, R. (1989). Fractal dimension of stream networks. *Water Resources Research, 25*(4), 735–741. https://doi.org/10.1029/WR025i004p00735
- Matsushita, M., & Fujikawa, H. (1990). Diffusion-limited growth in bacterial colony formation. *Physica A, 168*(1), 498–506. https://doi.org/10.1016/0378-4371(90)90402-E
- Mecke, K. R. (2000). Minkowski functionals in statistical physics. *LNP 554*, 111–184. https://doi.org/10.1007/3-540-45043-2_6
- Melton, M. A. (1958). Geometric properties of drainage systems. *Journal of Geology, 66*(1), 35–54. https://doi.org/10.1086/626490
- Nicolás-Carlock, J. R., Carrillo-Estrada, J. L., & Dossetti, V. (2019). Universal dimensionality function for Laplacian growth. *Scientific Reports, 9*, 1120. https://doi.org/10.1038/s41598-018-38084-3
- Niemeyer, L., Pietronero, L., & Wiesmann, H. J. (1984). Fractal dimension of dielectric breakdown. *Physical Review Letters, 52*(12), 1033–1036. https://doi.org/10.1103/PhysRevLett.52.1033
- Peckham, S. D. (1995). Self-similar trees and river networks (Tokunaga). *Water Resources Research, 31*(4), 1023–1029. https://doi.org/10.1029/94WR03155
- Plotnick, R. E., et al. (1996). Lacunarity analysis. *Physical Review E, 53*(5), 5461–5468. https://doi.org/10.1103/PhysRevE.53.5461
- Saffman, P. G., & Taylor, G. I. (1958). Penetration of a fluid in a Hele-Shaw cell. *Proc. R. Soc. A, 245*(1242), 312–329. https://doi.org/10.1098/rspa.1958.0085
- Sanderson, D. J., & Nixon, C. W. (2015). Topology in fracture-network characterization. *J. Structural Geology, 72*, 55–66. https://doi.org/10.1016/j.jsg.2015.01.005
- Shreve, R. L. (1966). Statistical law of stream numbers. *Journal of Geology, 74*(1), 17–37. https://doi.org/10.1086/627137
- Strahler, A. N. (1957). Quantitative analysis of watershed geomorphology. *Trans. AGU, 38*(6), 913–920. https://doi.org/10.1029/TR038i006p00913
- Witten, T. A., & Sander, L. M. (1981). Diffusion-limited aggregation. *Physical Review Letters, 47*(19), 1400–1403. https://doi.org/10.1103/PhysRevLett.47.1400

*Books / no-DOI (cite by edition):* Mandelbrot, B. B. (1983) *The Fractal Geometry of Nature*;
Rodríguez-Iturbe, I. & Rinaldo, A. (1997) *Fractal River Basins*; Vicsek, T. (1992) *Fractal Growth
Phenomena*; Hack, J. T. (1957) USGS PP 294-B.

---

## 7. Verification caveats (updated 2026-06-25 after independent fact-check audit)

- **Provisional / not peer-reviewed:** *HyPhy* (bioRxiv 2025.09.11.675604) — confirmed still
  preprint as of 2026-06-25; no published journal version found.
  <!-- FACT-CHECK: [CONFIRMED] HyPhy remains a bioRxiv preprint. Source: biorxiv.org/content/10.1101/2025.09.11.675604v1 -->
- **Unconfirmed tool names — do NOT cite without independent check:** "MycoMeter" (the commercial
  *Mycometer* is an enzymatic biomass assay, not an image tool), "CMEIAS" (bacterial, not hyphal),
  "Vaa3D-for-fungi" (generic Vaa3D repurposed; no canonical fungal citation).
  <!-- FACT-CHECK: [CONFIRMED] All three are correctly flagged as mis-attributions; no canonical fungal image-tool citations exist for them. -->
- **AnalyzeSkeleton citation confirmed:** DOI 10.1002/jemt.20829 (Arganda-Carreras et al. 2010,
  *Microscopy Research and Technique* 73(11):1019–1029) is the citation the plugin developers
  themselves designate on the ImageJ/Fiji documentation page. The paper is a histological
  3D-reconstruction study (mammary gland tissue); the plugin authors chose it as the canonical
  software citation despite the mismatch of subject matter. The DOI and bibliographic details in
  §6 are correct.
  <!-- FACT-CHECK: [CONFIRMED] DOI verified via imagej.net/plugins/analyze-skeleton/ and imagejdocu.list.lu; paper subject confirmed via PubMed PMID 20232465. -->
- **FFT (Vidal-Diez de Ulzurrun) corrected in §6:** the original entry cited journal "Fungal
  Genetics and Biology" and DOI 10.1016/j.fgb.2018.11.005; that DOI resolves to an entirely
  different paper (Zhang et al. 2019, reference-gene normalization in wood-decomposing fungi).
  Corrected to *PLOS Computational Biology, 15*(10):e1007428, DOI 10.1371/journal.pcbi.1007428.
  <!-- FACT-CHECK: [REPLACED] See §6 correction note. -->
- **Papagianni 2006 authorship corrected in §6:** DOI 10.1186/1475-2859-5-5 is sole-authored by
  Papagianni; "Papagianni & Mattey" was incorrect. Table rows citing "Papagianni & Mattey 2006"
  refer to the same sole-authored paper and should be read as "Papagianni 2006". The separate
  co-authored paper (Papagianni & Mattey 2006, DOI 10.1186/1475-2859-5-3) concerns citric acid
  fermentation morphology and is a different work.
  <!-- FACT-CHECK: [CONFIRMED] Source: PMC1382250; PubMed PMID 16472407. -->
- **Melanization table row author corrected:** "Lyra 2025 (PUPMCR)" was incorrect; the PUPMCR
  package is by Zapanta, N.R. et al. (Deocaris group). No "Lyra 2025" fungal melanization paper
  was found. Table cell updated to "Zapanta et al. 2025 (PUPMCR)".
  <!-- FACT-CHECK: [REPLACED] Source: doi.org/10.1093/biomethods/bpaf004; PMC11825390. -->
- **PUPMCR DOI now confirmed:** 10.1093/biomethods/bpaf004 (*Biology Methods and Protocols,
  10*(1):bpaf004, 2025). PMC11825390 remains the correct PMC identifier.
  <!-- FACT-CHECK: [CONFIRMED] Source: academic.oup.com/biomethods/article/10/1/bpaf004/7952017 -->
- **AngioQuant DOI confirmed:** 10.1109/TMI.2004.837339 (*IEEE TMI, 24*(4):549–553, 2005) is
  correct. (The DOI uses a 2004 manuscript number despite the 2005 publication date; PubMed
  PMID 15822812 confirms.)
  <!-- FACT-CHECK: [CONFIRMED] Source: pubmed.ncbi.nlm.nih.gov/15822812/ -->
- **Retinal/OCT-A secondary DOIs:** Durbin 2017 (10.1001/jamaophthalmol.2017.0080), Kim 2016
  (10.1167/iovs.15-18904), and Patton 2006 (10.1111/j.1469-7580.2005.00395.x) are all
  confirmed by PubMed/journal lookup. Perez-Rovira VAMPIRE (2011) correct DOI is
  10.1109/IEMBS.2011.6090918 (pages 3391–3394, IEEE EMBC 2011); the earlier draft DOI
  10.1109/IEMBS.2011.6090724 maps to an unrelated optic-nerve-head segmentation paper —
  do not use. VAMPIRE is not in the §6 reference list; any external publication must cite
  the correct DOI.
  <!-- FACT-CHECK: [CORRECTED] VAMPIRE DOI: was 6090724 (wrong paper), confirmed correct is 6090918. Sources: pubmed.ncbi.nlm.nih.gov/22255067/; researchportal.hw.ac.uk -->
- **Linde 2021 confirmed:** *MethodsX, 8*, 101218, DOI 10.1016/j.mex.2021.101218. ✓
  <!-- FACT-CHECK: [CONFIRMED] Source: sciencedirect.com/science/article/pii/S2215016121000108; PMC8374203 -->
- **Headline spot-checks (no retraction found):** Sholl 1953 (PMC1244622, *J. Anat.* 87(4):387–406),
  Kanari 2018 TMD (DOI 10.1007/s12021-017-9341-1, *Neuroinformatics* 16:3–13), Frangi 1998
  (DOI 10.1007/BFb0056195, MICCAI LNCS 1496:130–137), Matsushita & Fujikawa 1990
  (DOI 10.1016/0378-4371(90)90402-E, *Physica A* 168:498–506), Zudaire 2011 AngioTool
  (DOI 10.1371/journal.pone.0027385, *PLoS ONE* 6(11):e27385), Seethepalli 2021 RhizoVision
  (DOI 10.1093/aobpla/plab056, *AoB PLANTS* 13(6):plab056), and Strahler 1957
  (DOI 10.1029/TR038i006p00913, *Trans. AGU* 38(6):913–920) — all confirmed; no retraction
  or editorial-concern notices found.
  <!-- FACT-CHECK: [CONFIRMED] All seven spot-checked references resolve correctly. -->
- **Method-origin attributions** (Obert box-counting, Cox & Thomas radius-of-gyration, Boddy &
  Donnelly network FD, Patankar ultrasonic index, Fitter 1987 topological framework, Tokunaga 1978
  original) are cited *as referenced inside* verified papers, not independently fetched.
- **Pre-DOI / book sources** (Sholl 1953, Mandelbrot 1983, Hack 1957, Tokunaga 1978) are cited by
  stable identifier or edition.
- No retraction/editorial-concern flags were found for any headline tool/method paper during this
  audit. Items not checked for retractions: the 2025 preprint (HyPhy) and the most recent 2025
  primary papers (Moore, Vicente, Wang, Camy, Cesbron, Lyra/PUPMCR).
