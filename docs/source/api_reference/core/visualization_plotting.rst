Visualization & Plotting
========================

.. currentmodule:: phenotypic

.. autoproperty:: Image.plot

The ``plot`` property provides quality-of-life visualizations for pipeline development,
specifically designed for colony detection parameter tuning on arrayed microbe cultures
on agar plates. These methods help understand how morphological operations, size filtering,
and spatial patterns affect detection results.

All methods support flexible data requirements, automatically detecting whether labeled
objects (objmap) or binary masks (objmask) are available.

Morphological Progression Analysis
-----------------------------------

.. automethod:: Image.plot.morph_progression

**Purpose:** Visualize how morphological operations (erosion, dilation, opening, closing)
affect colony boundaries as kernel size increases.

**Use Case:** Identify critical transition points where small colonies vanish, distinct
colonies merge, or unexpected artifacts appear. This is essential for selecting optimal
morphological parameters in colony detection pipelines.

**Output:** Grid of images with color-coded boundary overlays for each kernel size.

**Interpretation Guide:**

- **Small colonies vanishing:** Operation too aggressive → reduce kernel size or use gentler preprocessing
- **Colonies merging unexpectedly:** Kernel too large → reduce size or improve initial detection
- **Stable transitions:** Minimal boundary changes between consecutive sizes → robust parameter region
- **Edge artifacts:** Boundary behavior differs at image edges → consider cropping or border removal

**Domain Context (Agar Plates):**

For arrayed microbial cultures, morphological operations are used to:

- Remove agar texture artifacts (opening with small disk)
- Merge colony fragments from uneven growth (closing)
- Remove uncertain edge pixels (erosion)
- Recover faint boundaries (dilation)

By observing boundary evolution across kernel sizes, you can empirically determine
appropriate parameters for your imaging conditions, colony size distribution, and
inter-colony spacing.

**Example:**

.. code-block:: python

    from phenotypic import Image
    from phenotypic.detect import OtsuDetector
    
    # Load and detect colonies
    img = Image.imread('plate.jpg')
    detector = OtsuDetector()
    detected = detector.apply(img)
    
    # Visualize opening operation across kernel sizes
    fig, axes = detected.plot.morph_progression(
        operation='opening',
        kernel_sizes=[1, 3, 5, 7, 9, 11, 13, 15],
        shape='disk'
    )
    fig.suptitle('Opening: Colony Boundary Changes')
    
    # Compare with closing
    fig2, axes2 = detected.plot.morph_progression(
        operation='closing',
        kernel_sizes=[1, 3, 5, 7, 9, 11],
        shape='disk'
    )
    fig2.suptitle('Closing: Gap Filling Progression')

Structural Response Curves
---------------------------

.. automethod:: Image.plot.structural_response_curve

**Purpose:** Quantify how object metrics (count, total area, mean size) respond to
morphological kernel size changes. Provides data-driven parameter selection by identifying
stable plateaus (robust) vs. steep slopes (sensitive).

**Use Case:** Select morphological parameters that balance noise removal with colony
preservation. Stable regions (low derivative) indicate robust parameter choices.

**Output:** Line plot with metric vs. kernel size; optional derivative curve showing
sensitivity.

**Interpretation Guide:**

- **Plateaus (low derivative):** Stable, robust parameters insensitive to small adjustments
- **Steep slopes (high derivative):** Parameter-sensitive zones where small changes dramatically affect results
- **For noise removal:** Select kernel where count stabilizes after initial drop
- **For fragment merging:** Select kernel where count drops to expected value without area explosion

**Domain Context (Colony Phenotyping):**

The structural response curve reveals trade-offs in detection pipelines:

- **Opening for noise removal:** Count drops sharply at small sizes (dust removed), then plateaus (real colonies preserved)
- **Closing for fragment merging:** Area increases while count decreases (fragments merge into single colonies)
- **Optimal kernel:** First stable plateau after transition region

**Example:**

.. code-block:: python

    detected = detector.apply(img)
    
    # Plot count response to opening
    fig, ax = detected.plot.structural_response_curve(
        operation='opening',
        kernel_range=(1, 20),
        metric='count',
        show_derivative=True
    )
    ax.set_title('Colony Count vs. Opening Kernel Size')
    
    # Find optimal kernel: first plateau after noise removal
    # Look for region where derivative approaches zero

Boundary Displacement Analysis
-------------------------------

.. automethod:: Image.plot.boundary_displacement

**Purpose:** Visualize spatial sensitivity to morphological parameters as a heatmap
showing where boundary displacement is highest. Identifies illumination-dependent
detection and edge effects.

**Use Case:** Detect systematic biases in detection due to uneven illumination,
vignetting, or edge artifacts.

**Output:** Heatmap showing boundary displacement magnitude at each location; statistics
panel with displacement metrics.

**Interpretation Guide:**

- **High displacement at edges:** Edge effects or border artifacts → apply border removal
- **High displacement following lighting patterns:** Vignetting or illumination gradient → apply flat-field correction
- **Uniform displacement:** Parameter-sensitive but systematic → acceptable if magnitude is small
- **Clustered displacement:** Specific biological/technical issues (small colonies, contamination, touching colonies)

**Domain Context (Agar Plates):**

Boundary displacement patterns reveal imaging artifacts common in plate scanners:

- **Radial gradient:** Lens vignetting or uneven illumination from center to edges
- **Corners darker:** Extreme vignetting requiring illumination correction
- **Edge displacement:** Boundary effects from detection algorithms
- **Random clusters:** Small/faint colonies more sensitive to parameters

**Example:**

.. code-block:: python

    detected = detector.apply(img)
    
    # Visualize displacement from opening
    fig, ax = detected.plot.boundary_displacement(
        operation='opening',
        kernel_sizes=[3, 7],
        reference_size=1,
        cmap='hot'
    )
    fig.suptitle('Boundary Sensitivity to Opening')
    
    # High displacement indicates regions requiring correction

Object Size Distribution
------------------------

.. automethod:: Image.plot.size_distribution

**Purpose:** Comprehensive size analysis with 6-panel visualization showing histogram,
CDF, sensitivity curve, and filtered previews at predetermined thresholds.

**Use Case:** Select minimum/maximum size thresholds to remove dust, debris, and
artifacts while preserving real colonies.

**Output:** 6-panel figure with histogram+KDE, CDF, sensitivity curve, and 3 threshold
previews (5th, 50th, and 95th percentiles by default).

**Interpretation Guide (each panel):**

- **Histogram:** Bimodal distributions suggest distinct populations (colonies vs. debris)
- **CDF:** Curves separating reveal transition from artifact-dominated to colony-dominated sizes
- **Sensitivity:** Plateaus indicate robust threshold regions where small changes don't affect count
- **Previews:** Green (accepted objects), red (rejected) for visual threshold validation

**Domain Context (Agar Plates):**

Typical colony size distributions:

- **50-5000 pixels:** Real colonies at standard plate scanner resolution
- **<50 pixels:** Usually dust, agar texture, or condensation
- **>5000 pixels:** Merged colonies, large mutants, or contamination
- **Bimodal:** Distinct artifact and colony populations
- **Log-normal:** Typical for exponential growth with size variation

**Example:**

.. code-block:: python

    detected = detector.apply(img)

    # Static 6-panel analysis
    fig, axes = detected.plot.size_distribution(
        thresholds=None,  # Auto-select 5th, 50th, 95th percentiles
        log_scale=True
    )
    fig.suptitle('Colony Size Distribution Analysis')

Size Viewer (Interactive)
-------------------------

.. automethod:: Image.plot.size_viewer

**Purpose:** Interactive size analysis with real-time threshold adjustment using
widgets for live preview and parameter tuning.

**Use Case:** Fine-tune size thresholds interactively to balance artifact removal
with colony preservation.

**Output:** Base 6-panel visualization plus ipywidgets slider for threshold selection
with live preview updates.

**Requirements:** ipywidgets package installed and Jupyter notebook environment.

**Interpretation Guide:** Same as static version, but with real-time feedback from
slider adjustments.

**Domain Context (Agar Plates):** Same size distribution patterns as static version,
but allows interactive exploration of threshold effects.

**Example:**

.. code-block:: python

    # Interactive mode (requires ipywidgets + Jupyter)
    fig, axes, widgets = detected.plot.size_distribution_interactive()
    # Use slider to adjust threshold and see immediate effect
    # on preview images. Find value that removes artifacts
    # while preserving all real colonies.


Spatial Size Distribution Map
-----------------------------

.. automethod:: Image.plot.spatial_size_map

**Purpose:** Pseudo-color map showing each object colored by its size relative to a
center value (median, mean, percentile, or absolute). Detects position effects,
illumination gradients, and contamination.

**Use Case:** Identify systematic spatial biases in colony growth or detection:
radial gradients (temperature), row/column patterns (pipetting errors), clusters
(contamination).

**Output:** Pseudo-color image with diverging colormap (blue = smaller, red = larger);
colorbar with annotations; statistics on spatial variation.

**Interpretation Guide (Color Patterns):**

- **Uniform color:** Homogeneous colony sizes (ideal)
- **Radial gradient (center vs. edges):** Temperature gradient, illumination gradient, evaporation effects
- **Row/column pattern:** Systematic pipetting volume gradient, plate tilt, incubator shelf effect
- **Random patches:** Contamination clusters or local growth defects
- **Edge effects:** Detection artifacts from image boundaries

**Mode Selection:**

- **median (default):** Robust to outliers, best for general use
- **mean:** Sensitive to outliers, use with symmetric distributions
- **percentile:** Highlight specific deviations (e.g., value=75 emphasizes smallest 25%)
- **absolute:** Compare to biological expectation from pilot experiments

**Domain Context (High-Throughput Screening):**

Spatial patterns reveal technical and biological effects:

- **Temperature:** Incubator edges cooler → slower growth → smaller colonies at periphery
- **Evaporation:** Plate edges dry faster → osmotic stress → smaller colonies
- **Pipetting:** Volume gradient across multichannel pipette → dose-response pattern
- **Contamination:** Localized clusters of abnormally large colonies

**Returns:** Metadata dictionary with spatial statistics for programmatic analysis.

**Example:**

.. code-block:: python

    detected = detector.apply(img)
    
    # Visualize size relative to median
    fig, ax, metadata = detected.plot.spatial_size_map(
        mode='median',
        robust=True,
        cmap='RdBu_r'
    )
    ax.set_title('Colony Size Deviation from Median')
    
    # Check spatial variation magnitude
    print(f"CV: {metadata['cv']:.2f}")
    print(f"Range: {metadata['value_range']}")

Size-Intensity Correlation
---------------------------

.. automethod:: Image.plot.size_scatter

**Purpose:** Scatter plot correlating object size (x-axis) with mean intensity (y-axis),
with optional color-coding by secondary feature. Distinguishes true colonies from dust,
condensation, merged objects via feature relationships.

**Use Case:** Identify artifact types based on size-intensity relationships and derive
multi-feature filtering rules.

**Output:** Scatter plot with optional marginal histograms; log-log regression line
with confidence band; color-coded by secondary feature.

**Interpretation Guide (Regression Slope in Log-Log Space):**

- **Slope ≈ 1:** Integrated density constant (typical for uniform colonies)
- **Slope ≈ 0:** Intensity independent of size (threshold-dominated detection)
- **Slope < 0:** **Artifacts!** Small bright spots are dust, condensation, or optical defects
- **Slope > 1:** 3D growth effects, merged colonies, or saturated bright objects

**Outlier Patterns:**

- **Small + very bright:** Dust, lens artifacts, condensation → filter by size AND intensity
- **Large + very dim:** Over-segmentation, shadows, plate edges → raise threshold or apply border removal
- **High intensity_std (color):** Merged/heterogeneous objects → use watershed segmentation
- **Low solidity (color):** Irregular shapes (merged, debris, artifacts with holes)
- **High eccentricity (color):** Elongated objects (merged in rows, spreading phenotypes)

**Color-by Options:**

- **intensity_std:** High values indicate heterogeneous/merged objects
- **solidity:** Low values indicate irregular shapes (merged, touching, debris)
- **eccentricity:** High values indicate elongated objects (merged, spreading)

**Domain Context (Colony Quality Control):**

Feature correlations reveal detection issues:

- **Negative slope:** Optical artifacts dominating small size range
- **Positive slope > 1.5:** Merged colonies or 3D growth (use watershed)
- **Outliers above trend:** Bright mutants or contamination
- **Outliers below trend:** Faint colonies near detection threshold
- **Cluster separation:** Distinct artifact and colony populations in feature space

**Example:**

.. code-block:: python

    detected = detector.apply(img)
    
    # Basic size-intensity scatter
    fig, ax, regression = detected.plot.size_scatter(
        color_by='intensity_std',
        show_regression=True,
        show_marginals=True
    )
    ax.set_title('Colony Feature Space')
    
    # Interpret slope
    slope = regression['slope']
    if slope < 0:
        print("Warning: Negative slope suggests optical artifacts")
    elif slope > 1.5:
        print("Warning: High slope suggests merged colonies")
    
    # Identify outliers for manual inspection
    # High intensity_std (red points) = heterogeneous/merged

