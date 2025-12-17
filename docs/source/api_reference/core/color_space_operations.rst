Color Space Operations
======================

.. currentmodule:: phenotypic

.. autoproperty:: Image.color

The ``color`` property provides unified access to multiple color space representations
through the ColorAccessor interface. This groups together device-dependent (HSV) and
CIE standard color spaces (XYZ, XYZ-D65, L*a*b*, xy chromaticity).

All color space conversions are computed on-demand and cached to avoid redundant
computations. The parent Image configuration (illuminant, observer, gamma) is used
consistently across all transformations.

CIE XYZ Color Space
-------------------

.. autoproperty:: Image.color.XYZ

Access the CIE XYZ color space representation computed under the image's configured
illuminant. XYZ is a device-independent color space that forms the basis for many
other color space transformations.

**Use cases:**

- Color science calculations
- Intermediate representation for color space conversions
- Device-independent color analysis

**Shape:** ``(height, width, 3)`` where channels are X, Y (luminance), and Z

**Example:**

.. code-block:: python

    img = Image.imread('sample.jpg')
    
    # Get full XYZ array
    xyz_array = img.color.XYZ[:]
    
    # Extract Y channel (luminance)
    luminance = img.color.XYZ[..., 1]
    
    # Slice specific region
    roi_xyz = img.color.XYZ[100:200, 100:200, :]

CIE XYZ under D65 Illuminant
-----------------------------

.. autoproperty:: Image.color.XYZ_D65

Access XYZ representation specifically under D65 (standard daylight) illuminant viewing
conditions. If the image uses a different illuminant (e.g., D50), chromatic adaptation
is automatically applied.

D65 is the CIE standard daylight illuminant with a color temperature of approximately
6504 K, commonly used in photography and display technology.

**Use cases:**

- Standardized color comparison across imaging systems
- Display calibration and color management
- Consistent color analysis regardless of original illuminant

**Shape:** ``(height, width, 3)`` where channels are X, Y, Z under D65 conditions

**Example:**

.. code-block:: python

    img = Image.imread('photo.jpg')
    
    # Get D65-adapted XYZ
    xyz_d65 = img.color.XYZ_D65[:]
    
    # Use for standardized comparisons
    luminance_d65 = img.color.XYZ_D65[..., 1]

CIE L*a*b* Color Space
----------------------

.. autoproperty:: Image.color.Lab

Access the perceptually uniform CIE L*a*b* color space. Lab is designed to approximate
human visual perception, making it ideal for color analysis, color correction, and
calculating perceptually meaningful color differences.

**Channels:**

- **L\*** (lightness): 0 (black) to 100 (white), perceptual brightness
- **a\*** (green-red opponent): Negative = green, positive = red
- **b\*** (blue-yellow opponent): Negative = blue, positive = yellow

Because Lab is perceptually uniform, Euclidean distances in Lab space correspond to
perceptual color differences (ΔE) as seen by human observers.

**Use cases:**

- Color difference calculations (ΔE)
- Perceptually meaningful color analysis
- Color-based segmentation for biological samples
- Lightness adjustment independent of hue

**Shape:** ``(height, width, 3)`` where channels are L*, a*, b*

**Example:**

.. code-block:: python

    import numpy as np
    img = Image.imread('plate.jpg')
    
    # Access Lab color space
    lab = img.color.Lab[:]
    L = img.color.Lab[..., 0]  # Lightness (0-100)
    a = img.color.Lab[..., 1]  # Green (-) to Red (+)
    b = img.color.Lab[..., 2]  # Blue (-) to Yellow (+)
    
    # Calculate color difference from reference
    reference_lab = np.array([50, 0, 0])  # Mid-gray
    delta_e = np.sqrt(
        (lab[..., 0] - reference_lab[0])**2 +
        (lab[..., 1] - reference_lab[1])**2 +
        (lab[..., 2] - reference_lab[2])**2
    )
    
    # Find similar colors (ΔE < 5)
    similar_mask = delta_e < 5

CIE xy Chromaticity
-------------------

.. autoproperty:: Image.color.xy

Access 2D chromaticity coordinates derived from CIE XYZ. Chromaticity expresses color
independently of luminance, isolating hue and saturation information.

The xy coordinates are computed as: ``x = X / (X + Y + Z)``, ``y = Y / (X + Y + Z)``

This normalized representation is device-independent and widely used for visualizing
color spaces on the CIE 1931 chromaticity diagram.

**Use cases:**

- Color analysis without brightness variation
- Gamut visualization on chromaticity diagram
- Hue/saturation analysis independent of luminance
- Color science research and calibration

**Shape:** ``(height, width, 2)`` where channels are x and y chromaticity (range [0, 1])

**Example:**

.. code-block:: python

    import matplotlib.pyplot as plt
    img = Image.imread('sample.jpg')
    
    # Get chromaticity coordinates
    xy_coords = img.color.xy[:]
    x = img.color.xy[..., 0]
    y = img.color.xy[..., 1]
    
    # Plot on CIE 1931 chromaticity diagram
    plt.scatter(x.flatten(), y.flatten(), 
                c=img.rgb[:].reshape(-1, 3)/255, s=1)
    plt.xlabel('x chromaticity')
    plt.ylabel('y chromaticity')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

HSV Color Space
---------------

.. autoproperty:: Image.color.hsv

Access the HSV (Hue, Saturation, Value) color space representation. HSV is device-dependent
but intuitive for human color selection and manipulation. Particularly useful for
color-based filtering and hue-specific analysis.

**Channels (all normalized to [0, 1]):**

- **H (hue)**: Color type, 0 to 1 (corresponds to 0° to 360°)
- **S (saturation)**: Color intensity/purity, 0 (grayscale) to 1 (pure color)
- **V (value)**: Brightness/luminosity, 0 (black) to 1 (brightest)

HSV is computed from RGB and is device-dependent (unlike CIE color spaces), but more
intuitive for operations like selecting all red pixels or adjusting hue.

.. note::
   HSV is only available for RGB images. Attempting to access this property on
   grayscale-only images will raise an AttributeError.

**Use cases:**

- Color-based filtering (e.g., select all red colonies)
- Hue selection for specific phenotypes
- Saturation-based object discrimination
- Intuitive color manipulation

**Shape:** ``(height, width, 3)`` where channels are H, S, V (each in [0, 1])

**Example:**

.. code-block:: python

    img = Image.imread('colored_colonies.jpg')
    
    # Access HSV components
    hsv = img.color.hsv[:]
    hue = img.color.hsv[..., 0]  # 0 to 1
    saturation = img.color.hsv[..., 1]
    brightness = img.color.hsv[..., 2]
    
    # Convert hue to degrees
    hue_degrees = hue * 360
    
    # Extract red pixels (hue near 0° or 360°)
    red_mask = (hue_degrees < 30) | (hue_degrees > 330)
    
    # Extract highly saturated colors
    saturated_mask = saturation > 0.5
    
    # Combine criteria
    red_saturated = red_mask & saturated_mask






