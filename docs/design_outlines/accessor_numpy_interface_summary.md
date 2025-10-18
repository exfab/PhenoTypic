# Numpy Array Interface Implementation for Accessors

## Summary

Added `__array__` dunder method support to all accessor classes in PhenoTypic, enabling numpy functions to work directly with accessor objects.

## Implementation

### Base Class Change

**File:** `src/phenotypic/core/_image_parts/accessor_abstracts/_image_accessor_base.py`

Added `__array__()` method to `ImageAccessorBase` class. This method:
- Returns the underlying array via `_subject_arr` property
- Supports optional `dtype` parameter for type conversion
- Supports optional `copy` parameter for NumPy 2.0+ compatibility
- Automatically propagates to all subclasses

### Affected Accessor Classes

All accessors now support the numpy array interface:

#### Regular Accessors (via `ImageAccessorBase`)
- **ImageArray** (`_array_accessor.py`) - RGB/multichannel images
- **ImageMatrix** (`_matrix_accessor.py`) - Grayscale representation
- **ImageEnhancedMatrix** (`_enh_matrix_accessor.py`) - Enhanced matrix
- **HsvAccessor** (`_hsv_accessor.py`) - HSV color space
- **GridAccessor** (`_grid_accessor.py`) - Grid-based data

#### Color Space Accessors (via `ColorSpaceAccessor` → `ImageAccessorBase`)
- **XyzAccessor** (`_xyz_accessor.py`) - CIE XYZ color space
- **XyzD65Accessor** (`_xyz_d65_accessor.py`) - XYZ under D65 illuminant
- **CieLabAccessor** (`_cielab_accessor.py`) - CIE L*a*b* color space
- **xyChromaticityAccessor** (`_chromaticity_xy_accessor.py`) - Chromaticity coordinates

#### Already Had `__array__` (retained existing implementations)
- **ObjectMap** (`_objmap_accessor.py`) - Labeled object map
- **ObjectMask** (`_objmask_accessor.py`) - Binary object mask

### Not Applicable
- **MeasurementAccessor** - Not an array-based accessor
- **MetadataAccessor** - Not an array-based accessor
- **ObjectsAccessor** - Returns Image objects, not arrays

## Usage Examples

### Basic Numpy Functions

```python
import numpy as np
from phenotypic import Image

# Load an image
img = Image("path/to/image.png")

# Apply numpy functions directly on accessors
mean_value = np.mean(img.matrix)
max_value = np.max(img.array)
std_dev = np.std(img.hsv)
total_pixels = np.sum(img.objmask)

# Works with color space accessors too
xyz_mean = np.mean(img.CieXYZ)
lab_std = np.std(img.CieLab)
```

### Advanced Operations

```python
# Reshape operations
flat = np.reshape(img.matrix, -1)

# Clipping
clipped = np.clip(img.matrix, 0.2, 0.8)

# Percentiles
median = np.percentile(img.matrix, 50)

# Concatenation
combined = np.concatenate([img.matrix, img.enh_matrix], axis=0)

# Type conversion
float_array = np.array(img.array, dtype=np.float32)
```

### Note on Comparison Operations

Accessor objects don't support direct comparison operators. Convert to array first:

```python
# ❌ This won't work:
# result = np.where(img.matrix > 0.5, 1, 0)

# ✅ Do this instead:
matrix_arr = np.array(img.matrix)
result = np.where(matrix_arr > 0.5, 1, 0)
```

## Testing

Created comprehensive test suite: `tests/test_accessor_numpy_interface.py`

**Test Coverage:**
- 31 tests covering all major accessor types
- Basic numpy functions (sum, mean, max, min, std)
- Advanced operations (reshape, clip, concatenate, percentile)
- Type conversion and copy parameter handling
- Consistency checks between `__array__` and direct access

**Run tests:**
```bash
uv run pytest tests/test_accessor_numpy_interface.py -v
```

## Benefits

1. **Cleaner Code:** Use numpy functions directly without explicit slicing
2. **Consistency:** All accessors work uniformly with numpy
3. **Compatibility:** Follows numpy array interface protocol
4. **Future-Proof:** Supports NumPy 2.0+ copy parameter
5. **Type Safety:** Optional dtype conversion for specific use cases

## Backward Compatibility

✅ Fully backward compatible - existing code continues to work:
- `accessor[:]` still works
- `.copy()` methods unchanged
- All existing functionality preserved
