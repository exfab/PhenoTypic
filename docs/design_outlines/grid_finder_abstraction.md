# GridFinder Abstraction Design

## Overview

This document describes the refactored `GridFinder` abstraction that enables multiple strategies for defining grids in
the PhenoTypic package.

## Architecture

### Base Class: `GridFinder`

The `GridFinder` abstract base class defines the interface that all grid finding strategies must implement:

#### Abstract Methods (Must Implement)

1. **`_operate(image: Image) -> pd.DataFrame`**
    - Main processing method that returns complete grid information
    - Called by the public `measure()` method
    - **Implementation pattern**: Calculate edges, then call `self._get_grid_info()`

2. **`get_row_edges(image: Image) -> np.ndarray`**
    - Returns array of row edge coordinates
    - Length should be `nrows + 1`

3. **`get_col_edges(image: Image) -> np.ndarray`**
    - Returns array of column edge coordinates
    - Length should be `ncols + 1`

#### Concrete Helper Methods (Available to All Subclasses)

1. **`_get_grid_info(image: Image, row_edges: np.ndarray, col_edges: np.ndarray) -> pd.DataFrame`** ✅ **IMPLEMENTED**
    - Assembles complete grid information from edges
    - Adds row/column numbers, intervals, section indices, and section numbers
    - This is the key method that eliminates code duplication
    - **Usage**: Call this from your `_operate()` method after calculating edges

2. **Helper methods for adding grid metadata:**
    - `_add_row_number_info()`
    - `_add_row_interval_info()`
    - `_add_col_number_info()`
    - `_add_col_interval_info()`
    - `_add_section_interval_info()`
    - `_add_section_number_info()`
    - `_clip_row_edges()` / `_clip_col_edges()`

## Implementations

### 1. OptimalBinsGridFinder ✅ **IMPLEMENTED**

**Purpose:** Automatically finds optimal grid placement by minimizing the error between object centroids and grid bin
midpoints.

**Key Features:**

- Uses optimization to find best row/column padding
- Calculates edges based on object bounding boxes
- Minimizes mean squared error between object and bin midpoints

**Implementation Pattern:**

```python
def _operate(self, image: Image) -> pd.DataFrame:
    # 1. Calculate optimal edges using optimization
    row_edges = self.get_row_edges(image)
    col_edges = self.get_col_edges(image)

    # 2. Use base class helper to assemble grid info
    return super()._get_grid_info(image=image, row_edges=row_edges, col_edges=col_edges)
```

**Status:** Refactored to use the new `_get_grid_info()` helper method, eliminating ~60 lines of duplicated code.

### 2. ManualGridFinder ✅ **IMPLEMENTED**

**Purpose:** Allows users to directly specify grid coordinates without any automatic calculation.

**Key Features:**

- User provides exact row and column edge coordinates
- No optimization or calculation performed
- Complete manual control over grid placement

**Usage Example:**

```python
import numpy as np
from phenotypic.grid import ManualGridFinder

# Define a 3x4 grid with specific coordinates
row_edges = np.array([0, 100, 200, 300])  # 3 rows
col_edges = np.array([0, 80, 160, 240, 320])  # 4 columns

finder = ManualGridFinder(row_edges=row_edges, col_edges=col_edges)
grid_info = finder.measure(image)
```

**Implementation Pattern:**

```python
def __init__(self, row_edges: np.ndarray, col_edges: np.ndarray):
    self._row_edges = np.asarray(row_edges, dtype=int)
    self._col_edges = np.asarray(col_edges, dtype=int)
    self.nrows = len(self._row_edges) - 1
    self.ncols = len(self._col_edges) - 1


def _operate(self, image: Image) -> pd.DataFrame:
    # Simply use predefined edges with base class helper
    return self._get_grid_info(image=image, row_edges=self._row_edges, col_edges=self._col_edges)


def get_row_edges(self, image: Image) -> np.ndarray:
    return self._row_edges.copy()


def get_col_edges(self, image: Image) -> np.ndarray:
    return self._col_edges.copy()
```

**Status:** Fully implemented and available in `phenotypic.grid.ManualGridFinder`.

## Creating New GridFinder Implementations

To create a new grid finding strategy:

### Step 1: Inherit from GridFinder

```python
from phenotypic.ABC_ import GridFinder
import numpy as np
import pandas as pd


class MyCustomGridFinder(GridFinder):
    def __init__(self, nrows: int, ncols: int, **custom_params):
        self.nrows = nrows
        self.ncols = ncols
        # Add your custom parameters
```

### Step 2: Implement Edge Calculation Methods

```python
    def get_row_edges(self, image: Image) -> np.ndarray:
    # Your logic to calculate row edges
    # Must return array of length (nrows + 1)
    row_edges =  # ... your calculation ...
    return row_edges


def get_col_edges(self, image: Image) -> np.ndarray:
    # Your logic to calculate column edges
    # Must return array of length (ncols + 1)
    col_edges =  # ... your calculation ...
    return col_edges
```

### Step 3: Implement _operate Method

```python
    def _operate(self, image: Image) -> pd.DataFrame:
    # Get edges using your methods
    row_edges = self.get_row_edges(image)
    col_edges = self.get_col_edges(image)

    # Use base class helper to assemble grid info
    return super()._get_grid_info(image=image, row_edges=row_edges, col_edges=col_edges)
```

## Example Use Cases for New Implementations

### 1. EqualSpacingGridFinder

Divide image into equal-sized grid cells:

```python
def get_row_edges(self, image: Image) -> np.ndarray:
    return np.linspace(0, image.shape[0], self.nrows + 1, dtype=int)


def get_col_edges(self, image: Image) -> np.ndarray:
    return np.linspace(0, image.shape[1], self.ncols + 1, dtype=int)
```

### 2. MarkerBasedGridFinder

Use fiducial markers or reference points to define grid:

```python
def __init__(self, nrows: int, ncols: int, marker_positions: dict):
    self.nrows = nrows
    self.ncols = ncols
    self.markers = marker_positions


def get_row_edges(self, image: Image) -> np.ndarray:
# Calculate edges based on detected marker positions
# ...
```

### 3. TemplateMatchingGridFinder

Match against a known grid template:

```python
def __init__(self, nrows: int, ncols: int, template: np.ndarray):
    self.nrows = nrows
    self.ncols = ncols
    self.template = template


def get_row_edges(self, image: Image) -> np.ndarray:
# Use template matching to find grid alignment
# ...
```

## Benefits of This Design

1. **Code Reuse:** The `_get_grid_info()` method eliminates ~60 lines of duplicated grid assembly logic
2. **Flexibility:** Easy to create new grid finding strategies by implementing just 3 methods
3. **Consistency:** All grid finders produce the same DataFrame structure
4. **Separation of Concerns:** Edge calculation logic is separate from grid assembly logic
5. **Testability:** Each component can be tested independently

## Migration Guide

For existing code using `OptimalBinsGridFinder`:

- **No changes required** - the public API remains the same
- Internal implementation is cleaner but functionality is identical

For new implementations:

- Follow the pattern in `ManualGridFinder` as a template
- Focus on implementing edge calculation logic
- Let the base class handle grid assembly

## Testing

When implementing a new GridFinder:

1. **Test edge calculation:**
   ```python
   def test_row_edges():
       finder = MyCustomGridFinder(nrows=3, ncols=4)
       edges = finder.get_row_edges(image)
       assert len(edges) == 4  # nrows + 1
       assert np.all(edges[:-1] < edges[1:])  # monotonically increasing
   ```

2. **Test grid assembly:**
   ```python
   def test_grid_info():
       finder = MyCustomGridFinder(nrows=3, ncols=4)
       grid_info = finder.measure(image)
       assert 'row_num' in grid_info.columns
       assert 'col_num' in grid_info.columns
       assert 'section_num' in grid_info.columns
   ```

3. **Test integration:**
   ```python
   def test_with_real_image():
       finder = MyCustomGridFinder(nrows=8, ncols=12)
       grid_info = finder.measure(real_image)
       assert len(grid_info) == len(real_image.objects)
   ```
