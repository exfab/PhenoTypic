# The `_operate()` Contract

Every PhenoTypic operation implements a single method: `_operate()`. This
page specifies the full contract that implementations must follow.

## Signature

```python
def _operate(self, image: Image) -> Image:
```

## Rules

1. **Return the image.** Always return the `image` argument, even if you
   modified it in place. The pipeline relies on the return value.

2. **Access data through accessors.** Use `image.detect_mat[:]`,
   `image.objmask[:]`, etc. Do not access internal `_data` attributes
   directly.

3. **Respect the read/write contract.** Each ABC type specifies which
   accessors it may read and write. An enhancer that writes to `objmap`
   is violating its contract.

4. **Store parameters on `self`.** The constructor should assign all
   configuration to instance attributes. The `_operate()` method
   accesses them via `self.param_name`.

5. **Be deterministic.** Given the same image and parameters, the
   operation should produce the same result. Use fixed random seeds
   if randomness is involved.

6. **Preserve GridImage type.** If the input is a `GridImage`, the
   output must also be a `GridImage` with grid state preserved. The
   base class handles this automatically when you return the same
   `image` object.

## Common Patterns

### Read-Transform-Write (Enhancer)

```python
def _operate(self, image):
    mat = image.detect_mat[:]
    transformed = some_function(mat, self.param)
    image.detect_mat[:] = transformed
    return image
```

### Detect-Label-Write (Detector)

```python
def _operate(self, image):
    mat = image.detect_mat[:]
    mask = mat > self.threshold
    labeled = label(mask).astype(np.uint16)
    image.objmask[:] = mask
    image.objmap[:] = labeled
    return image
```

### Read-Measure-Return (Measurement)

```python
def _operate(self, image):
    objmap = image.objmap[:]
    rows = []
    for prop in regionprops(objmap):
        rows.append({"Object_Label": prop.label, "MyMetric": ...})
    return pd.DataFrame(rows)
```
