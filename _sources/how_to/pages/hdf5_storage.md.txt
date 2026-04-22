# Store Results in HDF5

Save processed images and their intermediate states to HDF5 for efficient
storage and retrieval.

## Save a Processed Image

```python
import phenotypic as pht

image = pht.Image.imread("plate.png")
# ... process image ...
image.save2hdf5("processed_plate.h5")
```

## Save Pipeline Intermediates

Capture the image state after each pipeline operation:

```python
pipeline = pht.ImagePipeline(ops=[...])
result = pipeline.apply_with_intermediates(
    image,
    output_dir="intermediates/"
)
```

Each intermediate is saved as a separate HDF5 file in `output_dir/`,
named after the operation that produced it.

## Why HDF5

- **Compact:** Compressed binary format, much smaller than TIFF/PNG for
  processed images
- **Fast I/O:** Random access to any data component without loading the
  entire file
- **Self-describing:** Metadata, image arrays, masks, and object maps
  stored together in one file
