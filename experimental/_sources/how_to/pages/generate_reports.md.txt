# Generate HTML Processing Reports

Create visual reports summarizing pipeline results for sharing with
collaborators.

## Using the CLI

The CLI automatically generates overlay images and measurement summaries
in the output directory:

```bash
python -m phenotypic --pipeline pipeline.json --input /plates/ -o /output/
```

**Output structure:**

```
output/
├── overlays/          # Detection overlay images
├── measurements.csv   # Combined measurements for all plates
├── summary.html       # Processing summary report
└── checkpoints/       # Resume state
```

## Programmatic Report Generation

For custom reports, combine PhenoTypic's visualization methods with your
preferred reporting tool:

```python
import phenotypic as pht

image = pht.Image.imread("plate.png")
pipeline = pht.ImagePipeline.from_json("pipeline.json")
result = pipeline.apply(image)

# Save overlay
fig, ax = result.show(overlay=True, show_labels=True)
fig.savefig("overlay.png", dpi=150, bbox_inches="tight")

# Save measurements
df = pipeline.apply_and_measure(image)
df.to_csv("measurements.csv")
```
