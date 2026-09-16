# Real-GPU smoke run: nested Sam2 staged vs single pass

The question: on real plates with the real deployment pipeline
(`F1gfd5.json.pht-pipe`), where `Sam2` sits nested inside a `CompositeDetector`,
does the staged SLURM engine produce **the same objmaps and measurements** as
running the pipeline in one pass?

Both arms run from ONE frozen checkout (`make_gate_tree.sh <sha> gpusmoke`,
then the `gpu` extra synced into it), so the code under test is identical and
any difference is the staging itself.

| file | role |
|---|---|
| `make_inputs.sh` | symlink a small subset of the Linzer test set, one folder per dataset |
| `submit_staged.sbatch` | arm A: `python -m phenotypic --slurm ... --gpu-slurm ...`, the production path |
| `run_single_pass.sbatch` | arm B: `pipeline.apply_and_measure` per image inside one GPU job |
| `single_pass_reference.py` | arm B's worker: writes `<stem>.objmap.npy` and `<stem>.measurements.parquet` |
| `compare_arms.py` | per image: objmap equality, then measurement columns by max abs difference |

Arm B deliberately does not use the CLI: with this branch, a local CLI run of a
nested-GPU pipeline is itself staged, so it would not be an independent
single-pass reference.

GPU inference is not guaranteed bit-identical across batch composition, so the
comparison reports the size of any difference rather than asserting zero; read
the numbers, not only the exit code.
