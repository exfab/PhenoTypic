# Parallel Measure-Only Marker Publication

## Goal

Make a legacy `--mode measure` SLURM resume start remeasurement promptly while preserving durable per-image success markers.

## Problem

The UCR-029 state predates required success markers. The launcher attempted to migrate all 6,657 HDFs synchronously before submitting work. It also calculated HDF identities twice during selection and script generation. Those serial reads made the job appear stuck before any Slurm worker was submitted.

## Design

Measure-only is a deliberate remeasurement of every discovered HDF. Its ordinary worker already remeasures one HDF and publishes its success marker atomically on successful completion. Therefore a separate marker-migration array would repeat the same work and delay useful computation.

For measure-only resumes only:

- Skip the legacy synchronous marker-migration preflight.
- Do not calculate HDF work IDs during remaining-work selection.
- Do not calculate expected work IDs, input hashes, or pipeline hashes when rendering the measurement array script.
- Submit all discovered HDFs to the normal chunked SLURM array. Each worker calculates its identity and publishes its marker after remeasurement.

Normal full and process resumes retain their exact immutable identity checks and legacy marker-migration behavior.

## Verification

Focused unit tests prove that measure-only selection and script rendering do not hash HDFs before dispatch, that legacy migration is skipped only for this mode, and that normal scripts retain their identity flags. A live UCR-029 run submitted its array in 47 seconds; the first worker remeasured its HDF and exited successfully.
