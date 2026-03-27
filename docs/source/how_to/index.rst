How-To Guides
=============

Task-oriented recipes that solve a specific problem. Each guide is standalone
-- you can jump directly to the one you need.

.. toctree::
   :maxdepth: 1
   :caption: Correction & Preprocessing

   notebooks/correct_color_cast
   notebooks/correct_grid_rotation
   notebooks/enhance_low_contrast
   notebooks/crop_and_pad
   notebooks/denoise_low_light

.. toctree::
   :maxdepth: 1
   :caption: Detection & Refinement

   notebooks/choose_detection_algorithm
   notebooks/refine_noisy_boundaries
   notebooks/manual_grid_detection
   notebooks/combine_detectors
   notebooks/merge_fragmented_detections

.. toctree::
   :maxdepth: 1
   :caption: Measurement & Analysis

   notebooks/measure_colony_size_intensity
   notebooks/growth_curves
   notebooks/correct_edge_effects
   notebooks/fit_logistic_growth
   notebooks/assess_image_quality

.. toctree::
   :maxdepth: 1
   :caption: CLI & Infrastructure

   pages/cli_batch_processing
   pages/slurm_pipelines
   pages/parameter_sweeps
   pages/serialize_pipelines
   pages/hdf5_storage
   pages/generate_reports
