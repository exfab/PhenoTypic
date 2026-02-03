from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple, TYPE_CHECKING, Iterator
from itertools import product
from pathlib import Path

from abc import ABC, abstractmethod
from phenotypic.abc_ import ImageOperation
from phenotypic import ImagePipeline

if TYPE_CHECKING:
    from phenotypic.tools_.typing_ import GridSearchSaveData, GridSearchConfig
    from phenotypic import Image
    from phenotypic._core._image_parts.accessor_abstracts import ImageAccessorBase


class PipelineGridSearchBase(ABC):
    def __init__(self,
                 pipe_cfgs: Dict[str, GridSearchConfig],
                 output_dir: Path | str | None = None,
                 data2save: GridSearchSaveData = None,
                 ):
        """Initialize base grid search infrastructure and validate configurations.

        This is the abstract base class for grid search implementations. It provides
        configuration validation, parameter sweep generation, and result directory
        management. Subclasses (PipeGridSearchJoblib, PipeGridSearchSubmitit) implement
        concrete execution strategies.

        The class generates all combinations of provided parameters, creates output
        directory structures, and prepares pipelines for execution. Actual pipeline
        processing is delegated to subclasses via `process()` or `submitit()` methods.

        Args:
            pipe_cfgs: Dictionary mapping pipeline configuration names to operation lists.
                Each value is a list of ``(operation, params_dict)`` tuples where:

                - ``operation`` is an ImageOperation instance
                - ``params_dict`` maps parameter names to lists of values to test

                Example:
                    ``{"Pipeline1": [(GaussianBlur(), {"sigma": [1.0, 2.0, 3.0]}), ...]}``

                The grid search will generate pipelines for all combinations of parameter
                values. Parameters must correspond to operation attributes (constructor
                arguments or instance variables).

            output_dir: Directory where results will be saved. If None, uses current
                working directory. Must exist and be writable. The directory structure
                is created automatically:

                - ``output_dir/data/PipelineName/PipelineName_0/`` - First parameter combo
                - ``output_dir/data/PipelineName/PipelineName_1/`` - Second parameter combo
                - etc.

            data2save: Set of image layers to persist to disk after pipeline execution.
                Valid options: ``{"rgb", "gray", "detect_mat", "objmask", "objmap", "map2rgb"}``.

                - ``rgb``: Original color image (if available) as TIFF
                - ``gray``: Grayscale luminance as TIFF
                - ``detect_mat``: Detection matrix as TIFF
                - ``objmask``: Binary detection mask as PNG
                - ``objmap``: Labeled object map as PNG
                - ``map2rgb``: Object map rendered as RGB overlay as PNG

                Defaults to all layers. Use subset to minimize disk usage for large
                batch processing. Each pipeline also saves its processed Image as pickle
                for later re-analysis.

        Raises:
            ValueError: If ``output_dir`` doesn't exist, is not a directory, or
                pipeline configurations are invalid (malformed tuples, invalid
                parameter names, etc.).

        Output Structure:

            After initialization, the following directory structure is created:

            .. code-block:: text

                output_dir/
                ├── data/
                │   ├── PipelineName1/
                │   │   ├── PipelineName1_0/           # Parameter combo 0
                │   │   │   ├── PipelineName1_0.json   # Serialized pipeline
                │   │   │   ├── rgb.tiff               # (if in data2save)
                │   │   │   ├── objmask.png            # (if in data2save)
                │   │   │   └── PipelineName1_0_Image.pkl
                │   │   ├── PipelineName1_1/           # Parameter combo 1
                │   │   │   └── ...
                │   │   └── ...
                │   └── PipelineName2/
                │       └── ...
                ├── OriginalImage.pkl                  # Preserved after processing
                ├── OriginalRGB.tif                    # (only if image had RGB)
                └── OriginalGray.tif

        Implementation Notes:

            - Configuration validation occurs during ``__init__``, catching errors early
            - Each pipeline configuration gets its own subdirectory in ``data/``
            - Parameter combinations are generated using ``itertools.product``
            - Pipelines are serialized as JSON, enabling independent re-execution
            - The base class is abstract; use ``PipeGridSearch`` (joblib) or
              ``PipeGridSearchSubmitit`` (SLURM) for actual processing

        Example:

            .. code-block:: python

                from phenotypic.util._pipeline_grid_search import PipeGridSearch
                from phenotypic.enhance import GaussianBlur, CLAHE
                from phenotypic.detect import OtsuDetector

                # Define parameter grid
                pipe_cfgs = {
                    "DetectionPipeline": [
                        (GaussianBlur(), {"sigma": [1.0, 2.0, 3.0]}),
                        (CLAHE(), {"clip_limit": [1.5, 2.0]}),
                        (OtsuDetector(), {"ignore_zeros": [True, False]}),
                    ]
                }

                # Initialize (validates configs, creates directory structure)
                gs = PipeGridSearch(
                    pipe_cfgs=pipe_cfgs,
                    output_dir="/data/results",
                    data2save={"detect_mat", "objmask"}  # Save 2 layers to minimize disk
                )

                # Then use subclass-specific method
                image = GridImage.imread("plate.jpg", nrows=8, ncols=12)
                gs.process(image, njobs=-1)  # For PipeGridSearch
                # OR
                # gs.submitit(image, slurm_config={...})  # For PipeGridSearchSubmitit
        """
        for cfg in pipe_cfgs.values():
            self._validate_pipe_cfgs(cfg)

        """pipe_cfgs example:
            pipe_cfgs = {
                "PipelineName": [
                    (GaussianBlur(), {"sigma": [1, 2, 3]}),
                    (OtsuDetector(), {"ignore_zeros": [True, False]}),
                ]
            }
        """
        self.pipe_cfgs: Dict[str, GridSearchConfig] = pipe_cfgs

        data2save = {"rgb", "gray", "detect_mat", "objmask", "objmap", "map2rgb"} \
            if data2save is None \
            else set(data2save)
        self._validate_layers(data2save)
        self.data2save: GridSearchSaveData = data2save

        self.output_dir: Path = Path(output_dir) \
            if output_dir \
            else Path.cwd()
        if (not self.output_dir.exists()) | (self.output_dir.is_dir() is False):
            raise ValueError("Output directory does not exist or is not a directory")
        self.data_dir.mkdir(exist_ok=True)

    @property
    def data_dir(self) -> Path:
        return self.output_dir / "data"

    @property
    def _image_pkl_path(self) -> Path:
        return self.output_dir / "OriginalImage.pkl"

    @property
    def _original_rgb_path(self) -> Path:
        return self.output_dir / "OriginalRGB.tif"

    @property
    def _original_gray_path(self) -> Path:
        return self.output_dir / "OriginalGray.tif"

    def _prep_image(self, image: Image) -> None:
        """Preserve original image and create output directory structure.

        Saves the original image as a pickle file (for per-pipeline reloading during
        parallel processing) and as TIFF files for manual inspection. Creates all
        output subdirectories based on pipeline configurations.

        This method is called automatically by `process()` or `submitit()` before
        any pipeline execution. For parallel execution, each worker process reloads
        the pickled image to ensure memory isolation.

        Args:
            image: Image or GridImage to preserve. RGB is only saved if the image
                contains RGB data; grayscale is always saved.

        Side Effects:
            - Saves ``output_dir/OriginalImage.pkl`` (used by worker processes)
            - Saves ``output_dir/OriginalRGB.tif`` (if image has RGB, for inspection)
            - Saves ``output_dir/OriginalGray.tif`` (always saved, for inspection)
            - Creates directory structure in ``output_dir/data/`` for all pipeline
              configurations and parameter combinations
        """
        image.save2pickle(self._image_pkl_path)

        if image.rgb.isempty() is False:
            image.rgb.imsave(self._original_rgb_path)
        image.gray.imsave(self._original_gray_path)
        self._make_output_folders()

    def _make_output_folders(self) -> None:
        """Create hierarchical output directories and serialize pipelines.

        For each pipeline configuration, generates all parameter combinations and
        creates output directories. Each directory contains a serialized JSON
        representation of its pipeline configuration for later re-execution or analysis.

        Directory structure created:
        ``output_dir/data/ConfigName/ConfigName_0/ConfigName_0.json``
        ``output_dir/data/ConfigName/ConfigName_1/ConfigName_1.json``
        etc.

        Each JSON file is fully self-contained and can be loaded independently:

        .. code-block:: python

            pipe = ImagePipeline.from_json(json_data="path/to/PipelineName_0.json")
            result = pipe.apply(image)
        """
        manifest = {}
        for pipe_config_name in self.pipe_cfgs.keys():
            manifest[pipe_config_name] = {}
            pipe_config_path = self.data_dir / pipe_config_name
            pipe_config_path.mkdir(exist_ok=True)
            pipe_param_sweep = self._get_param_sweep(
                    name=pipe_config_name,
                    pipe_cfg=self.pipe_cfgs[pipe_config_name],
            )
            for pipe in pipe_param_sweep:
                manifest[pipe_config_name][pipe.name] = pipe.to_json_str()
                pipe_subdir = pipe_config_path / pipe.name
                pipe_subdir.mkdir(exist_ok=True)
                pipe.to_json(pipe_subdir / f"{pipe.name}.json")

        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @staticmethod
    def _get_param_sweep(
            name: str,
            pipe_cfg: GridSearchConfig) -> Iterator[ImagePipeline]:
        """
        Generates an iterable of `ImagePipeline` objects, each configured with a unique
        combination of parameters derived from the grid search configuration.

        This static method is used to perform exhaustive parameter sweeps over the
        defined parameter space for image processing pipelines. Parameters are applied
        to operations defined in the pipeline configuration, resulting in pipelines
        tailored to specific parameter sets. The primary application is in processing
        microbial colonies on solid media agar, wherein variations in parameters can
        alter results such as colony detection accuracy, separation of overlapping
        colonies, or noise suppression.

        Args:
            name (str):
                A base name for each generated pipeline. This name will be suffixed with
                a unique identifier corresponding to each parameter combination. Adjusting
                this value provides context to the generated pipelines, which can be
                useful when analyzing or benchmarking different configurations.
            pipe_cfg (GridSearchConfig):
                Contains the operations and their corresponding parameter ranges for
                sweeping. Changing this configuration directly influences the number of
                pipelines generated, as well as the variety of parameter configurations
                applied. Modifications to these parameters could impact the pipeline's
                behavior when processing images, affecting outputs like identifying
                colony boundaries, resolving complex overlaps, and suppressing lighting
                artifacts.

        Yields:
            Iterator[ImagePipeline]:
                An iterator of `ImagePipeline` objects, each instantiated with a distinct
                parameter combination. Each pipeline is constructed with unique operational
                parameters, enabling systematic analysis of image processing configurations.
        """
        ops, params = PipelineGridSearchBase._unpack_ops_tuples(pipe_cfg)
        param_sweep = PipelineGridSearchBase._generate_param_combinations(params)
        for idx, param_set in enumerate(param_sweep):
            yield ImagePipeline(
                    ops=[op.__class__(**param_set[op_idx])
                         for op_idx, op in enumerate(ops)],
                    name=f"{name}_{idx}"
            )

    @staticmethod
    def _validate_pipe_cfgs(
            pipe_cfgs: GridSearchConfig,
    ) -> None:
        """Validate pipeline configuration structure and parameter names.

        Performs comprehensive validation of the pipeline configuration:

        1. **Structure validation:** Ensures pipe_cfgs is a non-empty list of tuples
        2. **Format validation:** Each tuple is (operation, params_dict)
        3. **Semantic validation:** Parameter names exist as operation attributes

        This method is called automatically during ``__init__`` for each pipeline
        configuration. Errors are caught early to provide clear messages before
        resource-intensive processing begins.

        Args:
            pipe_cfgs: List of ``(operation, params_dict)`` tuples. Example:

                .. code-block:: python

                    [
                        (GaussianBlur(), {"sigma": [1.0, 2.0, 3.0]}),
                        (OtsuDetector(), {"ignore_zeros": [True, False]}),
                    ]

                - Each operation is an ImageOperation instance
                - Each params_dict maps parameter names to lists of values
                - Parameter names must exist as operation attributes (e.g., ``GaussianBlur.sigma``)

        Raises:
            ValueError:
                - If ``pipe_cfgs`` is not a list
                - If list is empty
                - If any element is not a 2-tuple
                - If second element of tuple is not a dict
                - If any parameter name doesn't exist on the operation
                  (provides helpful list of available attributes)
        """
        # Validate pipe_cfgs format
        if not isinstance(pipe_cfgs, list):
            raise ValueError(f"pipe_cfgs must be a list, got {type(pipe_cfgs)}")

        if not pipe_cfgs:
            raise ValueError("pipe_cfgs cannot be empty")

        for idx, item in enumerate(pipe_cfgs):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError(
                        f"pipe_cfgs[{idx}]: Each element must be a tuple (operation, params_dict), "
                        f"got {type(item)}"
                )

            op, params = item
            if not isinstance(params, dict):
                raise ValueError(
                        f"pipe_cfgs[{idx}]: Second element of tuple must be a dict, "
                        f"got {type(params)}"
                )

        # Unpack tuples
        operations, parameters = PipelineGridSearchBase._unpack_ops_tuples(pipe_cfgs)

        # Verify parameter names exist as operation attributes
        for op_idx, (op, params) in enumerate(zip(operations, parameters)):
            for param_name in params.keys():
                if not hasattr(op, param_name):
                    raise ValueError(
                            f"Operation {op_idx} ({op.__class__.__name__}) has no "
                            f"attribute '{param_name}'. Available attributes: "
                            f"{[a for a in dir(op) if not a.startswith('_')]}"
                    )

    @staticmethod
    def _validate_layers(data2save: List[str]) -> None:
        """Validate requested image data layers for output saving.

        Checks that all requested layers exist and can be persisted. Called automatically
        during ``__init__`` if ``data2save`` is provided.

        Args:
            data2save: Set of layer names to save. Must be subset of:

                - ``"rgb"``: Original color image (TIFF)
                - ``"gray"``: Grayscale luminance (TIFF)
                - ``"detect_mat"``: Detection matrix for detection (TIFF)
                - ``"objmask"``: Binary detection mask (PNG)
                - ``"objmap"``: Labeled object map (PNG)
                - ``"map2rgb"``: Object map as RGB overlay (PNG)

        Raises:
            ValueError: If any invalid layer names are provided. Error message lists
                valid options.

        Example:

            .. code-block:: python

                # Valid - saves only detection results
                gs = PipeGridSearch(..., data2save={"objmask", "objmap"})

                # Invalid - "depth" is not a valid layer
                gs = PipeGridSearch(..., data2save={"objmask", "depth"})
                # ValueError: Invalid DataAccessors: {'depth'}. Must be subset of {...}
        """
        # Validate GridSearchSaveData
        valid_layers = {"rgb", "gray", "detect_mat", "objmask", "objmap", "map2rgb"}
        invalid = set(data2save) - valid_layers
        if invalid:
            raise ValueError(
                    f"Invalid DataAccessors: {invalid}. Must be subset of {valid_layers}"
            )

    @staticmethod
    def _unpack_ops_tuples(
            ops: List[Tuple[ImageOperation, Dict[str, List[Any]]]],
    ) -> Tuple[List[ImageOperation], List[Dict[str, List[Any]]]]:
        """Unpack list of (operation, params) tuples into separate lists.

        Args:
            ops: List of (operation, params_dict) tuples

        Returns:
            Tuple of (operations_list, parameters_list)
        """
        operations = []
        parameters = []

        for op, params in ops:
            operations.append(op)
            parameters.append(params)

        return operations, parameters

    @staticmethod
    def _generate_param_combinations(
            parameters: List[Dict[str, List[Any]]],
    ) -> List[Tuple[Dict[str, Any], ...]]:
        """Generate all parameter combinations using itertools.product.

        Args:
            parameters: List of dicts mapping param names to lists of values

        Returns:
            List of tuples, each tuple contains one dict per operation
        """
        param_combinations = []
        for params_dict in parameters:
            if not params_dict:  # Handle empty dict (no params to vary)
                param_combinations.append([{}])
            else:
                keys = list(params_dict.keys())
                values = list(params_dict.values())
                combos = [dict(zip(keys, vals)) for vals in product(*values)]
                param_combinations.append(combos)

        # Generate all pipeline combinations
        return list(product(*param_combinations))

    @staticmethod
    def _process_single_pipe_dir(pipe_dir: Path,
                                 image_pkl_path: Path,
                                 data2save: GridSearchSaveData):
        """Process a single pipeline configuration and save results.

        This method is called by parallel workers (joblib or submitit) to execute
        a single parameter combination's pipeline. Each worker:

        1. Loads the original image from pickle (shared read-only reference)
        2. Loads the pipeline configuration from JSON
        3. Applies the pipeline to the image (in-place for memory efficiency)
        4. Saves requested data layers to disk

        This is a **static method** to ensure workers have no state dependencies and
        can run independently in separate processes.

        Args:
            pipe_dir: Output directory for this pipeline configuration.
                Must contain exactly one ``.json`` file (the serialized pipeline).
                Results are saved to this directory.
            image_pkl_path: Path to the pickled original image. Used by all workers
                to load a fresh copy of the image.
            data2save: Set of layer names to persist (e.g., ``{"objmask", "objmap"}``)

        Raises:
            ValueError: If image pickle doesn't exist, pipe_dir is invalid, or
                multiple JSON files are found in pipe_dir.

        Side Effects:
            - Loads pickled image from ``image_pkl_path``
            - Loads JSON pipeline from ``pipe_dir/*.json``
            - Modifies the image in-place (applies pipeline)
            - Saves result image and requested layers to ``pipe_dir/``

        Note:
            Designed for parallel execution. Workers should not share state or
            modify shared data structures. Each worker gets its own image copy
            to ensure memory isolation.

        Examples:
            This method is called internally by workers in `process()` and `submitit()`.
            For manual re-execution of a single pipeline:

            .. code-block:: python

                # Load result image and pipeline from a completed grid search
                result_dir = Path("/results/data/Pipeline1/Pipeline1_5")

                image = Image.load_pickle(Path("/results/OriginalImage.pkl"))
                pipe = ImagePipeline.from_json(result_dir / "Pipeline1_5.json")
                pipe.apply(image, inplace=True)
                objmask = image.objmask[:]  # Access results
        """
        import phenotypic as pht

        if not image_pkl_path.exists():
            raise ValueError(f"{image_pkl_path} does not exist")

        if not isinstance(pipe_dir, Path) and not pipe_dir.is_dir():
            raise ValueError("pipe_dir must be a Path object "
                             "and existing directory")

        # load in image
        image = pht.Image.load_pickle(image_pkl_path)

        # load in pipeline
        pipe_json = list(pipe_dir.glob("*.json"))
        if len(pipe_json) != 1: raise ValueError("multiple ImagePipeline json files in "
                                                 "individual pipeline config directory")
        pipe_json = pipe_json[0]
        pipe = pht.ImagePipeline.from_json(json_data=pipe_json)

        # apply pipeline to image in place to save mem
        pipe.apply(image=image, inplace=True)
        PipelineGridSearchBase._save_data(image=image,
                                          savedir=pipe_dir,
                                          data2save=data2save)

    @staticmethod
    def _save_data(image: Image,
                   savedir: Path,
                   data2save: GridSearchSaveData):
        # Save the new image as a pkl
        image.save2pickle(savedir / f"{savedir.stem}_Image.pkl")

        # save outputs based on data2save
        for data_name in data2save:

            extra_params = {}
            # set the suffix
            if data_name in {"objmap", "objmask", "map2rgb"}:
                curr_suffix = "png"
            elif data_name in {"rgb", "gray", "detect_mat"}:
                curr_suffix = "tiff"
            else:
                raise ValueError(f"Unknown data2save value: {data_name}")

            # Switches to objmap and sets the rgb params
            if data_name == "map2rgb":
                extra_params["use_label2rgb"] = True
                data_name = "objmap"

            # get the accessor
            acc: ImageAccessorBase = getattr(image, data_name)
            acc.imsave(savedir / f"{data_name}.{curr_suffix}", **extra_params)
