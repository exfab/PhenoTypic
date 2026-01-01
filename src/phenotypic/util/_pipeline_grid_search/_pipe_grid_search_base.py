from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Literal, Tuple, TYPE_CHECKING, Iterator

from phenotypic import ImagePipeline

if TYPE_CHECKING:
    from phenotypic.tools.typing_ import DataAccessors, GridSearchConfig

from pathlib import Path
from abc import ABC, abstractmethod

from phenotypic.abc_ import ImageOperation


class PipelineGridSearchBase(ABC):
    def __init__(self,
                 pipe_cfgs: Dict[str, GridSearchConfig],
                 output_dir: Path | str | None = None,
                 data_layers: DataAccessors = None,
                 ):
        """
        Execute parameter grid search with parallel pipelines and directory-based output.

        Generates all combinations of provided parameters, executes ImagePipeline for each
        combination in parallel, and saves results to organized directory structure with
        an interactive HTML viewer.

        Args:
            pipe_cfgs: Dictionary of pipeline configuration lists. Each dict has the key
                as the name of the pipeline and the values as the list of tuples of
                ImageOperation and parameter dictionary pairs to search. Example:
                {"MyPipeline":[(GaussianBlur(), {"sigma": [1, 2, 3]})]}
            output_dir: Directory for saving all results. Will be created if it doesn't
                exist. Results are organized into subdirectories per pipeline
                configuration.
            data_layers: Which image data to save. Valid options: "rgb", "gray",
               "enh_gray", "objmask", "objmap". Defaults to all available layers.

        Implementation Notes:
            - pipe_cfgs should be processed sequentially. Each cfg will get its own
                output directory
            - Following the parameter sweep of each cfg, each parameter sweep will
                have its own subfolder in the parent cfg folder

        Output Structure:
            output_dir/
            ├── Manifest.json (holds the param_sweep name to ImagePipeline mapping)
            ├── OriginalRGB.tif (Only if the image was originally RGB)
            ├── OriginalGray.tif
            ├── pipe_cfg1/
            │   ├── param_sweep1/
            │   │   └── <Data Layer Images>...
            │   └── param_sweep2/
            └── pipe_cfg2/
                └── ...

        Example:

            .. code-block:: python

                from phenotypic.util._pipeline_grid_search import PipelineGridSearchBase

                pipe_cfgs = {
                    "PipelineName": [
                        (GaussianBlur(), {"sigma": [1, 2, 3]}),
                        (OtsuDetector(), {"ignore_zeros": [True, False]}),
                    ]
                }
                gs = PipelineGridSearchBase(pipe_cfgs=pipe_cfgs,)


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

        data_layers = ["rgb", "gray", "enh_gray", "objmask", "objmap"] \
            if data_layers is None \
            else data_layers
        self._validate_layers(data_layers)
        self.data_layers: DataAccessors = data_layers

        self.output_dir: Path = Path(output_dir) \
            if output_dir \
            else Path.cwd()

    @staticmethod
    def _get_param_sweep(
            name: str,
            pipe_cfg: GridSearchConfig) -> Iterator[ImagePipeline]:
        """
        Generates an iterable of `ImagePipeline` objects, each configured with a unique
        combination of parameters derived from the grid search configuration.

        This static method is utilized to perform exhaustive parameter sweeps over the
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
            yield ImagePipeline(ops=[op.__class__(**param_set[op_idx])
                                     for op_idx, op in enumerate(ops)],
                                name=f"{name}_{idx}"
                                )

    @staticmethod
    def _validate_pipe_cfgs(
            pipe_cfgs: GridSearchConfig,
    ) -> None:
        """Validate all inputs before processing.

        Args:
            pipe_cfgs: List of (operation, params_dict) tuples

        Raises:
            ValueError: If inputs are invalid or malformed
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
    def _validate_layers(data_layers: List[str]) -> None:
        """Validate all layers present in data_layers.
        Args:
        """
        # Validate DataAccessors
        valid_layers = {"rgb", "gray", "enh_gray", "objmask", "objmap"}
        invalid = set(data_layers) - valid_layers
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
