from __future__ import annotations

import importlib.util
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Union, Optional, NamedTuple

import numpy as np

from phenotypic.tools_.constants_ import OBJECT

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image
    from phenotypic._core._image_pipeline import ImagePipeline

import pandas as pd
from typing import Dict, List
import inspect
import time
import sys

from phenotypic.abc_ import MeasureFeatures, BaseOperation, ImageOperation
from phenotypic.abc_._post_measurement import PostMeasurement
from phenotypic.tools_.mixin import LazyWidgetMixin

logger = logging.getLogger("ImagePipeline")


def _layers_modified_by(operation: BaseOperation) -> tuple[str, ...] | None:
    """Return layer names modified by this operation, or None for read-only ops.

    Args:
        operation: An operation instance from the pipeline.

    Returns:
        Tuple of layer name strings that this operation modifies,
        or ``None`` if the operation is read-only (e.g. measurements).
    """
    from phenotypic.abc_ import (
        ImageCorrector,
        ImageEnhancer,
        MeasureFeatures,
        ObjectDetector,
        ObjectRefiner,
    )

    if isinstance(operation, MeasureFeatures):
        return None
    if isinstance(operation, ImageCorrector):
        return ("rgb", "gray", "detect_mat", "objmap")
    if isinstance(operation, ImageEnhancer):
        return ("detect_mat",)
    if isinstance(operation, (ObjectDetector, ObjectRefiner)):
        return ("objmap",)
    return ("rgb", "gray", "detect_mat", "objmap")


class IntermediateResult(NamedTuple):
    """Result of ``apply_with_intermediates``.

    Attributes:
        image: The final processed image (same type as the input).
        intermediates: Dictionary mapping operation names to image snapshots
            taken after each operation. Values are ``Image`` copies when
            results are kept in memory, or ``None`` when saved to disk.
    """

    image: Union[GridImage, Image]
    intermediates: Dict[str, Optional[Image]]


class ImagePipelineCore(BaseOperation, LazyWidgetMixin):
    """
    Represents a handler for processing and measurement queues used in Image operations
    and feature extraction tasks.

    This class manages two queues: a processing queue and a measurement queue. The processing
    queue contains Image operations that are applied sequentially to an Image. The measurement
    queue contains feature extractors that are used to analyze an Image and produce results
    as a pandas DataFrame. Both queues are optional and can be specified as dictionaries. If not
    provided, empty queues are initialized by default to enable flexibility in pipeline
    construction and usage.

    Attributes:
        name (str): A unique identifier for this pipeline. Defaults to a randomly
            generated UUID4 string if not provided during initialization.
        desc (str): Pipeline description accessed via property. Returns the class docstring
            if no custom description was set. Can be modified via the property setter.
        _ops (Dict[str, ImageOperation]): A dictionary where keys are string
            identifiers and values are `ImageOperation` objects representing operations to apply
            to an Image.
        _meas (Dict[str, MeasureFeatures]): A dictionary where keys are string
            identifiers and values are `FeatureExtractor` objects for extracting features
            from images.
    """

    def __init__(
            self,
            ops: List[ImageOperation | ImagePipeline] | Dict[str, ImageOperation | ImagePipeline] | None = None,
            meas: List[MeasureFeatures] | Dict[str, MeasureFeatures] | None = None,
            post: List[PostMeasurement] | Dict[str, PostMeasurement] | None = None,
            benchmark: bool = False,
            verbose: bool = False,
            name: Optional[str] = None,
            desc: Optional[str] = None,
            reset: bool = False,
    ):
        """
        This class represents a processing and measurement interface for Image operations
        and feature extraction. It initializes operational and measurement queues based
        on the provided dictionaries.

        Args:
            ops: A list or dictionary of ImageOperation or ImagePipeline objects.
                If a list, class names are used as keys. If a dictionary, keys are
                operation names (strings) and values are ImageOperation or ImagePipeline
                objects responsible for performing specific Image processing tasks.
            meas: An optional dictionary where the keys are feature names
                (strings) and the values are FeatureExtractor objects responsible for
                extracting specific features.
            benchmark: A flag indicating whether to track execution times for operations
                and measurements. Defaults to False.
            verbose: A flag indicating whether to print progress information when
                benchmark mode is on. Defaults to False.
            name: An optional string identifier for this pipeline. If not provided,
                a randomly generated UUID4 string will be assigned automatically.
            desc: An optional description for this pipeline. If not provided, the
                class docstring will be used when accessing the desc property.
            reset: Default reset behavior for the apply() method. When True, the image
                will be reset before applying operations. Can be overridden per-call
                in apply() and apply_and_measure(). Defaults to False.
        """
        # If pipe_cfgs is a list of operations convert to a dictionary
        self._ops: Dict[str, ImageOperation] = {}
        if ops is not None:
            self.set_ops(ops)

        self._meas: Dict[str, MeasureFeatures] = {}
        if meas is not None:
            self.set_meas(meas)

        self._post: Dict[str, PostMeasurement] = {}
        if post is not None:
            self.set_post(post)

        # Store benchmark, verbose, and reset flags
        self._benchmark = benchmark
        self._verbose = verbose
        self._reset = reset

        # Set pipeline name (generate UUID4 if not provided)
        self.name = name if name is not None else str(uuid.uuid4())

        # Store description as protected attribute
        self._desc = desc

        # Initialize dictionaries to store execution times
        self._operation_times: Dict[str, float] = {}
        self._measurement_times: Dict[str, float] = {}

        # Initialize dictionaries to store memory usage
        self._operation_memory: Dict[str, float] = {}
        self._measurement_memory: Dict[str, float] = {}
        self._operation_rss: Dict[str, float] = {}
        self._measurement_rss: Dict[str, float] = {}

    @property
    def desc(self) -> str:
        """Get pipeline description. Returns class docstring if no description set."""
        if self._desc is not None:
            return self._desc
        # Return the actual class's docstring (e.g., ImagePipeline's docstring)
        # This uses self.__class__.__doc__ to get the docstring of the instantiated class
        return self.__class__.__doc__ or ""

    @desc.setter
    def desc(self, value: Optional[str]):
        """Set pipeline description."""
        self._desc = value

    def set_ops(self, ops: List[ImageOperation | ImagePipeline] | Dict[str, ImageOperation | ImagePipeline]):
        """
        Sets the operations to be performed. The operations can be passed as either a list of
        ImageOperation or ImagePipeline instances or a dictionary mapping operation names to
        ImageOperation or ImagePipeline instances. This method ensures that each operation in
        the list has a unique name. Raises a TypeError if the input is neither a list nor a dictionary.

        Args:
            ops (List[ImageOperation | ImagePipeline] | Dict[str, ImageOperation | ImagePipeline]):
                A list of ImageOperation or ImagePipeline objects, or a dictionary where keys are
                operation names and values are ImageOperation or ImagePipeline objects.

        Raises:
            TypeError: If the input is not a list or a dictionary.
        """
        # If pipe_cfgs is a list of ImageOperation
        if isinstance(ops, list):
            op_names = [x.__class__.__name__ for x in ops]
            op_names = self.__make_unique(op_names)
            self._ops = {op_names[i]: ops[i] for i in range(len(ops))}
        # If pipe_cfgs is a dictionary
        elif isinstance(ops, dict):
            self._ops = ops
        else:
            raise TypeError(
                    f"pipe_cfgs must be a list or a dictionary, got {type(ops)}")

    def set_meas(
            self, measurements: List[MeasureFeatures] | Dict[str, MeasureFeatures]
    ):
        """
        Sets the measurements to be used for further computation. The input can be either
        a list of `MeasureFeatures` objects or a dictionary with string keys and `MeasureFeatures`
        objects as values.

        The method processes the given input to construct a dictionary mapping measurement names
        to `MeasureFeatures` instances. If a list is passed, unique class names of the
        `MeasureFeatures` instances in the list are used as keys.

        Args:
            measurements (List[MeasureFeatures] | Dict[str, MeasureFeatures]): A collection
                of measurement features either as a list of `MeasureFeatures` objects, where
                class names are used as keys for dictionary creation, or as a dictionary where
                keys are predefined strings and values are `MeasureFeatures` objects.

        Raises:
            TypeError: If the `measurements` argument is neither a list nor a dictionary.
        """
        if isinstance(measurements, list):
            measurement_names = [
                x.__class__.__name__
                for x in measurements
                if isinstance(x, MeasureFeatures)
            ]
            measurement_names = self.__make_unique(measurement_names)
            self._meas = {
                measurement_names[i]: measurements[i] for i in range(len(measurements))
            }
        elif isinstance(measurements, dict):
            self._meas = measurements
        else:
            raise TypeError(
                    f"measurements must be a list or a dictionary, got {type(measurements)}"
            )

    def set_post(
            self, post: List[PostMeasurement] | Dict[str, PostMeasurement]
    ):
        """Set the post-measurement transforms.

        Args:
            post: A list or dictionary of PostMeasurement objects.
                If a list, class names are used as keys.

        Raises:
            TypeError: If post is neither a list nor a dictionary.
        """
        if isinstance(post, list):
            post_names = [
                x.__class__.__name__
                for x in post
                if isinstance(x, PostMeasurement)
            ]
            post_names = self.__make_unique(post_names)
            self._post = {
                post_names[i]: post[i] for i in range(len(post))
            }
        elif isinstance(post, dict):
            self._post = post
        else:
            raise TypeError("post must be a list or dictionary")

    def get_ops(self) -> Dict[str, ImageOperation]:
        """Get a copy of the operations dictionary.

        Returns a shallow copy to prevent accidental mutation of internal state.

        Returns:
            Dict[str, ImageOperation]: Dictionary mapping operation names to
                ImageOperation instances.
        """
        return dict(self._ops)

    def get_meas(self) -> Dict[str, MeasureFeatures]:
        """Get a copy of the measurements dictionary.

        Returns a shallow copy to prevent accidental mutation of internal state.

        Returns:
            Dict[str, MeasureFeatures]: Dictionary mapping measurement names to
                MeasureFeatures instances.
        """
        return dict(self._meas)

    @staticmethod
    def __make_unique(class_names):
        """
        Ensures uniqueness of strings in the given list by appending numeric suffixes when duplicates are
        found. If duplicates exist, subsequent occurrences of the duplicate string are modified by adding a
        numeric suffix to make them unique.

        Args:
            class_names (List[str]): A list of strings where duplicates may exist.

        Returns:
            List[str]: A new list of strings where each string is guaranteed to be unique.

        Raises:
            None
        """
        seen = {}
        result = []

        for s in class_names:
            if s not in seen:
                seen[s] = 0
                result.append(s)
            else:
                seen[s] += 1
                new_s = f"{s}_{seen[s]}"
                while new_s in seen:
                    seen[s] += 1
                    new_s = f"{s}_{seen[s]}"
                seen[new_s] = 0
                result.append(new_s)

        return result

    @staticmethod
    def _get_process_rss_mb() -> float:
        """Return current process RSS in megabytes.

        Returns:
            float: Resident set size in MB, or ``nan`` if unavailable.
        """
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            return float("nan")

    def _run_operations(
        self,
        img: Image,
        on_op_complete: Optional[Callable[[int, str, Image, ImageOperation], None]] = None,
    ) -> None:
        """Execute all queued operations on *img* in order.

        Args:
            img: The image to process (modified in place).
            on_op_complete: Optional callback invoked after each successful
                operation with ``(index, op_name, img, operation)``.
        """
        # Reset operation times and memory for new apply run if benchmarking
        if self._benchmark:
            self._operation_times = {}
            self._operation_memory = {}
            self._operation_rss = {}

        # Create progress bar if verbose and benchmark are enabled
        if self._benchmark and self._verbose:
            has_tqdm = importlib.util.find_spec("tqdm") is not None
            if has_tqdm:
                from tqdm import tqdm

                total_ops = len(self._ops)
                pbar = tqdm(
                    total=total_ops, desc="Applying operations", file=sys.stdout
                )
            else:
                print("Applying operations...")
        else:
            has_tqdm = False

        for i, (key, operation) in enumerate(self._ops.items()):
            logger.debug("Applying operation [%d/%d]: %s", i + 1, len(self._ops), key)
            try:
                # Update progress bar description with current operation
                if self._benchmark and self._verbose:
                    if has_tqdm:
                        pbar.set_description(f"Operation: {key}")
                    else:
                        print(f"  Applying operation: {key}")

                # Measure execution time and memory if benchmarking is enabled
                if self._benchmark:
                    rss_before = self._get_process_rss_mb()
                    start_time = time.time()

                sig = inspect.signature(operation.apply)

                apply_params = {}
                if "inplace" in sig.parameters:
                    apply_params["inplace"] = True

                if "reset" in sig.parameters:
                    apply_params["reset"] = (
                        False  # Prevents intermediate pipelines from resetting progress
                    )

                # Propagate benchmark flag to nested pipelines
                nested_was_benchmarking = None
                if self._benchmark and isinstance(operation, ImagePipelineCore):
                    nested_was_benchmarking = operation._benchmark
                    operation._benchmark = True

                # Apply actual operation
                operation.apply(img, **apply_params)

                # Restore nested pipeline benchmark flag
                if nested_was_benchmarking is not None:
                    operation._benchmark = nested_was_benchmarking

                # Store execution time and memory if benchmarking is enabled
                if self._benchmark:
                    self._operation_times[key] = time.time() - start_time
                    rss_after = self._get_process_rss_mb()
                    self._operation_memory[key] = rss_after - rss_before
                    self._operation_rss[key] = rss_after

                    if self._verbose:
                        delta = self._operation_memory[key]
                        delta_str = f"{delta:+.1f} MB"
                        if has_tqdm:
                            pbar.set_postfix(
                                time=f"{self._operation_times[key]:.4f}s",
                                mem=delta_str,
                            )
                            pbar.update(1)
                        else:
                            print(
                                f"    Completed in {self._operation_times[key]:.4f} seconds"
                                f" ({delta_str})"
                            )

                if on_op_complete is not None:
                    on_op_complete(i, key, img, operation)

            except Exception as exc:
                if self._benchmark and self._verbose and has_tqdm:
                    pbar.close()
                op_class = type(operation).__name__
                raise type(exc)(
                    f"[{op_class}] (step {i + 1}/{len(self._ops)}, "
                    f"key='{key}'): {exc}"
                ) from exc

        # Close the progress bar if it exists
        if self._benchmark and self._verbose and has_tqdm:
            pbar.close()

    def apply(
            self, image: Image, inplace: bool = False, reset: Optional[bool] = None
    ) -> Union[GridImage, Image]:
        """
        The class provides an interface to process and apply a series of operations on
        an Image. The operations are maintained in a queue and executed sequentially
        when applied to the given Image.

        Args:
            image (Image): The arr Image to be processed. The type `Image` refers to
                an instance of the Image object to which transformations are applied.
            inplace (bool, optional): A flag indicating whether to apply the
                transformations directly on the provided Image (`True`) or create a
                copy of the Image before performing transformations (`False`). Defaults
                to `False`.
            reset (bool, optional): Whether to reset the image before applying the pipeline.
                If None (default), uses the pipeline's reset setting from __init__.
                If explicitly set to True or False, overrides the pipeline setting.
        """
        effective_reset = reset if reset is not None else self._reset
        img = image if inplace else image.copy()
        if effective_reset:
            img.reset()
        self._run_operations(img)
        return img

    def apply_with_intermediates(
        self,
        image: Image,
        inplace: bool = False,
        reset: Optional[bool] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> IntermediateResult:
        """Apply the pipeline and capture a snapshot of the image after each operation.

        Behaves identically to :meth:`apply` (respecting *inplace*, *reset*,
        benchmark timing, and verbose/tqdm progress) but additionally records
        the image state after every operation completes.

        Args:
            image: The input image to process.
            inplace: If ``True`` the image is modified in place; otherwise a
                copy is made first.  Defaults to ``False``.
            reset: Whether to reset the image before applying operations.
                ``None`` (default) uses the pipeline-level setting.
            output_dir: Optional directory path.  When provided, each
                intermediate image is persisted to an HDF5 file inside this
                directory (created automatically) and the corresponding dict
                value is set to ``None`` to conserve memory.  When ``None``,
                intermediates are kept in memory as ``Image`` copies.

        Returns:
            IntermediateResult: A named tuple containing the final image and a
            dictionary mapping operation names to intermediate snapshots (or
            ``None`` when *output_dir* is used).
        """
        effective_reset = reset if reset is not None else self._reset
        img = image if inplace else image.copy()
        if effective_reset:
            img.reset()

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        intermediates: Dict[str, Optional[Image]] = {}

        if output_dir is not None:
            # Save initial base with all layers (pre-pipeline state)
            _all_layers = ("rgb", "gray", "detect_mat", "objmap")
            img.copy().save_intermediate_layers(
                output_dir / "base_00.h5", layers=_all_layers,
            )

        def _capture(i: int, key: str, current: Image, operation: ImageOperation) -> None:
            if output_dir is not None:
                layers = _layers_modified_by(operation)

                if layers is None:
                    # Read-only op (MeasureFeatures, GridFinder): no file
                    intermediates[key] = None
                elif len(layers) == 4:
                    # Corrector: emit a new base with all layers
                    current.copy().save_intermediate_layers(
                        output_dir / f"base_{i:02d}.h5", layers=layers,
                    )
                    intermediates[key] = None
                else:
                    # Delta: save only modified layers
                    current.copy().save_intermediate_layers(
                        output_dir / f"{i:02d}_{key}.h5", layers=layers,
                    )
                    intermediates[key] = None
            else:
                intermediates[key] = current.copy()

        self._run_operations(img, on_op_complete=_capture)
        return IntermediateResult(image=img, intermediates=intermediates)

    def measure(self, image: Image, include_metadata=True) -> pd.DataFrame:
        """
        Measures properties of a given image and optionally includes metadata. The method performs
        measurements using a set of predefined measurement operations. If benchmarking is enabled,
        the execution time of each measurement is recorded. When verbose mode is active, detailed
        logging of the measurement process is displayed. A progress bar is used to track progress
        if the tqdm library is available.

        Args:
            image (Image): The image object for which measurements are performed. It must support
                the `info` method and optionally a `grid` or `objects` attribute.
            include_metadata (bool, optional): Indicates whether metadata should be included in
                the measurements. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the results of all performed measurements combined
                on the same index.

        Raises:
            Exception: An exception is raised if a measurement operation fails while being
                applied to the image.
        """
        # Reset measurement times and memory for new measure run if benchmarking
        if self._benchmark:
            self._measurement_times = {}
            self._measurement_memory = {}
            self._measurement_rss = {}

        # Print message if verbose and benchmark are enabled
        if self._benchmark and self._verbose:
            print("Measuring image properties...")

        # Get image info and measure time/memory if benchmarking is enabled
        if self._benchmark:
            rss_before = self._get_process_rss_mb()
            start_time = time.time()
            measurements = [image.info(include_metadata=include_metadata)]
            self._measurement_times["image_info"] = time.time() - start_time
            rss_after = self._get_process_rss_mb()
            self._measurement_memory["image_info"] = rss_after - rss_before
            self._measurement_rss["image_info"] = rss_after

            # Print execution time if verbose and benchmark are enabled
            if self._verbose:
                delta = self._measurement_memory["image_info"]
                print(
                        f"  Image info: {self._measurement_times['image_info']:.4f} seconds"
                        f" ({delta:+.1f} MB)"
                )
        else:
            measurements = [
                image.grid.info(include_metadata=include_metadata)
                if hasattr(image, "grid")
                else image.objects.info(include_metadata=include_metadata)
            ]

        # Create progress bar if verbose and benchmark are enabled
        if self._benchmark and self._verbose:
            has_tqdm = importlib.util.find_spec("tqdm") is not None
            if has_tqdm:
                from tqdm import tqdm

                # Create a tqdm instance without items to manually update it
                total_measurements = len(self._meas)
                pbar = tqdm(
                        total=total_measurements,
                        desc="Applying measurements",
                        file=sys.stdout,
                )
            else:
                # If tqdm is not available, fall back to simple printing
                print("Applying measurements...")
        else:
            has_tqdm = False

        # perform measurements
        for i, (key, measurement) in enumerate(self._meas.items()):
            logger.debug("Running measurement [%d/%d]: %s", i + 1, len(self._meas), key)
            try:
                # Update progress bar description with current measurement
                if self._benchmark and self._verbose:
                    if has_tqdm:
                        pbar.set_description(f"Measurement: {key}")
                    else:
                        print(f"  Applying measurement: {key}")

                # Measure execution time and memory if benchmarking is enabled
                if self._benchmark:
                    rss_before = self._get_process_rss_mb()
                    start_time = time.time()

                    # Measurement is taken here
                    measurements.append(measurement.measure(image))
                    self._measurement_times[key] = time.time() - start_time
                    rss_after = self._get_process_rss_mb()
                    self._measurement_memory[key] = rss_after - rss_before
                    self._measurement_rss[key] = rss_after

                    # Print execution time if verbose and benchmark are enabled
                    if self._verbose:
                        delta = self._measurement_memory[key]
                        delta_str = f"{delta:+.1f} MB"
                        if has_tqdm:
                            pbar.set_postfix(
                                    time=f"{self._measurement_times[key]:.4f}s",
                                    mem=delta_str,
                            )
                            pbar.update(1)
                        else:
                            print(
                                    f"    Completed in {self._measurement_times[key]:.4f} seconds"
                                    f" ({delta_str})"
                            )
                else:
                    measurements.append(measurement.measure(image))
            except Exception as exc:
                if self._benchmark and self._verbose and has_tqdm:
                    pbar.close()
                meas_class = type(measurement).__name__
                raise type(exc)(
                    f"[{meas_class}] (step {i + 1}/{len(self._meas)}, "
                    f"key='{key}'): {exc}"
                ) from exc

        # Close the progress bar if it exists
        if self._benchmark and self._verbose and has_tqdm:
            pbar.close()

        df = self._merge_on_object_labels(measurements)

        # Apply post-measurement transforms
        for key, post_op in self._post.items():
            logger.debug("Running post-measurement transform: %s", key)
            df = post_op.apply(df)

        return df

    def apply_and_measure(
            self,
            image: Image,
            inplace: bool = False,
            reset: Optional[bool] = None,
            include_metadata: bool = True,
    ) -> pd.DataFrame:
        """
        Applies processing to the given image and measures the results.

        This function first applies a processing method to the supplied image,
        adjusting it based on the given parameters. After processing, the
        resulting image is measured, and a DataFrame containing the measurement
        data is returned.

        Args:
            image (Image): The image to process and measure.
            inplace (bool): Whether to modify the original image directly or
                work on a copy. Default is False.
            reset (bool, optional): Whether to reset any previous processing on the image
                before applying the current method. If None (default), uses the pipeline's
                reset setting. If explicitly set, overrides the pipeline setting.
            include_metadata (bool): Whether to include metadata in the
                measurement results. Default is True.

        Returns:
            pd.DataFrame: A DataFrame containing measurement data for the
            processed image.
        """
        img = self.apply(image=image, inplace=inplace, reset=reset)
        return self.measure(image=img, include_metadata=include_metadata)

    def benchmark_results(self) -> pd.DataFrame:
        """Return execution times and memory usage for operations and measurements.

        This method should be called after applying the pipeline on an image to get
        the execution times and memory consumption of the different processes.

        When an operation is itself an ``ImagePipelineCore`` (nested pipeline),
        its sub-operations are expanded as indented sub-rows beneath the parent
        entry with names like ``"ParentOp > ChildOp"``.

        Returns:
            pd.DataFrame: A DataFrame with columns ``Process Type``,
                ``Process Name``, ``Execution Time (s)``, ``Memory Delta (MB)``,
                and ``RSS After (MB)``.
        """
        columns = [
            "Process Type",
            "Process Name",
            "Execution Time (s)",
            "Memory Delta (MB)",
            "RSS After (MB)",
        ]

        data: List[Dict[str, object]] = []

        # Add operation rows (with nested expansion)
        for op_name, op_time in self._operation_times.items():
            data.append({
                "Process Type"      : "Operation",
                "Process Name"      : op_name,
                "Execution Time (s)": op_time,
                "Memory Delta (MB)" : self._operation_memory.get(op_name, float("nan")),
                "RSS After (MB)"    : self._operation_rss.get(op_name, float("nan")),
            })

            # Expand nested pipeline sub-operations
            operation = self._ops.get(op_name)
            if (
                isinstance(operation, ImagePipelineCore)
                and operation._operation_times
            ):
                for sub_name, sub_time in operation._operation_times.items():
                    data.append({
                        "Process Type"      : "Operation",
                        "Process Name"      : f"  {op_name} > {sub_name}",
                        "Execution Time (s)": sub_time,
                        "Memory Delta (MB)" : operation._operation_memory.get(
                            sub_name, float("nan")
                        ),
                        "RSS After (MB)"    : operation._operation_rss.get(
                            sub_name, float("nan")
                        ),
                    })

        # Add measurement rows
        for meas_name, meas_time in self._measurement_times.items():
            data.append({
                "Process Type"      : "Measurement",
                "Process Name"      : meas_name,
                "Execution Time (s)": meas_time,
                "Memory Delta (MB)" : self._measurement_memory.get(
                    meas_name, float("nan")
                ),
                "RSS After (MB)"    : self._measurement_rss.get(
                    meas_name, float("nan")
                ),
            })

        if not data:
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(data)

        # Total row: sum top-level times and deltas only (exclude sub-rows)
        top_level_mask = ~df["Process Name"].str.startswith("  ")
        total_time = df.loc[top_level_mask, "Execution Time (s)"].sum()
        total_delta = df.loc[top_level_mask, "Memory Delta (MB)"].sum()
        # RSS After for total: last top-level RSS value
        top_level_rss = df.loc[top_level_mask, "RSS After (MB)"]
        final_rss = top_level_rss.iloc[-1] if len(top_level_rss) > 0 else float("nan")

        total_row = pd.DataFrame([{
            "Process Type"      : "Total",
            "Process Name"      : "All Processes",
            "Execution Time (s)": total_time,
            "Memory Delta (MB)" : total_delta,
            "RSS After (MB)"    : final_rss,
        }])
        df = pd.concat([df, total_row], ignore_index=True)

        return df

    @staticmethod
    def _merge_on_object_labels(dataframes_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Merge multiple DataFrames only if share object labels

        Args:
            dataframes_list: List of pandas DataFrames to merge

        Returns:
            Merged DataFrame containing only the data from DataFrames with matching index names

        Raises:
            ValueError: If no DataFrames are provided or if no matching index names are found
        """
        if not dataframes_list or not all(
                [isinstance(x, pd.DataFrame) for x in dataframes_list]
        ):
            raise ValueError("No DataFrames provided")
        new_df = dataframes_list[0]
        if new_df.index.name == OBJECT.LABEL:
            new_df = new_df.reset_index(drop=False)

        if len(dataframes_list) > 1:
            for df in dataframes_list[1:]:
                if df.index.name == OBJECT.LABEL:
                    df = df.reset_index(drop=False)

                cols_to_merge_on = [OBJECT.LABEL]  # Resets each new other df

                for col_new_df in new_df.columns:
                    if col_new_df != OBJECT.LABEL:  # skip the object label
                        for col_other_df in df.columns:
                            if col_new_df == col_other_df and np.all(
                                    df[col_new_df] == df[col_other_df]
                            ):
                                cols_to_merge_on.append(col_other_df)

                new_df = new_df.merge(df, on=cols_to_merge_on, suffixes=("", "_merged"))

        return new_df
