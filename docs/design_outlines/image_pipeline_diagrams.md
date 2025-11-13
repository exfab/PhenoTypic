# ImagePipeline and ImageSet UML & Interaction Diagrams

This document contains comprehensive UML and interaction diagrams for the `ImagePipeline` and `ImageSet` classes in the
PhenoTypic framework.

## UML Class Diagram

```mermaid
classDiagram
    class ImageOperation {
        +apply(image, inplace=False): Image
        +_operate(image): Image
        +_apply_to_single_image(): Image
    }

    class ImagePipelineCore {
        -_ops: Dict[str, ImageOperation]
        -_meas: Dict[str, MeasureFeatures]
        -_benchmark: bool
        -_verbose: bool
        +__init__(ops, meas, benchmark, verbose)
        +set_ops(ops): void
        +set_meas(measurements): void
        +apply(image, inplace, reset): Image
        +measure(image, include_metadata): DataFrame
        +apply_and_measure(image, inplace, reset, include_metadata): DataFrame
        +benchmark_results(): DataFrame
    }

    class ImagePipelineBatch {
        -num_workers: int
        -verbose: bool
        -memblock_factor: float
        -timeout: int
        +__init__(ops, meas, num_workers, verbose, memblock_factor, benchmark, timeout)
        +apply(image, inplace, reset): Image|None
        +measure(image, include_metadata, verbose): DataFrame
        +apply_and_measure(image, inplace, reset, include_metadata): DataFrame|None
        +_coordinator(image_set, mode, num_workers): DataFrame|None
        +_allocate_measurement_datasets(imageset): void
        +_producer(): void
        +_writer(): void
        +_worker(): void
    }

    class ImagePipeline {
        +apply(image, inplace, reset): Image|None
        +measure(image, include_metadata, verbose): DataFrame
        +apply_and_measure(image, inplace, reset, include_metadata): DataFrame|None
    }

    class ImageSetCore {
        -name: str
        -_src_path: Path
        -_out_path: Path
        -_overwrite: bool
        -grid_finder: GridFinder|None
        -hdf_: HDF
        +__init__(name, grid_finder, src, outpath, default_mode, overwrite)
        +add_image(image, overwrite): void
        +get_image_names(): List[str]
        +get_image(image_name): Image
        +iter_images(): Iterator[Image]
        +close(): void
    }

    class ImageSetStatus {
        +__init__(name, grid_finder, src, outpath, overwrite)
        +reset_status(image_names, str|None): void
        +get_status(image_names, str|None): DataFrame
        +_add_image2group(group, image, overwrite): void
    }

    class ImageSetMeasurements {
        -_measurement_accessor: SetMeasurementAccessor
        +__init__(name, grid_finder, src, outpath, overwrite)
        +measurements: SetMeasurementAccessor
        +get_measurement(image_names, str|None): DataFrame
    }

    class ImageSet {
        +__init__(name, grid_finder, src, outpath, overwrite)
    }

    class BaseOperation {
        +_get_matched_operation_args(): Dict
    }

    class MeasureFeatures {
        +measure(image): DataFrame
        +_operate(image): DataFrame
        +_ensure_array(scipy_output): np.array
        +_calculate_center_of_mass(array, objmap): np.ndarray
        +_calculate_max(array, objmap): np.ndarray
        +_calculate_mean(array, objmap): np.ndarray
        +_calculate_median(array, objmap): np.ndarray
        +_calculate_minimum(array, objmap): np.ndarray
        +_calculate_stddev(array, objmap): np.ndarray
        +_calculate_sum(array, objmap): np.ndarray
        +_calculate_variance(array, objmap): np.ndarray
        +_calculate_coeff_variation(array, objmap): np.ndarray
        +_calculate_extrema(array, objmap): tuple
        +_calculate_min_extrema(array, objmap): tuple
        +_calculate_max_extrema(array, objmap): tuple
        +_funcmap2objects(func, out_dtype, array, objmap, default, pass_positions): np.ndarray
        +_calculate_q1(array, objmap, method): np.ndarray
        +_calculate_q3(array, objmap, method): np.ndarray
        +_calculate_iqr(array, objmap, method, nan_policy): np.ndarray
    }

    class SetMeasurementAccessor {
        +__init__(imageset)
        +get_measurements(): DataFrame
        +save_measurements(measurements): void
    }

    class HDF {
        +reader(): HDF5Reader
        +writer(): HDF5Writer
        +swmr_reader(): HDF5SWMRReader
        +swmr_writer(): HDF5SWMRWriter
        +safe_writer(): HDF5SafeWriter
        +get_root_group(handler): HDF5Group
        +get_data_group(handler): HDF5Group
        +get_status_subgroup(handler, image_name): HDF5Group
        +get_image_measurement_subgroup(handler, image_name): HDF5Group
        +save_frame_update(group, dataframe, start, require_swmr): void
        +preallocate_frame_layout(group, dataframe, chunks, compression, preallocate, string_fixed_length, require_swmr): void
    }

    class GridFinder {
        +__init__()
        +imread(path): Image
    }

    class Image {
        +name: str
        +grid: Grid
        +objects: ObjectMap
        +copy(): Image
        +reset(): void
        +info(include_metadata): Dict
        +_load_from_hdf5_group(group): Image
        +_save_image2hdfgroup(group, compression, compression_opts): void
    }

%% Inheritance relationships
    ImageOperation <|-- ImagePipelineCore
    ImagePipelineCore <|-- ImagePipelineBatch
    ImagePipelineBatch <|-- ImagePipeline
    ImageSetCore <|-- ImageSetStatus
    ImageSetStatus <|-- ImageSetMeasurements
    ImageSetMeasurements <|-- ImageSet
    BaseOperation <|-- ImageOperation
    BaseOperation <|-- MeasureFeatures
%% Composition relationships
    ImagePipelineCore *-- ImageOperation: contains
    ImagePipelineCore *-- MeasureFeatures: contains
    ImageSetMeasurements *-- SetMeasurementAccessor: contains
    ImageSetCore *-- HDF: contains
    ImageSetCore *-- GridFinder: contains
    SetMeasurementAccessor *-- HDF: uses
%% Usage relationships
    ImagePipeline --> Image: processes
    ImagePipeline --> ImageSet: processes
    ImageSet --> Image: contains
    ImageSet --> HDF: uses
%% Note about relationships
    note for ImagePipeline "Can process both single Image and ImageSet objects"
    note for ImageSet "Manages collections of images in HDF5 format"
```

## Single Image Processing Interaction Diagram

```mermaid
sequenceDiagram
    participant User
    participant ImagePipeline
    participant ImagePipelineCore
    participant ImageOperation
    participant MeasureFeatures
    participant Image

    User->>ImagePipeline: apply_and_measure(image)
    ImagePipeline->>ImagePipelineCore: apply_and_measure(image, inplace, reset, include_metadata)
    ImagePipelineCore->>ImagePipelineCore: apply(image, inplace, reset)

    loop For each operation in _ops
        ImagePipelineCore->>ImageOperation: apply(image, inplace=True)
        ImageOperation->>ImageOperation: _operate(image)
        ImageOperation-->>ImagePipelineCore: modified_image
    end

    ImagePipelineCore->>ImagePipelineCore: measure(image, include_metadata)

    loop For each measurement in _meas
        ImagePipelineCore->>MeasureFeatures: measure(image)
        MeasureFeatures->>MeasureFeatures: _operate(image)
        MeasureFeatures-->>ImagePipelineCore: measurement_dataframe
    end

    ImagePipelineCore->>ImagePipelineCore: _merge_on_object_labels(dataframes)
    ImagePipelineCore-->>ImagePipeline: final_dataframe
    ImagePipeline-->>User: final_dataframe
```

## ImageSet Parallel Processing Interaction Diagram

```mermaid
sequenceDiagram
    participant User
    participant ImagePipelineBatch
    participant ImageSet
    participant Coordinator
    participant Producer
    participant Worker
    participant Writer
    participant HDF

    User->>ImagePipelineBatch: apply_and_measure(imageset)
    ImagePipelineBatch->>Coordinator: _coordinator(imageset, mode="apply_and_measure")

    Coordinator->>Coordinator: _allocate_measurement_datasets(imageset)
    Coordinator->>HDF: writer()
    HDF-->>Coordinator: writer_handle

    Coordinator->>Producer: start_thread(target=_producer)
    Coordinator->>Writer: start_thread(target=_writer)
    Coordinator->>Worker: start_processes(target=_worker, num_workers)

    Producer->>HDF: swmr_reader()
    HDF-->>Producer: reader_handle

    loop For each image_name in image_names
        Producer->>ImageSet: get_image(image_name)
        ImageSet->>HDF: get_data_group(reader)
        HDF-->>ImageSet: image_group
        ImageSet->>ImageSet: image._load_from_hdf5_group(group)
        ImageSet-->>Producer: image
        Producer->>Coordinator: put(image_pkl) to work_queue
    end

    Producer->>Coordinator: put(None) to work_queue * num_workers

    Worker->>Coordinator: get() from work_queue
    Worker->>Worker: pickle.loads(image_pkl)
    Worker->>Worker: apply operations to image
    Worker->>Worker: measure features from image
    Worker->>Coordinator: put((image_name, image_bytes, meas_bytes)) to results_queue

    Writer->>Coordinator: get() from results_queue
    Writer->>Writer: pickle.loads(image_bytes)
    Writer->>Writer: pickle.loads(meas_bytes)

    Writer->>HDF: swmr_writer()
    HDF-->>Writer: writer_handle
    Writer->>HDF: get_data_group(writer)
    HDF-->>Writer: image_group
    Writer->>Image: _save_image2hdfgroup(group)
    Writer->>HDF: get_image_measurement_subgroup(writer, image_name)
    HDF-->>Writer: meas_group
    Writer->>HDF: save_frame_update(meas_group, meas)

    Coordinator->>Coordinator: join threads and processes
    Coordinator->>ImageSet: get_measurement()
    ImageSet->>HDF: reader()
    HDF-->>ImageSet: reader_handle
    ImageSet->>HDF: get_image_measurement_subgroup(reader, image_name)
    HDF-->>ImageSet: meas_dataframes
    ImageSet->>ImageSet: pd.concat(measurements)
    ImageSet-->>Coordinator: final_dataframe
    Coordinator-->>ImagePipelineBatch: final_dataframe
    ImagePipelineBatch-->>User: final_dataframe
```

## ImageSet Measurement Retrieval Interaction Diagram

```mermaid
sequenceDiagram
    participant User
    participant ImageSet
    participant SetMeasurementAccessor
    participant HDF
    participant Image

    User->>ImageSet: get_measurement(image_names)
    ImageSet->>SetMeasurementAccessor: get_measurements()

    alt image_names is None
        ImageSet->>ImageSet: get_image_names()
        ImageSet-->>ImageSet: all_image_names
    end

    loop For each name in image_names
        SetMeasurementAccessor->>HDF: reader()
        HDF-->>SetMeasurementAccessor: reader_handle

        SetMeasurementAccessor->>HDF: get_image_group(reader, name)
        HDF-->>SetMeasurementAccessor: image_group

        SetMeasurementAccessor->>HDF: get_image_measurement_subgroup(reader, name)
        HDF-->>SetMeasurementAccessor: meas_subgroup

        alt measurements exist
            SetMeasurementAccessor->>HDF: load_frame(meas_subgroup)
            HDF-->>SetMeasurementAccessor: measurement_dataframe

            %% Add metadata from protected group
            SetMeasurementAccessor->>HDF: get_protected_metadata_subgroup(reader, name)
            HDF-->>SetMeasurementAccessor: prot_metadata_group
            SetMeasurementAccessor->>SetMeasurementAccessor: add metadata columns

            %% Add metadata from public group
            SetMeasurementAccessor->>HDF: get_public_metadata_subgroup(reader, name)
            HDF-->>SetMeasurementAccessor: pub_metadata_group
            SetMeasurementAccessor->>SetMeasurementAccessor: add metadata columns

            SetMeasurementAccessor-->>ImageSet: measurement_dataframe
        end
    end

    ImageSet->>ImageSet: pd.concat(measurements)
    ImageSet-->>User: final_dataframe
```

## Key Design Patterns Illustrated

### 1. **Template Method Pattern**

- `ImagePipelineCore` defines the skeleton of the algorithm in `apply_and_measure()`
- Subclasses like `ImagePipelineBatch` implement specific steps like `_coordinator()`

### 2. **Strategy Pattern**

- `ImageOperation` and `MeasureFeatures` define interfaces for interchangeable algorithms
- Concrete implementations can be swapped at runtime

### 3. **Producer-Consumer Pattern**

- `ImagePipelineBatch` uses multiple threads/processes:
    - **Producer**: Loads images from HDF5 and queues them for processing
    - **Consumers (Workers)**: Process images in parallel
    - **Writer**: Saves results back to HDF5

### 4. **Composite Pattern**

- `ImageSet` acts as a container for multiple `Image` objects
- Both single images and image sets can be processed uniformly by pipelines

### 5. **Adapter Pattern**

- `SetMeasurementAccessor` adapts HDF5 operations to a measurement-specific interface

### 6. **Factory Method Pattern**

- `ImageSet.get_image()` creates appropriate `Image` or `GridImage` objects based on `grid_finder`

## Performance Considerations

1. **Memory Management**: Uses memory block factors to prevent OOM errors
2. **HDF5 SWMR**: Single Writer Multiple Reader mode for concurrent access
3. **Multiprocessing**: Cross-platform spawn context for parallel processing
4. **Threading**: Separate threads for producer, writer, and coordinator
5. **Lazy Loading**: Images are loaded from HDF5 only when needed

## Error Handling

- Graceful handling of processing failures
- Status tracking for each image (processed/measured)
- Logging at multiple levels (debug, info, error)
- Exception propagation with context information

## Cross-Platform Compatibility

- Uses `multiprocessing.get_context('spawn')` for Windows/Linux/Mac compatibility
- Path handling with `pathlib.Path`
- File extension detection for various image formats
- Temporary file management for different operating systems
