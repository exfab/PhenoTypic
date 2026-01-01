# MultiPipelineGridSearch Detailed Data Flow

```mermaid
flowchart TD
    %% Input Data
    subgraph "INPUT DATA"
        Image[(Single Image object)]
        PipelineConfigs[(pipeline_configs: List[Dict])]
        Params[(Parameters: n_jobs, inplace, save_tiff_dir, etc.)]
    end

    %% Validation Phase
    subgraph "VALIDATION PHASE"
        ValidateConfigs[_validate_pipeline_configs<br/>Validate config structure]
        ValidateTiff[_validate_save_tiff_params<br/>Validate TIFF parameters]
    end

    %% Backend Decision
    subgraph "BACKEND DECISION"
        BackendCheck{backend == 'submitit'?}
        DisableTrie[Set optimize_shared_prefixes = False]
        EnforceTiff[Enforce save_tiff_dir required]
    end

    %% Main Processing Decision
    subgraph "EXECUTION STRATEGY"
        OptimizeCheck{optimize_shared_prefixes?}
    end

    %% Optimized Path
    subgraph "OPTIMIZED PATH (Trie-based)"
        ExpandConfigs[_expand_pipeline_configs_to_concrete<br/>Generate all parameter combinations]
        EstimateMemory[_estimate_pipeline_memory<br/>Calculate memory per pipeline]
        CalcBatching[_calculate_optimal_batch_size<br/>Determine batch size & parallelism]
        ProcessBatches[Process in batches with progress bar]

        subgraph "Per Batch Processing"
            GroupByPrefix[_group_pipelines_by_longest_prefix<br/>Create trie groups]
            ProcessTrieGroups[_process_trie_groups_sequentially<br/>Execute trie groups]
            ExecuteParallel[_execute_parallel_tasks<br/>Parallel execution within groups]
        end
    end

    %% Non-Optimized Path
    subgraph "NON-OPTIMIZED PATH (Linear)"
        ProcessEachConfig[Process each pipeline config]
        UnpackOps[_unpack_ops_tuples<br/>Separate operations & parameters]
        GenerateCombos[_generate_param_combinations<br/>Create parameter combinations]
        ExecuteLinearParallel[_execute_parallel_tasks<br/>Parallel execution per pipeline]
    end

    %% Result Processing
    subgraph "RESULT PROCESSING"
        CollectResults[Collect pipeline results]
        ModeCheck{save_tiff_dir?}

        subgraph "TIFF MODE"
            ExtractLayers[_extract_data_layers<br/>Get arrays from results]
            SaveTiff[_save_array_as_tiff<br/>Save each layer as TIFF]
            StoreConfigs[Store configs in all_configs dict]
            GenerateHtml[_create_trial_view_html<br/>Create overview page]
        end

        subgraph "NAPARI MODE"
            ExtractLayersNapari[_extract_data_layers<br/>Get arrays from results]
            AddOriginal[_add_original_layers<br/>Add reference layers]
            AddResults[_add_result_layer<br/>Add result layers to viewer]
            StoreConfigsNapari[Store configs in all_configs dict]
        end
    end

    %% Output
    subgraph "OUTPUT"
        TiffOutput[(Dict[str, str]: configs_dict)]
        NapariOutput[(Tuple[napari.Viewer, Dict[str, str]])]
    end

    %% Data Flow Connections
    Image --> ValidateConfigs
    PipelineConfigs --> ValidateConfigs
    Params --> ValidateTiff

    ValidateConfigs --> BackendCheck
    ValidateTiff --> BackendCheck

    BackendCheck -->|submitit| DisableTrie
    DisableTrie --> EnforceTiff
    EnforceTiff --> OptimizeCheck

    BackendCheck -->|joblib| OptimizeCheck

    OptimizeCheck -->|True| ExpandConfigs
    OptimizeCheck -->|False| ProcessEachConfig

    %% Optimized Path Flow
    ExpandConfigs --> EstimateMemory
    EstimateMemory --> CalcBatching
    CalcBatching --> ProcessBatches

    ProcessBatches --> GroupByPrefix
    GroupByPrefix --> ProcessTrieGroups
    ProcessTrieGroups --> ExecuteParallel
    ExecuteParallel --> CollectResults

    %% Non-Optimized Path Flow
    ProcessEachConfig --> UnpackOps
    UnpackOps --> GenerateCombos
    GenerateCombos --> ExecuteLinearParallel
    ExecuteLinearParallel --> CollectResults

    %% Result Processing Flow
    CollectResults --> ModeCheck

    ModeCheck -->|TIFF mode| ExtractLayers
    ExtractLayers --> SaveTiff
    SaveTiff --> StoreConfigs
    StoreConfigs --> GenerateHtml
    GenerateHtml --> TiffOutput

    ModeCheck -->|Napari mode| ExtractLayersNapari
    ExtractLayersNapari --> AddOriginal
    AddOriginal --> AddResults
    AddResults --> StoreConfigsNapari
    StoreConfigsNapari --> NapariOutput

    %% Styling
    classDef inputData fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef validation fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#ef6c00,stroke-width:2px
    classDef optimized fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef linear fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
    classDef results fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef output fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px

    class Image,PipelineConfigs,Params inputData
    class ValidateConfigs,ValidateTiff validation
    class BackendCheck,OptimizeCheck,ModeCheck decision
    class ExpandConfigs,EstimateMemory,CalcBatching,ProcessBatches,GroupByPrefix,ProcessTrieGroups,ExecuteParallel optimized
    class ProcessEachConfig,UnpackOps,GenerateCombos,ExecuteLinearParallel linear
    class CollectResults,ExtractLayers,SaveTiff,StoreConfigs,GenerateHtml,ExtractLayersNapari,AddOriginal,AddResults,StoreConfigsNapari results
    class TiffOutput,NapariOutput output
```
