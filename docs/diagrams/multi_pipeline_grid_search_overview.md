# MultiPipelineGridSearch Overview

```mermaid
flowchart LR
    %% Inputs
    subgraph "INPUTS"
        IMG[Image Object]
        CFGS[Pipeline Configs<br/>List[Dict[str, Any]]]
        OPTS[Options<br/>n_jobs, memory_limit,<br/>backend, etc.]
    end

    %% Main Function
    MPG[MultiPipelineGridSearch]

    %% Key Decision Points
    BK{Backend?}
    OPT{Optimize<br/>Shared Prefixes?}

    %% Processing Strategies
    subgraph "EXECUTION STRATEGIES"
        TRIE[Trie-based Optimization<br/>• Expand configs to concrete<br/>• Group by shared prefixes<br/>• Batch processing<br/>• Parallel execution]
        LINEAR[Linear Processing<br/>• Process each config<br/>• Generate param combos<br/>• Parallel per pipeline]
    end

    %% Memory Management
    MEM[Adaptive Batching<br/>• Memory estimation<br/>• Batch size calculation<br/>• Memory monitoring]

    %% Output Modes
    subgraph "OUTPUT MODES"
        NAPARI[Napari Viewer<br/>• Interactive visualization<br/>• Layer management<br/>• Real-time inspection]
        TIFF[TIFF Files<br/>• Batch saving<br/>• Memory efficient<br/>• HTML overview<br/>• Cluster compatible]
    end

    %% Outputs
    subgraph "OUTPUTS"
        VIEWER[(napari.Viewer<br/>+ configs_dict)]
        CONFIGS[(configs_dict<br/>TIFF paths)]
    end

    %% Flow Connections
    IMG --> MPG
    CFGS --> MPG
    OPTS --> MPG

    MPG --> BK
    BK -->|submitit| OPT
    BK -->|joblib| OPT

    OPT -->|True| TRIE
    OPT -->|False| LINEAR

    TRIE --> MEM
    MEM --> NAPARI
    MEM --> TIFF

    LINEAR --> NAPARI
    LINEAR --> TIFF

    NAPARI --> VIEWER
    TIFF --> CONFIGS

    %% Styling
    classDef inputClass fill:#e1f5fe,stroke:#01579b
    classDef mainClass fill:#f3e5f5,stroke:#7b1fa2,stroke-width:3px
    classDef decisionClass fill:#fff3e0,stroke:#ef6c00
    classDef strategyClass fill:#e8f5e8,stroke:#2e7d32
    classDef memoryClass fill:#fce4ec,stroke:#c2185b
    classDef outputClass fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px

    class IMG,CFGS,OPTS inputClass
    class MPG mainClass
    class BK,OPT decisionClass
    class TRIE,LINEAR strategyClass
    class MEM memoryClass
    class NAPARI,TIFF outputClass
    class VIEWER,CONFIGS outputClass
```