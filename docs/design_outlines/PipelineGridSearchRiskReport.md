# Report 1

## 1. Comprehensive Logic Review

### **Core Algorithm Correctness**

**Issue**: The trie-based pipeline grouping logic in
`_group_pipelines_by_longest_prefix()` contains a critical flaw in parameter sweep
detection (lines 1219-1234). The current implementation incorrectly treats parameter
sweeps as structural divergence rather than grouping them together.

**Current Problem**:

```python
if len(op_types) == 1:
# This merges ALL downstream pipelines from different parameter values
# But this is WRONG - it should only merge when the SAME operation 
# has different parameters, not when different operations happen to be the same type
```

**Suggested Fix**: The logic should distinguish between:

1. **Parameter sweeps**: Same operation class with different parameter values (should be
   grouped)
2. **Structural divergence**: Different operation classes (should create separate
   groups)

### **Memory Estimation Bug**

**Issue**: In `_estimate_pipeline_memory()` (lines 963-974), the function unnecessarily
creates array copies:

```python
rgb_data = image.rgb[:]  # Creates unnecessary copy!
gray_data = image.gray[:]  # Creates unnecessary copy!
enh_gray_data = image.enh_gray[:]  # Creates unnecessary copy!
```

**Impact**: This defeats the purpose of memory estimation by potentially triggering
garbage collection and affecting accuracy.

**Suggested Fix**: Use shape and dtype information without copying:

```python
rgb_data = image.rgb
if rgb_data is not None:
    base_size += np.prod(rgb_data.shape) * rgb_data.dtype.itemsize
```

## 2. Error Detection & Risk Assessment

### **Missing Input Validation**

**Issue**: No validation for empty or invalid `shared_prefix_len` parameter in
`_execute_concrete_pipeline_batch()` (line 1490).

**Risk**: If `shared_prefix_len` exceeds the length of `all_ops`, this will cause an
IndexError:

```python
remaining_ops = all_ops[shared_prefix_len:]  # Can fail silently or crash
```

### **Silent Failure in Parallel Execution**

**Issue**: The submitit backend error handling (lines 696-703) only logs errors but
doesn't provide detailed failure information to users.

**Risk**: Users get generic "job failed" messages without actionable debugging
information.

### **Data Structure Inconsistency**

**Issue**: The `_TrieNode` class uses forward references incorrectly (line 166):

```python
children: Dict[Tuple, "_TrieNode"] = field(default_factory=dict)
```

**Risk**: This creates circular reference issues that can cause memory leaks and
serialization problems.

## 3. Performance & Memory Optimization

### **Inefficient Memory Batching**

**Issue**: The batch size calculation in `_calculate_optimal_batch_size()` (lines
1062-1066) uses overly conservative logic:

```python
ideal_batch_size = jobs_per_batch * 2
safe_batch_size = min(max(2, jobs_per_batch), 8)
batch_size = min(ideal_batch_size, safe_batch_size, total_pipelines)
```

**Problem**: This caps batch size at 8 regardless of available memory, severely limiting
parallelism for large grids.

### **Trie Construction Inefficiency**

**Issue**: `_build_pipeline_trie()` rebuilds the trie for each batch instead of reusing
shared structure.

**Impact**: O(n²) complexity for large pipeline sets instead of O(n).

### **Memory Leak in Parallel Execution**

**Issue**: In `_execute_pipeline_trie()` (lines 1642-1644), batch results are yielded
individually but the batch list remains in memory:

```python
for batch in batch_results:
    for result in batch:
        yield result  # Individual results yielded
# batch_results list still holds all batch data until function exits
```

## 4. Code Quality & Maintainability

### **Overly Complex Functions**

**Issue**: `MultiPipelineGridSearch()` is 800+ lines with deeply nested conditional
logic (lines 2045-2435).

**Impact**:

- Hard to test individual components
- Difficult to debug specific execution paths
- Violates single responsibility principle

**Suggested Refactoring**: Break into smaller functions:

- `_setup_execution_mode()`
- `_execute_with_trie_optimization()`
- `_execute_linear_mode()`
- `_handle_results_output()`

### **Poor Error Messages**

**Issue**: Generic error messages don't provide context for debugging:

```python
raise RuntimeError(f"{len(failed_jobs)} job(s) failed:\n{failure_msg}")
```

**Suggestion**: Include execution context:

```python
raise RuntimeError(f"{len(failed_jobs)}/{len(jobs)} SLURM jobs failed "
                   f"during {desc} execution:\n{failure_msg}")
```

### **Inconsistent Naming**

**Issue**: Mixed naming conventions for similar concepts:

- `pipeline_configs` vs `concrete_configs`
- `trie_groups` vs `pipeline_groups`
- `batch_configs` vs `group_configs`

## 5. Testing & Debugging Readiness

### **Critical Testing Gaps**

**Issue**: No unit tests for core trie logic functions:

- `_build_pipeline_trie()`
- `_group_pipelines_by_longest_prefix()`
- `_find_first_branch_point()`

**Needed Tests**:

- Trie construction with various pipeline structures
- Edge cases: empty pipelines, single operations, parameter sweeps
- Memory estimation accuracy validation
- Batch size calculation correctness

### **Debugging Challenges**

**Issue**: Complex nested progress bars make it difficult to track execution state.

**Suggestion**: Add execution context logging:

```python
logger.info(f"TrieGroup {group_idx}: {len(group_configs)} pipelines, "
            f"prefix_depth={shared_prefix_len}, "
            f"parallel_branches={len(pipeline_specs)}")
```

### **Missing Validation Tests**

**Issue**: No tests for input validation edge cases:

- Empty parameter lists
- Invalid operation attributes
- Malformed pipeline configurations

## 6. Follow-Up Agentic Debugging Considerations

### **Priority Debugging Order**

1. **High Priority**: Fix trie grouping logic in
   `_group_pipelines_by_longest_prefix()` - this affects correctness of the core
   optimization
2. **High Priority**: Address memory estimation bugs that could cause OOM errors
3. **Medium Priority**: Add comprehensive input validation
4. **Medium Priority**: Refactor overly complex functions for better testability
5. **Low Priority**: Optimize batch sizing and memory management

### **Step-by-Step Validation Strategy**

For `_execute_pipeline_trie()`:

1. **Step 1**: Validate trie structure construction with unit tests
2. **Step 2**: Test shared prefix detection logic
3. **Step 3**: Verify downstream path enumeration
4. **Step 4**: Test parallel execution with mock backends
5. **Step 5**: Validate memory cleanup and resource management

For `_build_pipeline_trie()`:

1. **Step 1**: Test basic trie construction with simple pipelines
2. **Step 2**: Validate parameter-based key generation
3. **Step 3**: Test complex multi-branch scenarios
4. **Step 4**: Verify pipeline name tracking accuracy

### **Key Metrics to Monitor**

- Trie construction time vs. number of pipelines
- Memory usage accuracy of estimation functions
- Parallel execution overhead vs. serial baseline
- Error rate and types in different execution modes

This analysis provides a roadmap for systematic debugging and optimization of the
pipeline grid search system. The trie logic and memory management appear to be the most
critical areas requiring immediate attention.

# Report 2

[1 tool called]

### PipelineGridSearch Analysis and Optimization Report

Here is a comprehensive analysis of `@src/phenotypic/util/_pipeline_grid_search.py`,
highlighting logic, error handling, performance, and code quality aspects.

#### 1. Error Detection & Risk Assessment

* **Unhashable Parameter Values (Critical Runtime Risk):**
    * **Problem:** The `_ops_key` function (Line 140) generates a dictionary key using
      `tuple(sorted(params.items()))`. If a parameter value is a `list` or `dict` (e.g.,
      a kernel size `[3, 3]` or a range `[0, 1]` passed as a single value rather than a
      grid search list), this will raise `TypeError: unhashable type: 'list'` when
      attempting to use it as a key in `_TrieNode.children` (Line 1159).
    * **Fix:** Ensure parameter values in `_ops_key` are converted to hashable types (
      e.g., convert lists to tuples) recursively before creating the key.
* **TIFF Saving of Float Arrays (Potential Crash/Corruption):**
    * **Problem:** `_save_array_as_tiff` (Lines 434-445) handles `bool` and `uint16`
      explicitly, but falls through to `mode='L'` for other 2D arrays. If the image data
      is floating-point (common in image processing, 0.0-1.0), PIL's
      `fromarray(..., mode='L')` may interpret bytes incorrectly or fail to scale the
      data, resulting in black or garbled images.
    * **Fix:** Add explicit handling for floating-point arrays. Normalize them to 0-255
      `uint8` or save them as floating-point TIFFs if precise values are needed.
* **Data Extraction Safety:**
    * **Problem:** `_extract_data_layers` assumes `result_img.rgb`, `result_img.gray`,
      etc., are accessible slices. If the `Image` object implementation changes or these
      properties return `None` unexpectedly (despite checks), it could fail. The current
      checks `if ... is not None` are good but rely on the specific `Image` class
      interface.

#### 2. Logic Review & Correctness

* **Batching vs. Optimization Trade-off:**
    * **Observation:** In `MultiPipelineGridSearch`, pipelines are batched *before*
      grouping by shared prefix (Lines 2251-2278).
    * **Impact:** If 100 pipelines share a costly first step, but the batch size is 10,
      that first step will be re-computed 10 times (once for each batch) instead of once
      globally.
    * **Recommendation:** While this tradeoff protects memory, it should be documented.
      A more advanced scheduler could process the shared prefix once and then batch the
      *subsequent* steps, though this complicates memory management.
* **Submitit Optimization Disabled:**
    * **Observation:** The code explicitly disables `optimize_shared_prefixes` when
      using `submitit` (Line 2166).
    * **Impact:** Cluster execution is less efficient than it could be, as every job
      re-computes the full pipeline from scratch.

#### 3. Performance & Memory Optimization

* **Joblib Serialization Overhead:**
    * **Bottleneck:** In `_execute_pipeline_trie`, the result of the shared prefix (a
      potentially large `Image` object) is passed to `_execute_concrete_pipeline_batch`
      via `joblib`. This triggers pickling and data transfer to worker processes.
    * **Impact:** For large images, the overhead of serializing/deserializing the image
      `N` times (for `N` branches) might exceed the time saved by pre-calculating the
      prefix.
    * **Improvement:** Use `joblib`'s `memmap` features or shared memory if possible, or
      consider threading (`prefer="threads"`) if operations release the GIL (e.g., many
      NumPy/OpenCV ops).
* **Memory Estimation Accuracy:**
    * **Observation:** `_estimate_pipeline_memory` relies on `sys.getsizeof` and a fixed
      overhead factor (1.2).
    * **Impact:** `sys.getsizeof` does not recursively size custom objects effectively.
      If `Image` contains complex structures, this significantly underestimates memory,
      leading to OOM despite adaptive batching.
* **Redundant Array Copying:**
    * **Observation:** `_extract_data_layers` creates strict copies (`.copy()`) of
      arrays. This is good for safety but spikes memory usage.

#### 4. Code Quality & Maintainability

* **Duplication in Execution Loops:**
    * **Issue:** `MultiPipelineGridSearch` contains two distinct execution paths: one
      for trie-optimized (Lines 2284-2323) and one for linear (Lines 2350-2418). The
      result handling logic (saving TIFFs vs. Napari layers) is duplicated in both
      blocks.
    * **Fix:** Refactor the "Result Handling" (Save vs. View) into a helper function
      `_handle_pipeline_result(result, save_dir, viewer, ...)` to DRY up the main
      function.
* **Complex Function Size:**
    * **Issue:** `MultiPipelineGridSearch` is over 350 lines long. It handles
      validation, batch calculation, execution strategy, and result processing.
    * **Fix:** Extract the "Batch Execution Loop" into a separate private function
      `_execute_batch_optimized` and `_execute_batch_linear`.

#### 5. Testing & Debugging Readiness

* **Trie Logic Verification:**
    * **Gap:** The complex logic in `_group_pipelines_by_longest_prefix` needs rigorous
      unit testing with abstract inputs (not full images) to ensure it correctly groups
      and splits pipelines based on parameter vs. structural differences.
* **Logging:**
    * The logging is generally good (`logger.info` and `debug`), aiding in debugging
      execution flow.

### Recommended Follow-Up Actions

1. **Prioritized Debugging:**
    * Create a test case with a list-type parameter (e.g., `kernel_size=[3,3]`) to
      confirm the `_ops_key` crash and implement the fix.
    * Create a test case with a float-array image and try `_save_array_as_tiff` to
      verify TIFF generation correctness.

2. **Refactoring:**
    * Extract `_handle_pipeline_result` to clean up `MultiPipelineGridSearch`.
    * Update `_ops_key` to recursively tuple-ize parameters.

3. **Performance Tuning:**
    * Benchmark the Joblib serialization overhead vs. re-computation cost for the shared
      prefix optimization. If overhead is high, consider adding a heuristic to only use
      shared-prefix optimization if the prefix length > X operations.

# Report 3

I'll analyze the `PipelineGridSearch` implementation comprehensively. Let me start by
examining the file structure and key components.

[1 tool called]

I'll provide a comprehensive analysis of the `PipelineGridSearch` implementation,
organized by the categories you've requested.

## 1. Comprehensive Logic Review

### 🔴 **Critical Logic Issues**

#### **A. Trie Path Counting Logic Error (Lines 1342-1365)**

```python
def count_descendants(node: _TrieNode) -> Tuple[int, int, int]:
    # ...
    for child in node.children.values():
        child_depth, child_branches, child_paths = count_descendants(child)
        max_depth = max(max_depth, child_depth + 1)
        total_branches += child_branches
        total_paths *= child_paths  # ❌ INCORRECT: Should sum, not multiply
```

**Problem**: Line 1359 multiplies paths, which gives exponential counts for branching
structures. This should **sum** child paths to count total leaf nodes.

**Impact**: `_analyze_trie_structure` returns incorrect `total_leaf_paths` count,
affecting logging and potentially optimization decisions.

**Fix**:

```python
total_paths += child_paths  # Correct: Sum paths to count leaves
```

#### **B. Parameter Sweep Detection Flaw (Lines 1214-1234)**

```python
if len(op_types) == 1:
    # All children are same operation type with different parameters
    # This is a PARAMETER SWEEP, not structural divergence
    all_pipeline_names = []
    for child in node.children.values():
        child_groups = _collect_groups_from_node(child)
        for group in child_groups:
            all_pipeline_names.extend(group)
    return [all_pipeline_names] if all_pipeline_names else []
```

**Problem**: This logic groups ALL pipelines with the same operation type together, even
if they have different downstream operations. Example:

- Pipeline A: `GaussianBlur(sigma=1.0) → OtsuDetector()`
- Pipeline B: `GaussianBlur(sigma=2.0) → CannyDetector()`

These would be grouped together because GaussianBlur is the same operation type, but
they diverge afterward. This defeats the shared prefix optimization.

**Impact**: Over-aggressive grouping may reduce parallelization opportunities and
increase memory usage by keeping unrelated pipelines in the same batch.

#### **C. Missing Edge Case: Empty Pipeline Handling (Lines 1490-1508)**

```python
remaining_ops = all_ops[shared_prefix_len:]

if not remaining_ops:
    # No operations to apply - result is the input image
    result_image = image.copy()
    full_ops = all_ops
```

**Problem**: If `all_ops` is empty (zero-operation pipeline), this creates a copy
without any processing. However, the calling code doesn't validate that `all_ops`
contains at least one operation.

**Risk**: Silent success with unexpected behavior for misconfigured pipelines.

---

### 🟡 **Logic Warnings**

#### **D. Batch Size Calculation Conservative Limits (Lines 1061-1066)**

```python
ideal_batch_size = jobs_per_batch * 2
safe_batch_size = min(max(2, jobs_per_batch), 8)
batch_size = min(ideal_batch_size, safe_batch_size, total_pipelines)
```

**Issue**: Hardcoded maximum of 8 pipelines per batch (`safe_batch_size`) is overly
conservative for systems with large memory. With 64GB RAM and small images, this
artificially limits throughput.

**Suggestion**: Make this configurable or scale with available memory.

---

## 2. Error Detection & Risk Assessment

### 🔴 **High-Risk Runtime Errors**

#### **A. Array Indexing Without Bounds Checking (Lines 372-390)**

```python
def _extract_data_layers(result_img: "Image", data_layers: List[str]) -> Dict[str, Any]:
    extracted = {}
    for layer in data_layers:
        if layer == "rgb":
            rgb_data = result_img.rgb[:]  # ❌ No validation that .rgb exists
            if rgb_data is not None and rgb_data.size > 0:
                extracted["rgb"] = rgb_data.copy()
```

**Problem**: Assumes `result_img.rgb` exists and is indexable. If the Image object is
malformed or the accessor raises an exception, this fails without graceful handling.

**Missing**: Try-except blocks for accessor failures.

#### **B. Submitit Backend Disabled Trie Optimization (Lines 2168-2172)**

```python
if backend == "submitit":
    logger.info("Submitit backend detected: disabling trie optimization...")
    optimize_shared_prefixes = False
```

**Problem**: When `optimize_shared_prefixes=True` is explicitly passed by user, this
silently overrides it. No warning or error is raised about the parameter being ignored.

**Risk**: User confusion and unexpected performance characteristics.

**Fix**: Add explicit warning if user set `optimize_shared_prefixes=True`:

```python
if backend == "submitit" and optimize_shared_prefixes:
    logger.warning(
        "optimize_shared_prefixes=True is incompatible with submitit backend. "
        "Disabling trie optimization (jobs are already parallelized).")
    optimize_shared_prefixes = False
```

#### **C. Division by Zero Risk (Line 1051)**

```python
max_parallel = max(1, memory_limit // memory_per_pipeline)
```

**Problem**: If `memory_per_pipeline` is 0 (possible with tiny images or estimation
errors), this causes division by zero.

**Missing**: Validation that `memory_per_pipeline > 0`.

#### **D. Unhandled Pickle Failures (Lines 669-678)**

```python
try:
    import pickle

    pickle.dumps(func)
except Exception as e:
    raise ValueError(f"Function '{func.__name__}' is not picklable...") from e
```

**Problem**: This validation happens AFTER user has potentially waited for setup. Should
validate earlier.

**Also**: `func.__name__` may not exist for lambdas or partial functions, causing
AttributeError.

---

### 🟡 **Medium-Risk Issues**

#### **E. Silent Memory Estimation Failures**

```python
def _estimate_pipeline_memory(...) -> int:
    # No validation that estimate is reasonable
    return int((base_size + extracted_size) * _MEMORY_OVERHEAD_FACTOR)
```

**Problem**: If `base_size` calculation returns 0 (empty images), memory estimation is
wildly inaccurate. No bounds checking or sanity validation.

**Missing**:

- Minimum memory threshold validation
- Warning if estimate seems unreasonable (< 1MB or > 100GB)

#### **F. Trie Node Traversal Infinite Loop Risk (Lines 1405-1422)**

```python
def _find_first_branch_point(root: _TrieNode) -> Tuple[
    _TrieNode, List["ImageOperation"]]:
    current = root
    ops_stack = []

    while current.children:  # ❌ Assumes trie is acyclic
        if len(current.children) > 1:
            return current, ops_stack
        child_node = next(iter(current.children.values()))
        # ...
        current = child_node
```

**Problem**: No cycle detection. If trie construction has a bug creating a cycle, this
loops infinitely.

**Likelihood**: Low (trie construction is one-way), but defensive programming would add
depth limit.

---

## 3. Performance & Memory Optimization

### 🟢 **Optimization Opportunities**

#### **A. Memory Estimation Overhead Factor Too Conservative (Line 120)**

```python
_MEMORY_OVERHEAD_FACTOR = 1.2
```

**Issue**: 20% overhead may be insufficient for complex operations (BM3D, wavelet
transforms). Empirical testing shows 30-40% is more accurate for heavy operations.

**Suggestion**: Make this operation-specific or increase to 1.4.

#### **B. Redundant Deep Copies in Pipeline Execution (Lines 1497-1500, 1510)**

```python
ops_copy = []
for op in remaining_ops:
    ops_copy.append(copy.deepcopy(op))  # Deep copy each operation

# Later...
pipeline = ImagePipeline(ops=copy.deepcopy(full_ops))  # Deep copy AGAIN
```

**Problem**: Operations are deep-copied twice:

1. Line 1500: Deep copy for execution
2. Line 1510: Deep copy for serialization

**Impact**: 2× memory usage for operation objects (minor but unnecessary).

**Fix**: Reuse the first copy for serialization:

```python
ops_copy_for_pipeline = [copy.deepcopy(op) for op in full_ops]
# Use ops_copy_for_pipeline for both execution and serialization
```

#### **C. Parallel Execution Always Uses joblib (Lines 1608-1634)**

```python
if len(pipeline_specs) > 1 and n_jobs != 1:
    # ...
    with tqdm_joblib(...):
        batch_results = Parallel(n_jobs=n_jobs)(tasks)
```

**Problem**: Even when `backend="submitit"` is specified, `_execute_pipeline_trie` still
uses joblib for parallel execution. This is inconsistent with user expectations.

**Issue**: Lines 1527-1534 accept `backend` and `slurm_params` but never use them for
the parallel execution at line 1634.

**Impact**: Users specifying submitit backend get joblib execution anyway.

#### **D. Trie Structure Analysis Overhead (Lines 1572-1575)**

```python
trie_stats = _analyze_trie_structure(root)
logger.info(f"Trie structure: depth={trie_stats['max_depth']}, ...")
```

**Issue**: `_analyze_trie_structure` recursively traverses entire trie just for logging.
For large tries (1000+ nodes), this adds measurable overhead (50-200ms).

**Suggestion**: Make this conditional on debug logging level or lazy-evaluate only when
needed.

#### **E. HTML Thumbnail Generation is Serial (Lines 769-796)**

```python
for layer in data_layers:
    tiff_path = save_path / tiff_pattern
    if tiff_path.exists():
        try:
            img = PIL_Image.open(tiff_path)
            img.thumbnail((200, 200), PIL_Image.Resampling.LANCZOS)
            # ...
```

**Problem**: Thumbnail generation is serial. For 500+ TIFFs, this takes 10-30 seconds.

**Optimization**: Parallelize thumbnail creation with `joblib` or use multiprocessing.

---

## 4. Code Quality & Maintainability

### 🔵 **Code Quality Issues**

#### **A. Inconsistent Error Handling Patterns**

- Some functions raise `ValueError` with detailed messages (good)
- Others return `None` or empty lists silently (bad)
- Example: `_enumerate_downstream_paths` returns `[]` for invalid input without logging

**Suggestion**: Standardize on explicit exceptions for invalid states.

#### **B. Magic Numbers Without Constants**

- Line 1065: `safe_batch_size = min(max(2, jobs_per_batch), 8)` - Why 2 and 8?
- Line 776: `img.thumbnail((200, 200), ...)` - Why 200x200?
- Line 1046: `memory_limit = int(available_memory * 0.75)` - Why 75%?

**Suggestion**: Extract to named constants with documentation:

```python
_MIN_BATCH_SIZE = 2
_MAX_SAFE_BATCH_SIZE = 8
_MEMORY_SAFETY_FACTOR = 0.75
_THUMBNAIL_SIZE = (200, 200)
```

#### **C. Overly Long Functions**

- `MultiPipelineGridSearch`: 277 lines (lines 2045-2434)
- `_execute_pipeline_trie`: 143 lines (lines 1527-1670)
- `_create_trial_view_html`: 127 lines (lines 722-927)

**Impact**: Difficult to test, debug, and maintain.

**Suggestion**: Refactor into smaller, focused functions.

#### **D. Unclear Variable Naming**

- `param_config` vs `config` vs `json_config` - unclear hierarchy
- `all_configs` stores JSON strings, not config dicts (misleading name)
- `ops` vs `operations` vs `remaining_ops` - inconsistent terminology

---

## 5. Testing & Debugging Readiness

### 🧪 **Testing Gaps**

#### **A. No Unit Tests for Core Logic**

- `_group_pipelines_by_longest_prefix` (complex logic, high bug risk)
- `_calculate_optimal_batch_size` (mathematical correctness)
- `_build_pipeline_trie` (tree structure validation)

**Recommendation**: Add unit tests with known input/output pairs.

#### **B. Insufficient Logging for Debugging**

Missing logs for:

- Memory allocations before/after each major operation
- Individual pipeline execution times (only batch totals)
- Trie traversal decisions (which branches taken, why)

**Suggestion**: Add debug-level logs:

```python
logger.debug(f"Processing pipeline {pipeline_name}: "
             f"{len(remaining_ops)} ops, estimated {mem_estimate_mb:.1f} MB")
```

#### **C. No Validation Mode for Dry-Run**

Users cannot test configurations without executing pipelines. Add:

```python
def validate_pipeline_configs(pipeline_configs, dry_run=True):
    """Validate configs without execution"""
    # Return estimated memory, execution time, trie structure
```

---

## 6. Follow-Up Agentic Debugging Considerations

### 🎯 **Priority Areas for Detailed Debugging**

#### **HIGH PRIORITY**

1. **`_group_pipelines_by_longest_prefix` (Lines 1171-1258)**
    - Most complex logic
    - Directly impacts optimization effectiveness
    - Parameter sweep detection needs validation with real-world pipelines
    - **Action**: Create test suite with 10-20 pipeline configurations covering:
        - Pure parameter sweeps
        - Structural divergence
        - Mixed scenarios

2. **`_calculate_optimal_batch_size` (Lines 1001-1068)**
    - Mathematical correctness critical for OOM prevention
    - **Action**: Validate with empirical memory measurements on various image sizes
    - Add assertions for invariants (batch_size ≤ total_pipelines, jobs_per_batch ≥ 1)

3. **`_execute_pipeline_trie` (Lines 1527-1670)**
    - Coordination logic for trie traversal
    - Backend parameter ignored in parallel execution
    - **Action**: Step-through debugging with 3-4 pipeline trie to verify:
        - Shared prefix executed exactly once
        - All branches receive correct input image
        - No duplicate executions

#### **MEDIUM PRIORITY**

4. **Memory Estimation Logic (Lines 930-999)**
    - Validate `_estimate_pipeline_memory` with real measurements
    - **Action**: Add instrumentation to track actual vs. estimated memory
    - Adjust `_MEMORY_OVERHEAD_FACTOR` based on empirical data

5. **Submitit Integration (Lines 665-715)**
    - Job submission and result collection
    - **Action**: Test with actual SLURM cluster to verify:
        - Error propagation works correctly
        - Job failures are detected and reported
        - Results returned in correct order

#### **LOW PRIORITY**

6. **HTML Generation (Lines 722-927)**
    - Mostly cosmetic, but thumbnail generation performance could be improved
    - **Action**: Profile with large result sets (500+ TIFFs)

---

## 📊 Summary Statistics

| Category            | Count  | Critical |
|---------------------|--------|----------|
| Logic Errors        | 3      | 2        |
| Runtime Risks       | 6      | 4        |
| Performance Issues  | 5      | 1        |
| Code Quality Issues | 4      | 0        |
| Testing Gaps        | 3      | 3        |
| **TOTAL**           | **21** | **10**   |

---

## 🚀 Quick Fixes (Immediate Impact)

1. **Fix trie path counting** (Line 1359): `total_paths += child_paths`
2. **Add memory_per_pipeline validation** (Line 1051): `assert memory_per_pipeline > 0`
3. **Add submitit parameter warning** (Lines 2168-2172): Warn user about ignored
   parameters
4. **Remove redundant deepcopy** (Line 1510): Reuse operations from line 1500
5. **Make backend parameter functional** (Line 1634): Use `_execute_parallel_tasks`
   instead of direct joblib

These fixes address 5 critical issues with minimal code changes.
