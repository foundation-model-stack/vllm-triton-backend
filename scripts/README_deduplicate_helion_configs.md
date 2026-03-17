# Deduplicate Helion Configs Script

## Overview

This script removes duplicate helion configurations based on `block_sizes`, `grid_fissions`, and `pid_type` parameters, keeping only the best configuration from each duplicate group.

## Usage

```bash
python scripts/deduplicate_helion_configs.py <target_file_path>
```

### Parameters

- `target_file_path`: Path to the helion attention Python file containing configurations (usually `vllm/vllm/attention/ops/helion_unified_attention.py`)

## Example

```bash
python scripts/deduplicate_helion_configs.py vllm/vllm/attention/ops/helion_unified_attention.py
```

## Duplicate Detection Criteria

Configurations are considered duplicates if they have identical values for:
1. **block_sizes** - Tile sizes for computation
2. **grid_fissions** - Grid splitting strategy  
3. **pid_type** - Program ID type (e.g., 'flat', 'persistent_interleaved', 'persistent_blocked')

## Selection Strategy

When multiple configurations match the duplicate criteria, the script selects the best one using the following priority order:

### 1. Higher `num_warps` (Primary)
- **Rationale**: More warps = better parallelism and GPU utilization
- **Example**: Config with `num_warps=8` is preferred over `num_warps=4`

### 2. Higher `num_stages` (Secondary)
- **Rationale**: More pipeline stages = better pipeline efficiency and latency hiding
- **Example**: Config with `num_stages=7` is preferred over `num_stages=3`

### 3. Simpler `loop_orders` (Tertiary)
- **Rationale**: Lower dimensions first typically leads to better memory access patterns
- **Preference order** (best to worst):
  1. `[0, 1, 2]` - Sequential, best cache locality
  2. `[0, 2, 1]` - Middle dimension swapped
  3. `[1, 0, 2]` - First two swapped
  4. `[1, 2, 0]` - Rotated once
  5. `[2, 0, 1]` - Rotated twice
  6. `[2, 1, 0]` - Fully reversed, worst cache locality

### 4. Original Index (Tie-breaker)
- If all above criteria are equal, keeps the first occurrence

## Example Results

From the initial 39 configurations, the script identified 7 duplicate groups and reduced the total to 24 unique configurations:

### Duplicate Group Example

**Group**: `block_sizes=[1, 2], grid_fissions=[1], pid_type='flat'`

Found 4 configs:
- Config #1: `num_warps=4, num_stages=1, loop_orders=[[1, 0, 2]]` ✗ removed
- Config #2: `num_warps=4, num_stages=1, loop_orders=[[1, 0, 2]]` ✗ removed  
- Config #3: `num_warps=4, num_stages=1, loop_orders=[[1, 0, 2]]` ✗ removed
- Config #5: `num_warps=8, num_stages=1, loop_orders=[[1, 0, 2]]` ✓ **SELECTED**

**Selection reason**: Highest `num_warps` (8 vs 4)

## Output

The script provides:
1. **Detailed statistics** showing total configs, unique configs, and removed configs
2. **Duplicate group analysis** with selection reasoning for each group
3. **Updated file** with deduplicated configurations maintaining proper formatting and comments

## Benefits

- **Reduced autotuning time**: Fewer configurations to evaluate
- **Better performance**: Keeps configurations with better parallelism and pipeline efficiency
- **Cleaner codebase**: Removes redundant configurations
- **Maintains diversity**: Keeps unique configurations across different block sizes and strategies

## Integration with Other Scripts

This script works well with:
- `scripts/insert_helion_configs.py` - Extract and insert configs from benchmark results
- `scripts/extract_unique_snippets.py` - Extract configuration snippets from files

## Typical Workflow

1. Run benchmarks and generate CSV results
2. Use `insert_helion_configs.py` to extract and insert all configurations
3. Use `deduplicate_helion_configs.py` to remove duplicates and keep best configs
4. Configurations are now optimized for autotuning

## Notes

- The script preserves the file structure and comments
- Block size groupings are maintained in the output
- The original file is overwritten - consider backing up if needed
- Selection strategy is based on general performance principles but may need adjustment for specific workloads