# Insert Helion Configs Script

## Overview

This script automates the process of extracting unique configurations from benchmark CSV files and inserting them into the helion attention code.

## Usage

```bash
python scripts/insert_helion_configs.py <csv_file_path> <target_file_path>
```

### Parameters

- `csv_file_path`: Path to the CSV file containing benchmark results with configuration snippets
- `target_file_path`: Path to the helion attention Python file to update (usually `vllm/vllm/attention/ops/helion_unified_attention.py`)

## Example

```bash
python scripts/insert_helion_configs.py \
    /home/zrlngl/watsonx/zrl-triton-results-and-notebooks/micro_benchmarks/raw_data/NVIDIA_H100_80GB_HBM3/2026-03-17_07-37-42/test_prefix_vllm_v1_attention_final.csv \
    vllm/vllm/attention/ops/helion_unified_attention.py
```

## What It Does

1. **Extracts Configurations**: Uses `scripts/extract_unique_snippets.py` to extract unique `helion.Config()` configurations from the CSV file
2. **Groups by Block Sizes**: Organizes configurations by their `block_sizes` parameter for better readability
3. **Updates Target File**: Automatically inserts the configurations into the helion attention file, replacing any existing configuration list

## Output

The script will:
- Create a `nv_configs` list with all unique configurations
- Add comments indicating the block_sizes and count for each group
- Set `configs = nv_configs if torch.version.cuda else []`
- Enable the `configs` parameter in the `@helion.kernel` decorator

## Configuration Format

Each configuration includes parameters such as:
- `block_sizes`: Tile sizes for computation
- `grid_fissions`: Grid splitting strategy
- `indexing`: Memory access patterns
- `l2_groupings`: L2 cache grouping
- `load_eviction_policies`: Cache eviction policies
- `loop_orders`: Loop ordering strategy
- `num_stages`: Pipeline stages
- `num_warps`: Number of warps
- `pid_type`: Program ID type
- And more...

## Dependencies

- Python 3.6+
- `scripts/extract_unique_snippets.py` must be present in the scripts directory

## Notes

- The script preserves the structure of the target file
- Configurations are automatically sorted by block_sizes for consistency
- The script will report the total number of configurations inserted
- Make sure to backup your target file before running if needed

## Troubleshooting

If the script reports "No changes made. Pattern not found in file", ensure:
1. The target file contains the expected pattern: `# configs = nv_configs if torch.version.cuda else amd_configs`
2. The file has the `@helion.kernel(` decorator
3. The file path is correct

## Related Scripts

- `scripts/extract_unique_snippets.py`: Extracts and displays unique configuration snippets from files