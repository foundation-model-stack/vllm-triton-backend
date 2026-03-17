#!/usr/bin/env python3
"""
Script to extract unique configurations from a CSV file and insert them into helion attention code.

Usage:
    python scripts/insert_helion_configs.py <csv_file_path> <target_file_path>

Example:
    python scripts/insert_helion_configs.py \
        /home/zrlngl/watsonx/zrl-triton-results-and-notebooks/micro_benchmarks/raw_data/NVIDIA_H100_80GB_HBM3/2026-03-17_07-37-42/test_prefix_vllm_v1_attention_final.csv \
        vllm/vllm/attention/ops/helion_unified_attention.py

Created by: Bob (AI Assistant)
Date: 2026-03-17
"""

import sys
import subprocess
import re
from typing import List, Dict, Any
from collections import defaultdict


def run_extract_snippets(csv_file: str) -> str:
    """
    Run the extract_unique_snippets.py script and capture output.
    
    Args:
        csv_file: Path to the CSV file
        
    Returns:
        The script output as a string
    """
    print(f"Extracting configurations from: {csv_file}")
    result = subprocess.run(
        ['python', 'scripts/extract_unique_snippets.py', csv_file],
        capture_output=True,
        text=True,
        check=True
    )
    return result.stdout


def parse_configs_from_output(output: str) -> List[str]:
    """
    Parse helion.Config() statements from the script output.
    
    Args:
        output: The output from extract_unique_snippets.py
        
    Returns:
        List of config strings
    """
    configs = []
    lines = output.split('\n')
    
    for line in lines:
        if line.strip().startswith('helion.Config('):
            configs.append('    ' + line.strip())
    
    return configs


def group_configs_by_block_sizes(output: str) -> Dict[str, List[str]]:
    """
    Parse and group configs by block_sizes from the script output.
    
    Args:
        output: The output from extract_unique_snippets.py
        
    Returns:
        Dictionary mapping block_sizes to list of config strings
    """
    grouped = defaultdict(list)
    current_block_sizes = None
    lines = output.split('\n')
    
    for line in lines:
        # Check for block_sizes header
        if line.startswith('block_sizes:'):
            current_block_sizes = line.split(':', 1)[1].strip()
        elif line.strip().startswith('helion.Config(') and current_block_sizes:
            grouped[current_block_sizes].append('    ' + line.strip())
    
    return dict(grouped)


def generate_config_list(output: str) -> str:
    """
    Generate the full config list with comments.
    
    Args:
        output: The output from extract_unique_snippets.py
        
    Returns:
        String containing the full config list
    """
    grouped = group_configs_by_block_sizes(output)
    
    config_lines = ['nv_configs = [']
    
    # Sort by extracting first number from block_sizes
    def get_sort_key(item):
        key = item[0]
        match = re.search(r'\[(\d+)', key)
        if match:
            return int(match.group(1))
        return 0
    
    for block_sizes, configs in sorted(grouped.items(), key=get_sort_key):
        config_lines.append(f'    # block_sizes: {block_sizes} - {len(configs)} config{"s" if len(configs) != 1 else ""}')
        for config in configs:
            config_lines.append(config)
        config_lines.append('')
    
    # Remove last empty line and close the list
    if config_lines[-1] == '':
        config_lines.pop()
    config_lines.append(']')
    
    return '\n'.join(config_lines)


def update_helion_file(target_file: str, config_list: str):
    """
    Update the helion attention file with new configs.
    
    Args:
        target_file: Path to the helion attention file
        config_list: The generated config list string
    """
    print(f"Reading target file: {target_file}")
    with open(target_file, 'r') as f:
        content = f.read()
    
    # Find the section to replace
    # Look for the pattern: # configs = ... followed by @helion.kernel
    pattern = r'(# configs = nv_configs if torch\.version\.cuda else amd_configs\n\n)(.*?)(@helion\.kernel\()'
    
    replacement = r'\1' + config_list + '\n\nconfigs = nv_configs if torch.version.cuda else []\n\n' + r'\3'
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content == content:
        print("Warning: No changes made. Pattern not found in file.")
        return False
    
    print(f"Writing updated content to: {target_file}")
    with open(target_file, 'w') as f:
        f.write(new_content)
    
    return True


def main():
    """Main function to run the script."""
    if len(sys.argv) != 3:
        print("Usage: python scripts/insert_helion_configs.py <csv_file_path> <target_file_path>")
        print("\nExample:")
        print("  python scripts/insert_helion_configs.py \\")
        print("    /path/to/test_prefix_vllm_v1_attention_final.csv \\")
        print("    vllm/vllm/attention/ops/helion_unified_attention.py")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    target_file = sys.argv[2]
    
    # Step 1: Extract configurations
    output = run_extract_snippets(csv_file)
    
    # Step 2: Generate config list
    config_list = generate_config_list(output)
    
    # Step 3: Update target file
    success = update_helion_file(target_file, config_list)
    
    if success:
        print("\n" + "=" * 80)
        print("SUCCESS: Configurations have been inserted into the helion attention file.")
        print("=" * 80)
        
        # Count configs
        num_configs = config_list.count('helion.Config(')
        print(f"\nTotal configurations inserted: {num_configs}")
    else:
        print("\n" + "=" * 80)
        print("ERROR: Failed to update the file. Please check the file format.")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
