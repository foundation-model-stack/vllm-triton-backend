#!/usr/bin/env python3
"""
Script to deduplicate helion configurations based on block_sizes, grid_fissions, and pid_type.

Selection strategy for duplicates:
1. Prefer higher num_warps (more parallelism)
2. Prefer higher num_stages (better pipeline efficiency)
3. Prefer simpler loop_orders (lower dimensions first)

Usage:
    python scripts/deduplicate_helion_configs.py <target_file_path>

Example:
    python scripts/deduplicate_helion_configs.py vllm/vllm/attention/ops/helion_unified_attention.py

Created by: Bob (AI Assistant)
Date: 2026-03-17
"""

import sys
import re
from typing import List, Dict, Tuple
from collections import defaultdict


def parse_config(config_str: str, index: int) -> Dict:
    """Parse a config string and extract key parameters."""
    # Extract all parameters
    block_sizes = re.search(r'block_sizes=(\[[^\]]+\])', config_str)
    grid_fissions = re.search(r'grid_fissions=(\[[^\]]+\])', config_str)
    pid_type = re.search(r"pid_type='([^']+)'", config_str)
    num_warps = re.search(r'num_warps=(\d+)', config_str)
    num_stages = re.search(r'num_stages=(\d+)', config_str)
    loop_orders = re.search(r'loop_orders=(\[\[([^\]]+)\]\])', config_str)
    
    # Parse loop order for ranking
    loop_order_list = None
    if loop_orders:
        loop_order_str = loop_orders.group(2)
        loop_order_list = [int(x.strip()) for x in loop_order_str.split(',')]
    
    return {
        'index': index,
        'config_str': config_str,
        'block_sizes': block_sizes.group(1) if block_sizes else None,
        'grid_fissions': grid_fissions.group(1) if grid_fissions else None,
        'pid_type': pid_type.group(1) if pid_type else None,
        'num_warps': int(num_warps.group(1)) if num_warps else 0,
        'num_stages': int(num_stages.group(1)) if num_stages else 0,
        'loop_orders': loop_orders.group(1) if loop_orders else None,
        'loop_order_list': loop_order_list,
    }


def rank_loop_order(loop_order_list: List[int]) -> int:
    """
    Rank loop order by simplicity (lower dimensions first).
    Lower rank = better (simpler).
    """
    if not loop_order_list:
        return 999
    
    # Define preference order
    order_preference = {
        (0, 1, 2): 0,
        (0, 2, 1): 1,
        (1, 0, 2): 2,
        (1, 2, 0): 3,
        (2, 0, 1): 4,
        (2, 1, 0): 5,
    }
    
    return order_preference.get(tuple(loop_order_list), 999)


def select_best_config(configs: List[Dict]) -> Dict:
    """
    Select the best config from a group of duplicates.
    
    Selection criteria (in order):
    1. Higher num_warps (more parallelism)
    2. Higher num_stages (better pipeline)
    3. Simpler loop_orders (lower dimensions first)
    """
    if len(configs) == 1:
        return configs[0]
    
    # Sort by: num_warps (desc), num_stages (desc), loop_order rank (asc)
    sorted_configs = sorted(
        configs,
        key=lambda c: (
            -c['num_warps'],  # Higher is better
            -c['num_stages'],  # Higher is better
            rank_loop_order(c['loop_order_list']),  # Lower rank is better
            c['index']  # Tie-breaker: keep first occurrence
        )
    )
    
    return sorted_configs[0]


def deduplicate_configs(config_strings: List[str]) -> Tuple[List[str], Dict]:
    """
    Deduplicate configs and return selected ones with statistics.
    
    Returns:
        Tuple of (selected_config_strings, statistics_dict)
    """
    # Parse all configs
    parsed_configs = []
    for i, config_str in enumerate(config_strings, 1):
        parsed = parse_config(config_str, i)
        if parsed['block_sizes'] and parsed['grid_fissions'] and parsed['pid_type']:
            parsed_configs.append(parsed)
    
    # Group by (block_sizes, grid_fissions, pid_type)
    groups = defaultdict(list)
    for config in parsed_configs:
        key = (config['block_sizes'], config['grid_fissions'], config['pid_type'])
        groups[key].append(config)
    
    # Select best from each group
    selected_configs = []
    duplicate_groups = []
    
    for key, configs_in_group in groups.items():
        best_config = select_best_config(configs_in_group)
        selected_configs.append(best_config)
        
        if len(configs_in_group) > 1:
            duplicate_groups.append({
                'key': key,
                'configs': configs_in_group,
                'selected': best_config,
            })
    
    # Sort selected configs by original index to maintain order
    selected_configs.sort(key=lambda c: c['index'])
    
    # Prepare statistics
    stats = {
        'total_configs': len(parsed_configs),
        'unique_configs': len(selected_configs),
        'removed_configs': len(parsed_configs) - len(selected_configs),
        'duplicate_groups': duplicate_groups,
    }
    
    return [c['config_str'] for c in selected_configs], stats


def update_file_with_deduplicated_configs(file_path: str, selected_configs: List[str]):
    """Update the file with deduplicated configs."""
    # Read file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find the config list section
    start_idx = None
    end_idx = None
    
    for i, line in enumerate(lines):
        if 'nv_configs = [' in line:
            start_idx = i
        elif start_idx is not None and line.strip() == ']':
            end_idx = i
            break
    
    if start_idx is None or end_idx is None:
        print("Error: Could not find config list in file")
        return False
    
    # Build new config section
    new_lines = ['nv_configs = [\n']
    
    # Group configs by block_sizes for comments
    current_block_sizes = None
    config_count = 0
    
    for config_str in selected_configs:
        # Extract block_sizes for grouping
        match = re.search(r'block_sizes=(\[[^\]]+\])', config_str)
        if match:
            block_sizes = match.group(1)
            if block_sizes != current_block_sizes:
                if current_block_sizes is not None:
                    new_lines.append('    \n')
                current_block_sizes = block_sizes
                config_count = 1
                # Count total configs with this block_sizes
                total_with_bs = sum(1 for c in selected_configs if f'block_sizes={block_sizes}' in c)
                new_lines.append(f'    # block_sizes: {block_sizes} - {total_with_bs} config{"s" if total_with_bs != 1 else ""}\n')
            else:
                config_count += 1
        
        new_lines.append(f'    helion.Config({config_str}),\n')
    
    new_lines.append(']\n')
    
    # Replace old config section with new one
    new_file_lines = lines[:start_idx] + new_lines + lines[end_idx+1:]
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(new_file_lines)
    
    return True


def print_statistics(stats: Dict):
    """Print deduplication statistics."""
    print("=" * 100)
    print("DEDUPLICATION STATISTICS")
    print("=" * 100)
    print(f"Total configs: {stats['total_configs']}")
    print(f"Unique configs: {stats['unique_configs']}")
    print(f"Removed configs: {stats['removed_configs']}")
    print()
    
    if stats['duplicate_groups']:
        print("=" * 100)
        print("DUPLICATE GROUPS AND SELECTIONS")
        print("=" * 100)
        print()
        
        for i, group in enumerate(stats['duplicate_groups'], 1):
            block_sizes, grid_fissions, pid_type = group['key']
            print(f"Group {i}: block_sizes={block_sizes}, grid_fissions={grid_fissions}, pid_type='{pid_type}'")
            print(f"  Found {len(group['configs'])} configs, selected 1")
            print()
            
            for config in group['configs']:
                selected_marker = " ✓ SELECTED" if config['index'] == group['selected']['index'] else " ✗ removed"
                print(f"  Config #{config['index']}{selected_marker}")
                print(f"    - num_warps: {config['num_warps']}")
                print(f"    - num_stages: {config['num_stages']}")
                print(f"    - loop_orders: {config['loop_orders']}")
            print()
            
            # Explain selection
            selected = group['selected']
            print(f"  Selection reason:")
            if selected['num_warps'] == max(c['num_warps'] for c in group['configs']):
                print(f"    - Highest num_warps ({selected['num_warps']})")
            if selected['num_stages'] == max(c['num_stages'] for c in group['configs']):
                print(f"    - Highest num_stages ({selected['num_stages']})")
            print(f"    - Loop order rank: {rank_loop_order(selected['loop_order_list'])}")
            print()
            print("-" * 100)
            print()


def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/deduplicate_helion_configs.py <target_file_path>")
        print("\nExample:")
        print("  python scripts/deduplicate_helion_configs.py vllm/vllm/attention/ops/helion_unified_attention.py")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print(f"Reading file: {file_path}")
    
    # Read file and extract configs
    with open(file_path, 'r') as f:
        content = f.read()
    
    config_pattern = r'helion\.Config\((.*?)\),'
    config_strings = re.findall(config_pattern, content, re.DOTALL)
    
    if not config_strings:
        print("Error: No configs found in file")
        sys.exit(1)
    
    print(f"Found {len(config_strings)} configs")
    print()
    
    # Deduplicate
    selected_configs, stats = deduplicate_configs(config_strings)
    
    # Print statistics
    print_statistics(stats)
    
    # Update file
    print("=" * 100)
    print("Updating file with deduplicated configs...")
    success = update_file_with_deduplicated_configs(file_path, selected_configs)
    
    if success:
        print(f"✓ Successfully updated {file_path}")
        print(f"✓ Reduced from {stats['total_configs']} to {stats['unique_configs']} configs")
    else:
        print("✗ Failed to update file")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
