#!/usr/bin/env python3
"""
Script to extract unique snippets from a text file.
Snippets start with "One can hardcode" and end with "--" or EOF.
Extracts Python dict configs and groups them by block_sizes.

Created by: Bob (AI Assistant)
Date: 2026-01-27
"""

import sys
import subprocess
import ast
import re
from typing import List, Set, Dict, Any
from collections import defaultdict
from pprint import pformat


def run_grep_and_get_output(file_path: str) -> str:
    """
    Run grep -C2 hardcode on the file and return the output.
    
    Args:
        file_path: Path to the text file to grep
        
    Returns:
        The grep output as a string
    """
    print("=" * 80)
    print("Running: grep -C2 'hardcode' on the file")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            ['grep', '-C2', 'hardcode', file_path],
            capture_output=True,
            text=True,
            check=False  # Don't raise exception if grep finds no matches
        )
        
        if result.returncode == 0:
            # print("Grep output:")
            # print(result.stdout)
            # print("=" * 80)
            # print()
            return result.stdout
        elif result.returncode == 1:
            print("No matches found for 'hardcode' in the file.")
            print("=" * 80)
            print()
            return ""
        else:
            print(f"grep command failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr}", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print("Error: 'grep' command not found. Make sure grep is installed.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error running grep: {e}", file=sys.stderr)
        sys.exit(1)


def extract_snippets_from_content(content: str) -> List[str]:
    """
    Extract all snippets from content that start with "One can hardcode"
    and end with "--" or end of content.
    
    Args:
        content: The text content to process (from grep output)
        
    Returns:
        List of extracted snippets
    """
    snippets = []
    
    if not content:
        return snippets
    
    # Split content into lines for processing
    lines = content.split('\n')
    
    current_snippet = []
    in_snippet = False
    
    for line in lines:
        # Check if this line starts a new snippet
        if "One can hardcode" in line:
            # If we were already in a snippet, save it first
            if in_snippet and current_snippet:
                snippets.append('\n'.join(current_snippet))
            
            # Start new snippet
            current_snippet = [line]
            in_snippet = True
        elif in_snippet:
            # Check if this line ends the snippet
            if line.strip() == "--":
                current_snippet.append(line)
                snippets.append('\n'.join(current_snippet))
                current_snippet = []
                in_snippet = False
            else:
                # Continue collecting lines for current snippet
                current_snippet.append(line)
    
    # If we're still in a snippet at EOF, save it
    if in_snippet and current_snippet:
        snippets.append('\n'.join(current_snippet))
    
    return snippets


def get_unique_snippets(snippets: List[str]) -> List[str]:
    """
    Filter snippets to return only unique ones.
    
    Args:
        snippets: List of all extracted snippets
        
    Returns:
        List of unique snippets
    """
    seen: Set[str] = set()
    unique_snippets = []
    
    for snippet in snippets:
        # Normalize whitespace for comparison
        normalized = snippet.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_snippets.append(snippet)
    
    return unique_snippets


def extract_dict_from_snippet(snippet: str) -> Dict[str, Any]:
    """
    Extract Python dict from snippet between '@helion.kernel(config=' and ', static_shapes='.
    
    Args:
        snippet: The snippet text to parse
        
    Returns:
        Parsed Python dict, or empty dict if not found or invalid
    """
    pattern = r'@helion\.kernel\(config=helion\.Config\((.+?)\),\s*static_shapes='
    match = re.search(pattern, snippet, re.DOTALL)
    
    if not match:
        return {}
    
    config_str = match.group(1).strip()
    
    # Convert key=value syntax to 'key': value for dict literal
    # Match word characters followed by = (but not ==, !=, etc.)
    dict_str = re.sub(r'(\w+)=', r'"\1":', config_str)
    
    # Wrap in braces to make it a dict
    dict_str = '{' + dict_str + '}'
    
    try:
        # Use ast.literal_eval to safely parse Python dict literal
        config = ast.literal_eval(dict_str)
        return config
    except (ValueError, SyntaxError) as e:
        print(f"Warning: Failed to parse dict: {e}", file=sys.stderr)
        print(f"Original string: {config_str[:200]}...", file=sys.stderr)
        print(f"Converted string: {dict_str[:200]}...", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"Warning: Unexpected error parsing config: {e}", file=sys.stderr)
        print(f"Original string: {config_str[:200]}...", file=sys.stderr)
        return {}


def get_grouping_key(block_sizes: Any) -> str:
    """
    Generate a grouping key from block_sizes.
    If block_sizes has exactly 3 entries and the 2nd value >= 1st value,
    ignore the 2nd value in grouping (treat as wildcard).
    
    Args:
        block_sizes: The block_sizes value (list, tuple, etc.)
        
    Returns:
        String key for grouping
    """
    if isinstance(block_sizes, (list, tuple)) and len(block_sizes) == 3:
        # Extract the three values
        val1, val2, val3 = block_sizes[0], block_sizes[1], block_sizes[2]
        
        # If 2nd value >= 1st value, ignore 2nd value in grouping
        if val2 >= val1:
            return f"[{val1}, *, {val3}]"
        else:
            return f"[{val1}, {val2}, {val3}]"
    else:
        # For other cases, use the full representation
        return str(block_sizes)


def group_configs_by_block_sizes(snippets: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract Python dict configs from snippets and group them by block_sizes attribute.
    Uses advanced grouping: if block_sizes has 3 entries and 2nd >= 1st, ignore 2nd in grouping.
    
    Args:
        snippets: List of unique snippets
        
    Returns:
        Dictionary mapping block_sizes grouping key to list of configs
    """
    grouped = defaultdict(list)
    
    for snippet in snippets:
        config = extract_dict_from_snippet(snippet)
        
        if not config:
            continue
        
        # Get block_sizes attribute
        block_sizes = config.get('block_sizes')
        
        if block_sizes is not None:
            # Generate grouping key with special logic
            grouping_key = get_grouping_key(block_sizes)
            grouped[grouping_key].append(config)
    
    return dict(grouped)


def dict_to_key_value_format(config: Dict[str, Any]) -> str:
    """
    Convert a dict to key=value format string (Python function call style).
    
    Args:
        config: Dictionary to convert
        
    Returns:
        String in key=value format
    """
    items = []
    for key, value in config.items():
        # Format the value appropriately
        if isinstance(value, str):
            value_str = f"'{value}'"
        elif isinstance(value, (list, tuple)):
            value_str = str(value)
        else:
            value_str = str(value)
        
        items.append(f"{key}={value_str}")
    
    return ", ".join(items)


def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python extract_unique_snippets.py <file_path>")
        print("Example: python extract_unique_snippets.py input.txt")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Run grep and get output
    grep_output = run_grep_and_get_output(file_path)
    
    # Extract all snippets from grep output
    all_snippets = extract_snippets_from_content(grep_output)
    
    if not all_snippets:
        print("No snippets found in the file.")
        return
    
    # Get unique snippets
    unique_snippets = get_unique_snippets(all_snippets)
    
    # Print snippet summary
    print(f"Found {len(all_snippets)} total snippet(s)")
    print(f"Found {len(unique_snippets)} unique snippet(s)\n")
    print("=" * 80)
    
    # Extract and group configs by block_sizes
    grouped_configs = group_configs_by_block_sizes(unique_snippets)
    
    if not grouped_configs:
        print("\nNo valid configs found in snippets.")
        return
    
    # Print grouped results
    print(f"\n### Configs Grouped by block_sizes ###")
    print(f"Found {len(grouped_configs)} different block_sizes value(s)\n")
    
    # Sort by extracting the first value from block_sizes key
    def get_sort_key(item):
        key = item[0]
        # Extract first number from patterns like "[16, *, 4]" or "[16, 8, 4]"
        import re
        match = re.search(r'\[(\d+)', key)
        if match:
            return int(match.group(1))
        return 0
    
    for block_sizes_key, configs in sorted(grouped_configs.items(), key=get_sort_key):
        print("=" * 80)
        print(f"\nblock_sizes: {block_sizes_key}")
        print(f"Number of configs: {len(configs)}\n")
        
        for i, config in enumerate(configs, 1):
            print(f"Config {i}:")
            print(f"helion.Config({dict_to_key_value_format(config)}),")
            print()
        
        print("=" * 80)


if __name__ == "__main__":
    main()