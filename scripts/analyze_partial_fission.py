"""Analyze generated Triton code for different grid fission configurations."""
from __future__ import annotations

import torch

import helion
from helion._testing import DEVICE, code_and_output
import helion.language as hl


@helion.kernel(autotune_effort="none")
def add_2d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = x.new_empty(x.size())
    for tile_m, tile_n in hl.tile([x.size(0), x.size(1)]):
        result[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n]
    return result


@helion.kernel(autotune_effort="none")
def add_3d_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    result = x.new_empty(x.size())
    for tile_a, tile_b, tile_c in hl.tile([x.size(0), x.size(1), x.size(2)]):
        result[tile_a, tile_b, tile_c] = (
            x[tile_a, tile_b, tile_c] + y[tile_a, tile_b, tile_c]
        )
    return result


def main():
    x2d = torch.randn(64, 64, device=DEVICE)
    y2d = torch.randn(64, 64, device=DEVICE)

    x3d = torch.randn(8, 64, 128, device=DEVICE)
    y3d = torch.randn(8, 64, 128, device=DEVICE)

    configs = [
        ("2D no fission", add_2d_kernel, (x2d, y2d), dict(block_sizes=[32, 32], grid_fissions=[[0, 0]])),
        ("2D full fission last dim", add_2d_kernel, (x2d, y2d), dict(block_sizes=[32, 32], grid_fissions=[[0, -1]])),
        ("2D partial fission last dim (factor=2)", add_2d_kernel, (x2d, y2d), dict(block_sizes=[32, 32], grid_fissions=[[0, 2]])),
        ("3D partial fission last dim (factor=4)", add_3d_kernel, (x3d, y3d), dict(block_sizes=[8, 16, 32], grid_fissions=[[0, 0, 4]])),
        ("3D mixed: partial on dim1, full on dim2", add_3d_kernel, (x3d, y3d), dict(block_sizes=[8, 16, 32], grid_fissions=[[0, 2, -1]])),
    ]

    for label, kernel_fn, args, config_kwargs in configs:
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"  Config: {config_kwargs}")
        print(f"{'='*80}\n")
        code, result = code_and_output(kernel_fn, args, **config_kwargs)
        print(code)

        # Verify correctness
        expected = args[0] + args[1]
        torch.testing.assert_close(result, expected)
        print(f"\n  -> Correctness: PASSED")


if __name__ == "__main__":
    main()
