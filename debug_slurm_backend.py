#!/usr/bin/env python3
"""
Debug script to validate SLURM backend selection in PhenoTypic CLI.

This script tests whether the CLI correctly selects the SLURM backend
when --slurm-args are provided.

Usage:
    python debug_slurm_backend.py

The script will:
1. Test ExecutionConfig.is_slurm_mode() logic
2. Test create_execution_strategy() backend selection
3. Show what would happen with your specific SLURM arguments
"""

from pathlib import Path
from phenotypic._cli._cli_types import ExecutionConfig
from phenotypic._cli._cli_execution_strategies import (
    create_execution_strategy,
    AutonomousSLURMStrategy,
    LocalParallelStrategy,
)
from phenotypic._cli._cli_output_manager import OutputManager


def test_slurm_backend_selection():
    """Test SLURM backend selection logic."""
    print("=" * 70)
    print("PhenoTypic SLURM Backend Selection Validator")
    print("=" * 70)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Test 1: Local mode (no SLURM args)
        print("\n[Test 1] Local mode (no SLURM args)")
        print("-" * 70)
        config_local = ExecutionConfig(
            pipeline_json=Path("pipeline.json"),
            input_path=Path("."),
            output_dir=tmpdir,
            image_type="GridImage",
            nrows=8,
            ncols=12,
            bit_depth=None,
            n_jobs=-1,
            slurm_args={},  # Empty = local mode
            force_local=False,
            wait=False,
            save_rgb=False,
            save_gray=False,
            save_enh_gray=False,
            save_objmask=False,
            save_objmap=False,
            save_objmap_rgb=False,
            rgb_ext=".tiff",
            gray_ext=".tiff",
            enh_gray_ext=".tiff",
            objmask_ext=".png",
            objmap_ext=".png",
            objmap_rgb_ext=".png",
            include_dataset_column=False,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

        is_slurm = config_local.is_slurm_mode()
        print(f"  slurm_args: {config_local.slurm_args}")
        print(f"  force_local: {config_local.force_local}")
        print(f"  is_slurm_mode(): {is_slurm}")
        print(f"  Expected: False, Got: {is_slurm}")
        print(f"  ✓ PASS" if not is_slurm else f"  ✗ FAIL")

        manager = OutputManager(
            base_dir=tmpdir,
            save_layers={},
            extensions={},
            include_dataset_column=False,
        )
        strategy = create_execution_strategy(config_local, manager)
        print(f"  Strategy type: {type(strategy).__name__}")
        print(f"  Expected: LocalParallelStrategy, Got: {type(strategy).__name__}")
        print(
            f"  ✓ PASS"
            if isinstance(strategy, LocalParallelStrategy)
            else f"  ✗ FAIL"
        )

        # Test 2: SLURM mode (with SLURM args)
        print("\n[Test 2] SLURM mode (with SLURM args)")
        print("-" * 70)
        config_slurm = ExecutionConfig(
            pipeline_json=Path("pipeline.json"),
            input_path=Path("."),
            output_dir=tmpdir,
            image_type="GridImage",
            nrows=8,
            ncols=12,
            bit_depth=None,
            n_jobs=-1,
            slurm_args={"slurm_partition": "compute", "mem_gb": 16},
            force_local=False,
            wait=False,
            save_rgb=False,
            save_gray=False,
            save_enh_gray=False,
            save_objmask=False,
            save_objmap=False,
            save_objmap_rgb=False,
            rgb_ext=".tiff",
            gray_ext=".tiff",
            enh_gray_ext=".tiff",
            objmask_ext=".png",
            objmap_ext=".png",
            objmap_rgb_ext=".png",
            include_dataset_column=False,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

        is_slurm = config_slurm.is_slurm_mode()
        print(f"  slurm_args: {config_slurm.slurm_args}")
        print(f"  force_local: {config_slurm.force_local}")
        print(f"  is_slurm_mode(): {is_slurm}")
        print(f"  Expected: True, Got: {is_slurm}")
        print(f"  ✓ PASS" if is_slurm else f"  ✗ FAIL")

        strategy = create_execution_strategy(config_slurm, manager)
        print(f"  Strategy type: {type(strategy).__name__}")
        print(f"  Expected: AutonomousSLURMStrategy, Got: {type(strategy).__name__}")
        print(
            f"  ✓ PASS"
            if isinstance(strategy, AutonomousSLURMStrategy)
            else f"  ✗ FAIL"
        )

        # Test 3: force_local overrides SLURM
        print("\n[Test 3] force_local=True overrides SLURM args")
        print("-" * 70)
        config_force_local = ExecutionConfig(
            pipeline_json=Path("pipeline.json"),
            input_path=Path("."),
            output_dir=tmpdir,
            image_type="GridImage",
            nrows=8,
            ncols=12,
            bit_depth=None,
            n_jobs=-1,
            slurm_args={"slurm_partition": "compute"},  # Has SLURM args
            force_local=True,  # But force_local is True
            wait=False,
            save_rgb=False,
            save_gray=False,
            save_enh_gray=False,
            save_objmask=False,
            save_objmap=False,
            save_objmap_rgb=False,
            rgb_ext=".tiff",
            gray_ext=".tiff",
            enh_gray_ext=".tiff",
            objmask_ext=".png",
            objmap_ext=".png",
            objmap_rgb_ext=".png",
            include_dataset_column=False,
            dry_run=False,
            sample=None,
            resume=False,
            retry_failures=False,
            skip_validation=False,
        )

        is_slurm = config_force_local.is_slurm_mode()
        print(f"  slurm_args: {config_force_local.slurm_args}")
        print(f"  force_local: {config_force_local.force_local}")
        print(f"  is_slurm_mode(): {is_slurm}")
        print(f"  Expected: False (force_local overrides), Got: {is_slurm}")
        print(f"  ✓ PASS" if not is_slurm else f"  ✗ FAIL")

        strategy = create_execution_strategy(config_force_local, manager)
        print(f"  Strategy type: {type(strategy).__name__}")
        print(
            f"  Expected: LocalParallelStrategy, Got: {type(strategy).__name__}"
        )
        print(
            f"  ✓ PASS"
            if isinstance(strategy, LocalParallelStrategy)
            else f"  ✗ FAIL"
        )

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(
        """
The SLURM backend selection logic works as follows:

1. If --slurm-args are provided (non-empty dict), use SLURM mode
2. If --force-local is used, use local mode regardless of --slurm-args
3. If neither, use local mode (default)

To use SLURM backend when deploying on an HPCC:
  - Ensure you are passing --slurm-args with at least one parameter
  - Example: --slurm-args slurm_partition=compute --slurm-args mem_gb=16
  - Do NOT pass --force-local flag
  - Check the CLI output - it should display "Backend: SLURM Cluster"

For debugging:
  - Run this script: python debug_slurm_backend.py
  - Check CLI output with --dry-run to preview without executing
  - Example: python -m phenotypic pipeline.json ./images \\
      --slurm-args slurm_partition=compute \\
      --dry-run
    """
    )


if __name__ == "__main__":
    test_slurm_backend_selection()
