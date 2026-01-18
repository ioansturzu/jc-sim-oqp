#!/usr/bin/env python3
"""
Test script for benchmark_runner module.

This script validates that the benchmark runner works correctly before
submitting to SLURM.
"""

import sys
from pathlib import Path

# Add the examples and src directories to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root / 'src'))
sys.path.insert(0, str(repo_root / 'examples'))

from benchmark_runner import (
    parse_benchmark_params,
    run_benchmark_from_params,
)


def test_parse_params():
    """Test parameter parsing."""
    print("Testing parameter parsing...")
    
    # Test valid inputs
    test_cases = [
        ("error:100", ("error", 100)),
        ("time:5", ("time", 5)),
        ("error:1000", ("error", 1000)),
    ]
    
    for param_str, expected in test_cases:
        result = parse_benchmark_params(param_str)
        assert result == expected, f"Failed for {param_str}: got {result}, expected {expected}"
        print(f"  ✓ {param_str} -> {result}")
    
    # Test invalid inputs
    invalid_cases = ["invalid", "error", "error:abc", "unknown:5"]
    for param_str in invalid_cases:
        try:
            parse_benchmark_params(param_str)
            print(f"  ✗ {param_str} should have raised an error")
        except ValueError:
            print(f"  ✓ {param_str} correctly raised ValueError")
    
    print("Parameter parsing tests passed!\n")


def test_small_benchmark():
    """Run a small benchmark to verify functionality."""
    print("Running small test benchmark...")
    
    output_dir = Path(__file__).parent.parent / "test_results"
    
    # Test error benchmark with minimal parameters
    print("\n  Testing error benchmark (ntraj=10)...")
    try:
        results = run_benchmark_from_params(
            param_string="error:10",
            output_dir=output_dir,
            task_id=999,
        )
        
        print(f"  ✓ Error benchmark completed")
        print(f"    RMSE: {results['rmse']:.6e}")
        print(f"    Runtime: {results['runtime_stochastic']:.2f}s")
        
        # Verify output files exist
        task_dir = output_dir / "task_999_error_10"
        assert (task_dir / "results.json").exists(), "results.json not created"
        assert (task_dir / "trajectory_data.npz").exists(), "trajectory_data.npz not created"
        print(f"  ✓ Output files created in {task_dir}")
        
    except Exception as e:
        print(f"  ✗ Error benchmark failed: {e}")
        raise
    
    # Test time benchmark with minimal parameters
    print("\n  Testing time benchmark (n_atoms=1)...")
    try:
        results = run_benchmark_from_params(
            param_string="time:1",
            output_dir=output_dir,
            task_id=998,
        )
        
        print(f"  ✓ Time benchmark completed")
        print(f"    Exact time: {results['time_exact']:.2f}s")
        print(f"    Stoch time: {results['time_stochastic']:.2f}s")
        
        # Verify output files exist
        task_dir = output_dir / "task_998_time_1"
        assert (task_dir / "results.json").exists(), "results.json not created"
        print(f"  ✓ Output files created in {task_dir}")
        
    except Exception as e:
        print(f"  ✗ Time benchmark failed: {e}")
        raise
    
    print("\nSmall benchmark tests passed!\n")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing benchmark_runner module")
    print("=" * 70 + "\n")
    
    try:
        test_parse_params()
        test_small_benchmark()
        
        print("=" * 70)
        print("All tests passed! ✓")
        print("=" * 70)
        print("\nThe benchmark runner is ready for SLURM deployment.")
        print("You can now submit: sbatch run_benchmark.batch")
        
    except Exception as e:
        print(f"\n{'=' * 70}")
        print(f"Tests failed: {e}")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
