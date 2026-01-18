# Examples Directory

This directory contains example scripts and benchmark utilities for JC-SIM-OQP.

## Files

### Example Simulations

- **demo.py** - Basic demonstration of the simulator
- **simulation_suite.py** - Comprehensive simulation examples
- **tavis_cummings.py** - Tavis-Cummings model examples
- **benchmark_scaling.py** - Original benchmark script (now superseded by benchmark_runner.py)

### Benchmark Runner (New)

- **benchmark_runner.py** - Modular benchmark library for SLURM parallel execution

## Benchmark Runner

The `benchmark_runner.py` module provides a clean, modular interface for running benchmarks either standalone or via SLURM job arrays.

### Features

- **Modular design**: Clean separation of benchmark logic from execution environment
- **CLI interface**: Run benchmarks from command line with simple syntax
- **Python API**: Import and use in custom scripts
- **SLURM-ready**: Designed for efficient parallel execution
- **Reproducible**: Fixed seeding for deterministic results

### Usage

#### Command Line

```bash
# Error convergence benchmark
uv run python benchmark_runner.py error:1000 output_dir/

# Timing benchmark
uv run python benchmark_runner.py time:5 output_dir/

# With task ID (for SLURM)
uv run python benchmark_runner.py error:1000 output_dir/ --task-id 42
```

#### Python API

```python
from benchmark_runner import run_error_benchmark, run_time_benchmark
from pathlib import Path

# Error benchmark
results = run_error_benchmark(
    ntraj=1000,
    output_dir=Path("results"),
    seed=42,
)

# Time benchmark
results = run_time_benchmark(
    n_atoms=5,
    output_dir=Path("results"),
    ntraj=100,
)
```

### Output Format

Each benchmark run creates:
- `results.json` - Numerical results and metadata
- `trajectory_data.npz` - Full time series data (error benchmarks only)

Example `results.json` for error benchmark:
```json
{
  "benchmark_type": "error",
  "ntraj": 1000,
  "rmse": 1.23e-4,
  "runtime_stochastic": 12.34,
  "runtime_exact": 5.67,
  "seed": 42,
  "params": {
    "n_atoms": 1,
    "t_max": 20.0,
    "n_steps": 200,
    "N": 10
  }
}
```

### Integration with SLURM

The benchmark runner is designed to work seamlessly with SLURM job arrays:

```bash
# In your SLURM script
uv run python examples/benchmark_runner.py \
    "${PARAM_STRING}" \
    "${OUTPUT_DIR}" \
    --task-id "${SLURM_ARRAY_TASK_ID}"
```

**Complete documentation**: See [docs/benchmarks.md](../docs/benchmarks.md) for full SLURM setup guide and [docs/benchmark_system.md](../docs/benchmark_system.md) for architecture details.

### Testing

Before submitting to SLURM, test locally:

```bash
# From repository root
uv run test_benchmark_runner.py
```

This validates that:
- Module imports work correctly
- Parameter parsing functions properly
- Benchmarks run and produce expected output files
- Results are saved in the correct format

## Legacy Files

- **benchmark_scaling.py** - Original monolithic benchmark script. Superseded by the modular `benchmark_runner.py` but kept for reference.
