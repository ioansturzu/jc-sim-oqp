# HPC Benchmarking Guide

```{eval-rst}
.. meta::
   :description: SLURM-based parallel benchmark system for JC-SIM-OQP
   :keywords: quantum simulation, benchmarking, SLURM, HPC, scaling
```

## Overview

Parallel benchmark system for JC-SIM-OQP designed to maximize HPC cluster utilization. Runs 120 parameter combinations (error convergence + timing benchmarks) with multiple replicates for statistical robustness.

This system is optimized for production HPC environments and can scale from small workstations to large clusters.

## System Design

**Core Components:**
- `examples/benchmark_runner.py` - Modular benchmark library
- `run_benchmark.batch` - SLURM array job (120 tasks)
- `aggregate_results.py` - Results analysis and plotting
- `test_benchmark_runner.py` - Local validation

**HPC Optimization:**
- 2 CPUs per task → 36 jobs/node on 72-core nodes
- 120 simultaneous jobs on 2-node cluster
- Single-threaded tasks with minimal overhead
- ~5-30 min per task

## Statistical Methodology

The benchmark system implements statistical analysis:

1.  **Replicate Aggregation**: All benchmarks automatically aggregate 5+ independent replicates per parameter point.
2.  **Confidence Intervals**: Convergence plots display the **Mean RMSE** with a shaded **$1\sigma$ Standard Deviation** region.
3.  **Scaling Fits**:
    *   **Time Scaling**: Exponential fits $T(N) \propto C^N$ for Exact solvers vs Linear $T(N) \propto N$ for Stochastic.
    *   **Error Scaling**: Power-law fits $Error \propto N_{traj}^{-0.5}$ to verify the $1/\sqrt{N}$ Monte Carlo convergence.
4.  **Visualization**: Dashboard generation uses `viridis` colormaps for multi-trajectory density analysis.

## Quick Start

```bash
# 1. Test locally (optional but recommended)
uv run python test_benchmark_runner.py

# 2. Submit 120 jobs to SLURM
sbatch run_benchmark.batch

# 3. Monitor progress
squeue -u $USER
watch -n 10 'ls benchmark_results/task_*/results.json | wc -l'

# 4. Analyze results (after completion)
uv run aggregate_results.py
```

**Output:** Plots in `benchmark_results/plots/` showing error convergence and timing scaling.

## Configuration

### Cluster Resources

Current settings optimized for 2-node × 72-core cluster:

```bash
#SBATCH --array=0-119              # 120 parallel tasks
#SBATCH --cpus-per-task=2          # 36 jobs/node (72 total)
#SBATCH --time=4-00:00:00          # Max 4 days
```

### Parameter Space

**Error benchmarks** (55 runs): ntraj = 10, 50, 100, 500, 1000, 2000, 3000, 5000, 10000, 20000, 50000 (5 replicates each)

**Time benchmarks** (65 runs): n_atoms = 1-18 (3+ replicates each)

### Customization

Edit `PARAMS` array in `run_benchmark.batch` to add/remove parameter combinations. Ensure array size matches:
```bash
#SBATCH --array=0-N  # N = len(PARAMS) - 1
```

### Cluster Modules

Adjust for your system:
```bash
module load gcc/YOUR_VERSION python/YOUR_VERSION
```

## Benchmark Types

**Error Convergence** - Monte Carlo accuracy vs trajectory count. Expected: Error ∝ 1/√(ntraj)

**Timing Scaling** - Exact vs stochastic solver performance vs system size

## Output Structure

```
benchmark_results/
├── task_0_error_10/
│   ├── results.json
│   └── trajectory_data.npz
├── task_1_error_10/
├── ...
└── plots/
    ├── error_convergence.png
    ├── time_scaling.png
    └── benchmark_dashboard.png
```

## Advanced Usage

**Re-run failed tasks:**
```bash
sbatch --array=5,7,12 run_benchmark.batch
```

**Test single parameter:**
```bash
uv run python examples/benchmark_runner.py error:1000 test_results/
```

**Custom analysis:**
```python
import json
data = json.load(open('benchmark_results/task_0_error_10/results.json'))
print(f"RMSE: {data['rmse']}, Runtime: {data['runtime_stochastic']}s")
```

## Troubleshooting

**Import errors:** Ensure `jc_sim_oqp` installed: `uv sync`

**SLURM_NTASKS unbound:** Fixed in current version (uses `--cpus-per-task` instead)

**Out of memory:** Large n_atoms (>15) may need `#SBATCH --mem=8G`

**Check job status:** `sacct -j <JOBID> --format=JobID,State,MaxRSS,Elapsed`

## Performance Notes

- Single-threaded tasks with 2 CPU allocation prevent oversubscription
- 72 jobs run simultaneously on 2×72-core cluster
- Expected total runtime: 30-60 minutes for all 120 tasks
- Storage needed: ~500 MB for all results

## Related Documentation

- {doc}`benchmark_system` - Detailed system architecture and design
- {doc}`tutorial` - Basic usage and examples
- {doc}`api` - Python API reference

## Files in Repository

- `examples/benchmark_runner.py` - Core benchmark library
- `run_benchmark.batch` - SLURM job script
- `aggregate_results.py` - Results analysis
- `test_benchmark_runner.py` - Local validation
- `benchmark_helpers.sh` - Shell convenience functions
