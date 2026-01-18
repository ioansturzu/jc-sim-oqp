# Benchmark System Architecture

```{eval-rst}
.. meta::
   :description: Architecture and design of the HPC benchmark system
   :keywords: system design, HPC, parallel computing, quantum simulation
```

## Purpose

Scaled parallel benchmark suite for JC-SIM-OQP quantum simulations. Maximizes HPC cluster utilization to characterize solver performance across parameter space.

This document describes the technical architecture, design decisions, and optimization strategies.

## Design Goals

1. **Maximum parallelism** - Simultaneous execution of all array tasks
2. **Efficient resource use** - Optimized CPU allocation per task
3. **Statistical robustness** - Multiple replicates per parameter
4. **Minimal overhead** - Modular Python library, clean SLURM script
5. **Easy scaling** - Simple parameter array modification

## System Architecture

```
User → test_benchmark_runner.py (local validation)
     ↓
     → run_benchmark.batch (SLURM: Array tasks)
     ↓
     → benchmark_runner.py (core library)
     ↓
     → jc_sim_oqp solvers (ExactSolver, StochasticSolver)
     ↓
     → results/ (JSON + NPZ files, output directories)
     ↓
     → aggregate_results.py (analysis + plots)
```

## Components

### benchmark_runner.py
**Location:** `examples/benchmark_runner.py`  
**Purpose:** Modular benchmark library with CLI  
**Functions:** `run_error_benchmark()`, `run_time_benchmark()`, `run_benchmark_from_params()`

### run_benchmark.batch  
**Purpose:** SLURM array job  
**Config:** Configurable CPUs/task and constraints  
**Parameters:** Error benchmarks (ntraj sweeps), time benchmarks (system size sweeps)

### aggregate_results.py
**Purpose:** Post-processing and visualization  
**Output:** Error convergence, timing scaling, combined dashboard plots

### test_benchmark_runner.py
**Purpose:** Local validation before HPC submission  
**Tests:** Parameter parsing, small benchmarks, output verification

## Workflow

### Development/Testing Phase

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Run validation tests
uv run test_benchmark_runner.py

# 3. Test single benchmark
uv run python examples/benchmark_runner.py error:50 test_results/

# 4. Verify output
ls -la test_results/
```

### Production Phase

```bash
# 1. Submit to SLURM
sbatch run_benchmark.batch

# 2. Monitor progress
squeue -u $USER
watch -n 5 'ls benchmark_results/task_*/results.json | wc -l'

# 3. Analyze results
uv run aggregate_results.py

# 4. View plots
ls benchmark_results/plots/
```

### Using Helper Functions

```bash
# Source helper functions
source benchmark_helpers.sh

# Show available commands
show_help

# Run workflow
check_venv
test_benchmark
submit_slurm
check_status
count_results
plot_results
```

## Benchmark Types

### Error Benchmark
**Measures**: Monte Carlo convergence
**Parameters**: Number of trajectories (ntraj)
**Output**: RMSE vs exact solution
**Expected**: Error ∝ 1/√(ntraj)

### Time Benchmark
**Measures**: Computational scaling
**Parameters**: System size (n_atoms)
```bash
# Local testing
uv run test_benchmark_runner.py

# Submit to HPC
sbatch run_benchmark.batch

# Monitor (concurrent execution)
watch -n 10 'ls benchmark_results/task_*/results.json | wc -l'

# Analyze (after completion)
uv run aggregate_results.py
```

## Parameter Space

**Error benchmarks:**  
ntraj sweeps × replicates

**Time benchmarks:**  
n_atoms sweeps × replicates

**Total:** Full coverage of parameter space

## HPC Optimization

**Typical setup (HPC cluster):**
- Configurable CPUs per task
- Optimized for node density
- Simultaneous execution across available nodes
- Efficient runtime scaling
import numpy as np
data = np.load('trajectory_data.npz')
times = data['times']
exact = data['exact']
stochastic = data['stochastic']
```

## System Advantages

### Modular Design
The architecture separates concerns into distinct, testable components:

*   **Testable**: Unit-testable components allow for rapid local debugging without SLURM overhead.
*   **Reusable**: The `benchmark_runner` library exposes a clean Python API for use in custom scripts and pipelines.
*   **Maintainable**: Standard Python code (docstrings, type hints) replaces complex shell logic.

### Scalability
The system supports linear scaling with available resources:

*   **Independent Execution**: Each parameter combination runs as an isolated process.
*   **Flexible Allocation**: Resource requirements (CPU/RAM) can be tuned per-task.
*   **Zero Overhead**: Direct Python execution minimizes job startup latency compared to containerized solutions.

## Performance

### Parallelization Strategy
- Each parameter combination runs independently
- No inter-task communication required
- Maximum parallelism up to array size
- Scales linearly with available nodes

### Resource Efficiency
- Single-node jobs minimize overhead
- No container overhead (as requested)
- Configurable CPU/memory per task
- Results saved incrementally

### Typical Runtime
- Error benchmark (ntraj=1000): ~10-30 seconds
- Time benchmark (n_atoms=5): ~5-20 seconds
- Full array: Variable depending on cluster load

## Troubleshooting

### Import Errors
```bash
# Check installation
uv run python -c "import jc_sim_oqp; print(jc_sim_oqp.__file__)"

# Verify PYTHONPATH
echo $PYTHONPATH
```

### Module Not Found
```bash
# Run from repository root
cd /path/to/jc-sim-oqp
uv run python examples/benchmark_runner.py ...
```

### SLURM Errors
```bash
# Check logs
tail benchmark_JOBID_TASKID.err

# Verify modules
module avail gcc python

# Test locally first
uv run test_benchmark_runner.py
```

## Future Scaling Roadmap

The system is designed to evolve from single-node calibration to massive-scale production runs.

### Phase 1: Parameter Sweeps (Current)
*   **Architecture**: Single-node or multi-node SLURM Job Arrays.
*   **Parallelism**: One MPI rank per parameter combination.
*   **Scale**: ~100-500 concurrent cores.
*   **Use Case**: Generating error convergence curves and timing capability limits.

### Phase 2: Distributed Stochastic Solvers (Planned)
*   **Architecture**: Multi-node MPI (per-trajectory parallelization).
*   **Parallelism**: One MCWF trajectory per core, aggregating results across nodes.
*   **Scale**: ~1,000+ cores.
*   **Goal**: Simulate $N=100+$ atoms where a single time step is fast, but $N_{traj}=10^5$ is required for convergence.

### Phase 3: Cloud Bursting (Large-Scale)
*   **Architecture**: Integration with SkyPilot or Ray for cloud orchestration.
*   **Parallelism**: Dynamic scaling of worker nodes on AWS/GCP spot instances.
*   **Scale**: ~10,000+ cores.
*   **Goal**: Large-scale calibration of commercial Neutral Atom QPU devices.

## Documentation

- **SLURM_BENCHMARK_GUIDE.md** - Comprehensive SLURM guide
- **examples/README.md** - Examples directory documentation
- **benchmark_runner.py** - Inline docstrings and type hints
- **This file** - Architecture and workflow overview

## Quick Reference

```bash
# Test locally
uv run test_benchmark_runner.py

# Run single benchmark
uv run python examples/benchmark_runner.py error:100 output/

# Submit to SLURM
sbatch run_benchmark.batch

# Check status
squeue -u $USER

# Analyze results
uv run aggregate_results.py

# Helper functions
source benchmark_helpers.sh && show_help
```

For detailed usage instructions, see {doc}`benchmarks`.
