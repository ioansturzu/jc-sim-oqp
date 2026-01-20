# JC-Sim-OQP: Operational Quantum Physics Simulation

**A Rigorous "Digital Twin" for Neutral Atom Cavity QED**

`jc-sim-oqp` is a high-performance Python package designed to simulate the interaction of Light and Matter at the fundamental level. It implements the **Jaynes-Cummings Model** and its multi-atom extension (**Tavis-Cummings**), serving as a verification tool for Neutral Atom quantum computing architectures.

Unlike standard textbook simulations, this package is built on **Operational Quantum Physics (OQP)** principles, specifically modeling realistic measurement backaction, open quantum systems (Lindblad master equation), and stochastic quantum trajectories.

## Features

*   **HPC-Scaled Benchmarking (Core Feature):**
    *   **Parallel Execution:** Built for SLURM clusters, capable of executing 120+ concurrent calibration jobs with linear scaling.
    *   **Performance:** Observed 100x speedup of Stochastic methods over Exact solvers for $N > 8$ atoms.
    *   **Automated Analytics:** Generation of convergence and timing plots with statistical confidence intervals.
*   **Three Physics Engines:**
    *   **Exact Solver:** Full density matrix simulation ($d^2$) using the Lindblad Master Equation.
    *   **Stochastic Solver:** Monte Carlo Wavefunction method ($N_{traj} \times d$) for scalable simulations of large systems ($N_{atoms} \ge 8$).
    *   **Dispersive Solver:** Effective Hamiltonian for fast readout simulation in the far-detuned regime ($|\Delta| \gg g$).
*   **Multi-Atom Scalability:** Supports Tavis-Cummings interactions for studying collective effects.
*   **Software Engineering:** Built with `uv`, `pytest`, and `sphinx` for reproducibility.

## Installation

This project is managed with [uv](https://github.com/astral-sh/uv).

```bash
# Clone the repository
git clone https://github.com/ioansturzu/jc-sim-oqp.git
cd jc-sim-oqp

# Install dependencies and the package
uv sync
```

For debugging (editable mode), where changes to the code apply immediately without reinstalling:
```bash
uv pip install -e .
```

Alternatively, you can install it using standard pip:
```bash
pip install .
```

## Quick Start / workflows

### 1. Run the Demo
Simulate a single atom undergoing Vacuum Rabi Oscillations:
```bash
uv run python examples/demo.py
```

### 2. Run Tests
Verify the physics engine against analytical predictions (Rabi splitting, dispersive shifts):
```bash
uv run pytest
```

### 3. Build Documentation
Generate the full HTML documentation (Theory, API, Tutorials):
```bash
# Install doc dependencies
uv sync --dev

# Build docs
uv run sphinx-build -b html docs docs/_build/html

# Open (Linux)
xdg-open docs/_build/html/index.html
```

### 4. Run Benchmarks

**Local benchmarks:**
```bash
uv run python examples/benchmark_scaling.py
```

**HPC-scaled benchmarks** (SLURM clusters):
```bash
# Test locally first
uv run test_benchmark_runner.py

# Submit 120-job array to cluster
sbatch run_benchmark.batch

# Analyze results
uv run aggregate_results.py
```

See [HPC Benchmarking Guide](docs/benchmarks.md) for detailed setup and scaling instructions.

## Theory
This software implements the rigorous derivation of the Lindblad Master Equation from the microscopic Reservoir Model. Theoretical details can be found in `docs/theory.md` or the generated documentation.

## License
MIT License.
