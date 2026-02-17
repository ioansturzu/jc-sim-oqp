# Welcome to JC-Sim-OQP

**JC-Sim-OQP** is a simulation package for **Cavity Quantum Electrodynamics (Cavity QED)** systems, aiming to support **Neutral Atom** architectures.

It can serve as a "Digital Twin" for the fundamental interactions between light and matter, merging theoretical foundations with production-grade software engineering.

## Key Features

*   **HPC Benchmarking:** Parallel benchmark suite for scaling analysis on SLURM clusters.
*   **Four Solvers:**
    *   **ExactSolver:** Full Lindblad Master Equation ($N \times N$ density matrix).
    *   **StochasticSolver:** Monte Carlo Wavefunction (Quantum Trajectories).
    *   **SteadyStateSolver:** Efficient long-time limit calculation under drive.
    *   **DispersiveSolver:** High-detuning dispersive Hamiltonian for readout.
*   **Advanced Physics:** Purcell factors ($F_p$), Transmon dispersive shifts ($\chi$), and analytical reflection spectra.
*   **Multi-Atom Support:** Verified collective effects (Tavis-Cummings) for $N > 20$ atoms using stochastic methods.
*   **Physics-First Design:** Implements **Operational Quantum Physics (OQP)** principles.

## Getting Started

```{toctree}
:maxdepth: 2
:caption: Contents:

theory
tutorial
api
benchmarks
benchmark_system
```

## Quick Install

```bash
uv pip install jc-sim-oqp
# or
pip install .
```

## Indices and tables

*   {ref}`genindex`
*   {ref}`modindex`
*   {ref}`search`
