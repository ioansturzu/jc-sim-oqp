# Welcome to JC-Sim-OQP

**JC-Sim-OQP** is a physics-aware simulation package for **Cavity Quantum Electrodynamics (Cavity QED)** systems, specifically designed for **Neutral Atom** architectures.

It serves as a "Digital Twin" for the fundamental interactions between light and matter, merging deep theoretical foundations with production-grade software engineering.

## Key Features

*   **Three Solvers:**
    *   **ExactSolver:** Full Lindblad Master Equation simulation ($N \times N$ density matrix).
    *   **StochasticSolver:** Monte Carlo Wavefunction (Quantum Trajectories) for scalable simulations ($N$ vector).
    *   **DispersiveSolver:** Effective Hamiltonian for the dispersive readout regime.
*   **Multi-Atom Support:** Scalable to $N$ atoms (Verified up to $N=8$ for Master Equation, $N>20$ for Stochastic).
*   **Physics-First Design:** Implements **Operational Quantum Physics (OQP)** principles, including Dephasing, Thermal Baths, and Observables.

## Getting Started

```{toctree}
:maxdepth: 2
:caption: Contents:

theory
tutorial
api
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
