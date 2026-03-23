
# Tutorials

## 1. Minimal Working Example: Rabi Oscillations

The `examples/demo.py` script demonstrates the most fundamental phenomenon in Cavity QED: **Vacuum Rabi Oscillations**.
It simulates a single atom initialized in $|e\rangle$ interacting with an empty cavity $|0\rangle$.

### Code

```python
from jc_sim_oqp.io import SimParams
from jc_sim_oqp.solvers import ExactSolver
import numpy as np

# 1. Configure
params = SimParams()
params.g = 0.05 * 2 * np.pi  # Strong Coupling
params.kappa = 0.005         # Weak Dissipation

# 2. Run
solver = ExactSolver(params)
result = solver.run()
```

### Expected Output
You will see the atomic population oscillating coherently between $|e\rangle$ and $|g\rangle$ as it exchanges a single photon with the cavity. The amplitude decays slowly due to cavity loss $\kappa$.

## 2. Advanced Simulation: Entanglement & Trajectories

The `examples/simulation_suite.py` script explores the **Operational** aspects of the theory, specifically Measurement and Entanglement.

### Key Concepts demonstrated:
1.  **Tavis-Cummings Model:** Scales to **N=2 Atoms**.
2.  **Entanglement Entropy:** Calculates the Von Neumann entropy $S(\rho_{atom})$ to quantify the non-separability of the Atom-Cavity state.
3.  **Quantum Trajectories:** Visualizes 2000 individual "experiments" (Monte Carlo runs) where the wavefunction acts as a "cloud" of specific outcomes.

### Visualization
The script generates `advanced_simulation.png` showing:
*   **Top Standard:** The ensemble average (Master Equation) in black.
*   **Top Cloud:** The stochastic trajectories in faint color (Quantum Jumps).
*   **Bottom:** The entanglement entropy oscillating in time, proving strong correlations.

## 3. Scaling Benchmark

The `examples/benchmark_scaling.py` script puts the software to the test.
It runs a comparison between the **Exact Solver** (Master Equation) and **Stochastic Solver** (Trajectories) for $N=1$ to $N=8$ atoms.

### Results
*   **Accuracy:** The Trajectory average converges to the Exact result with error $1/\sqrt{N_{traj}}$.
*   **Runtime:** The Stochastic approach scales linearly with N, while the Exact solver scales exponentially.

## 4. Advanced Physics: Purcell Enhancement

The `jc_sim_oqp.physics.purcell` module provides tools to characterize the dissipative environment.

```python
from jc_sim_oqp.physics.purcell import purcell_factor, beta_factor

g = 0.1 * 2 * np.pi
kappa = 0.01 * 2 * np.pi
gamma = 0.001 * 2 * np.pi

Fp = purcell_factor(g, kappa, gamma)
beta = beta_factor(g, kappa, gamma)

print(f"Purcell Factor: {Fp:.2f}")
print(f"Beta Factor: {beta:.4f}")
```

## 5. Steady-State Scans & Reflection Spectra

To simulate CW measurements (like Vacuum Rabi Splitting), use the `SteadyStateSolver`.

```python
from jc_sim_oqp.solvers import SteadyStateSolver
from jc_sim_oqp.io import SimParams

params = SimParams(g=0.1*2*np.pi, kappa=0.01*2*np.pi, gamma=0.001*2*np.pi)
solver = SteadyStateSolver(params)

# Calculate steady state under weak drive
rho_ss = solver.run(drive_amp=0.001)
```

For automated sweeps over detuning, see the `solvers.scanners` module.

## 6. HPC Benchmarking

For production-scale performance analysis on HPC clusters, see the {doc}`benchmarks` guide. This system runs 120 parallel benchmark jobs to characterize:

- **Error convergence** vs trajectory count (10 to 50,000 trajectories)
- **Timing scaling** vs system size (1 to 18 atoms)
- **Statistical analysis** with multiple replicates

The benchmark system is optimized for SLURM clusters and can utilize up to 72 simultaneous jobs on a 2-node system.

**Quick start:**
```bash
# Local test
uv run test_benchmark_runner.py

# HPC submission
sbatch run_benchmark.batch
```

See {doc}`benchmarks` and {doc}`benchmark_system` for complete documentation.
*   **Speed:** The Exact solver becomes intractably slow around $N=7$ ($>12s$) or $N=8$ ($>70s$). The Stochastic solver remains fast ($<10s$), proving it is the only viable path for large-scale Neutral Atom simulations (N > 50).
