import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.solvers import ExactSolver, StochasticSolver


def benchmark_error():
    """Verify that Monte Carlo error scales with 1/sqrt(N_traj)."""
    params = SimParams()
    params.n_atoms = 1
    params.t_max = 20.0 # Shorter for speed
    params.n_steps = 200
    params.N = 10

    # 1. Exact Solution (Ground Truth)
    solver_exact = ExactSolver(params)
    res_exact = solver_exact.run()
    # Explicit deep copy of the expectation data to prevent mutation interference
    exact_data = np.array(res_exact.expect[0], copy=True)

    # 2. Stochastic Solutions
    ntraj_list = [10, 50, 100, 500, 1000, 5000, 10000]
    errors = []

    for ntraj in ntraj_list:
        # Seeded run for deterministic proof
        solver_mc = StochasticSolver(params, ntraj=ntraj)
        res_mc = solver_mc.run(seed=42)

        mc_data = np.array(res_mc.expect[0], copy=True)

        # Calculate RMSE (Root Mean Square Error over time)
        rmse = np.sqrt(np.mean((exact_data - mc_data)**2))
        errors.append(rmse)

    # 3. Plot
    _fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(ntraj_list, errors, 'bo-', label='Simulation Error')

    # Reference 1/sqrt(N) line
    C = errors[2] * np.sqrt(ntraj_list[2])
    ref = [C / np.sqrt(n) for n in ntraj_list]
    ax.loglog(ntraj_list, ref, 'k--', label=r'Theory $1/\sqrt{N_{traj}}$')

    ax.set_xlabel('Number of Trajectories ($N_{traj}$)')
    ax.set_ylabel('RMSE (vs Master Equation)')
    ax.set_title('Convergence of Quantum Trajectories (Monte Carlo)')
    ax.grid(True, which="both", ls="-", alpha=0.5)
    ax.legend()

    output_file = Path(__file__).parent / "benchmark_error.png"
    plt.savefig(output_file)

    return ntraj_list, errors


def benchmark_time():
    """Compare exact solver runtime vs stochastic solver scaling."""
    # Loop over N_atoms up to 8
    n_atoms_list = [1, 2, 3, 4, 5, 6, 7, 8]

    times_exact = []
    times_stoch = []

    # Params
    params = SimParams()
    params.t_max = 5.0
    params.n_steps = 50
    params.N = 5 # Small cavity truncation

    ntraj_fixed = 100

    for n_atoms in n_atoms_list:
        params.n_atoms = n_atoms

        # Exact
        if n_atoms <= 8:
             pass

        t0 = time.perf_counter()
        try:
            ExactSolver(params).run()
            dt_exact = time.perf_counter() - t0
        except Exception:  # noqa: BLE001
            dt_exact = None # Marker for failure/timeout

        if dt_exact is not None:
            times_exact.append(dt_exact)
        else:
            times_exact.append(np.nan)

        # Stochastic
        t0 = time.perf_counter()
        StochasticSolver(params, ntraj=ntraj_fixed).run()
        dt_stoch = time.perf_counter() - t0
        times_stoch.append(dt_stoch)

    return n_atoms_list, times_exact, times_stoch

def run_all():
    """Run both error and time benchmarks and plot combined results."""
    # Run 1: Error
    ntraj_list, errors = benchmark_error()

    # Run 2: Time
    n_atoms_list, times_exact, times_stoch = benchmark_time()

    # Combined Plot
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Error
    ax1.loglog(ntraj_list, errors, 'bo-', label='Simulation Error')
    # Ref
    ref = [errors[0] * np.sqrt(ntraj_list[0]) / np.sqrt(n) for n in ntraj_list]
    ax1.loglog(ntraj_list, ref, 'k--', label=r'Theory $1/\sqrt{N_{traj}}$')
    ax1.set_xlabel('Number of Trajectories ($N_{traj}$)')
    ax1.set_ylabel('RMSE vs Exact')
    ax1.set_title('Accuracy Convergence')
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # Plot 2: Time
    filtered_exact = [(n, t) for n, t in zip(n_atoms_list, times_exact) if not np.isnan(t)]
    nx, tx = zip(*filtered_exact)
    ax2.semilogy(nx, tx, 'r-o', label='Exact Master Eq')
    ax2.semilogy(n_atoms_list, times_stoch, 'b-s', label='Stochastic (100 traj)')

    ax2.set_xlabel('Number of Atoms ($N$)')
    ax2.set_ylabel('Runtime (s)')
    ax2.set_title('Computational Scaling')
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    output_file = Path(__file__).parent / "benchmark_dashboard.png"
    plt.savefig(output_file)

if __name__ == "__main__":
    run_all()
