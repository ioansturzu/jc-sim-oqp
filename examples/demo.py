
import matplotlib.pyplot as plt
import numpy as np

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics import get_operators
from jc_sim_oqp.solvers import ExactSolver, StochasticSolver


def main():
    """Run a demo of Rabi oscillations using exact and stochastic solvers."""
    params = SimParams()
    params.wc = 1.0     # Resonant
    params.wa = 1.0
    params.g = 0.05     # Strong coupling
    params.kappa = 0.005 # Weak cavity decay
    params.gamma = 0.005 # Weak atomic decay
    params.n_steps = 200
    params.t_max = 200  # Several Rabi cycles
    params.N = 15       # Truncation


    # 1. Exact Solution (Master Equation)
    exact_solver = ExactSolver(params)
    res_exact = exact_solver.run()

    # Extract expectation values (Atom e state is index 1 of the list returned by core)
    # index 0 is photons, index 1 is atom excited probability.
    n_a_exact = res_exact.expect[1]

    # 2. Stochastic Solution (Trajectories)
    ntraj = 50
    stoch_solver = StochasticSolver(params, ntraj=ntraj)
    res_stoch = stoch_solver.run()

    n_a_stoch = np.array(res_stoch.expect[1])

    # 3. Compare
    # (Visual comparison only in this version)

    # 4. Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    times = res_exact.times

    ax.plot(times, n_a_exact, 'b-', label="Exact (Atom)", linewidth=2)
    ax.plot(times, n_a_stoch, 'r--', label=f"Stochastic (Atom, N={ntraj})", alpha=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Excited State Population")
    ax.set_title("Exact vs. Stochastic Solver: Damped Rabi Oscillations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_file = "examples/rabi_demo.png"
    fig.savefig(output_file)
    fig.savefig(output_file)

    _a, _sm_list = get_operators(params.N) # Default n_atoms=1

if __name__ == "__main__":
    main()
