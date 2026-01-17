
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.solvers import ExactSolver


def main():
    """Simulate Tavis-Cummings dynamics for 2 atoms."""
    # 1. Setup Parameters
    params = SimParams()
    params.n_atoms = 2
    params.wc = 1.0 * 2 * np.pi
    params.wa = 1.0 * 2 * np.pi
    params.g  = 0.05 * 2 * np.pi
    params.kappa = 0.0 # No dissipation for clear Rabi oscillations
    params.gamma = 0.0
    params.t_max = 50.0
    params.n_steps = 1000
    params.N = 5 # Sufficient for vacuum/1-photon


    # 2. Run Exact Simulation
    solver = ExactSolver(params)
    result = solver.run()

    # 3. Analyze Results
    # ExactSolver returns [n_photons, n_total_excitation]
    n_photons = result.expect[0]
    n_excitation = result.expect[1]

    # 4. Plot
    tlist = params.tlist
    _fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tlist, n_excitation, 'b-', label='Total Atomic Excitation')
    ax.plot(tlist, n_photons, 'r--', label='Photon Number')

    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Population')
    ax.set_title(f'Tavis-Cummings Dynamics (N={params.n_atoms})')
    ax.legend()
    ax.grid(True)

    # Save
    output_file = Path(__file__).parent / "tavis_cummings.png"
    plt.savefig(output_file)

if __name__ == "__main__":
    main()
