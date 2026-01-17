from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from qutip import entropy_vn, expect, mcsolve, mesolve

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics import get_collapse_operators, get_initial_state, get_operators, jc_hamiltonian


def main():
    """Run an advanced simulation of a 2-atom Tavis-Cummings model."""
    # 1. Setup Parameters for Strong Coupling
    params = SimParams()
    params.n_atoms = 2
    params.wc = 1.0 * 2 * np.pi  # 1.0 GHz
    params.wa = 1.0 * 2 * np.pi
    params.g  = 0.05 * 2 * np.pi # 50 MHz coupling
    # Dissipation: High Cooperativity C = g^2 / (kappa*gamma)
    params.kappa = 0.005 * 2 * np.pi # 5 MHz
    params.gamma = 0.001 * 2 * np.pi # 1 MHz

    params.t_max = 50.0  # ns (Simulation time)
    params.n_steps = 500
    params.N = 10        # Hilbert space truncation

    # Effective coupling for N atoms is sqrt(N)*g

    # 2. Run Exact Solver (Density Matrix)

    a, sm_list = get_operators(params.N, n_atoms=params.n_atoms)

    # jc_hamiltonian accepts list
    H = jc_hamiltonian(params.wc, params.wa, params.g, a, sm_list)
    c_ops = get_collapse_operators(params.kappa, params.gamma, 0.0, a, sm_list)
    psi0 = get_initial_state(params.N, n_atoms=params.n_atoms) # |0, e, e...>
    tlist = np.linspace(0, params.t_max, params.n_steps)

    # Return states!
    # e_ops empty to get states
    res_exact_states = mesolve(H, psi0, tlist, c_ops, [])

    # 3. Compute Entanglement Entropy (Von Neumann)
    # S(rho_atom) = -Tr(rho_atom ln rho_atom)
    entropies = []
    # Atom indices are 1, 2, ... n_atoms. Cavity is 0.
    atom_indices = list(range(1, params.n_atoms + 1))

    for state in res_exact_states.states:
        if state.isket:
            rho = state * state.dag()
        else:
            rho = state
        # Trace out cavity (0), keep all atoms
        rho_atoms = rho.ptrace(atom_indices)
        entropies.append(entropy_vn(rho_atoms, base=np.e))

    # 5. Visualization
    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Population Dynamics & Jumps
    # Use qutip.expect explicitly

    # Observable: Total Atomic Excitation = sum(sigma+ sigma-)
    n_atoms_op = sum(sm.dag()*sm for sm in sm_list)

    n_a_exact = expect(n_atoms_op, res_exact_states.states)
    ax1.plot(tlist, n_a_exact, 'k-', lw=2, label='Ensemble Average (Master Eq)')

    # Plot trajectories
    ntraj_plot = 2000

    # Helper for parallel execution
    def run_traj(_seed):
        # Set seed for reproducibility if needed, but mcsolve handles randomness
        # Must recreate operators inside worker or pass them? best to pass H/ops if picklable.
        # qutip Qobjs are picklable.
        return mcsolve(H, psi0, tlist, c_ops, [n_atoms_op], ntraj=1).expect[0]

    # Run in parallel
    all_trajs = Parallel(n_jobs=-1)(delayed(run_traj)(i) for i in range(ntraj_plot))

    cmap = cm.get_cmap('viridis')
    for i, data in enumerate(all_trajs):
        # Use a color from the colormap
        color = cmap(i / ntraj_plot)
        # Optimized visibility: alpha 0.05 for dense plots
        ax1.plot(tlist, data, color=color, lw=0.8, alpha=0.05, ls='-', label=None)

    # Compute and plot empirical average
    empirical_avg = np.mean(all_trajs, axis=0)
    ax1.plot(tlist, empirical_avg, color='cyan', lw=2.0, ls='--', label=f'Empirical Avg ({ntraj_plot} runs)')

    ax1.set_ylabel(f'Total Atom Excitation (N={params.n_atoms})')
    ax1.set_title(f'Quantum Trajectories: {params.n_atoms}-Atom Tavis-Cummings Model')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Entanglement Entropy
    ax2.plot(tlist, entropies, 'b-', lw=2, label='Von Neumann Entropy $S(\\rho_{atom})$')
    ax2.set_ylabel('Entropy (nats)')
    ax2.set_xlabel('Time (ns)')
    ax2.set_title('Atom-Cavity Entanglement Dynamics')
    ax2.fill_between(tlist, 0, entropies, color='b', alpha=0.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    # Save relative to the script location to ensure it works from anywhere
    output_file = Path(__file__).parent / "advanced_simulation.png"

    plt.savefig(output_file)

if __name__ == "__main__":
    main()
