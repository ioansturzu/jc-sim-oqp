from qutip import mcsolve
from qutip.solver import Result

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics import (
    get_collapse_operators,
    get_initial_state,
    get_operators,
    jc_hamiltonian,
)


class StochasticSolver:
    """Solver using Monte Carlo Quantum Trajectories (mcsolve)."""

    def __init__(self, params: SimParams, ntraj: int = 500):
        self.params = params
        self.ntraj = ntraj

    def run(self, seed: int | None = None) -> Result:
        """Execute the simulation.

        Args:
            seed (int, optional): Random seed for reproducibility.
        """
        a, sm_list = get_operators(self.params.N, n_atoms=self.params.n_atoms)
        psi0 = get_initial_state(self.params.N, n_atoms=self.params.n_atoms)

        H = jc_hamiltonian(
            self.params.wc,
            self.params.wa,
            self.params.g,
            a,
            sm_list,
            use_rwa=self.params.use_rwa,
        )

        c_ops = get_collapse_operators(
            self.params.kappa,
            self.params.gamma,
            self.params.n_th_a,
            a,
            sm_list,
            gamma_phi=self.params.gamma_phi,
        )

        n_atoms_op = sum(sm.dag() * sm for sm in sm_list)
        e_ops = [a.dag() * a, n_atoms_op]

        # mcsolve returns a Result object similar to mesolve but with trajectory data
        return mcsolve(
            H, psi0, self.params.tlist, c_ops, e_ops=e_ops, ntraj=self.ntraj,
            seeds=seed
        )
