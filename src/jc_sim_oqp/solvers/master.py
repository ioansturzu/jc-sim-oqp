from qutip import mesolve
from qutip.solver import Result

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics import (
    get_collapse_operators,
    get_initial_state,
    get_operators,
    jc_hamiltonian,
)


class ExactSolver:
    """Solver using the exact Jaynes-Cummings Hamiltonian and Master Equation."""

    def __init__(self, params: SimParams):
        self.params = params

    def run(self) -> Result:
        """Execute the simulation."""
        # 1. Setup operators and state
        a, sm_list = get_operators(self.params.N, n_atoms=self.params.n_atoms)
        psi0 = get_initial_state(self.params.N, n_atoms=self.params.n_atoms)

        # 2. Hamiltonian
        H = jc_hamiltonian(
            self.params.wc,
            self.params.wa,
            self.params.g,
            a,
            sm_list,
            use_rwa=self.params.use_rwa,
        )

        # 3. Collapse operators
        c_ops = get_collapse_operators(
            self.params.kappa,
            self.params.gamma,
            self.params.n_th_a,
            a,
            sm_list,
            gamma_phi=self.params.gamma_phi,
        )

        # 4. Evolve
        # Observables: Photon number, Total Atom Excitation
        n_atoms_op = sum(sm.dag() * sm for sm in sm_list)
        e_ops = [a.dag() * a, n_atoms_op]

        return mesolve(H, psi0, self.params.tlist, c_ops, e_ops=e_ops)
