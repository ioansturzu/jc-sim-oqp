from qutip import mesolve
from qutip.solver import Result

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics import (
    dispersive_hamiltonian,
    get_collapse_operators,
    get_initial_state,
    get_operators,
)


class DispersiveSolver:
    """Solver using the Dispersive Hamiltonian (effective model).

    Valid when ``|Delta| >> g * sqrt(n)``.
    """

    def __init__(self, params: SimParams):
        self.params = params

    def run(self) -> Result:
        """Execute the simulation."""
        a, sm_list = get_operators(self.params.N, n_atoms=self.params.n_atoms)
        psi0 = get_initial_state(self.params.N, n_atoms=self.params.n_atoms)

        # Use dispersive Hamiltonian
        H = dispersive_hamiltonian(
            self.params.wc, self.params.wa, self.params.g, a, sm_list
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
        return mesolve(H, psi0, self.params.tlist, c_ops, e_ops=e_ops)
