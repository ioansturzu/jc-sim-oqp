import numpy as np
from qutip import Qobj, Result, mesolve

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

    def run(
        self,
        psi0: Qobj | None = None,
        tlist: np.ndarray | None = None,
        options: dict | None = None,
    ) -> Result:
        """Execute the simulation.
        
        Args:
            psi0 (Qobj, optional): Initial state vector or density matrix.
            tlist (np.ndarray, optional): Time steps for simulation.
            options (dict, optional): Solver options for mesolve.
        """
        # 1. Setup operators and state
        a, sm_list = get_operators(self.params.N, n_atoms=self.params.n_atoms)
        if psi0 is None:
            psi0 = get_initial_state(self.params.N, n_atoms=self.params.n_atoms)
        
        if tlist is None:
            tlist = self.params.tlist

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
            n_th_q=self.params.n_th_q,
        )

        # 4. Evolve
        # Observables: Photon number, Total Atom Excitation
        n_atoms_op = sum(sm.dag() * sm for sm in sm_list)
        e_ops = [a.dag() * a, n_atoms_op]

        return mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, options=options)


class SteadyStateSolver:
    """Solver for finding the steady state of the Jaynes-Cummings system."""

    def __init__(self, params: SimParams):
        self.params = params

    def run(self, drive_amp: float = 0.0) -> Qobj:
        """Calculate the steady state density matrix.
        
        Args:
            drive_amp (float): Coherent drive amplitude on the cavity.
                               H_drive = i * drive_amp * (a.dag() - a)

        Returns:
            qutip.Qobj: Steady state density matrix (rho_ss).
        """
        from qutip import steadystate

        # 1. Setup operators
        a, sm_list = get_operators(self.params.N, n_atoms=self.params.n_atoms)

        # 2. Hamiltonian
        H = jc_hamiltonian(
            self.params.wc,
            self.params.wa,
            self.params.g,
            a,
            sm_list,
            use_rwa=self.params.use_rwa,
        )
        
        # Add drive if present
        if drive_amp != 0.0:
            H += 1j * drive_amp * (a.dag() - a)

        # 3. Collapse operators
        c_ops = get_collapse_operators(
            self.params.kappa,  # Uses the property (kappa_in + kappa_sc)
            self.params.gamma,
            self.params.n_th_a,
            a,
            sm_list,
            gamma_phi=self.params.gamma_phi,
            n_th_q=self.params.n_th_q,
        )

        # 4. Solve for steady state
        return steadystate(H, c_ops)
