from __future__ import annotations

from typing import TYPE_CHECKING

from qutip import mesolve

from jc_sim_oqp.physics import (
    dispersive_hamiltonian,
    get_collapse_operators,
    get_initial_state,
    get_operators,
)

if TYPE_CHECKING:
    from qutip.solver import Result

    from jc_sim_oqp.backends.protocol import QuantumBackend
    from jc_sim_oqp.backends.result import SimResult
    from jc_sim_oqp.io import SimParams


class DispersiveSolver:
    """Solver using the Dispersive Hamiltonian (effective model).

    Valid when ``|Delta| >> g * sqrt(n)``.

    Args:
        params: Simulation parameters.
        backend: Optional backend implementing ``QuantumBackend``.
    """

    def __init__(
        self, params: SimParams, backend: QuantumBackend | None = None
    ):
        self.params = params
        self.backend = backend

    def run(self) -> Result | SimResult:
        """Execute the simulation."""
        if self.backend is not None:
            return self._run_via_backend()
        return self._run_inline()

    def _run_via_backend(self) -> SimResult:
        backend = self.backend
        a, sm_list = backend.build_operators(self.params.N, self.params.n_atoms)
        psi0 = backend.build_initial_state(self.params.N, self.params.n_atoms)

        H = backend.build_hamiltonian(
            self.params, a, sm_list, variant="dispersive"
        )
        c_ops = backend.build_collapse_ops(self.params, a, sm_list)

        n_atoms_op = sum(sm.dag() * sm for sm in sm_list)
        e_ops = [a.dag() * a, n_atoms_op]
        return backend.mesolve(H, psi0, self.params.tlist, c_ops, e_ops)

    def _run_inline(self) -> Result:
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
