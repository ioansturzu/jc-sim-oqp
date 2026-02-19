from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from qutip import Qobj, Result, mesolve

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics import (
    get_collapse_operators,
    get_initial_state,
    get_operators,
    jc_hamiltonian,
)

if TYPE_CHECKING:
    from jc_sim_oqp.backends.protocol import QuantumBackend
    from jc_sim_oqp.backends.result import SimResult


class ExactSolver:
    """Solver using the exact Jaynes-Cummings Hamiltonian and Master Equation.

    Args:
        params: Simulation parameters.
        backend: Optional backend implementing ``QuantumBackend``.
            When ``None`` (the default), the solver uses inline QuTiP
            calls directly — identical to the original behavior.
    """

    def __init__(
        self, params: SimParams, backend: QuantumBackend | None = None
    ):
        self.params = params
        self.backend = backend

    def run(
        self,
        psi0: Qobj | None = None,
        tlist: np.ndarray | None = None,
        options: dict[str, Any] | None = None,
    ) -> Result | SimResult:
        """Execute the simulation.

        Args:
            psi0: Initial state vector or density matrix.
            tlist: Time steps for simulation.
            options: Solver options.

        Returns:
            ``qutip.Result`` when no backend is set (default),
            ``SimResult`` when using an explicit backend.
        """
        if self.backend is not None:
            return self._run_via_backend(psi0, tlist, options)
        return self._run_inline(psi0, tlist, options)


    def _run_via_backend(
        self,
        psi0: Any | None,
        tlist: np.ndarray | None,
        options: dict[str, Any] | None,
    ) -> SimResult:
        backend = self.backend
        a, sm_list = backend.build_operators(self.params.N, self.params.n_atoms)

        if psi0 is None:
            psi0 = backend.build_initial_state(self.params.N, self.params.n_atoms)
        if tlist is None:
            tlist = self.params.tlist

        H = backend.build_hamiltonian(self.params, a, sm_list, variant="jc")
        c_ops = backend.build_collapse_ops(self.params, a, sm_list)

        n_atoms_op = sum(sm.dag() * sm for sm in sm_list)
        e_ops = [a.dag() * a, n_atoms_op]

        return backend.mesolve(H, psi0, tlist, c_ops, e_ops, options)


    def _run_inline(
        self,
        psi0: Qobj | None,
        tlist: np.ndarray | None,
        options: dict[str, Any] | None,
    ) -> Result:

        a, sm_list = get_operators(self.params.N, n_atoms=self.params.n_atoms)
        if psi0 is None:
            psi0 = get_initial_state(self.params.N, n_atoms=self.params.n_atoms)

        if tlist is None:
            tlist = self.params.tlist


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
            n_th_q=self.params.n_th_q,
        )


        n_atoms_op = sum(sm.dag() * sm for sm in sm_list)
        e_ops = [a.dag() * a, n_atoms_op]

        return mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, options=options)


class SteadyStateSolver:
    """Solver for finding the steady state of the Jaynes-Cummings system.

    Args:
        params: Simulation parameters.
        backend: Optional backend implementing ``QuantumBackend``.
            When ``None`` (the default), the solver uses inline QuTiP
            calls directly.
    """

    def __init__(
        self, params: SimParams, backend: QuantumBackend | None = None
    ):
        self.params = params
        self.backend = backend

    def run(self, drive_amp: float = 0.0) -> Qobj:
        """Calculate the steady state density matrix.

        Args:
            drive_amp: Coherent drive amplitude on the cavity.
                       ``H_drive = i * drive_amp * (a† − a)``

        Returns:
            Steady state density matrix ρ_ss.
        """
        if self.backend is not None:
            return self._run_via_backend(drive_amp)
        return self._run_inline(drive_amp)


    def _run_via_backend(self, drive_amp: float) -> Qobj:
        backend = self.backend
        a, sm_list = backend.build_operators(self.params.N, self.params.n_atoms)
        H = backend.build_hamiltonian(self.params, a, sm_list, variant="jc")

        if drive_amp != 0.0:
            H += 1j * drive_amp * (a.dag() - a)

        c_ops = backend.build_collapse_ops(self.params, a, sm_list)
        return backend.steadystate(H, c_ops)


    def _run_inline(self, drive_amp: float) -> Qobj:
        from qutip import steadystate


        a, sm_list = get_operators(self.params.N, n_atoms=self.params.n_atoms)


        H = jc_hamiltonian(
            self.params.wc,
            self.params.wa,
            self.params.g,
            a,
            sm_list,
            use_rwa=self.params.use_rwa,
        )


        if drive_amp != 0.0:
            H += 1j * drive_amp * (a.dag() - a)


        c_ops = get_collapse_operators(
            self.params.kappa,
            self.params.gamma,
            self.params.n_th_a,
            a,
            sm_list,
            gamma_phi=self.params.gamma_phi,
            n_th_q=self.params.n_th_q,
        )


        return steadystate(H, c_ops)
