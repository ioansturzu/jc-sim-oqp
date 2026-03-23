from __future__ import annotations

from typing import TYPE_CHECKING

from qutip import mcsolve

from jc_sim_oqp.physics import (
    get_collapse_operators,
    get_initial_state,
    get_operators,
    jc_hamiltonian,
)

if TYPE_CHECKING:
    from qutip.solver import Result

    from jc_sim_oqp.backends.protocol import QuantumBackend
    from jc_sim_oqp.io import SimParams


class StochasticSolver:
    """Solver using Monte Carlo Quantum Trajectories (mcsolve).

    Args:
        params: Simulation parameters.
        ntraj: Number of trajectories.
        backend: Optional backend.  Currently only the inline QuTiP
            path is supported; passing a backend raises
            ``NotImplementedError``.
    """

    def __init__(
        self,
        params: SimParams,
        ntraj: int = 500,
        backend: QuantumBackend | None = None,
    ):
        self.params = params
        self.ntraj = ntraj
        self.backend = backend

    def run(self, seed: int | None = None) -> Result:
        """Execute the simulation.

        Args:
            seed: Random seed for reproducibility.
        """
        if self.backend is not None:
            raise NotImplementedError(
                "StochasticSolver does not yet support external backends. "
                "mcsolve is a QuTiP-specific solver."
            )
        return self._run_inline(seed)

    def _run_inline(self, seed: int | None) -> Result:
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

        return mcsolve(
            H, psi0, self.params.tlist, c_ops, e_ops=e_ops, ntraj=self.ntraj,
            seeds=seed
        )

