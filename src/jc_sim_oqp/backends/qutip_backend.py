"""QuTiP backend — wraps the existing jc_sim_oqp physics layer.

This backend delegates all operator construction, Hamiltonian building,
and ODE integration to the functions already living in
``jc_sim_oqp.physics`` and ``qutip.mesolve``.  It adds no new physics;
it simply exposes the existing code through the ``QuantumBackend``
protocol so that solvers can be backend-agnostic.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from qutip import Qobj, mesolve, steadystate

from jc_sim_oqp.backends.result import SimResult
from jc_sim_oqp.physics import (
    dispersive_hamiltonian,
    get_collapse_operators,
    get_initial_state,
    get_operators,
    jc_hamiltonian,
)

if TYPE_CHECKING:
    from jc_sim_oqp.io import SimParams


class QuTiPBackend:
    """Concrete backend powered by QuTiP ≥ 5."""

    name: str = "qutip"


    def build_operators(
        self, n_cavity: int, n_atoms: int = 1
    ) -> tuple[Qobj, list[Qobj]]:
        """Construct cavity and atom operators via ``physics.operators``."""
        return get_operators(n_cavity, n_atoms=n_atoms)

    def build_initial_state(
        self, n_cavity: int, n_atoms: int = 1
    ) -> Qobj:
        """Construct the default initial state |e, …, 0⟩."""
        return get_initial_state(n_cavity, n_atoms=n_atoms)

    def build_hamiltonian(
        self,
        params: SimParams,
        a: Qobj,
        sm_list: list[Qobj],
        *,
        variant: str = "jc",
    ) -> Qobj:
        """Build a Hamiltonian from ``SimParams``.

        Args:
            variant: ``"jc"`` or ``"dispersive"``.
        """
        if variant == "dispersive":
            return dispersive_hamiltonian(
                params.wc, params.wa, params.g, a, sm_list
            )
        return jc_hamiltonian(
            params.wc, params.wa, params.g, a, sm_list,
            use_rwa=params.use_rwa,
        )

    def build_collapse_ops(
        self, params: SimParams, a: Qobj, sm_list: list[Qobj]
    ) -> list[Qobj]:
        """Build collapse operators from ``SimParams``."""
        return get_collapse_operators(
            params.kappa,
            params.gamma,
            params.n_th_a,
            a,
            sm_list,
            gamma_phi=params.gamma_phi,
            n_th_q=params.n_th_q,
        )

    def mesolve(
        self,
        H: Qobj,
        psi0: Qobj,
        tlist: np.ndarray,
        c_ops: list[Qobj],
        e_ops: list[Qobj],
        options: dict[str, Any] | None = None,
    ) -> SimResult:
        """Integrate the master equation via ``qutip.mesolve``."""
        t0 = time.perf_counter()
        result = mesolve(H, psi0, tlist, c_ops, e_ops=e_ops, options=options)
        wall = time.perf_counter() - t0

        return SimResult(
            times=np.asarray(result.times),
            expect=[np.asarray(e) for e in result.expect],
            states=result.states if result.states else None,
            wall_time=wall,
            backend_name=self.name,
        )


    def steadystate(
        self, H: Qobj, c_ops: list[Qobj]
    ) -> Qobj:
        """Find the steady-state density matrix via ``qutip.steadystate``."""
        return steadystate(H, c_ops)
