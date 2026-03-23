"""SciPy backend — pure NumPy/SciPy Lindblad master equation solver.

Implements the ``QuantumBackend`` protocol using only NumPy and
``scipy.integrate.solve_ivp``.  All QuTiP objects are converted to
dense NumPy arrays before integration.  This backend is intended
for *correctness validation*, not performance.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.integrate import solve_ivp

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


def _to_dense(obj: Any) -> np.ndarray:
    """Convert a QuTiP Qobj (or plain array) to a dense numpy matrix."""
    if hasattr(obj, "full"):
        return obj.full()
    return np.asarray(obj)


def _build_liouvillian(
    H: np.ndarray, c_ops: list[np.ndarray]
) -> np.ndarray:
    """Build the Liouvillian superoperator as a dense matrix.

    For a system of dimension *d*, the density matrix ρ is vectorised
    into a column vector of length d² (column-stacking order), and the
    Liouvillian acts as a d²×d² matrix::

        L vec(ρ) = vec( -i[H, ρ] + Σ_k D[C_k](ρ) )

    where the dissipator is::

        D[C](ρ) = C ρ C† − ½ {C†C, ρ}
    """
    d = H.shape[0]
    I = np.eye(d)

    # Unitary part: -i(H⊗I - I⊗Hᵀ)
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))

    for C in c_ops:
        Cd = C.conj().T
        CdC = Cd @ C
        # C ⊗ C* vec(ρ) = vec(C ρ C†)
        L += np.kron(C, C.conj())
        # -½ (C†C ⊗ I + I ⊗ (C†C)ᵀ)
        L -= 0.5 * (np.kron(CdC, I) + np.kron(I, CdC.T))

    return L


class ScipyBackend:
    """Lindblad master equation solver using ``scipy.integrate.solve_ivp``.

    This backend converts all QuTiP operators to dense NumPy arrays
    and integrates the vectorised Lindblad equation with the BDF
    method (suitable for stiff ODEs).
    """

    name: str = "scipy"

    def build_operators(
        self, n_cavity: int, n_atoms: int = 1
    ) -> tuple[Any, list[Any]]:
        """Construct operators via the physics layer (returns Qobj)."""
        return get_operators(n_cavity, n_atoms=n_atoms)

    def build_initial_state(
        self, n_cavity: int, n_atoms: int = 1
    ) -> Any:
        """Construct the default initial state |e, …, 0⟩."""
        return get_initial_state(n_cavity, n_atoms=n_atoms)

    def build_hamiltonian(
        self,
        params: SimParams,
        a: Any,
        sm_list: list[Any],
        *,
        variant: str = "jc",
    ) -> Any:
        """Build a Hamiltonian from ``SimParams``."""
        if variant == "dispersive":
            return dispersive_hamiltonian(
                params.wc, params.wa, params.g, a, sm_list
            )
        return jc_hamiltonian(
            params.wc, params.wa, params.g, a, sm_list,
            use_rwa=params.use_rwa,
        )

    def build_collapse_ops(
        self, params: SimParams, a: Any, sm_list: list[Any]
    ) -> list[Any]:
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
        H: Any,
        psi0: Any,
        tlist: np.ndarray,
        c_ops: list[Any],
        e_ops: list[Any],
        options: dict[str, Any] | None = None,
    ) -> SimResult:
        """Integrate the Lindblad master equation via SciPy.

        Args:
            H: Hamiltonian (Qobj or array).
            psi0: Initial state — ket or density matrix.
            tlist: Time points at which to store the solution.
            c_ops: Collapse operators.
            e_ops: Operators whose expectation values are recorded.
            options: Solver options.  Recognised keys:
                ``method`` (str, default ``"BDF"``),
                ``rtol`` / ``atol`` (float).
        """
        opts = options or {}
        method = opts.get("method", "BDF")
        rtol = opts.get("rtol", 1e-8)
        atol = opts.get("atol", 1e-10)

        H_np = _to_dense(H)
        c_np = [_to_dense(c) for c in c_ops]
        e_np = [_to_dense(e) for e in e_ops]

        psi0_np = _to_dense(psi0)
        if psi0_np.ndim == 2 and psi0_np.shape[1] == 1:
            rho0 = psi0_np @ psi0_np.conj().T
        elif psi0_np.ndim == 2:
            rho0 = psi0_np
        else:
            rho0 = np.outer(psi0_np, psi0_np.conj())

        d = rho0.shape[0]

        L = _build_liouvillian(H_np, c_np)

        y0 = rho0.ravel(order="F")

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            return L @ y

        t0 = time.perf_counter()
        sol = solve_ivp(
            rhs,
            (tlist[0], tlist[-1]),
            y0,
            t_eval=tlist,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        wall = time.perf_counter() - t0

        expect = [np.empty(len(tlist)) for _ in e_np]
        for k, y_col in enumerate(sol.y.T):
            rho_t = y_col.reshape((d, d), order="F")
            for j, O in enumerate(e_np):
                expect[j][k] = np.real(np.trace(O @ rho_t))

        return SimResult(
            times=np.asarray(tlist),
            expect=expect,
            states=None,
            wall_time=wall,
            backend_name=self.name,
        )
