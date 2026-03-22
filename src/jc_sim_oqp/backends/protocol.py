"""Protocol defining the interface that any simulation backend must satisfy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

    from jc_sim_oqp.backends.result import SimResult
    from jc_sim_oqp.io import SimParams


@runtime_checkable
class QuantumBackend(Protocol):
    """Protocol for a quantum simulation backend.

    Any backend (QuTiP, SciPy, dynamiqs, …) must implement these four
    methods so it can be used interchangeably by the solver classes.
    """

    def build_operators(
        self, n_cavity: int, n_atoms: int
    ) -> tuple[Any, list[Any]]:
        """Construct the cavity and atom operators.

        Args:
            n_cavity: Fock space truncation for the cavity.
            n_atoms: Number of two-level atoms.

        Returns:
            (a, sm_list): Cavity annihilation operator and list of atom
            lowering operators, in the backend's native representation.
        """
        ...

    def build_hamiltonian(
        self,
        params: SimParams,
        a: Any,
        sm_list: list[Any],
        *,
        variant: str = "jc",
    ) -> Any:
        """Build a Hamiltonian from simulation parameters.

        Args:
            params: Simulation parameters (frequencies, coupling, …).
            a: Cavity operator (from ``build_operators``).
            sm_list: Atom operators (from ``build_operators``).
            variant: ``"jc"`` for Jaynes-Cummings, ``"dispersive"``
                     for the dispersive approximation.

        Returns:
            Hamiltonian in the backend's native format.
        """
        ...

    def build_collapse_ops(
        self, params: SimParams, a: Any, sm_list: list[Any]
    ) -> list[Any]:
        """Build the collapse (Lindblad jump) operators.

        Args:
            params: Simulation parameters (decay rates, thermal photons, …).
            a: Cavity operator.
            sm_list: Atom operators.

        Returns:
            List of collapse operators.
        """
        ...

    def mesolve(
        self,
        H: Any,
        psi0: Any,
        tlist: np.ndarray,
        c_ops: list[Any],
        e_ops: list[Any],
        options: dict[str, Any] | None = None,
    ) -> SimResult:
        """Integrate the Lindblad master equation.

        Args:
            H: Hamiltonian.
            psi0: Initial state (ket or density matrix).
            tlist: Array of time points.
            c_ops: Collapse operators.
            e_ops: Expectation-value operators.
            options: Backend-specific solver options.

        Returns:
            A ``SimResult`` containing times and expectation values.
        """
        ...
