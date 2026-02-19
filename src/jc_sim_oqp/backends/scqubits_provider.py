"""scqubits Hamiltonian provider for transmon-cavity systems.

Uses scqubits' validated transmon model to construct Hamiltonians
that can be passed to the existing QuTiP-based solvers.  This is
*not* a full ``QuantumBackend`` — scqubits builds Hamiltonians
but delegates time-domain dynamics to QuTiP.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scqubits
from qutip import Qobj


@dataclass
class TransmonCavityParams:
    """Parameters for a transmon-cavity system in the scqubits convention.

    Attributes:
        EJ: Josephson energy (GHz).
        EC: Charging energy (GHz).
        ng: Gate charge offset (dimensionless).
        ncut: Charge basis cutoff for the transmon.
        n_cavity: Fock-space truncation for the cavity.
        wc: Cavity frequency (GHz).
        g: Coupling strength (GHz).
        n_levels: Number of transmon levels to retain after truncation.
    """

    EJ: float = 20.0
    EC: float = 0.3
    ng: float = 0.0
    ncut: int = 30
    n_cavity: int = 10
    wc: float = 5.0
    g: float = 0.1
    n_levels: int = 4


class ScqubitsHamiltonianProvider:
    """Construct validated transmon-cavity Hamiltonians via scqubits.

    The returned Hamiltonian is a ``qutip.Qobj`` and can be used
    directly with ``QuTiPBackend.mesolve`` or the existing solvers.
    """

    def build_transmon(
        self, params: TransmonCavityParams
    ) -> scqubits.Transmon:
        """Create a bare scqubits Transmon object."""
        return scqubits.Transmon(
            EJ=params.EJ,
            EC=params.EC,
            ng=params.ng,
            ncut=params.ncut,
            truncated_dim=params.n_levels,
        )

    def build_oscillator(
        self, params: TransmonCavityParams
    ) -> scqubits.Oscillator:
        """Create a bare scqubits Oscillator (cavity)."""
        return scqubits.Oscillator(
            E_osc=params.wc,
            truncated_dim=params.n_cavity,
        )

    def build_hilbert_space(
        self, params: TransmonCavityParams
    ) -> scqubits.HilbertSpace:
        """Build the composite transmon-cavity HilbertSpace with coupling.

        The coupling is the standard charge-photon interaction:
        ``g * n̂_transmon ⊗ (a + a†)``

        Returns:
            scqubits.HilbertSpace with the interaction added.
        """
        tmon = self.build_transmon(params)
        osc = self.build_oscillator(params)

        hilbert_space = scqubits.HilbertSpace([tmon, osc])
        hilbert_space.add_interaction(
            g_strength=params.g,
            op1=(tmon.n_operator, tmon),
            op2=(osc.creation_operator, osc),
            add_hc=True,
        )
        return hilbert_space

    def hamiltonian(self, params: TransmonCavityParams) -> Qobj:
        """Build the full transmon-cavity Hamiltonian as a ``qutip.Qobj``.

        This is the primary entry point.  The returned object can be
        passed to ``mesolve``, ``eigenenergies``, etc.
        """
        hs = self.build_hilbert_space(params)
        return hs.hamiltonian()

    def bare_transmon_spectrum(
        self, params: TransmonCavityParams, n_evals: int = 4
    ) -> np.ndarray:
        """Eigenvalues of the bare (uncoupled) transmon.

        Returns:
            1-D array of the first ``n_evals`` eigenvalues (GHz).
        """
        tmon = self.build_transmon(params)
        return tmon.eigenvals(evals_count=n_evals)

    def dressed_spectrum(
        self, params: TransmonCavityParams, n_evals: int = 10
    ) -> np.ndarray:
        """Eigenvalues of the full coupled system.

        Returns:
            1-D array of the first ``n_evals`` eigenvalues (GHz).
        """
        H = self.hamiltonian(params)
        return np.sort(np.real(H.eigenenergies()))[:n_evals]

    def dispersive_shift_from_spectrum(
        self, params: TransmonCavityParams
    ) -> float:
        """Extract the dispersive shift χ from the dressed spectrum.

        Computes ``χ = (ω_{e,1} − ω_{e,0}) − (ω_{g,1} − ω_{g,0})``
        where the subscripts label (qubit state, photon number).

        Uses eigenvector overlaps with bare states to robustly identify
        the dressed eigenstates, regardless of parameter regime.
        """
        from qutip import basis, tensor

        H = self.hamiltonian(params)
        evals, evecs = H.eigenstates()

        n_t = params.n_levels
        n_c = params.n_cavity

        bare_states = {
            "g0": tensor(basis(n_t, 0), basis(n_c, 0)),
            "g1": tensor(basis(n_t, 0), basis(n_c, 1)),
            "e0": tensor(basis(n_t, 1), basis(n_c, 0)),
            "e1": tensor(basis(n_t, 1), basis(n_c, 1)),
        }

        def _find_dressed_energy(bare_ket: Qobj) -> float:
            overlaps = np.array([
                abs(evec.overlap(bare_ket)) ** 2 for evec in evecs
            ])
            return float(evals[np.argmax(overlaps)])

        E = {label: _find_dressed_energy(ket) for label, ket in bare_states.items()}

        return (E["e1"] - E["e0"]) - (E["g1"] - E["g0"])
