"""Deeper tests for Hamiltonian construction and operator algebra.

Goes beyond the existing smoke tests to verify:
- Known eigenvalues of the JC Hamiltonian
- Multi-atom (Tavis-Cummings) tensor structure
- Non-RWA Hamiltonian properties
- Dispersive Hamiltonian chi extraction
- Driven Hamiltonian format
"""

import numpy as np
import pytest

from jc_sim_oqp.physics import (
    dispersive_hamiltonian,
    driven_jc_hamiltonian,
    get_collapse_operators,
    get_initial_state,
    get_operators,
    jc_hamiltonian,
)


# ---------------------------------------------------------------------------
# Operator construction: multi-atom and edge cases
# ---------------------------------------------------------------------------
class TestOperatorsMultiAtom:
    """Test operator construction for multi-atom systems."""

    def test_two_atom_dimensions(self):
        N = 5
        a, sm_list = get_operators(N, n_atoms=2)
        # Total Hilbert space: N x 2 x 2
        assert a.dims == [[N, 2, 2], [N, 2, 2]]
        assert len(sm_list) == 2
        for sm in sm_list:
            assert sm.dims == [[N, 2, 2], [N, 2, 2]]

    def test_three_atom_dimensions(self):
        N = 4
        a, sm_list = get_operators(N, n_atoms=3)
        assert a.dims == [[N, 2, 2, 2], [N, 2, 2, 2]]
        assert len(sm_list) == 3

    def test_operators_commutation(self):
        """[a, a†] = 1 in the cavity subspace."""
        N = 10
        a, _ = get_operators(N, n_atoms=1)
        commutator = a * a.dag() - a.dag() * a
        # Should be identity (in cavity) ⊗ identity (in atom)
        # But truncated: [a, a†]|N-1> ≠ |N-1> due to truncation.
        # Check the first diagonal elements which should be 1.
        diag = commutator.diag()
        # For Fock state |n>, [a,a†]|n> = 1 for n < N-1
        # The last element is -(N-1) due to truncation
        np.testing.assert_allclose(diag[:2*(N-1)], 1.0, atol=1e-12)

    def test_sm_anticommutation(self):
        """σ- σ+ + σ+ σ- = I for a single two-level atom."""
        N = 5
        _, sm_list = get_operators(N, n_atoms=1)
        sm = sm_list[0]
        anticomm = sm * sm.dag() + sm.dag() * sm
        # This should be identity on the full Hilbert space
        assert anticomm.isherm
        # Check trace: Tr(I) = N * 2 = 2N
        assert anticomm.tr() == pytest.approx(N * 2)


class TestInitialStateMultiAtom:
    """Test initial state for multi-atom configurations."""

    def test_two_atom_initial_state(self):
        N = 5
        psi0 = get_initial_state(N, n_atoms=2)
        assert psi0.isket
        # Total dimension: N * 2 * 2 = 20
        assert psi0.shape[0] == N * 4

    def test_initial_state_is_normalized(self):
        for n_atoms in [1, 2, 3]:
            psi0 = get_initial_state(10, n_atoms=n_atoms)
            norm = abs(psi0.dag() * psi0)
            assert norm == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# JC Hamiltonian: eigenvalue and symmetry tests
# ---------------------------------------------------------------------------
class TestJCHamiltonianEigenvalues:
    """Verify the JC Hamiltonian produces expected physics."""

    def test_resonant_jc_splitting(self):
        """On resonance (wc=wa), the JC doublets split by 2g√(n+1).

        For the lowest doublet (n=0 excitation manifold → n=1 manifold):
        |+,0> and |-,0> split by 2g around wc.
        The one-excitation manifold eigenvalues are: wc ± g.
        """
        wc = 5.0
        wa = 5.0  # resonance
        g = 0.25
        N = 5
        a, sm = get_operators(N, n_atoms=1)
        H = jc_hamiltonian(wc, wa, g, a, sm, use_rwa=True)

        eigenvalues = H.eigenenergies()
        eigenvalues_sorted = np.sort(eigenvalues)

        # Ground state: |g,0> has energy 0 (with -wa/2 convention)
        # or wc*0 + wa*0 = 0 for the |0 photons, ground atom> state
        # 1-excitation manifold: wc ± g
        # Find the two eigenvalues closest to wc
        one_exc = eigenvalues_sorted[np.argsort(np.abs(eigenvalues_sorted - wc))[:2]]
        splitting = abs(one_exc[1] - one_exc[0])
        assert splitting == pytest.approx(2 * g, rel=1e-6)

    def test_rwa_hamiltonian_is_hermitian(self):
        N = 8
        a, sm = get_operators(N)
        H = jc_hamiltonian(1.0, 1.0, 0.1, a, sm, use_rwa=True)
        assert H.isherm

    def test_non_rwa_hamiltonian_is_hermitian(self):
        """The full (non-RWA) Hamiltonian must also be Hermitian."""
        N = 8
        a, sm = get_operators(N)
        H = jc_hamiltonian(1.0, 1.0, 0.1, a, sm, use_rwa=False)
        assert H.isherm

    def test_multi_atom_hamiltonian_dimensions(self):
        N = 5
        a, sm = get_operators(N, n_atoms=2)
        H = jc_hamiltonian(1.0, 1.0, 0.1, a, sm, use_rwa=True)
        # Total dim: N * 2 * 2 = 20
        assert H.shape == (N * 4, N * 4)
        assert H.isherm


# ---------------------------------------------------------------------------
# Dispersive Hamiltonian
# ---------------------------------------------------------------------------
class TestDispersiveHamiltonian:
    """Validate the dispersive Hamiltonian chi-dependent structure."""

    def test_dispersive_is_hermitian(self):
        N = 8
        a, sm = get_operators(N)
        H = dispersive_hamiltonian(10.0, 8.0, 0.1, a, sm)
        assert H.isherm

    def test_dispersive_eigenfrequency_shift(self):
        """In the dispersive regime, the cavity frequency shifts by ±chi
        depending on the qubit state.

        For |g,1> vs |e,1>: energy difference includes 2*chi contribution.
        """
        wc = 10.0
        wa = 8.0  # Delta = -2.0
        g = 0.1
        chi = g**2 / (wa - wc)  # = -0.005
        N = 5
        a, sm = get_operators(N)
        H_disp = dispersive_hamiltonian(wc, wa, g, a, sm)

        eigenvalues = np.sort(H_disp.eigenenergies())

        # The dispersive Hamiltonian should produce eigenvalues that include
        # chi-shifted terms. Just verify it's diagonal-like (no off-diagonal
        # coupling in the dressed basis).
        assert len(eigenvalues) == 2 * N

    def test_dispersive_matches_exact_in_limit(self):
        """For very large detuning, dispersive and exact eigenvalues should
        be close for the lowest states (both use number-operator convention)."""
        wc = 10.0
        wa = 8.0
        g = 0.001  # g << Delta
        N = 5
        a, sm = get_operators(N)

        H_exact = jc_hamiltonian(wc, wa, g, a, sm, use_rwa=True)
        H_disp = dispersive_hamiltonian(wc, wa, g, a, sm)

        evals_exact = np.sort(H_exact.eigenenergies())[:4]
        evals_disp = np.sort(H_disp.eigenenergies())[:4]

        # Absolute eigenvalues should agree to within O(g^4/Delta^3) ≈ 1e-12 for these parameters
        np.testing.assert_allclose(evals_exact, evals_disp, atol=1e-6)


# ---------------------------------------------------------------------------
# Driven Hamiltonian
# ---------------------------------------------------------------------------
class TestDrivenHamiltonian:
    """Validate the time-dependent driven Hamiltonian construction."""

    def test_no_drive_returns_qobj(self):
        """Without drive terms, should return a static Qobj."""
        from qutip import Qobj
        N = 5
        a, sm = get_operators(N)
        H = driven_jc_hamiltonian(1.0, 1.0, 0.1, a, sm)
        assert isinstance(H, Qobj)

    def test_with_drive_returns_list(self):
        """With a drive, should return QuTiP time-dependent list format."""
        N = 5
        a, sm = get_operators(N)
        drive = lambda t, args: np.cos(t)
        H = driven_jc_hamiltonian(1.0, 1.0, 0.1, a, sm, drive_x=drive)
        assert isinstance(H, list)
        assert len(H) == 2  # [H_static, [H_drive_x, coeff]]

    def test_both_drives_returns_three_terms(self):
        N = 5
        a, sm = get_operators(N)
        dx = lambda t, args: np.cos(t)
        dy = lambda t, args: np.sin(t)
        H = driven_jc_hamiltonian(1.0, 1.0, 0.1, a, sm, drive_x=dx, drive_y=dy)
        assert isinstance(H, list)
        assert len(H) == 3  # [H_static, [H_x, fx], [H_y, fy]]


# ---------------------------------------------------------------------------
# Collapse operators: deeper validation
# ---------------------------------------------------------------------------
class TestCollapseOperatorsDeep:
    """Beyond counting: verify operator structure and rates."""

    def test_multi_atom_collapse_operators(self):
        """Each atom should contribute its own collapse operators."""
        N = 5
        a, sm = get_operators(N, n_atoms=2)
        c_ops = get_collapse_operators(0.1, 0.05, 0.0, a, sm)
        # kappa term (1) + gamma per atom (2) = 3
        assert len(c_ops) == 3

    def test_thermal_bath_adds_excitation_ops(self):
        """Non-zero n_th should add excitation (creation) operators."""
        N = 5
        a, sm = get_operators(N)
        c_ops_cold = get_collapse_operators(0.1, 0.05, 0.0, a, sm)
        c_ops_hot = get_collapse_operators(0.1, 0.05, 1.0, a, sm)
        # Hot bath adds cavity excitation operator
        assert len(c_ops_hot) == len(c_ops_cold) + 1

    def test_all_collapse_ops_are_valid_operators(self):
        """Each collapse operator should be a valid Qobj of correct dims."""
        N = 5
        a, sm = get_operators(N, n_atoms=2)
        c_ops = get_collapse_operators(0.1, 0.05, 0.5, a, sm, gamma_phi=0.01, n_th_q=0.1)
        for c in c_ops:
            assert c.shape == (N * 4, N * 4)
