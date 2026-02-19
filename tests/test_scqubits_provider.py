"""Tests for the scqubits Hamiltonian provider."""

import numpy as np
import pytest

from jc_sim_oqp.backends.scqubits_provider import (
    ScqubitsHamiltonianProvider,
    TransmonCavityParams,
)


@pytest.fixture()
def provider():
    return ScqubitsHamiltonianProvider()


@pytest.fixture()
def default_params():
    return TransmonCavityParams()


class TestTransmonCavityParams:
    """Verify parameter defaults are sensible."""

    def test_defaults(self):
        p = TransmonCavityParams()
        assert p.EJ == 20.0
        assert p.EC == 0.3
        assert p.n_cavity == 10
        assert p.n_levels == 4


class TestBareTransmon:
    """Validate bare transmon spectrum against known limits."""

    def test_eigenvalues_count(self, provider, default_params):
        evals = provider.bare_transmon_spectrum(default_params, n_evals=4)
        assert len(evals) == 4

    def test_transition_frequency_positive(self, provider, default_params):
        evals = provider.bare_transmon_spectrum(default_params, n_evals=2)
        omega_01 = evals[1] - evals[0]
        assert omega_01 > 0

    def test_anharmonicity_negative(self, provider, default_params):
        """Transmon anharmonicity α = ω₁₂ − ω₀₁ should be ≈ −EC."""
        evals = provider.bare_transmon_spectrum(default_params, n_evals=3)
        omega_01 = evals[1] - evals[0]
        omega_12 = evals[2] - evals[1]
        alpha = omega_12 - omega_01
        assert alpha < 0
        np.testing.assert_allclose(alpha, -default_params.EC, rtol=0.15)


class TestDressedSystem:
    """Validate the coupled transmon-cavity system."""

    def test_hamiltonian_is_hermitian(self, provider, default_params):
        H = provider.hamiltonian(default_params)
        assert H.isherm

    def test_hamiltonian_dimension(self, provider, default_params):
        H = provider.hamiltonian(default_params)
        dim = default_params.n_levels * default_params.n_cavity
        assert H.shape == (dim, dim)

    def test_dressed_spectrum_ordered(self, provider, default_params):
        evals = provider.dressed_spectrum(default_params, n_evals=8)
        assert np.all(np.diff(evals) >= 0)

    def test_vacuum_rabi_splitting_at_resonance(self, provider):
        """At resonance (ωa ≈ ωc), the splitting should be ≈ 2g."""
        bare_evals = provider.bare_transmon_spectrum(
            TransmonCavityParams(), n_evals=2
        )
        omega_01 = bare_evals[1] - bare_evals[0]

        params = TransmonCavityParams(wc=omega_01, g=0.1, n_cavity=15)
        evals = provider.dressed_spectrum(params, n_evals=6)

        spacings = np.diff(evals)
        min_spacing = np.min(spacings[spacings > 0.01])
        np.testing.assert_allclose(min_spacing, 2 * params.g, rtol=0.3)


class TestDispersiveShift:
    """Validate the numerically extracted dispersive shift."""

    def _make_dispersive_params(self, provider, g=0.05):
        """Build params in the dispersive regime (ωc << ωa)."""
        bare = provider.bare_transmon_spectrum(TransmonCavityParams(), n_evals=2)
        omega_a = bare[1] - bare[0]
        return TransmonCavityParams(wc=omega_a - 2.0, g=g, n_cavity=10)

    def test_chi_is_small(self, provider):
        """In the dispersive regime, |χ| should be much smaller than g."""
        params = self._make_dispersive_params(provider, g=0.05)
        chi = provider.dispersive_shift_from_spectrum(params)
        assert abs(chi) < params.g
        assert abs(chi) > 0

    def test_chi_scales_as_g_squared(self, provider):
        """χ ∝ g²:  doubling g should roughly quadruple χ.

        This is a self-consistency check that does not depend on the
        two-level approximation (which doesn't apply to a transmon).
        """
        params_1 = self._make_dispersive_params(provider, g=0.03)
        params_2 = self._make_dispersive_params(provider, g=0.06)

        chi_1 = provider.dispersive_shift_from_spectrum(params_1)
        chi_2 = provider.dispersive_shift_from_spectrum(params_2)

        ratio = chi_2 / chi_1
        np.testing.assert_allclose(ratio, 4.0, rtol=0.3)
