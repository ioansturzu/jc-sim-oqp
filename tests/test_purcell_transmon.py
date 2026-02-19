"""Tests for pure-math physics formulas: purcell.py, transmon.py, spectra.py.

These modules are backend-agnostic (numpy only) and can be validated against
known analytical results from textbook cavity QED.
"""

import numpy as np
import pytest

from jc_sim_oqp.physics.purcell import (
    beta_factor,
    cavity_enhanced_decay,
    cooperativity,
    purcell_factor,
)
from jc_sim_oqp.physics.transmon import (
    critical_photon_number,
    dispersive_shift,
    purcell_limit_t1,
)


# ---------------------------------------------------------------------------
# purcell.py
# ---------------------------------------------------------------------------
class TestPurcellFactor:
    """Validate F_p = 4g^2 / (kappa * gamma)."""

    def test_known_value(self):
        # g=0.1, kappa=1.0, gamma=0.01 → F_p = 4*0.01 / 0.01 = 4.0
        fp = purcell_factor(g=0.1, kappa=1.0, gamma=0.01)
        assert fp == pytest.approx(4.0)

    def test_zero_coupling_gives_zero(self):
        assert purcell_factor(g=0.0, kappa=1.0, gamma=0.01) == 0.0

    def test_zero_kappa_returns_zero(self):
        """Guard: division by zero should return 0, not raise."""
        assert purcell_factor(g=0.1, kappa=0.0, gamma=0.01) == 0.0

    def test_zero_gamma_returns_zero(self):
        assert purcell_factor(g=0.1, kappa=1.0, gamma=0.0) == 0.0

    def test_scaling_with_g_squared(self):
        """Doubling g should quadruple F_p."""
        fp1 = purcell_factor(g=0.1, kappa=1.0, gamma=0.01)
        fp2 = purcell_factor(g=0.2, kappa=1.0, gamma=0.01)
        assert fp2 == pytest.approx(4.0 * fp1)


class TestCooperativity:
    """Validate eta_d = 2g^2 / (kappa * gamma_d)."""

    def test_no_dephasing(self):
        # gamma_d = gamma/2 = 0.005
        # eta = 2 * 0.01 / (1.0 * 0.005) = 4.0
        eta = cooperativity(g=0.1, kappa=1.0, gamma=0.01)
        assert eta == pytest.approx(4.0)

    def test_with_dephasing_reduces_cooperativity(self):
        eta_no_deph = cooperativity(g=0.1, kappa=1.0, gamma=0.01, gamma_deph=0.0)
        eta_with_deph = cooperativity(g=0.1, kappa=1.0, gamma=0.01, gamma_deph=0.01)
        assert eta_with_deph < eta_no_deph

    def test_zero_params(self):
        assert cooperativity(g=0.0, kappa=1.0, gamma=0.01) == 0.0
        assert cooperativity(g=0.1, kappa=0.0, gamma=0.01) == 0.0


class TestBetaFactor:
    """Validate beta = F_p / (1 + F_p)."""

    def test_range_zero_to_one(self):
        beta = beta_factor(g=0.1, kappa=1.0, gamma=0.01)
        assert 0.0 <= beta <= 1.0

    def test_large_fp_approaches_one(self):
        """Very strong coupling → beta ≈ 1."""
        beta = beta_factor(g=10.0, kappa=0.01, gamma=0.01)
        assert beta > 0.99

    def test_zero_coupling_gives_zero(self):
        assert beta_factor(g=0.0, kappa=1.0, gamma=0.01) == 0.0

    def test_known_value(self):
        # F_p = 4.0 from test above → beta = 4/5 = 0.8
        beta = beta_factor(g=0.1, kappa=1.0, gamma=0.01)
        assert beta == pytest.approx(0.8)


class TestCavityEnhancedDecay:
    """Validate gamma_cav = gamma * (1 + F_p)."""

    def test_enhancement_factor(self):
        gamma = 0.01
        Fp = 4.0
        result = cavity_enhanced_decay(gamma, Fp)
        assert result == pytest.approx(gamma * (1 + Fp))

    def test_no_enhancement(self):
        """F_p = 0 should return the bare gamma."""
        assert cavity_enhanced_decay(0.05, 0.0) == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# transmon.py
# ---------------------------------------------------------------------------
class TestDispersiveShift:
    """Validate chi = g^2 / Delta."""

    def test_known_value(self):
        # g=0.1, Delta=2.0 → chi = 0.01/2.0 = 0.005
        chi = dispersive_shift(g=0.1, detuning=2.0)
        assert chi == pytest.approx(0.005)

    def test_sign_with_negative_detuning(self):
        """Negative detuning should give negative chi."""
        chi = dispersive_shift(g=0.1, detuning=-2.0)
        assert chi == pytest.approx(-0.005)

    def test_zero_detuning_returns_inf(self):
        chi = dispersive_shift(g=0.1, detuning=0.0)
        assert chi == np.inf


class TestCriticalPhotonNumber:
    """Validate n_crit = Delta^2 / (4 * g^2)."""

    def test_known_value(self):
        # Delta=2, g=0.1 → n_crit = 4 / 0.04 = 100
        n_crit = critical_photon_number(g=0.1, detuning=2.0)
        assert n_crit == pytest.approx(100.0)

    def test_zero_coupling_returns_inf(self):
        assert critical_photon_number(g=0.0, detuning=2.0) == np.inf

    def test_larger_g_gives_smaller_ncrit(self):
        """Stronger coupling breaks dispersive approx at fewer photons."""
        n1 = critical_photon_number(g=0.1, detuning=2.0)
        n2 = critical_photon_number(g=0.2, detuning=2.0)
        assert n2 < n1


class TestPurcellLimitT1:
    """Validate T1_P = 1 / ((g/Delta)^2 * kappa)."""

    def test_known_value(self):
        # g=0.1, Delta=2.0, kappa=1.0 → Gamma_P = (0.05)^2 * 1.0 = 0.0025
        # T1 = 400
        t1 = purcell_limit_t1(g=0.1, detuning=2.0, kappa=1.0)
        assert t1 == pytest.approx(400.0)

    def test_zero_detuning_returns_zero(self):
        """On resonance, Purcell decay is instantaneous (T1 → 0)."""
        assert purcell_limit_t1(g=0.1, detuning=0.0, kappa=1.0) == 0.0

    def test_stronger_coupling_shorter_t1(self):
        t1_weak = purcell_limit_t1(g=0.01, detuning=2.0, kappa=1.0)
        t1_strong = purcell_limit_t1(g=0.1, detuning=2.0, kappa=1.0)
        assert t1_strong < t1_weak


# ---------------------------------------------------------------------------
# spectra.py
# ---------------------------------------------------------------------------
from jc_sim_oqp.physics.spectra import (
    reflection_coefficient_corrected,
    reflection_coefficient_naive,
)


class TestReflectionCoefficientNaive:
    """Validate the naive semiclassical reflection coefficient."""

    def test_empty_cavity_on_resonance(self):
        """g=0, on resonance: empty cavity reflection."""
        r = reflection_coefficient_naive(
            detuning=0.0, g=0.0, kappa_in=0.5, kappa_sc=0.5, gamma=0.01
        )
        # Empty cavity on resonance: r should be real
        assert isinstance(r, complex)

    def test_overcoupled_empty_cavity(self):
        """Overcoupled (kappa_in >> kappa_sc), g=0, on resonance → r ≈ -1."""
        r = reflection_coefficient_naive(
            detuning=0.0, g=0.0, kappa_in=1.0, kappa_sc=0.0, gamma=0.01
        )
        # r = 1 - kappa_in/tilde_kappa = 1 - kappa_in/(kappa/2) = 1 - 2 = -1
        assert abs(r - (-1.0)) < 0.01

    def test_strong_atom_on_resonance_reflects(self):
        """With a strongly coupled atom on resonance, more light is reflected."""
        r_empty = reflection_coefficient_naive(
            detuning=0.0, g=0.0, kappa_in=0.5, kappa_sc=0.5, gamma=0.01
        )
        r_atom = reflection_coefficient_naive(
            detuning=0.0, g=1.0, kappa_in=0.5, kappa_sc=0.5, gamma=0.01
        )
        # Atom changes the reflection
        assert abs(r_atom) != pytest.approx(abs(r_empty), abs=0.01)


class TestReflectionCoefficientCorrected:
    """Validate the corrected (dephasing-aware) reflection power."""

    def test_raises_for_nonzero_detuning(self):
        with pytest.raises(NotImplementedError):
            reflection_coefficient_corrected(
                detuning=1.0, g=0.1, kappa_in=0.5, kappa_sc=0.5,
                gamma=0.01, gamma_deph=0.01
            )

    def test_returns_float(self):
        result = reflection_coefficient_corrected(
            detuning=0.0, g=0.1, kappa_in=0.5, kappa_sc=0.5,
            gamma=0.01, gamma_deph=0.01
        )
        assert isinstance(result, float)

    def test_physical_range(self):
        """Reflected power should be between 0 and ~1 for passive system."""
        result = reflection_coefficient_corrected(
            detuning=0.0, g=0.1, kappa_in=0.5, kappa_sc=0.5,
            gamma=0.01, gamma_deph=0.01
        )
        assert 0.0 <= result <= 1.0 + 1e-6
