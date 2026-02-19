"""Tests for SimParams dataclass: defaults, properties, and type coercion."""

import numpy as np
import pytest

from jc_sim_oqp.io import SimParams


class TestSimParamsDefaults:
    """Verify default values match the documented JC model conventions."""

    def test_default_resonance(self):
        """Default cavity and atom frequencies should be equal (resonant)."""
        p = SimParams()
        assert p.wc == p.wa

    def test_default_coupling_weaker_than_frequencies(self):
        """Default g should be much smaller than wc (weak coupling)."""
        p = SimParams()
        assert p.g < p.wc

    def test_default_hilbert_space_size(self):
        p = SimParams()
        assert p.N == 15
        assert p.n_atoms == 1

    def test_default_rwa_enabled(self):
        p = SimParams()
        assert p.use_rwa is True

    def test_default_thermal_occupation_zero(self):
        p = SimParams()
        assert p.n_th_a == 0.0
        assert p.n_th_q == 0.0


class TestSimParamsPostInit:
    """Verify __post_init__ enforces integer types on dimension parameters."""

    def test_float_n_atoms_cast_to_int(self):
        p = SimParams(n_atoms=2.0)
        assert isinstance(p.n_atoms, int)
        assert p.n_atoms == 2

    def test_float_N_cast_to_int(self):
        p = SimParams(N=10.0)
        assert isinstance(p.N, int)
        assert p.N == 10

    def test_numpy_float_cast(self):
        """Ensure numpy scalar types are also cast correctly."""
        p = SimParams(N=np.float64(20), n_atoms=np.int32(3))
        assert isinstance(p.N, int)
        assert isinstance(p.n_atoms, int)
        assert p.N == 20
        assert p.n_atoms == 3


class TestSimParamsKappa:
    """Verify the derived kappa property."""

    def test_kappa_is_sum_of_components(self):
        p = SimParams(kappa_in=0.003, kappa_sc=0.002)
        assert p.kappa == pytest.approx(0.005)

    def test_kappa_zero_when_both_zero(self):
        p = SimParams(kappa_in=0.0, kappa_sc=0.0)
        assert p.kappa == 0.0

    def test_kappa_with_only_input_coupling(self):
        p = SimParams(kappa_in=0.1, kappa_sc=0.0)
        assert p.kappa == pytest.approx(0.1)


class TestSimParamsTlist:
    """Verify the time vector generation."""

    def test_tlist_length(self):
        p = SimParams(t_max=100.0, n_steps=500)
        assert len(p.tlist) == 500

    def test_tlist_endpoints(self):
        p = SimParams(t_max=50.0, n_steps=100)
        assert p.tlist[0] == pytest.approx(0.0)
        assert p.tlist[-1] == pytest.approx(50.0)

    def test_tlist_uniform_spacing(self):
        p = SimParams(t_max=10.0, n_steps=11)
        dt = np.diff(p.tlist)
        np.testing.assert_allclose(dt, dt[0], rtol=1e-12)

    def test_tlist_returns_ndarray(self):
        p = SimParams()
        assert isinstance(p.tlist, np.ndarray)
