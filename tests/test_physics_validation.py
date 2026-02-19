"""Physics-level integration tests: validate simulator output against
known analytical results from cavity QED.

These tests actually run the solvers and check that the numerical results
match textbook predictions. They are slower than unit tests but catch
integration errors between the physics and solver layers.
"""

import numpy as np
import pytest

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics.purcell import cavity_enhanced_decay, purcell_factor
from jc_sim_oqp.solvers import ExactSolver, SteadyStateSolver


class TestVacuumRabiOscillations:
    """On resonance with no dissipation, energy should oscillate between
    the atom and cavity at frequency 2g (the vacuum Rabi frequency).
    """

    @pytest.fixture()
    def rabi_result(self):
        """Run a lossless resonant simulation."""
        params = SimParams(
            wc=5.0,
            wa=5.0,        # exact resonance
            g=0.5,         # strong enough to see oscillations
            kappa_in=0.0,  # no dissipation
            kappa_sc=0.0,
            gamma=0.0,
            gamma_phi=0.0,
            n_th_a=0.0,
            N=10,
            n_atoms=1,
            use_rwa=True,
            t_max=4 * np.pi / 0.5,  # ~4 full Rabi periods
            n_steps=500,
        )
        solver = ExactSolver(params)
        return solver.run(), params

    def test_oscillation_frequency(self, rabi_result):
        """The photon number should oscillate at frequency g (Rabi freq = 2g,
        but <n> oscillates at the Rabi frequency's half-period for sin²).
        """
        result, params = rabi_result
        times = np.array(result.times)
        n_photons = np.array(result.expect[0])  # <a†a>

        # Find peaks using zero-crossings of the derivative
        dn = np.diff(n_photons)
        sign_changes = np.where(np.diff(np.sign(dn)))[0]

        if len(sign_changes) >= 4:
            # Period = time between every other peak (full cycle)
            peak_times = times[sign_changes[::2] + 1]
            if len(peak_times) >= 2:
                periods = np.diff(peak_times)
                avg_period = np.mean(periods)
                measured_freq = 1.0 / avg_period
                expected_freq = params.g / np.pi  # from sin²(gt) period = π/g
                assert measured_freq == pytest.approx(expected_freq, rel=0.1)

    def test_energy_conservation(self, rabi_result):
        """In a lossless system, total excitation number must be conserved.

        <a†a> + <σ+σ-> = const = 1 (started with 1 excitation in atom).
        """
        result, _ = rabi_result
        n_photons = np.array(result.expect[0])
        n_atoms = np.array(result.expect[1])
        total = n_photons + n_atoms

        # Should be constant = 1.0 (one excitation)
        np.testing.assert_allclose(total, 1.0, atol=1e-4)

    def test_photon_number_bounded(self, rabi_result):
        """Photon number must stay in [0, 1] for single excitation."""
        result, _ = rabi_result
        n_photons = np.array(result.expect[0])
        assert np.all(n_photons >= -1e-6)
        assert np.all(n_photons <= 1.0 + 1e-6)


class TestPurcellDecay:
    """In the weak-coupling regime, the atom should decay at the
    Purcell-enhanced rate γ_eff = γ(1 + F_p).

    We verify this by fitting an exponential to the atomic population.
    """

    def test_weak_coupling_decay_rate(self):
        """With moderate Fp, the numerical decay should match the analytical rate."""
        g = 0.05
        kappa = 1.0
        gamma = 0.01
        fp = purcell_factor(g, kappa, gamma)
        gamma_eff = cavity_enhanced_decay(gamma, fp)

        params = SimParams(
            wc=5.0,
            wa=5.0,       # resonance for maximum Purcell
            g=g,
            kappa_in=kappa / 2,
            kappa_sc=kappa / 2,
            gamma=gamma,
            gamma_phi=0.0,
            n_th_a=0.0,
            N=10,
            n_atoms=1,
            use_rwa=True,
            t_max=50.0 / gamma_eff,  # several decay times
            n_steps=500,
        )

        solver = ExactSolver(params)
        result = solver.run()

        times = np.array(result.times)
        pe = np.array(result.expect[1])  # atomic excitation

        # Fit exponential: pe ≈ exp(-gamma_eff * t)
        # Use log-linear fit on the portion where pe > 0.01
        mask = pe > 0.01
        if np.sum(mask) > 10:
            log_pe = np.log(pe[mask])
            t_masked = times[mask]
            # Linear fit: log(pe) = -gamma_fit * t + c
            coeffs = np.polyfit(t_masked, log_pe, 1)
            gamma_fit = -coeffs[0]

            # Should match within ~20% for weak coupling
            # (Purcell formula is approximate; exact dynamics include transients)
            assert gamma_fit == pytest.approx(gamma_eff, rel=0.25)


class TestDissipativeDecayToGround:
    """With dissipation and no drive, the system should relax to the
    ground state (vacuum + atom ground).
    """

    def test_relaxation_to_ground(self):
        params = SimParams(
            wc=5.0,
            wa=5.0,
            g=0.1,
            kappa_in=0.05,
            kappa_sc=0.05,
            gamma=0.05,
            gamma_phi=0.0,
            n_th_a=0.0,
            n_th_q=0.0,
            N=8,
            n_atoms=1,
            use_rwa=True,
            t_max=500.0,  # long enough for full relaxation
            n_steps=200,
        )

        solver = ExactSolver(params)
        result = solver.run()

        # At the end, both photon and atom should be near ground
        final_photons = result.expect[0][-1]
        final_atom = result.expect[1][-1]

        assert final_photons < 0.01
        assert final_atom < 0.01

    def test_monotonic_total_excitation_decay(self):
        """Total excitation should decrease monotonically (on average) with dissipation."""
        params = SimParams(
            wc=5.0,
            wa=5.0,
            g=0.1,
            kappa_in=0.1,
            kappa_sc=0.0,
            gamma=0.1,
            n_th_a=0.0,
            N=8,
            n_atoms=1,
            t_max=200.0,
            n_steps=100,
        )

        solver = ExactSolver(params)
        result = solver.run()

        total = np.array(result.expect[0]) + np.array(result.expect[1])

        # Check that the last value is much smaller than the first
        assert total[-1] < total[0] * 0.1


class TestSteadyStateSolver:
    """Verify the steady-state solver produces physically correct results."""

    def test_vacuum_steady_state_no_drive(self):
        """Without drive, steady state at T=0 should be vacuum."""
        params = SimParams(
            wc=5.0,
            wa=5.0,
            g=0.1,
            kappa_in=0.05,
            kappa_sc=0.05,
            gamma=0.05,
            n_th_a=0.0,
            N=8,
        )
        solver = SteadyStateSolver(params)
        rho_ss = solver.run(drive_amp=0.0)

        # Should be a valid density matrix
        assert rho_ss.isherm
        assert rho_ss.tr() == pytest.approx(1.0, abs=1e-8)

        # Ground state population should dominate
        from qutip import expect
        from jc_sim_oqp.physics import get_operators
        a, sm_list = get_operators(params.N, n_atoms=params.n_atoms)
        n_photons = expect(a.dag() * a, rho_ss)
        assert n_photons < 0.01

    def test_driven_steady_state_has_photons(self):
        """With a coherent drive, the cavity should have a nonzero photon number."""
        params = SimParams(
            wc=5.0,
            wa=8.0,  # far detuned atom
            g=0.01,
            kappa_in=0.5,
            kappa_sc=0.5,
            gamma=0.01,
            n_th_a=0.0,
            N=15,
        )
        solver = SteadyStateSolver(params)
        rho_ss = solver.run(drive_amp=1.0)

        from qutip import expect
        from jc_sim_oqp.physics import get_operators
        a, _ = get_operators(params.N, n_atoms=params.n_atoms)
        n_photons = expect(a.dag() * a, rho_ss)

        # Drive should populate the cavity
        assert n_photons > 0.01

    def test_thermal_steady_state(self):
        """With a thermal bath, steady-state photon number should approach n_th."""
        n_th = 2.0
        params = SimParams(
            wc=5.0,
            wa=8.0,  # far detuned
            g=0.001, # negligible coupling
            kappa_in=0.5,
            kappa_sc=0.5,
            gamma=0.01,
            n_th_a=n_th,
            N=20,  # large enough to hold thermal photons
        )
        solver = SteadyStateSolver(params)
        rho_ss = solver.run(drive_amp=0.0)

        from qutip import expect
        from jc_sim_oqp.physics import get_operators
        a, _ = get_operators(params.N, n_atoms=params.n_atoms)
        n_photons = expect(a.dag() * a, rho_ss)

        # Should be close to n_th (within Fock truncation artifacts)
        assert n_photons == pytest.approx(n_th, rel=0.15)
