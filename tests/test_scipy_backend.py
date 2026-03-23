"""Tests for the SciPy Lindblad backend.

Validates protocol conformance and numerical agreement with QuTiP.
"""

import numpy as np
import pytest

from jc_sim_oqp.backends.protocol import QuantumBackend
from jc_sim_oqp.backends.qutip_backend import QuTiPBackend
from jc_sim_oqp.backends.scipy_backend import ScipyBackend
from jc_sim_oqp.io import SimParams


@pytest.fixture
def scipy_backend():
    return ScipyBackend()


@pytest.fixture
def qutip_backend():
    return QuTiPBackend()


@pytest.fixture
def resonant_params():
    """Resonant JC system with moderate dissipation."""
    return SimParams(
        wc=1.0 * 2 * np.pi,
        wa=1.0 * 2 * np.pi,
        g=0.05 * 2 * np.pi,
        kappa_in=0.0025,
        kappa_sc=0.0025,
        gamma=0.005,
        n_th_a=0.0,
        N=5,
        n_atoms=1,
        t_max=25.0,
        n_steps=200,
        use_rwa=True,
    )


class TestProtocolConformance:
    """ScipyBackend must satisfy the QuantumBackend protocol."""

    def test_is_quantum_backend(self, scipy_backend):
        assert isinstance(scipy_backend, QuantumBackend)

    def test_has_name(self, scipy_backend):
        assert scipy_backend.name == "scipy"


class TestLiouvillian:
    """Verify the Liouvillian construction."""

    def test_trace_preservation(self, scipy_backend, resonant_params):
        """Tr(ρ) must remain 1 at all times."""
        a, sm_list = scipy_backend.build_operators(
            resonant_params.N, resonant_params.n_atoms
        )
        psi0 = scipy_backend.build_initial_state(
            resonant_params.N, resonant_params.n_atoms
        )
        H = scipy_backend.build_hamiltonian(resonant_params, a, sm_list)
        c_ops = scipy_backend.build_collapse_ops(resonant_params, a, sm_list)

        d = resonant_params.N * 2 ** resonant_params.n_atoms
        identity = np.eye(d)
        result = scipy_backend.mesolve(
            H, psi0, resonant_params.tlist, c_ops, [identity]
        )

        np.testing.assert_allclose(result.expect[0], 1.0, atol=1e-6)


class TestAgreementWithQuTiP:
    """Expectation values from SciPy must match QuTiP to high precision."""

    def test_vacuum_rabi_oscillation(
        self, scipy_backend, qutip_backend, resonant_params
    ):
        """⟨n_cav⟩ and ⟨n_atom⟩ traces must agree to < 1e-4."""
        a, sm = scipy_backend.build_operators(
            resonant_params.N, resonant_params.n_atoms
        )
        psi0 = scipy_backend.build_initial_state(
            resonant_params.N, resonant_params.n_atoms
        )
        H = scipy_backend.build_hamiltonian(resonant_params, a, sm)
        c_ops = scipy_backend.build_collapse_ops(resonant_params, a, sm)

        n_cav = a.dag() * a
        n_atom = sum(s.dag() * s for s in sm)
        e_ops = [n_cav, n_atom]

        res_scipy = scipy_backend.mesolve(
            H, psi0, resonant_params.tlist, c_ops, e_ops
        )
        res_qutip = qutip_backend.mesolve(
            H, psi0, resonant_params.tlist, c_ops, e_ops
        )

        for j in range(2):
            np.testing.assert_allclose(
                res_scipy.expect[j],
                res_qutip.expect[j],
                atol=1e-4,
                err_msg=f"Mismatch in e_ops[{j}]",
            )

    def test_dissipative_decay(self, scipy_backend, qutip_backend):
        """System with strong damping should decay to ground state."""
        params = SimParams(
            wc=1.0 * 2 * np.pi,
            wa=1.0 * 2 * np.pi,
            g=0.05 * 2 * np.pi,
            kappa_in=0.05,
            kappa_sc=0.05,
            gamma=0.1,
            n_th_a=0.0,
            N=5,
            n_atoms=1,
            t_max=50.0,
            n_steps=300,
            use_rwa=True,
        )

        a, sm = scipy_backend.build_operators(params.N, params.n_atoms)
        psi0 = scipy_backend.build_initial_state(params.N, params.n_atoms)
        H = scipy_backend.build_hamiltonian(params, a, sm)
        c_ops = scipy_backend.build_collapse_ops(params, a, sm)

        n_cav = a.dag() * a
        n_atom = sum(s.dag() * s for s in sm)
        e_ops = [n_cav, n_atom]

        res_scipy = scipy_backend.mesolve(
            H, psi0, params.tlist, c_ops, e_ops
        )
        res_qutip = qutip_backend.mesolve(
            H, psi0, params.tlist, c_ops, e_ops
        )

        for j in range(2):
            np.testing.assert_allclose(
                res_scipy.expect[j],
                res_qutip.expect[j],
                atol=1e-4,
                err_msg=f"Mismatch in e_ops[{j}] for dissipative decay",
            )

        # Both backends should show decay to near zero
        assert res_scipy.expect[0][-1] < 0.01
        assert res_scipy.expect[1][-1] < 0.01

    def test_sim_result_fields(self, scipy_backend, resonant_params):
        """SimResult must contain well-formed fields."""
        a, sm = scipy_backend.build_operators(
            resonant_params.N, resonant_params.n_atoms
        )
        psi0 = scipy_backend.build_initial_state(
            resonant_params.N, resonant_params.n_atoms
        )
        H = scipy_backend.build_hamiltonian(resonant_params, a, sm)
        c_ops = scipy_backend.build_collapse_ops(resonant_params, a, sm)
        e_ops = [a.dag() * a]

        result = scipy_backend.mesolve(
            H, psi0, resonant_params.tlist, c_ops, e_ops
        )

        assert result.backend_name == "scipy"
        assert result.wall_time > 0
        assert len(result.times) == len(resonant_params.tlist)
        assert len(result.expect) == 1
        assert len(result.expect[0]) == len(resonant_params.tlist)
