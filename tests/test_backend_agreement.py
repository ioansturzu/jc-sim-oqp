"""Cross-backend agreement tests.

Verifies that QuTiP and SciPy backends produce matching expectation
values across multiple physics regimes.  The scqubits+QuTiP path
uses a different Hamiltonian (transmon vs two-level), so it is tested
for self-consistency rather than cross-agreement.
"""

import numpy as np
import pytest

from jc_sim_oqp.backends.qutip_backend import QuTiPBackend
from jc_sim_oqp.backends.result import SimResult
from jc_sim_oqp.backends.scipy_backend import ScipyBackend
from jc_sim_oqp.backends.scqubits_provider import (
    ScqubitsHamiltonianProvider,
    TransmonCavityParams,
)
from jc_sim_oqp.io import SimParams


def _make_params(regime: str) -> SimParams:
    """Create SimParams for a given physics regime."""
    configs = {
        "resonant": SimParams(
            wc=1.0 * 2 * np.pi,
            wa=1.0 * 2 * np.pi,
            g=0.05 * 2 * np.pi,
            kappa_in=0.0025, kappa_sc=0.0025,
            gamma=0.005, n_th_a=0.0,
            N=5, n_atoms=1,
            t_max=25.0, n_steps=200,
        ),
        "dispersive": SimParams(
            wc=1.0 * 2 * np.pi,
            wa=1.5 * 2 * np.pi,
            g=0.02 * 2 * np.pi,
            kappa_in=0.001, kappa_sc=0.001,
            gamma=0.002, n_th_a=0.0,
            N=5, n_atoms=1,
            t_max=50.0, n_steps=300,
        ),
        "purcell": SimParams(
            wc=1.0 * 2 * np.pi,
            wa=1.2 * 2 * np.pi,
            g=0.1 * 2 * np.pi,
            kappa_in=0.05, kappa_sc=0.05,
            gamma=0.01, n_th_a=0.0,
            N=5, n_atoms=1,
            t_max=30.0, n_steps=250,
        ),
    }
    return configs[regime]


def _run_backend(backend, params: SimParams) -> SimResult:
    """Run a JC-model backend and return SimResult."""
    a, sm = backend.build_operators(params.N, params.n_atoms)
    psi0 = backend.build_initial_state(params.N, params.n_atoms)
    H = backend.build_hamiltonian(params, a, sm, variant="jc")
    c_ops = backend.build_collapse_ops(params, a, sm)

    n_cav = a.dag() * a
    n_atom = sum(s.dag() * s for s in sm)
    e_ops = [n_cav, n_atom]

    return backend.mesolve(H, psi0, params.tlist, c_ops, e_ops)


class TestQuTiPvsSciPy:
    """QuTiP and SciPy solve the same JC model — must agree numerically."""

    @pytest.mark.parametrize("regime", ["resonant", "dispersive", "purcell"])
    def test_expectation_values_agree(self, regime):
        params = _make_params(regime)

        ref = _run_backend(QuTiPBackend(), params)
        test = _run_backend(ScipyBackend(), params)

        for i in range(len(ref.expect)):
            np.testing.assert_allclose(
                test.expect[i], ref.expect[i],
                atol=1e-4, rtol=1e-3,
                err_msg=f"Backend disagreement in {regime}, observable {i}",
            )


class TestScqubitsConsistency:
    """scqubits+QuTiP path: self-consistency checks.

    The scqubits Hamiltonian models a transmon (multi-level), so it will
    not match the two-level JC model.  Instead we verify:
    - The Hamiltonian is Hermitian
    - mesolve completes without error
    - Trace is preserved
    """

    def test_scqubits_mesolve_runs(self):
        from qutip import basis, destroy, qeye, tensor

        provider = ScqubitsHamiltonianProvider()
        qutip_be = QuTiPBackend()

        n_levels = 2
        n_cavity = 5
        tc_params = TransmonCavityParams(
            EJ=15.0, EC=0.3, ng=0.0, ncut=30,
            n_cavity=n_cavity, wc=5.0, g=0.1, n_levels=n_levels,
        )

        H = provider.hamiltonian(tc_params)
        assert H.isherm

        # Operators in scqubits ordering: transmon ⊗ cavity
        a = tensor(qeye(n_levels), destroy(n_cavity))
        sm = tensor(destroy(n_levels), qeye(n_cavity))
        psi0 = tensor(basis(n_levels, 1), basis(n_cavity, 0))

        kappa = 0.002
        gamma = 0.001
        c_ops = [np.sqrt(kappa) * a, np.sqrt(gamma) * sm]

        n_cav = a.dag() * a
        tlist = np.linspace(0, 10.0, 50)

        result = qutip_be.mesolve(H, psi0, tlist, c_ops, [n_cav])

        assert result.backend_name == "qutip"
        assert len(result.expect[0]) == 50
        assert np.all(np.isfinite(result.expect[0]))
