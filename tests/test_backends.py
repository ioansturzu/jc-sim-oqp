"""Tests for the backend abstraction layer.

Verifies that:
1. QuTiPBackend satisfies the QuantumBackend protocol
2. Solver results via the backend path match the inline (default) path
3. SimResult has the correct structure
"""

import numpy as np
import pytest

from jc_sim_oqp.backends import QuTiPBackend, SimResult
from jc_sim_oqp.backends.protocol import QuantumBackend
from jc_sim_oqp.io import SimParams
from jc_sim_oqp.solvers import DispersiveSolver, ExactSolver


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------
class TestProtocolConformance:
    """Verify QuTiPBackend satisfies the QuantumBackend protocol."""

    def test_isinstance_check(self):
        backend = QuTiPBackend()
        assert isinstance(backend, QuantumBackend)

    def test_has_required_methods(self):
        backend = QuTiPBackend()
        assert callable(getattr(backend, "build_operators", None))
        assert callable(getattr(backend, "build_hamiltonian", None))
        assert callable(getattr(backend, "build_collapse_ops", None))
        assert callable(getattr(backend, "mesolve", None))


# ---------------------------------------------------------------------------
# Backend path produces correct SimResult
# ---------------------------------------------------------------------------
class TestQuTiPBackendResult:
    """Verify the backend path returns a well-formed SimResult."""

    @pytest.fixture
    def backend_result(self):
        params = SimParams(N=5, t_max=10.0, n_steps=20)
        backend = QuTiPBackend()
        solver = ExactSolver(params, backend=backend)
        return solver.run()

    def test_returns_simresult(self, backend_result):
        assert isinstance(backend_result, SimResult)

    def test_has_times(self, backend_result):
        assert isinstance(backend_result.times, np.ndarray)
        assert len(backend_result.times) == 20

    def test_has_two_expectation_values(self, backend_result):
        assert backend_result.n_expect == 2
        for e in backend_result.expect:
            assert isinstance(e, np.ndarray)
            assert len(e) == 20

    def test_has_wall_time(self, backend_result):
        assert backend_result.wall_time > 0.0

    def test_backend_name(self, backend_result):
        assert backend_result.backend_name == "qutip"


# ---------------------------------------------------------------------------
# Backend path matches inline path
# ---------------------------------------------------------------------------
class TestBackendMatchesInline:
    """The backend-delegated path must produce identical physics
    to the original inline QuTiP path.
    """

    @pytest.fixture
    def params(self):
        return SimParams(
            wc=5.0,
            wa=5.0,
            g=0.1,
            kappa_in=0.05,
            kappa_sc=0.05,
            gamma=0.05,
            N=8,
            n_atoms=1,
            t_max=50.0,
            n_steps=100,
        )

    def test_exact_solver_agreement(self, params):
        """ExactSolver via backend must match ExactSolver inline."""
        inline = ExactSolver(params).run()
        backend_res = ExactSolver(params, backend=QuTiPBackend()).run()

        for i in range(2):
            np.testing.assert_allclose(
                backend_res.expect[i],
                np.asarray(inline.expect[i]),
                atol=1e-10,
                err_msg=f"Mismatch in observable {i}",
            )

    def test_dispersive_solver_agreement(self, params):
        """DispersiveSolver via backend must match inline."""
        # Need detuning for dispersive to be valid
        params.wc = 10.0
        params.wa = 8.0
        params.g = 0.01

        inline = DispersiveSolver(params).run()
        backend_res = DispersiveSolver(params, backend=QuTiPBackend()).run()

        for i in range(2):
            np.testing.assert_allclose(
                backend_res.expect[i],
                np.asarray(inline.expect[i]),
                atol=1e-10,
                err_msg=f"Mismatch in dispersive observable {i}",
            )

    def test_times_match(self, params):
        """Time arrays should be identical."""
        inline = ExactSolver(params).run()
        backend_res = ExactSolver(params, backend=QuTiPBackend()).run()

        np.testing.assert_array_equal(
            backend_res.times, np.asarray(inline.times)
        )
