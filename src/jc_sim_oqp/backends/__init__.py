"""Backends package — pluggable quantum simulation engines.

Usage::

    from jc_sim_oqp.backends import QuTiPBackend, SimResult

    backend = QuTiPBackend()
    a, sm = backend.build_operators(n_cavity=10, n_atoms=1)
    ...
"""

from jc_sim_oqp.backends.qutip_backend import QuTiPBackend
from jc_sim_oqp.backends.result import SimResult

__all__ = ["QuTiPBackend", "SimResult"]


def get_default_backend() -> QuTiPBackend:
    """Return the default simulation backend (QuTiP)."""
    return QuTiPBackend()
