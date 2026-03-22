"""Benchmark configuration — parameter grids for backend comparison.

Defines physics regimes and system sizes used by the benchmark runner
and the agreement test suite.
"""

from __future__ import annotations

import numpy as np

from jc_sim_oqp.io import SimParams


def resonant_config(n_cavity: int = 10, n_atoms: int = 1) -> SimParams:
    """Resonant regime: ωc = ωa, moderate coupling."""
    return SimParams(
        wc=1.0 * 2 * np.pi,
        wa=1.0 * 2 * np.pi,
        g=0.05 * 2 * np.pi,
        kappa_in=0.0025,
        kappa_sc=0.0025,
        gamma=0.005,
        n_th_a=0.0,
        N=n_cavity,
        n_atoms=n_atoms,
        t_max=25.0,
        n_steps=200,
        use_rwa=True,
    )


def dispersive_config(n_cavity: int = 10, n_atoms: int = 1) -> SimParams:
    """Dispersive regime: large detuning, weak effective coupling."""
    return SimParams(
        wc=1.0 * 2 * np.pi,
        wa=1.5 * 2 * np.pi,
        g=0.02 * 2 * np.pi,
        kappa_in=0.001,
        kappa_sc=0.001,
        gamma=0.002,
        n_th_a=0.0,
        N=n_cavity,
        n_atoms=n_atoms,
        t_max=50.0,
        n_steps=300,
        use_rwa=True,
    )


def purcell_config(n_cavity: int = 10, n_atoms: int = 1) -> SimParams:
    """Strong-coupling / Purcell regime: significant cavity decay."""
    return SimParams(
        wc=1.0 * 2 * np.pi,
        wa=1.2 * 2 * np.pi,
        g=0.1 * 2 * np.pi,
        kappa_in=0.05,
        kappa_sc=0.05,
        gamma=0.01,
        n_th_a=0.0,
        N=n_cavity,
        n_atoms=n_atoms,
        t_max=30.0,
        n_steps=250,
        use_rwa=True,
    )


REGIME_FACTORIES = {
    "resonant": resonant_config,
    "dispersive": dispersive_config,
    "purcell": purcell_config,
}

CAVITY_SIZES = [5, 10, 15]
