"""Example: transmon-cavity spectrum using scqubits.

Builds a transmon-cavity system using the ScqubitsHamiltonianProvider,
computes the bare and dressed spectra, and compares the numerically
extracted dispersive shift with the analytical two-level formula.
"""

import numpy as np

from jc_sim_oqp.backends.scqubits_provider import (
    ScqubitsHamiltonianProvider,
    TransmonCavityParams,
)
from jc_sim_oqp.physics.transmon import dispersive_shift


def main():
    provider = ScqubitsHamiltonianProvider()

    params = TransmonCavityParams(
        EJ=20.0,
        EC=0.3,
        ng=0.0,
        ncut=30,
        n_cavity=15,
        g=0.1,
        n_levels=4,
    )

    # --- Bare transmon spectrum ---
    bare = provider.bare_transmon_spectrum(params, n_evals=4)
    omega_01 = bare[1] - bare[0]
    omega_12 = bare[2] - bare[1]
    alpha = omega_12 - omega_01

    print("=== Bare Transmon ===")
    print(f"  Eigenvalues (GHz): {bare}")
    print(f"  ω₀₁ = {omega_01:.4f} GHz")
    print(f"  ω₁₂ = {omega_12:.4f} GHz")
    print(f"  Anharmonicity α = {alpha:.4f} GHz  (expect ≈ −EC = {-params.EC:.2f})")

    # --- Dressed spectrum (on resonance) ---
    params_res = TransmonCavityParams(
        EJ=params.EJ, EC=params.EC, ng=params.ng,
        ncut=params.ncut, n_cavity=params.n_cavity,
        wc=omega_01, g=params.g, n_levels=params.n_levels,
    )
    dressed = provider.dressed_spectrum(params_res, n_evals=8)

    print("\n=== Dressed Spectrum (on resonance, ωc = ω₀₁) ===")
    print(f"  First 8 eigenvalues: {np.round(dressed, 4)}")
    spacings = np.diff(dressed)
    min_gap = np.min(spacings[spacings > 0.01])
    print(f"  Minimum gap (vacuum Rabi): {min_gap:.4f} GHz  (expect ≈ 2g = {2*params.g:.2f})")

    # --- Dispersive shift comparison ---
    detuning = 2.0
    params_disp = TransmonCavityParams(
        EJ=params.EJ, EC=params.EC, ng=params.ng,
        ncut=params.ncut, n_cavity=params.n_cavity,
        wc=omega_01 - detuning, g=0.05, n_levels=params.n_levels,
    )

    chi_analytic = dispersive_shift(params_disp.g, detuning)
    chi_numerical = provider.dispersive_shift_from_spectrum(params_disp)

    print(f"\n=== Dispersive Shift (Δ = {detuning} GHz, g = {params_disp.g} GHz) ===")
    print(f"  Analytic (g²/Δ):  χ = {chi_analytic:.6f} GHz")
    print(f"  Numerical (scqubits): χ = {chi_numerical:.6f} GHz")
    print(f"  Relative error: {abs(chi_numerical - chi_analytic) / abs(chi_analytic) * 100:.1f}%")


if __name__ == "__main__":
    main()
