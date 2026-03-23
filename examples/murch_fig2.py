
import matplotlib.pyplot as plt
import numpy as np


def reproduce_murch_fig2():
    """Reproduce Figure 2c and 2d from Murch et al. (Nature 2013).
    
    Correlation between integrated weak measurement signal Vm and conditioned
    qubit tomography result (X, Y, Z).
    """
    # 1. Physical Parameters (from the paper)
    chi = 2 * np.pi * 0.49e6    # 0.49 MHz
    kappa = 2 * np.pi * 10.8e6  # 10.8 MHz
    tau = 1.8e-6                # 1.8 us
    eta = 0.49                  # Quantum efficiency
    t2_star = 20e-6             # 20 us
    gamma_env = 1.0 / t2_star   # Environmental dephasing
    
    # Delta V is the separation between |0> and |1> distributions.
    # Vm is normalized by Delta V in the paper's plots/equations usually.
    # In Fig 2, the x-axis is Vm in Volts.
    # Equation 1 uses S and Delta V.
    
    # Panel (c): Z-measurement (n_bar = 0.4)
    n_bar_z = 0.4
    
    # Measurement Strength S (Eq 1 caption / text)
    # S = 64 * tau * chi^2 * n_bar * eta / kappa
    # Note: The paper uses a specific SNR definition.
    # Let's calculate S directly.
    S_z = 64 * tau * chi**2 * n_bar_z * eta / kappa
    print(f"Calculated S_z = {S_z:.2f} (Paper Fig 2c says S=3.15)")
    
    # Dephasing rate gamma (Eq 2 caption)
    # gamma = 8 * chi^2 * n_bar * (1-eta) / kappa + 1/T2*
    gamma_z = 8 * chi**2 * n_bar_z * (1 - eta) / kappa + gamma_env
    print(f"Calculated gamma_z = {gamma_z:.2e} s^-1 (Paper says 2.8e5)")
    
    # Panel (d): Phi-measurement (n_bar = 0.46)
    n_bar_phi = 0.46
    S_phi = 64 * tau * chi**2 * n_bar_phi * eta / kappa
    gamma_phi = 8 * chi**2 * n_bar_phi * (1 - eta) / kappa + gamma_env
    print(f"Calculated S_phi = {S_phi:.2f} (Paper Fig 2d says S=3.62)")
    print(f"Calculated gamma_phi = {gamma_phi:.2e} s^-1 (Paper says 3.1e5)")

    # 2. Coordinate conditioned on Vm
    # We span Vm from -0.15 to 0.15 Volts (matching Fig 2 axes)
    # The paper uses DV = 0.12 V (estimated from Fig 1e histograms separation ~ 0.12V)
    DV = 0.12
    vm_range = np.linspace(-0.15, 0.15, 200)
    
    # Z-measurement Correlation (Fig 2c)
    # Eq (1): Z = tanh(Vm * S / 2*DV)
    # Eq (2): X = sqrt(1 - Z^2) * exp(-gamma * tau)
    # Y = 0 (for Z measurement initialized at X=1)
    
    z_z = np.tanh(vm_range * S_z / (2 * DV))
    x_z = np.sqrt(1 - z_z**2) * np.exp(-gamma_z * tau)
    y_z = np.zeros_like(vm_range)
    
    # Phi-measurement Correlation (Fig 2d)
    # In phi-measurement, Vm carries phase info.
    # Eq (3): X = cos(Vm * S / 2*DV) * exp(-gamma * tau)
    # Eq (4): Y = -sin(Vm * S / 2*DV) * exp(-gamma * tau)
    # Z = 0 (uncorrelated)
    
    x_phi = np.cos(vm_range * S_phi / (2 * DV)) * np.exp(-gamma_phi * tau)
    y_phi = -np.sin(vm_range * S_phi / (2 * DV)) * np.exp(-gamma_phi * tau)
    z_phi = np.zeros_like(vm_range)
    
    # 3. Create Simulation Points (Scatter plot mock to match the paper's dots)
    # In reality, these are averaged from 10^5 experiments.
    # We can add a bit of noise to represent finite sampling.
    vm_samples_z = np.linspace(-0.15, 0.15, 30)
    z_samples_z = np.tanh(vm_samples_z * S_z / (2 * DV)) + np.random.normal(0, 0.03, 30)
    x_samples_z = np.sqrt(1 - np.tanh(vm_samples_z * S_z / (2 * DV))**2) * np.exp(-gamma_z * tau) + np.random.normal(0, 0.03, 30)
    
    vm_samples_phi = np.linspace(-0.15, 0.15, 30)
    x_samples_phi = np.cos(vm_samples_phi * S_phi / (2 * DV)) * np.exp(-gamma_phi * tau) + np.random.normal(0, 0.03, 30)
    y_samples_phi = -np.sin(vm_samples_phi * S_phi / (2 * DV)) * np.exp(-gamma_phi * tau) + np.random.normal(0, 0.03, 30)

    # 4. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Figure 2c
    ax = axes[0]
    ax.plot(vm_range, z_z, 'k--', label='ZZ (Theory)')
    ax.plot(vm_range, x_z, 'b--', label='XZ (Theory)')
    ax.plot(vm_range, y_z, 'r:', label='YZ (Theory)')
    
    ax.scatter(vm_samples_z, z_samples_z, c='k', marker='o', s=10, alpha=0.5, label='ZZ (Sim)')
    ax.scatter(vm_samples_z, x_samples_z, c='b', marker='s', s=10, alpha=0.5, label='XZ (Sim)')
    
    ax.set_title('Figure 2c: Z-measurement Correlation')
    ax.set_xlabel('Integrated measurement signal $V_m$ (V)')
    ax.set_ylabel('Tomography result ($X^Z, Y^Z, Z^Z$)')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True)
    ax.legend()
    
    # Figure 2d
    ax = axes[1]
    ax.plot(vm_range, x_phi, 'b--', label='$X^\\phi$ (Theory)')
    ax.plot(vm_range, y_phi, 'r--', label='$Y^\\phi$ (Theory)')
    ax.plot(vm_range, z_phi, 'k:', label='$Z^\\phi$ (Theory)')
    
    ax.scatter(vm_samples_phi, x_samples_phi, c='b', marker='s', s=10, alpha=0.5, label='$X^\\phi$ (Sim)')
    ax.scatter(vm_samples_phi, y_samples_phi, c='r', marker='^', s=10, alpha=0.5, label='$Y^\\phi$ (Sim)')
    
    ax.set_title('Figure 2d: $\\phi$-measurement Correlation')
    ax.set_xlabel('Integrated measurement signal $V_m$ (V)')
    ax.set_ylabel('Tomography result ($X^\\phi, Y^\\phi, Z^\\phi$)')
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('murch_fig2_repro.png')
    print("Saved plot to murch_fig2_repro.png")

if __name__ == "__main__":
    reproduce_murch_fig2()
