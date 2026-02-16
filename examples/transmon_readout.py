
import numpy as np
import matplotlib.pyplot as plt
from qutip import mesolve, expect, tensor, basis, destroy, qeye

from jc_sim_oqp.physics import get_operators, dispersive_hamiltonian
from jc_sim_oqp.physics.transmon import dispersive_shift, purcell_limit_t1, critical_photon_number

def run_readout_simulation():
    """Simulate a Transmon readout to demonstrate dispersive shift and Purcell decay.
    
    Uses the Dispersive Hamiltonian approximation for efficient simulation of 
    readout dynamics, with Purcell decay added explicitly.
    """
    
    # 1. System Parameters (Typical cQED)
    wc = 2 * np.pi * 6.0  # Cavity 6 GHz
    wa = 2 * np.pi * 5.0  # Qubit  5 GHz (Detuning = 1 GHz)
    g  = 2 * np.pi * 0.1  # Coupling 100 MHz
    
    # Detuning Delta = wa - wc = -1.0 GHz
    delta = wa - wc
    
    # Calculate Dispersive Shift chi
    chi = dispersive_shift(g, delta)
    print(f"Dispersive Shift chi/2pi = {chi / (2*np.pi):.4f} GHz ({chi / (2*np.pi) * 1000:.1f} MHz)")
    
    # Setup Kappa (Readout Rate)
    kappa = 2 * np.pi * 0.01 # 10 MHz
    
    # Purcell Limit (calculated)
    t1_purcell = purcell_limit_t1(g, delta, kappa)
    gamma_purcell = 1.0 / t1_purcell
    print(f"Purcell-limited T1 = {t1_purcell / 1000:.2f} us")
    
    # Intrinsic qubit decay
    t1_int = 20000.0 # 20 us = 20,000 ns
    gamma_int = 1.0 / t1_int
    
    # Drive
    wd = wc
    epsilon = 2 * np.pi * 0.005 # Weak drive (5 MHz)
    
    # Time evolution
   # tlist = np.linspace(0, 1000.0, 1000) # 1 us, 1ns steps
    tlist = np.linspace(0, 10000.0, 100000)
    
    # 2. Build Hamiltonian and Operators
    N = 10 
    a = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2)) 
    
    # Dispersive Hamiltonian in frame rotating at `wd`.
    # H = (wc-wd) a^d a + 0.5*(wa-wd+chi)*sz + chi a^d a sz + eps(a^d - a)
    
    d_c = wc - wd # 0
    d_q = wa - wd # Physical Detuning (e.g. -1 GHz)
    
    # H_disp DOES NOT include drive.
    H_disp = dispersive_hamiltonian(
        wc=d_c,
        wa=d_q,
        g=g,
        a=a,
        sm=sm
    )
    
    # Add Drive
    # We use i*eps*(a^d - a) to create displacement
    H = H_disp + 1j * epsilon * (a.dag() - a)
    
    # 3. Collapse Operators
    c_ops = []
    c_ops.append(np.sqrt(kappa) * a)
    
    # Total Qubit Decay: Intrinsic + Purcell
    gamma_total = gamma_int + gamma_purcell
    c_ops.append(np.sqrt(gamma_total) * sm)
    
    # 4. Simulation
    psi0_g = tensor(basis(N, 0), basis(2, 0)) # |0>_ph, |g>_qubit
    psi0_e = tensor(basis(N, 0), basis(2, 1)) # |0>_ph, |e>_qubit
    
    # Standard mesolve (no special options needed as dynamics are slow/dispersive)
    output_g = mesolve(H, psi0_g, tlist, c_ops, [a, sm.dag()*sm])
    output_e = mesolve(H, psi0_e, tlist, c_ops, [a, sm.dag()*sm])
    
    # 5. Plotting
    a_g = output_g.expect[0]
    a_e = output_e.expect[0]
    
    sz_g = output_g.expect[1] # <sm^d sm> = Population of |e>
    sz_e = output_e.expect[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # IQ Plot
    ax = axes[0]
    ax.plot(np.real(a_g), np.imag(a_g), 'b-', label='Start |g>')
    ax.plot(np.real(a_e), np.imag(a_e), 'r-', label='Start |e>')
    ax.plot(np.real(a_g[-1]), np.imag(a_g[-1]), 'bo', markersize=8)
    ax.plot(np.real(a_e[-1]), np.imag(a_e[-1]), 'rx', markersize=8)
    ax.set_title('Readout Trajectories (IQ Plane)\nDispersive Limit')
    ax.set_xlabel('I = Re(<a>)')
    ax.set_ylabel('Q = Im(<a>)')
    ax.grid(True)
    ax.legend()
    ax.axis('equal')
    
    # Population Plot
    ax = axes[1]
    ax.plot(tlist, sz_g, 'b--', label='Excited Pop (Start |g>)')
    ax.plot(tlist, sz_e, 'r-', label='Excited Pop (Start |e>)')
    
    # Theoretical Decay
    # T1_elemental = 1/gamma_total (in ns)
    t1_eff_us = (1.0 / gamma_total) / 1000.0
    ax.plot(tlist, np.exp(-gamma_total * tlist), 'k:', label=f'Theory T1 eff = {t1_eff_us:.2f} us')
    
    ax.set_title('Qubit Excitation vs Time')
    ax.set_xlabel('Time (ns)')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('transmon_readout_iq.png')
    print("Saved plot to transmon_readout_iq.png")

if __name__ == "__main__":
    run_readout_simulation()
