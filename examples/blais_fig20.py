
import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, tensor, expect

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics import get_operators
from jc_sim_oqp.solvers import ExactSolver, SteadyStateSolver

def reproduce_blais_fig20():
    """Reproduce Figure 20 from Blais et al. (RMP 2021) with extreme fidelity.
    
    Units: ns for time, Rad/ns (angular GHz) for frequencies.
    """
    
    opts = {"nsteps": 30000, "method": "adams"}
    
    def run_time_domain(params, psi0, tlist):
        solver = ExactSolver(params)
        result = solver.run(psi0=psi0, tlist=tlist, options=opts)
        # result.expect[0] is <n>, result.expect[1] is Pe
        return result.expect[1], result.expect[0]

    def run_ss_scan(params, det_mhz_list, drive_mhz, thermalize_qubit=False):
        intensities = []
        drive_angular = 2 * np.pi * (drive_mhz * 1e-3)
        a_op, _ = get_operators(params.N, n_atoms=1)
        for d_mhz in det_mhz_list:
            d_angular = 2 * np.pi * (d_mhz * 1e-3)
            # Frame rotating at drive frequency
            p = SimParams(
                wc=-d_angular, wa=-d_angular, g=params.g,
                kappa_in=params.kappa_in, kappa_sc=params.kappa_sc,
                gamma=params.gamma, n_th_a=params.n_th_a, 
                n_th_q=params.n_th_a if thermalize_qubit else 0.0,
                N=params.N
            )
            ss_sol = SteadyStateSolver(p)
            rho_ss = ss_sol.run(drive_amp=drive_angular)
            intensities.append(abs(expect(a_op, rho_ss))**2)
        return np.array(intensities)

    C_E = 'skyblue'
    C_G = 'navy'

    # --- (a, b) Bad Cavity ---
    print("Regime: Bad Cavity...")
    params_bc = SimParams(
        wc=0, wa=0, g=2*np.pi*0.001,
        kappa_in=2*np.pi*0.005, kappa_sc=2*np.pi*0.005, # total 10 MHz
        gamma=0, n_th_a=0, N=5
    )
    tlist_bc = np.linspace(0, 2000, 1000)
    pe_e_bc, n_e_bc = run_time_domain(params_bc, tensor(basis(5,0), basis(2,1)), tlist_bc)
    pe_g1_bc, n_g1_bc = run_time_domain(params_bc, tensor(basis(5,1), basis(2,0)), tlist_bc)
    
    det_bc = np.linspace(-20, 20, 1000)
    resp_bc = run_ss_scan(params_bc, det_bc, 0.0005)

    # --- (c, d) Bad Qubit ---
    print("Regime: Bad Qubit...")
    params_bq = SimParams(
        wc=0, wa=0, g=2*np.pi*0.001,
        kappa_in=2*np.pi*1e-6, kappa_sc=2*np.pi*1e-6,
        gamma=2*np.pi*0.010, n_th_a=0, N=5
    )
    tlist_bq = np.linspace(0, 2000, 1000)
    pe_e_bq, n_e_bq = run_time_domain(params_bq, tensor(basis(5,0), basis(2,1)), tlist_bq)
    pe_g1_bq, n_g1_bq = run_time_domain(params_bq, tensor(basis(5,1), basis(2,0)), tlist_bq)
    
    det_bq = np.linspace(-2, 2, 1000)
    resp_bq = run_ss_scan(params_bq, det_bq, 0.0001)

    # --- (e, f) Strong Coupling ---
    print("Regime: Strong Coupling...")
    g_sc = 2*np.pi*0.100 
    
    params_sc_d = SimParams(wc=0, wa=0, g=g_sc, kappa_in=2*np.pi*0.00005, kappa_sc=2*np.pi*0.00005, 
                           gamma=2*np.pi*0.0001, n_th_a=0, N=15)
    params_sc_s = SimParams(wc=0, wa=0, g=g_sc, kappa_in=2*np.pi*0.0005, kappa_sc=2*np.pi*0.0005, 
                           gamma=2*np.pi*0.001, n_th_a=0, N=15)
    
    tlist_sc = np.linspace(0, 40, 1000)
    pe_e_sc_d, n_e_sc_d = run_time_domain(params_sc_d, tensor(basis(15,0), basis(2,1)), tlist_sc)
    pe_g1_sc_d, n_g1_sc_d = run_time_domain(params_sc_d, tensor(basis(15,1), basis(2,0)), tlist_sc)
    
    pe_e_sc_s, n_e_sc_s = run_time_domain(params_sc_s, tensor(basis(15,0), basis(2,1)), tlist_sc)
    pe_g1_sc_s, n_g1_sc_s = run_time_domain(params_sc_s, tensor(basis(15,1), basis(2,0)), tlist_sc)
    
    print("Panel (f) high-res scan (10k points)...")
    det_sc = np.linspace(-150, 150, 10000)
    drive_f = 0.002
    resp_vac = run_ss_scan(params_sc_d, det_sc, drive_f)
    params_sc_d.n_th_a = 0.35
    resp_th = run_ss_scan(params_sc_d, det_sc, drive_f, thermalize_qubit=False)

    # --- Save Simulation Results ---
    np.savez('examples/blais_fig20_sim.npz',
             tlist_bc=tlist_bc, pe_e_bc=pe_e_bc, pe_g1_bc=pe_g1_bc, resp_bc=resp_bc, det_bc=det_bc,
             tlist_bq=tlist_bq, pe_e_bq=pe_e_bq, n_e_bq=n_e_bq, pe_g1_bq=pe_g1_bq, n_g1_bq=n_g1_bq, resp_bq=resp_bq, det_bq=det_bq,
             tlist_sc=tlist_sc, pe_e_sc_d=pe_e_sc_d, pe_g1_sc_d=pe_g1_sc_d, pe_e_sc_s=pe_e_sc_s, pe_g1_sc_s=pe_g1_sc_s, 
             resp_vac=resp_vac, resp_th=resp_th, det_sc=det_sc)
    print("Saved simulation results to examples/blais_fig20_sim.npz")

    # --- Reproduction Plot ---
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    
    # (a)
    axes[0, 0].plot(tlist_bc, pe_e_bc, color=C_E, label=r'$|0, e\rangle$')
    axes[0, 0].plot(tlist_bc, pe_g1_bc, color=C_G, label=r'$|1, g\rangle$')
    axes[0, 0].set_title("(a) Bad-cavity limit"); axes[0, 0].set_ylabel(r"$P_e$")
    axes[0, 0].set_xlabel("t (ns)"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True)
    
    # (b)
    axes[0, 1].plot(det_bc, resp_bc / resp_bc.max(), color=C_G)
    axes[0, 1].set_title("(b) SS Response"); axes[0, 1].set_ylabel(r"$|\langle \hat{a} \rangle|^2$ (norm)")
    axes[0, 1].set_xlabel(r"$\delta_r/2\pi$ (MHz)"); axes[0, 1].grid(True)

    # (c)
    axes[1, 0].plot(tlist_bq, n_g1_bq, color=C_G, label=r'$|1, g\rangle$')
    axes[1, 0].plot(tlist_bq, n_e_bq, color=C_E, label=r'$|0, e\rangle$')
    axes[1, 0].set_title("(c) Bad-qubit limit"); axes[1, 0].set_ylabel(r"$\langle \hat{n} \rangle$")
    axes[1, 0].set_xlabel("t (ns)"); axes[1, 0].legend(fontsize=8); axes[1, 0].grid(True)
    
    # (d)
    axes[1, 1].plot(det_bq, resp_bq / resp_bq.max(), color=C_G)
    axes[1, 1].set_title("(d) SS Response")
    axes[1, 1].set_xlabel(r"$\delta_r/2\pi$ (MHz)"); axes[1, 1].grid(True)

    # (e)
    axes[2, 0].plot(tlist_sc, pe_e_sc_d, color=C_E, ls='--', label=r'$|0,e\rangle$ (0.1)')
    axes[2, 0].plot(tlist_sc, pe_g1_sc_d, color=C_G, ls='--', label=r'$|1,g\rangle$ (0.1)')
    axes[2, 0].plot(tlist_sc, pe_e_sc_s, color=C_E, ls='-', label=r'$|0,e\rangle$ (1.0)')
    axes[2, 0].plot(tlist_sc, pe_g1_sc_s, color=C_G, ls='-', label=r'$|1,g\rangle$ (1.0)')
    axes[2, 0].set_title("(e) Strong coupling"); axes[2, 0].set_ylabel(r"$P_e$")
    axes[2, 0].set_xlabel("t (ns)"); axes[2, 0].legend(fontsize=8); axes[2, 0].grid(True)

    # (f)
    vmax_vac = resp_vac.max()
    axes[2, 1].plot(det_sc, resp_vac / vmax_vac, color=C_G, label=r'$n_{th}=0$')
    axes[2, 1].plot(det_sc, resp_th / vmax_vac, color=C_E, label=r'$n_{th}=0.35$')
    axes[2, 1].set_title("(f) Vacuum Rabi Splitting"); axes[2, 1].set_ylabel(r"$|\langle \hat{a} \rangle|^2$ (norm)")
    axes[2, 1].set_xlabel(r"$\delta_r/2\pi$ (MHz)")
    axes[2, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=8)
    axes[2, 1].set_ylim(-0.05, 1.1); axes[2, 1].grid(True)

    fig.tight_layout()
    fig.savefig('blais_fig20_repro.png')
    plt.close(fig)
    print("Saved reproduction plot to blais_fig20_repro.png")

if __name__ == "__main__":
    reproduce_blais_fig20()
