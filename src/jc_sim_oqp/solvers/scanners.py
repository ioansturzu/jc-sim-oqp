
import numpy as np
from qutip import expect

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics import get_operators
from jc_sim_oqp.solvers.master import SteadyStateSolver

class SpectrumScanner:
    """Helper to scan parameters and compute spectra."""

    def __init__(self, params: SimParams):
        self.params = params

    def scan_detuning(
        self, 
        detunings: list[float], 
        drive_amp: float = 0.1,
        scan_type: str = "cavity"
    ) -> np.ndarray:
        """Scan detuning and return reflection coefficient.
        
        Reflection coefficient R = |<a_out>/<a_in>|^2
        <a_out> = <a_in> + sqrt(kappa_in) * <a>
        
        In the standard input-output formalism:
        Input field <a_in> is related to drive strength E by E = sqrt(kappa_in) * <a_in>.
        So <a_in> = E / sqrt(kappa_in).
        
        And <a_out> = E/sqrt(kappa_in) + sqrt(kappa_in)*<a>
        r = <a_out>/<a_in> = 1 + (kappa_in/E) * <a>
        
        Args:
            detunings (list[float]): List of detuning values (delta = wc - wl).
            drive_amp (float): Drive amplitude E.
            scan_type (str): "cavity" (scan wc).
                             
        Returns:
            np.ndarray: Array of reflection coefficients |r|^2.
        """
        original_wc = self.params.wc
        results = []
        
        # Operators for expectation values
        a, _ = get_operators(self.params.N, n_atoms=self.params.n_atoms)
        
        kappa_in = self.params.kappa_in
        # Check for zero kappa_in to avoid division by zero
        if kappa_in == 0:
            return np.zeros(len(detunings))

        epsilon = drive_amp
        a_in_amp = epsilon / np.sqrt(kappa_in)
        
        solver = SteadyStateSolver(self.params)

        for delta in detunings:
            # Update wc based on detuning (assuming laser is at 0 in rotating frame)
            # if we scan cavity detuning relative to fixed laser:
            # wc = laser_freq + delta
            # Let's assume laser freq is 0 in rotating frame if we don't have explicit laser freq.
            # But wait, SimParams has 'wa' and 'wc'. 
            # In rotating frame of laser:
            # H = delta_c * a^dag*a + delta_a * sz + ...
            # delta_c = wc - wl
            # So if we scan delta, we are setting wc = wl + delta.
            # But what is 'wl'? 
            # Usually we set 'wa' as reference or 'wd'.
            # Let's assume we are scanning 'wc' relative to 'wa'.
            # So if delta = 0, wc = wa.
            
            # self.params.wc is the cavity frequency.
            # self.params.wa is the atom frequency.
            # The Hamiltonian uses these absolute values? 
            # Looking at hamiltonians.py: 
            # H = wc * a.dag() * a + wa * sm.dag() * sm + ...
            # AND use_rwa=True (default).
            # This is NOT the rotating frame Hamiltonian! This is the LAB frame H.
            # 
            # If we want to simulate a DRIVEN system, we usually go to a rotating frame.
            # mesolve/steadystate works in lab frame too if time dependent?
            # But for steady state with drive, we MUST be in rotating frame of the drive
            # otherwise the Hamiltonian is time dependent H = H0 + E(e^-iwt a + ...).
            #
            # The current `jc_hamiltonian` returns Lab Frame H if we pass raw frequencies.
            # 
            # However, standard practice in QuTiP simulation of JC model often uses
            # detunings directly if we say we are in rotating frame.
            #
            # If `params.wc` and `params.wa` are ~ 2*pi*1.0 (as default in io.py),
            # these are extremely small for optical/microwave frequencies (usually GHz = 1e9).
            # So these are likely DETUNINGS or frequencies in a rotating frame already?
            #
            # Re-reading `io.py`:
            # wc: float = 1.0 * 2 * np.pi
            # wa: float = 1.0 * 2 * np.pi
            # g: float = 0.1 * 2 * np.pi
            #
            # These look like dummy values for a unitless simulation or normalized units.
            #
            # If we want to scan the "detuning" of the probe laser relative to the cavity:
            # In the frame of the PROBE laser (frequency w_L):
            # H_rot = (wc - w_L) a^d a + (wa - w_L) sz + g(...) + E(a^d + a)
            #
            # The solver's H is `wc * a^d a + wa * sz ...`.
            # So if we want to simulate the rotating frame, we should interpret `self.params.wc`
            # as `wc - w_L` and `self.params.wa` as `wa - w_L`.
            #
            # So, if we scan "detuning" acts as `wc - w_L`:
            # We set params.wc = delta.
            # We set params.wa = (wa - wc_center) + delta?
            #
            # Let's assume we want to scan the PROBE frequency w_L across the cavity/atom system.
            # Cavity freq W_C is fixed. Atom freq W_A is fixed.
            # We vary w_L.
            #
            # Params for solver should be:
            # effective_wc = W_C - w_L
            # effective_wa = W_A - w_L
            #
            # W_C, W_A are the fixed values in self.params (from __init__ or user).
            # We want to scan `delta_probe = w_L - W_C` (or similar).
            
            # Let's assume input `detunings` is `w_L - W_C`.
            # Then w_L = W_C + delta.
            # effective_wc = W_C - (W_C + delta) = -delta
            # effective_wa = W_A - (W_C + delta) = (W_A - W_C) - delta
            
            w_L_minus_wc = delta 
            
            # Effective detunings for the Hamiltonian (Rotating Frame)
            # The Hamiltonian function takes 'wc', 'wa'. We will pass effective values.
            
            # Save original values to restore later (though we working on a copy would be safer/slower)
            # Here we modify params in place, which is risky if assumed static.
            # But SpectrumScanner owns the params? No, it references them.
            # Let's interpret params.wc as the PHYSICAL cavity frequency W_C.
            
            W_C = original_wc
            W_A = self.params.wa
            
            # derived:
            eff_wc = -1.0 * delta   # (wc - w_L) = - (w_L - wc)
            eff_wa = W_A - W_C - delta # (wa - w_L) = wa - (wc + delta) = (wa - wc) - delta
            
            # We need to temporarily update params to feed the solver
            self.params.wc = eff_wc
            self.params.wa = eff_wa
            
            # Run solver
            rho_ss = solver.run(drive_amp=epsilon)
            
            # Calculate expectation <a>
            a_expect = expect(a, rho_ss)
            
            # Reflection coefficient r = 1 + (kappa_in / a_in_amp) * <a>
            # Note: This formula depends on convention of input-output relation.
            # a_out = a_in - sqrt(kappa)*a  OR  a_out = a_in + sqrt(kappa)*a ?
            # Kindem eq 4.7: a_out = sqrt(kappa_in)*a + a_in
            # So r = <a_out>/<a_in> = 1 + sqrt(kappa_in) * <a> / <a_in>
            # with <a_in> = epsilon / sqrt(kappa_in),
            # r = 1 + sqrt(kappa_in) * <a> / (epsilon / sqrt(kappa_in))
            #   = 1 + (kappa_in / epsilon) * <a>
            
            r = 1.0 + (kappa_in / epsilon) * a_expect
            results.append(abs(r)**2)
            
        # Restore parameters
        self.params.wc = original_wc
        self.params.wa = W_A
        
        return np.array(results)
