
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
        
        a, _ = get_operators(self.params.N, n_atoms=self.params.n_atoms)
        
        kappa_in = self.params.kappa_in
        if kappa_in == 0:
            return np.zeros(len(detunings))

        epsilon = drive_amp
        a_in_amp = epsilon / np.sqrt(kappa_in)
        
        solver = SteadyStateSolver(self.params)

        for delta in detunings:
            # Shift frequencies to the rotating frame of the probe drive (w_L)
            # w_L = wc_center + delta
            # eff_wc = wc_center - w_L = -delta
            # eff_wa = wa_center - w_L = (wa_center - wc_center) - delta
            
            W_C = original_wc
            W_A = self.params.wa
            
            eff_wc = -1.0 * delta
            eff_wa = (W_A - W_C) - delta
            
            self.params.wc = eff_wc
            self.params.wa = eff_wa
            
            rho_ss = solver.run(drive_amp=epsilon)
            a_expect = expect(a, rho_ss)
            
            # Compute the complex reflection coefficient r = <a_out>/<a_in>
            # Based on the input-output relation: a_out = a_in + sqrt(kappa_in) * a
            # and the drive relation: epsilon = sqrt(kappa_in) * a_in
            r = 1.0 + (kappa_in / epsilon) * a_expect
            results.append(abs(r)**2)
            
        self.params.wc = original_wc
        self.params.wa = W_A
        
        return np.array(results)
