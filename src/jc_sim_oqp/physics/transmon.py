
import numpy as np

def dispersive_shift(g: float, detuning: float) -> float:
    """Calculate the dispersive shift (chi).
    
    chi = g^2 / Delta

    Args:
        g (float): Coupling strength.
        detuning (float): Detuning Delta = wa - wc.

    Returns:
        float: Dispersive shift chi.
    """
    if detuning == 0:
        return np.inf
    return g**2 / detuning

def critical_photon_number(g: float, detuning: float) -> float:
    """Calculate the critical photon number (n_crit).
    
    n_crit = Delta^2 / (4 * g^2)
    This is the number of photons where the dispersive approximation breaks down.

    Args:
        g (float): Coupling strength.
        detuning (float): Detuning Delta = wa - wc.

    Returns:
        float: Critical photon number.
    """
    if g == 0:
        return np.inf
    return detuning**2 / (4 * g**2)

def purcell_limit_t1(g: float, detuning: float, kappa: float) -> float:
    """Calculate the Purcell-limited T1 time.
    
    Gamma_P = (g / Delta)^2 * kappa
    T1_P = 1 / Gamma_P

    Args:
        g (float): Coupling strength.
        detuning (float): Detuning Delta = wa - wc.
        kappa (float): Cavity decay rate.

    Returns:
        float: Purcell-limited T1.
    """
    if detuning == 0:
        return 0.0
    gamma_purcell = (g / detuning)**2 * kappa
    if gamma_purcell == 0:
        return np.inf
    return 1.0 / gamma_purcell
