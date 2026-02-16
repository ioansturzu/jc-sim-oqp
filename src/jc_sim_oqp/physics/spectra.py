
import numpy as np

def reflection_coefficient_naive(
    detuning: float, 
    g: float, 
    kappa_in: float, 
    kappa_sc: float, 
    gamma: float,
    gamma_deph: float = 0.0
) -> complex:
    """Calculate the complex reflection coefficient using the naive semiclassical model.
    
    This model assumes the field expectation value factorizes, which fails in the presence of dephasing.
    
    Args:
        detuning (float): Laser-cavity detuning (omega_L - omega_c).
                          Assumes laser-atom detuning is the same (omega_c = omega_a).
        g (float): Coupling strength.
        kappa_in (float): Input coupling rate.
        kappa_sc (float): Scattering loss rate.
        gamma (float): Atom decay rate.
        gamma_deph (float): Atom dephasing rate.

    Returns:
        complex: Reflection coefficient r = <a_out> / <a_in>
    """
    kappa = kappa_in + kappa_sc
    
    # In the bad cavity limit derivations (Eq 4.42 kindem_cavity.tex):
    # tilde_kappa = kappa/2 - i * detuning  (Note: sign of detuning depends on convention, 
    # kindem uses Delta_c = w_c - w_L. Here we use typically w_L - w_c. Let's stick to Kindem convention:
    # Delta_c = w_c - w_L. If detuning is w_L - w_c, then Delta_c = -detuning.)
    # Let's assume input 'detuning' is Delta_c = w_c - w_L. 
    
    tilde_kappa = kappa / 2 + 1j * detuning
    tilde_gamma = (gamma / 2 + gamma_deph) + 1j * detuning
    
    if abs(tilde_gamma * tilde_kappa) == 0:
        return -1.0 + 0j
        
    tilde_eta = g**2 / (tilde_gamma * tilde_kappa)
    
    # r = (1 + eta - kappa_in/tilde_kappa) / (1 + eta)
    return (1 + tilde_eta - kappa_in / tilde_kappa) / (1 + tilde_eta)

def reflection_coefficient_corrected(
    detuning: float, 
    g: float, 
    kappa_in: float, 
    kappa_sc: float, 
    gamma: float, 
    gamma_deph: float
) -> float:
    """Calculate the REFLECTED POWER using the corrected semiclassical model (Eq 4.60).
    
    This correctly accounts for dephasing by averaging photon numbers instead of fields.
    Returns |r|^2 directly. Currently only implemented for RESONANCE (detuning=0).
    
    Args:
        detuning (float): Detuning (must be 0 for now).
        g (float): Coupling strength.
        kappa_in (float): Input coupling rate.
        kappa_sc (float): Scattering loss rate.
        gamma (float): Atom decay rate.
        gamma_deph (float): Atom dephasing rate.

    Returns:
        float: Reflected power |r|^2.
    """
    if detuning != 0:
        raise NotImplementedError("Corrected reflection coefficient only implemented for zero detuning.")
        
    kappa = kappa_in + kappa_sc
    gamma_d = gamma / 2 + gamma_deph
    
    # Cooperativities
    # eta_d = 2 * g^2 / (kappa * gamma_d)
    # eta_s = 4 * g^2 / (kappa * gamma)
    if kappa * gamma_d == 0: eta_d = 0
    else: eta_d = 2 * g**2 / (kappa * gamma_d)
        
    if kappa * gamma == 0: eta_s = 0
    else: eta_s = 4 * g**2 / (kappa * gamma)
        
    term1 = (1 + eta_d) * (1 + eta_s)
    term2 = 4 * (1 + eta_s) * (kappa_in / kappa)
    term3 = (1 + eta_s - eta_d) * (2 * kappa_in / kappa)**2
    
    numerator = term1 - term2 + term3
    denominator = (1 + eta_d) * (1 + eta_s)
    
    return numerator / denominator
