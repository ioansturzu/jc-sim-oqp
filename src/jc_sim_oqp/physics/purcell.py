

def purcell_factor(g: float, kappa: float, gamma: float) -> float:
    """Calculate the Purcell factor (F_p) or cooperativity (eta).
    
    F_p = 4 * g^2 / (kappa * gamma)

    Args:
        g (float): Coupling strength.
        kappa (float): Total cavity decay rate.
        gamma (float): Emitter free-space decay rate.

    Returns:
        float: Purcell factor.
    """
    if kappa * gamma == 0:
        return 0.0
    return 4 * g**2 / (kappa * gamma)

def cooperativity(g: float, kappa: float, gamma: float, gamma_deph: float = 0.0) -> float:
    """Calculate the cooperativity (eta_d) including dephasing.
    
    eta_d = 2 * g^2 / (kappa * gamma_d)
    where gamma_d = gamma/2 + gamma_deph

    Args:
        g (float): Coupling strength.
        kappa (float): Total cavity decay rate.
        gamma (float): Emitter free-space decay rate (population decay).
        gamma_deph (float): Pure dephasing rate.

    Returns:
        float: Cooperativity.
    """
    gamma_d = gamma / 2 + gamma_deph
    if kappa * gamma_d == 0:
        return 0.0
    return 2 * g**2 / (kappa * gamma_d)

def beta_factor(g: float, kappa: float, gamma: float) -> float:
    """Calculate the beta factor (fraction of emission into cavity mode).
    
    beta = F_p / (1 + F_p)
    Assuming beta_freespace is small or included in F_p definition for this simple model.
    Technically, P_cav = (beta0 * Fp) / (1 + beta0 * Fp).
    Here we assume the transition of interest is the one being enhanced.

    Args:
        g (float): Coupling strength.
        kappa (float): Total cavity decay rate.
        gamma (float): Emitter free-space decay rate.

    Returns:
        float: Beta factor (0 to 1).
    """
    Fp = purcell_factor(g, kappa, gamma)
    return Fp / (1 + Fp)

def cavity_enhanced_decay(gamma: float, Fp: float) -> float:
    """Calculate the total cavity-enhanced decay rate.
    
    gamma_cav = gamma * (1 + Fp)

    Args:
        gamma (float): Free-space decay rate.
        Fp (float): Purcell factor.

    Returns:
        float: Enhanced decay rate.
    """
    return gamma * (1 + Fp)
