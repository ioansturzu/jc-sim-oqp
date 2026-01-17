
import numpy as np
from qutip import Qobj


def get_collapse_operators(
    kappa: float,
    gamma: float,
    n_th_a: float,
    a: Qobj,
    sm: Qobj,
    gamma_phi: float | None = None
) -> list[Qobj]:
    """Create the list of collapse operators for dissipation.

    Args:
        kappa (float): Cavity dissipation rate.
        gamma (float): Atom dissipation rate.
        n_th_a (float): Avg number of thermal bath excitations.
        a (qutip.Qobj): Cavity destruction operator.
        sm (qutip.Qobj): Atom lowering operator.
        gamma_phi (float, optional): Pure dephasing rate for the atom. Defaults to None.

    Returns:
        list: List of collapse operators.
    """
    c_ops = []

    # cavity relaxation
    rate = kappa * (1 + n_th_a)
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * a)

    # cavity excitation, if temperature > 0
    rate = kappa * n_th_a
    if rate > 0.0:
        c_ops.append(np.sqrt(rate) * a.dag())

    # Normalize to list
    sm_list = sm if isinstance(sm, list) else [sm]

    for sm_i in sm_list:
        # qubit relaxation
        rate = gamma
        if rate > 0.0:
            c_ops.append(np.sqrt(rate) * sm_i)

        # qubit pure dephasing
        if gamma_phi is not None and gamma_phi > 0.0:
            # sigma_z = |e><e| - |g><g|
            sz = sm_i.dag() * sm_i - sm_i * sm_i.dag()
            c_ops.append(np.sqrt(gamma_phi) * sz)

    return c_ops
