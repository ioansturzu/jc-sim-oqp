from qutip import Qobj


def jc_hamiltonian(
    wc: float, wa: float, g: float, a: Qobj, sm: list[Qobj] | Qobj, *, use_rwa: bool = True
) -> Qobj:
    """Construct the Jaynes-Cummings (or Tavis-Cummings) Hamiltonian.

    Args:
        wc (float): Cavity frequency.
        wa (float): Atom frequency.
        g (float): Coupling strength.
        a (qutip.Qobj): Cavity destruction operator.
        sm (Qobj | list[Qobj]): Atom lowering operator(s).
        use_rwa (bool): Whether to use the Rotating Wave Approximation.

    Returns:
        qutip.Qobj: Hamiltonian.
    """
    # Normalize to list
    sm_list = sm if isinstance(sm, list) else [sm]

    # Cavity energy
    H = wc * a.dag() * a

    # Add atoms
    for sm_i in sm_list:
        if use_rwa:
            H += wa * sm_i.dag() * sm_i + g * (a.dag() * sm_i + a * sm_i.dag())
        else:
            H += wa * sm_i.dag() * sm_i + g * (a.dag() + a) * (sm_i + sm_i.dag())
    return H


def dispersive_hamiltonian(wc: float, wa: float, g: float, a: Qobj, sm: list[Qobj] | Qobj) -> Qobj:
    """Construct the Dispersive Hamiltonian.

    Approximates the system when detuning ``Delta = wa - wc`` is large (``|Delta| >> g*sqrt(n)``).
    Includes the ac-Stark shift and Lamb shift.

    ``H_disp = wc * a^dag * a + sum_i [ (wa + chi)/2 * sigma_z^i + chi * a^dag * a * sigma_z^i ]``

    where ``chi = g^2 / Delta``.

    Args:
        wc (float): Cavity frequency.
        wa (float): Atom frequency.
        g (float): Coupling strength.
        a (qutip.Qobj): Cavity destruction operator.
        sm (Qobj | list[Qobj]): Atom lowering operator(s).

    Returns:
        qutip.Qobj: Dispersive Hamiltonian.
    """
    delta = wa - wc
    chi = g**2 / delta

    # Normalize to list
    sm_list = sm if isinstance(sm, list) else [sm]

    H = wc * a.dag() * a

    for sm_i in sm_list:
        # sigma_z = sm.dag * sm - sm * sm.dag
        sz = sm_i.dag() * sm_i - sm_i * sm_i.dag()
        H += 0.5 * (wa + chi) * sz + chi * a.dag() * a * sz

    return H
