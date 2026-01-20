from typing import Any, Callable

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


def driven_jc_hamiltonian(
    wc: float,
    wa: float,
    g: float,
    a: Qobj,
    sm: list[Qobj] | Qobj,
    drive_x: Callable[[float, Any], float] | str | None = None,
    drive_y: Callable[[float, Any], float] | str | None = None,
    *,
    use_rwa: bool = True,
) -> list[Any] | Qobj:
    """Construct a Time-Dependent Jaynes-Cummings Hamiltonian with qubit drive.

    H(t) = H_JC + drive_x(t) * sigma_x + drive_y(t) * sigma_y

    Args:
        wc (float): Cavity frequency.
        wa (float): Atom frequency.
        g (float): Coupling strength.
        a (qutip.Qobj): Cavity destruction operator.
        sm (Qobj | list[Qobj]): Atom lowering operator(s).
        drive_x (Callable | str | None): Time-dependent coeff for sigma_x drive.
        drive_y (Callable | str | None): Time-dependent coeff for sigma_y drive.
        use_rwa (bool): Whether to use RWA for the static JC part.

    Returns:
        list | Qobj: QuTiP Hamiltonian format [H_static, [H_drive, coeff], ...].
                     Returns just Qobj if no drive is present.
    """
    # Base static Hamiltonian
    H_static = jc_hamiltonian(wc, wa, g, a, sm, use_rwa=use_rwa)

    # If no drive, return static
    if drive_x is None and drive_y is None:
        return H_static

    # Normalize sm to list
    sm_list = sm if isinstance(sm, list) else [sm]

    # Build the time-dependent list format
    H_td = [H_static]

    # Driving terms (global drive on all atoms for this simple model)
    # sigma_x = sm + sm.dag()
    # sigma_y = -1j * (sm - sm.dag())
    total_sx = sum(sm_i + sm_i.dag() for sm_i in sm_list)
    total_sy = sum(-1j * (sm_i - sm_i.dag()) for sm_i in sm_list)

    if drive_x is not None:
        H_td.append([total_sx, drive_x])

    if drive_y is not None:
        H_td.append([total_sy, drive_y])

    return H_td

