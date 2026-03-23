from collections.abc import Callable
from typing import Any

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
    sm_list = sm if isinstance(sm, list) else [sm]

    H = wc * a.dag() * a

    for sm_i in sm_list:
        if use_rwa:
            H += wa * sm_i.dag() * sm_i + g * (a.dag() * sm_i + a * sm_i.dag())
        else:
            H += wa * sm_i.dag() * sm_i + g * (a.dag() + a) * (sm_i + sm_i.dag())
    return H


def dispersive_hamiltonian(wc: float, wa: float, g: float, a: Qobj, sm: list[Qobj] | Qobj) -> Qobj:
    """Construct the Dispersive Hamiltonian.

    Approximates the system when detuning ``Delta = wa - wc`` is large
    (``|Delta| >> g*sqrt(n)``).  Includes the ac-Stark shift and Lamb shift.

    Uses the **number-operator** convention (matching ``jc_hamiltonian``)::

        H = wc * a†a + (wa + chi) * σ+σ- + chi * a†a * σz

    where ``chi = g² / Delta`` and ``σz = 2 σ+σ- − I``.

    This is algebraically equivalent to the σz-centered form
    ``½(wa+chi) σz + chi a†a σz`` plus a constant ``½(wa+chi) I``,
    but keeps the ground-state energy at zero, consistent with
    ``jc_hamiltonian``.

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

    sm_list = sm if isinstance(sm, list) else [sm]

    H = wc * a.dag() * a

    for sm_i in sm_list:
        ne = sm_i.dag() * sm_i
        sz = sm_i.dag() * sm_i - sm_i * sm_i.dag()
        H += (wa + chi) * ne + chi * a.dag() * a * sz

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
    H_static = jc_hamiltonian(wc, wa, g, a, sm, use_rwa=use_rwa)

    if drive_x is None and drive_y is None:
        return H_static

    sm_list = sm if isinstance(sm, list) else [sm]

    H_td = [H_static]

    total_sx = sum(sm_i + sm_i.dag() for sm_i in sm_list)
    total_sy = sum(-1j * (sm_i - sm_i.dag()) for sm_i in sm_list)

    if drive_x is not None:
        H_td.append([total_sx, drive_x])

    if drive_y is not None:
        H_td.append([total_sy, drive_y])

    return H_td

