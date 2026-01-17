
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def exp_func(x: float | NDArray[np.float64], a: float, b: float, c: float) -> float | NDArray[np.float64]:
    """Exponential decay function for fitting."""
    return a * np.exp(-b * x) + c


def fit_decay(
    tlist: NDArray[np.float64], n_c: NDArray[np.float64]
) -> tuple[float, float, float] | None:
    """Fit an exponential decay to the peaks of the cavity photon number oscillation.

    Args:
        tlist (np.ndarray): Time vector.
        n_c (np.ndarray): Cavity photon number expectation values.

    Returns:
        tuple: (a, b, c) parameters of the fitted exponential a * exp(-b * t) + c.
               Returns None if fit fails or not enough peaks.
    """
    peaks, _ = find_peaks(n_c)
    if len(peaks) < 2:
        return None

    peak_times = tlist[peaks]
    peak_values = n_c[peaks]

    # Initial guess based on data
    # a ~ amplitude, b ~ decay rate, c ~ offset
    p0 = (float(np.max(peak_values) - np.min(peak_values)), 0.1, float(np.min(peak_values)))

    try:
        popt, _ = curve_fit(exp_func, peak_times, peak_values, p0=p0)
        return tuple(popt)
    except (RuntimeError, ValueError):
        # Curve fit failed to converge
        return None


def plot_rabi_oscillations(
    tlist: NDArray[np.float64],
    n_a: NDArray[np.float64],
    n_c: NDArray[np.float64],
    fit_params: tuple[float, float, float] | None = None,
) -> Figure:
    """Plot the vacuum Rabi oscillations and optionally the exponential fit.

    Args:
        tlist (np.ndarray): Time vector.
        n_a (np.ndarray): Atom excited state population.
        n_c (np.ndarray): Cavity photon number.
        fit_params (tuple, optional): (a, b, c) for exponential fit. Defaults to None.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))

    axes.plot(tlist, n_a, label="Atom excited state")
    axes.plot(tlist, n_c, label="Cavity")

    if fit_params is not None:
        a, b, c = fit_params
        axes.plot(tlist, exp_func(tlist, a, b, c), "r--", label=f"Fit: {a:.2f}e^(-{b:.3f}t) + {c:.2f}")

    axes.legend(loc=0)
    axes.set_xlabel("Time")
    axes.set_ylabel("Occupation probability")
    axes.set_title("Vacuum Rabi oscillations")
    axes.grid(True, alpha=0.3)

    return fig


def plot_cavity_fit(
    tlist: NDArray[np.float64],
    n_c: NDArray[np.float64],
    fit_params: tuple[float, float, float],
) -> Figure:
    """Plot the cavity photon number and the fitted exponential decay.

    Args:
        tlist (np.ndarray): Time vector.
        n_c (np.ndarray): Cavity photon number.
        fit_params (tuple): (a, b, c) for exponential fit.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    a, b, c = fit_params
    axes.plot(tlist, exp_func(tlist, a, b, c), "r--", label=f"Fit: {a:.2f}e^(-{b:.3f}t) + {c:.2f}")
    axes.plot(tlist, n_c, label="Cavity")

    axes.legend(loc=0)
    axes.set_xlabel("Time")
    axes.set_ylabel("Occupation probability")
    axes.set_title("Cavity Decay Fit")
    axes.grid(True, alpha=0.3)

    return fig
