"""Backend-agnostic simulation result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimResult:
    """Container for the output of a quantum simulation.

    This is intentionally backend-agnostic: all fields are plain Python
    or NumPy types so the result can be serialised, compared, or plotted
    without importing any quantum library.

    Attributes:
        times: 1-D array of evaluation time points.
        expect: List of 1-D arrays, one per observable, containing
            expectation values at each time point.
        states: Optional list of full quantum states at each time point,
            in the backend's native format.  May be ``None`` if the solver
            was configured to return only expectation values.
        wall_time: Wall-clock execution time of the solver (seconds).
        backend_name: Human-readable name of the backend that produced
            this result (e.g. ``"qutip"``, ``"scipy"``).
    """

    times: np.ndarray
    expect: list[np.ndarray]
    states: list[Any] | None = None
    wall_time: float = 0.0
    backend_name: str = ""



    @property
    def n_times(self) -> int:
        """Number of time points."""
        return len(self.times)

    @property
    def n_expect(self) -> int:
        """Number of expectation-value observables."""
        return len(self.expect)
