from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimParams:
    """Simulation parameters for the Jaynes-Cummings model."""

    wc: float = 1.0 * 2 * np.pi     # cavity frequency
    wa: float = 1.0 * 2 * np.pi     # atom frequency
    g: float = 0.1 * 2 * np.pi      # coupling strength
    kappa: float = 0.005            # cavity dissipation rate
    gamma: float = 0.05             # atom dissipation rate
    gamma_phi: float = 0.0          # atom dephasing rate
    N: int = 15                     # number of cavity fock states
    n_atoms: int = 1                # number of atoms
    n_th_a: float = 2.0             # avg number of thermal bath excitation
    use_rwa: bool = True            # toggle for rotating wave approximation
    t_max: float = 200.0            # simulation time length
    n_steps: int = 1000             # number of time steps

    @property
    def tlist(self) -> NDArray[np.float64]:
        """Generate the time vector for the simulation."""
        return np.linspace(0, self.t_max, self.n_steps)
