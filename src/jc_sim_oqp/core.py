from qutip.solver import Result

from .io import SimParams
from .solvers import ExactSolver


def run_simulation(params: SimParams) -> Result:
    """Run the Jaynes-Cummings simulation.

    Defaults to the ExactSolver (Master Equation).

    Args:
        params (SimParams): Simulation parameters.

    Returns:
        qutip.solver.Result: Simulation result.
    """
    solver = ExactSolver(params)
    return solver.run()
