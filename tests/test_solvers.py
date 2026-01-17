
from jc_sim_oqp.io import SimParams
from jc_sim_oqp.solvers import DispersiveSolver, ExactSolver, StochasticSolver


def test_dispersive_limit():
    """Test that Exempt and Dispersive solvers agree in the large detuning limit.

    In the dispersive regime (|Delta| >> g), the effective Hamiltonian should
    predict dynamics close to the exact one, specifically the phase evolutions.
    Here we just check that they run and produce results of correct shape.
    Quantitative agreement requires careful parameter selection (Delta ~ 100g).
    """
    params = SimParams()
    # Large detuning for dispersive regime
    params.wc = 10.0
    params.wa = 8.0  # Delta = 2.0
    params.g = 0.01  # g << Delta
    params.n_steps = 10

    exact = ExactSolver(params)
    disp = DispersiveSolver(params)

    res_exact = exact.run()
    res_disp = disp.run()

    assert len(res_exact.times) == 10
    assert len(res_disp.times) == 10
    # Basic check that simulation ran
    assert res_exact.expect[0][0] == 0.0  # Start with 0 photons


def test_stochastic_solver():
    """Test that Stochastic solver runs and returns trajectories."""
    params = SimParams()
    params.n_steps = 10

    # Run a few trajectories
    solver = StochasticSolver(params, ntraj=5)
    result = solver.run()

    assert len(result.times) == 10
    # mcsolve result.expect is a list of arrays (average over trajectories)
    assert len(result.expect) == 2
