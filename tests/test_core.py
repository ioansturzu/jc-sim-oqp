import numpy as np

from jc_sim_oqp.core import run_simulation
from jc_sim_oqp.io import SimParams


def test_run_simulation():
    # Use default params but override N for speed
    params = SimParams(N=5, t_max=10.0, n_steps=20)

    output = run_simulation(params)

    assert len(output.expect) == 2
    assert len(output.expect[0]) == len(params.tlist)
    assert len(output.expect[1]) == len(params.tlist)
    # Check probabilities are within [0, 1] (roughly, considering numerical errors)
    assert np.all(output.expect[1] <= 1.0 + 1e-6)
    assert np.all(output.expect[1] >= 0.0 - 1e-6)
