
from jc_sim_oqp.physics import (
    get_collapse_operators,
    get_initial_state,
    get_operators,
    jc_hamiltonian,
)


def test_operators_dimensions():
    """Test that operators have correct dimensions."""
    N = 10
    a, sm_list = get_operators(N)
    assert a.dims == [[N, 2], [N, 2]]
    # sm_list is now a list of Qobjs
    assert isinstance(sm_list, list)
    assert len(sm_list) == 1
    assert sm_list[0].dims == [[N, 2], [N, 2]]


def test_initial_state_dimensions():
    """Test that initial state has correct dimensions."""
    N = 10
    psi0 = get_initial_state(N)
    # Check shape primarily
    assert psi0.shape == (2 * N, 1)
    # Dims check - QuTiP 5+ might differ from QuTiP 4
    # Just check it's compatible with ket
    assert psi0.isket


def test_hamiltonian_hermitian():
    """Test that the Hamiltonian is hermitian."""
    wc = 1.0
    wa = 1.0
    g = 0.1
    N = 5
    a, sm = get_operators(N)
    H = jc_hamiltonian(wc, wa, g, a, sm, use_rwa=True)
    assert H.isherm


def test_collapse_operators():
    """Test that the correct number of collapse operators is returned."""
    kappa = 0.1
    gamma = 0.1
    N = 5
    n_th_a = 0.0
    a, sm = get_operators(N)
    c_ops = get_collapse_operators(kappa, gamma, n_th_a, a, sm)
    assert len(c_ops) == 2  # kappa and gamma, n_th_a is 0

    n_th_a = 1.0
    c_ops = get_collapse_operators(kappa, gamma, n_th_a, a, sm)
    assert len(c_ops) == 3  # kappa(1+n), kappa*n, gamma

    # New dephasing
    c_ops = get_collapse_operators(kappa, gamma, n_th_a, a, sm, gamma_phi=0.1)
    assert len(c_ops) == 4
