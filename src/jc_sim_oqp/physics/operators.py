from qutip import Qobj, basis, destroy, qeye, tensor


def get_initial_state(n_cavity: int, n_atoms: int = 1) -> Qobj:
    """Create the initial state: all atoms excited, cavity in vacuum.

    Args:
        n_cavity (int): Number of cavity Fock states.
        n_atoms (int): Number of atoms.

    Returns:
        qutip.Qobj: Initial state vector ``|e, e, ..., 0>``.
    """
    # Cavity in vacuum
    psi_cavity = basis(n_cavity, 0)

    # All atoms excited (level 1)
    psi_atoms = basis(2, 1)
    for _ in range(n_atoms - 1):
        psi_atoms = tensor(psi_atoms, basis(2, 1))

    return tensor(psi_cavity, psi_atoms)


def get_operators(n_cavity: int, n_atoms: int = 1) -> tuple[Qobj, list[Qobj]]:
    """Create the destruction and sigma-minus operators.

    Args:
        n_cavity (int): Number of cavity Fock states.
        n_atoms (int): Number of atoms.

    Returns:
        tuple: (a, sm_list)
               a: cavity destruction operator
               sm_list: list of atom lowering operators [sm1, sm2, ...]
    """
    # Cavity operator: a x I x I ...
    op_list = [destroy(n_cavity)]
    op_list.extend([qeye(2)] * n_atoms)
    a = tensor(*op_list)

    # Atom operators
    sm_list = []
    for i in range(n_atoms):
        # I x ... x sm_i x ... x I
        op_list = [qeye(n_cavity)]
        for j in range(n_atoms):
            if i == j:
                op_list.append(destroy(2))
            else:
                op_list.append(qeye(2))
        sm_list.append(tensor(*op_list))

    return a, sm_list
