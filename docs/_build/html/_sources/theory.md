
# Theoretical Framework

## The Jaynes-Cummings Model

The Jaynes-Cummings (JC) model describes the coherent interaction between a single two-level atom (qubit) and a single quantized mode of the electromagnetic field (cavity).

### Hamiltonian

In the **Rotating Wave Approximation (RWA)**, the Hamiltonian is given by:

$$
\hat{H}_{JC} = \hbar\omega_c \hat{a}^\dagger \hat{a} + \hbar\omega_a \hat{\sigma}^+\hat{\sigma}^- + \hbar g (\hat{a}^\dagger \hat{\sigma}^- + \hat{a} \hat{\sigma}^+)
$$

where:
*   $\omega_c$: Cavity frequency.
*   $\omega_a$: Atom transition frequency.
*   $g$: Vacuum Rabi coupling strength.
*   $\hat{a}, \hat{a}^\dagger$: Cavity annihilation/creation operators.
*   $\hat{\sigma}^+, \hat{\sigma}^-$: Atom raising/lowering operators.

### Dressed States

On resonance ($\Delta = \omega_a - \omega_c = 0$), the eigenstates of the system are the **Dressed States**:

$$
|n, \pm\rangle = \frac{1}{\sqrt{2}} (|n, g\rangle \pm |n-1, e\rangle)
$$

The energy splitting between these states is $2g\sqrt{n}$, known as the **Vacuum Rabi Splitting**.

## Open Quantum Systems

Real systems are never perfectly isolated. We treat the system as an **Open Quantum System** coupled to environmental reservoirs.

### Lindblad Master Equation

The evolution of the reduced density matrix $\rho$ is governed by the Lindblad Master Equation:

$$
\frac{d\hat{\rho}}{dt} = -\frac{i}{\hbar}[\hat{H}_{JC}, \hat{\rho}] + \sum_k \left( \hat{L}_k \hat{\rho} \hat{L}_k^\dagger - \frac{1}{2}\{\hat{L}_k^\dagger \hat{L}_k, \hat{\rho}\} \right)
$$

Our package implements the following continuous measurement channels (Collapse Operators $\hat{L}_k$):

| Channel | Operator | Description |
| :--- | :--- | :--- |
| **Cavity Decay** | $\sqrt{\kappa(1+\bar{n}_{th})} \hat{a}$ | Photon loss through mirrors. |
| **Thermal Bath** | $\sqrt{\kappa \bar{n}_{th}} \hat{a}^\dagger$ | Heating from environment. |
| **Spontaneous Emission** | $\sqrt{\gamma} \hat{\sigma}^-$ | Atom decay to free space. |
| **Pure Dephasing** | $\sqrt{\gamma_\phi} \hat{\sigma}_z$ | Elastic collisions/noise (Critical for Solid State). |

## Stochastic Unravelling (Quantum Trajectories)

Solving the Master Equation for $N$ atoms requires a density matrix of size $d^2 \approx (2^N)^2$. This scales exponentially ($2^{2N}$).
Using the **Minimal Dilation Theorem**, we can "unravel" the mixed state evolution into individual pure state trajectories $|\psi(t)\rangle$ subject to random Quantum Jumps.

1.  **Drift:** Between jumps, the system evolves under a non-Hermitian effective Hamiltonian:
    $$ \hat{H}_{eff} = \hat{H} - \frac{i\hbar}{2}\sum_k \hat{L}_k^\dagger \hat{L}_k $$
2.  **Jump:** With probability $\delta p = \langle \psi | \hat{L}^\dagger \hat{L} | \psi \rangle dt$, the state collapses:
    $$ |\psi\rangle \to \frac{\hat{L} |\psi\rangle}{||\hat{L} \psi||} $$

This method scales as $\approx N_{traj} \times 2^N$, offering a quadratic speedup over the Master Equation.
