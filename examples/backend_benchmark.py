"""Backend benchmark runner.

Compares QuTiP, SciPy, and scqubits+QuTiP backends across physics
regimes and system sizes.  Collects wall time, accuracy (L2 error
vs. QuTiP reference), and peak memory.

Usage::

    uv run examples/backend_benchmark.py
    uv run examples/backend_benchmark.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from jc_sim_oqp.backends.qutip_backend import QuTiPBackend
from jc_sim_oqp.backends.result import SimResult
from jc_sim_oqp.backends.scipy_backend import ScipyBackend
from jc_sim_oqp.io import SimParams

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_configs import CAVITY_SIZES, REGIME_FACTORIES


@dataclass
class BenchmarkResult:
    """Single benchmark measurement."""

    backend: str
    regime: str
    n_cavity: int
    hilbert_dim: int
    wall_time: float
    peak_memory_kb: float
    l2_error: float | None


def _run_jc_backend(
    backend: QuTiPBackend | ScipyBackend,
    params: SimParams,
) -> SimResult:
    """Run a JC-model backend (QuTiP or SciPy)."""
    a, sm = backend.build_operators(params.N, params.n_atoms)
    psi0 = backend.build_initial_state(params.N, params.n_atoms)
    H = backend.build_hamiltonian(params, a, sm, variant="jc")
    c_ops = backend.build_collapse_ops(params, a, sm)

    n_cav = a.dag() * a
    n_atom = sum(s.dag() * s for s in sm)
    e_ops = [n_cav, n_atom]

    return backend.mesolve(H, psi0, params.tlist, c_ops, e_ops)



def _l2_error(test: SimResult, ref: SimResult) -> float:
    """L2 norm of expectation-value differences, averaged over observables."""
    errors = []
    for t_exp, r_exp in zip(test.expect, ref.expect):
        diff = np.asarray(t_exp) - np.asarray(r_exp)
        errors.append(np.sqrt(np.mean(diff**2)))
    return float(np.mean(errors))


def run_benchmark(
    regimes: list[str] | None = None,
    sizes: list[int] | None = None,
) -> list[BenchmarkResult]:
    """Run the full benchmark grid."""
    regimes = regimes or list(REGIME_FACTORIES.keys())
    sizes = sizes or CAVITY_SIZES

    qutip_be = QuTiPBackend()
    scipy_be = ScipyBackend()

    results: list[BenchmarkResult] = []

    for regime_name in regimes:
        factory = REGIME_FACTORIES[regime_name]

        for n_cav in sizes:
            params = factory(n_cavity=n_cav)
            hdim = params.N * (2 ** params.n_atoms)

            print(f"\n--- {regime_name} | N={n_cav} | dim={hdim} ---")

            # QuTiP (reference)
            tracemalloc.start()
            ref = _run_jc_backend(qutip_be, params)
            _, peak_qt = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            results.append(BenchmarkResult(
                backend="qutip", regime=regime_name,
                n_cavity=n_cav, hilbert_dim=hdim,
                wall_time=ref.wall_time,
                peak_memory_kb=peak_qt / 1024,
                l2_error=0.0,
            ))
            print(f"  qutip:          {ref.wall_time:.4f}s")

            # SciPy
            tracemalloc.start()
            res_sp = _run_jc_backend(scipy_be, params)
            _, peak_sp = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            err_sp = _l2_error(res_sp, ref)
            results.append(BenchmarkResult(
                backend="scipy", regime=regime_name,
                n_cavity=n_cav, hilbert_dim=hdim,
                wall_time=res_sp.wall_time,
                peak_memory_kb=peak_sp / 1024,
                l2_error=err_sp,
            ))
            print(f"  scipy:          {res_sp.wall_time:.4f}s  err={err_sp:.2e}")


    return results


def main():
    parser = argparse.ArgumentParser(description="Backend benchmark runner")
    parser.add_argument(
        "--output", "-o", type=str, default="benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--regimes", nargs="+", default=None,
        help="Regimes to benchmark (default: all)",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=None,
        help="Cavity sizes (default: 5, 10, 15)",
    )
    args = parser.parse_args()

    results = run_benchmark(regimes=args.regimes, sizes=args.sizes)

    output = [asdict(r) for r in results]
    out_path = Path(args.output)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")

    print("\n=== Summary ===")
    print(f"{'Backend':<20} {'Regime':<14} {'N':>4} {'dim':>6} {'Time (s)':>10} {'L2 err':>10}")
    print("-" * 70)
    for r in results:
        err_str = f"{r.l2_error:.2e}" if r.l2_error is not None else "N/A"
        print(f"{r.backend:<20} {r.regime:<14} {r.n_cavity:>4} {r.hilbert_dim:>6} {r.wall_time:>10.4f} {err_str:>10}")


if __name__ == "__main__":
    main()
