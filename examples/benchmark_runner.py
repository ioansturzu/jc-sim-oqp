#!/usr/bin/env python3
"""Benchmark runner module for parallel SLURM execution.

This module provides clean interfaces for running individual benchmark tasks
that can be called from SLURM job arrays or standalone scripts.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.solvers import ExactSolver, StochasticSolver
from jc_sim_oqp.physics.purcell import purcell_factor, cavity_enhanced_decay


def run_error_benchmark(
    ntraj: int,
    output_dir: Path,
    seed: int = 42,
    n_atoms: int = 1,
    t_max: float = 20.0,
    n_steps: int = 200,
    cavity_n: int = 10,
) -> dict[str, Any]:
    """Run error convergence benchmark for a single trajectory count.
    
    Parameters
    ----------
    ntraj : int
        Number of stochastic trajectories to simulate
    output_dir : Path
        Directory to save results
    seed : int, optional
        Random seed for reproducibility
    n_atoms : int, optional
        Number of atoms in the system
    t_max : float, optional
        Maximum simulation time
    n_steps : int, optional
        Number of time steps
    cavity_n : int, optional
        Cavity truncation parameter
        
    Returns:
    -------
    dict
        Results dictionary containing RMSE, runtime, and parameters
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup parameters
    params = SimParams()
    params.n_atoms = n_atoms
    params.t_max = t_max
    params.n_steps = n_steps
    params.N = cavity_n

    print(f"Running error benchmark with ntraj={ntraj}")
    print(f"Parameters: n_atoms={n_atoms}, t_max={t_max}, n_steps={n_steps}, N={cavity_n}")

    # Compute exact solution (ground truth)
    print("Computing exact solution...")
    t0 = time.perf_counter()
    solver_exact = ExactSolver(params)
    res_exact = solver_exact.run()
    exact_time = time.perf_counter() - t0
    exact_data = np.array(res_exact.expect[0], copy=True)
    print(f"Exact solution computed in {exact_time:.2f}s")

    # Compute stochastic solution
    print(f"Computing stochastic solution with {ntraj} trajectories...")
    t0 = time.perf_counter()
    solver_mc = StochasticSolver(params, ntraj=ntraj)
    res_mc = solver_mc.run(seed=seed)
    stoch_time = time.perf_counter() - t0
    mc_data = np.array(res_mc.expect[0], copy=True)
    print(f"Stochastic solution computed in {stoch_time:.2f}s")

    # Calculate error metric
    rmse = np.sqrt(np.mean((exact_data - mc_data) ** 2))
    print(f"RMSE: {rmse:.6e}")

    # Prepare results
    results = {
        'benchmark_type': 'error',
        'ntraj': ntraj,
        'rmse': float(rmse),
        'runtime_stochastic': stoch_time,
        'runtime_exact': exact_time,
        'seed': seed,
        'params': {
            'n_atoms': params.n_atoms,
            't_max': params.t_max,
            'n_steps': params.n_steps,
            'N': params.N,
        }
    }

    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    # Save time series data
    data_file = output_dir / 'trajectory_data.npz'
    np.savez(
        data_file,
        times=res_mc.times,
        exact=exact_data,
        stochastic=mc_data,
    )
    print(f"Trajectory data saved to {data_file}")

    return results


def run_time_benchmark(
    n_atoms: int,
    output_dir: Path,
    ntraj: int = 100,
    t_max: float = 5.0,
    n_steps: int = 50,
    cavity_n: int = 5,
) -> dict[str, Any]:
    """Run timing benchmark for a single system size.
    
    Parameters
    ----------
    n_atoms : int
        Number of atoms in the system
    output_dir : Path
        Directory to save results
    ntraj : int, optional
        Number of stochastic trajectories (fixed for comparison)
    t_max : float, optional
        Maximum simulation time
    n_steps : int, optional
        Number of time steps
    cavity_n : int, optional
        Cavity truncation parameter
        
    Returns:
    -------
    dict
        Results dictionary containing timing data for both solvers
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup parameters
    params = SimParams()
    params.n_atoms = n_atoms
    params.t_max = t_max
    params.n_steps = n_steps
    params.N = cavity_n

    print(f"Running time benchmark with n_atoms={n_atoms}")
    print(f"Parameters: t_max={t_max}, n_steps={n_steps}, N={cavity_n}")

    # Calculate Hilbert space dimension
    hilbert_dim = (cavity_n + 1) * (2 ** n_atoms)
    print(f"Hilbert space dimension: {hilbert_dim}")

    # Safety check: Skip exact solver for large systems to avoid segfaults
    # Density matrix memory: dim^2 * 16 bytes (complex128)
    MAX_HILBERT_DIM = 16384  # ~4 GB density matrix, safe for most systems

    # Time exact solver
    time_exact = None
    exact_failed = False
    exact_skipped = False

    if hilbert_dim > MAX_HILBERT_DIM:
        print(f"SKIPPING exact solver: Hilbert dimension {hilbert_dim} exceeds safe limit {MAX_HILBERT_DIM}")
        print(f"  (Would require ~{(hilbert_dim**2 * 16) / (1024**3):.2f} GB for density matrix alone)")
        exact_skipped = True
        exact_failed = False
    else:
        try:
            print("Timing exact solver...")
            t0 = time.perf_counter()
            ExactSolver(params).run()
            time_exact = time.perf_counter() - t0
            print(f"Exact solver: {time_exact:.2f}s")
        except Exception as e:
            exact_failed = True
            print(f"Exact solver failed: {e}")

    # Time stochastic solver
    print(f"Timing stochastic solver with {ntraj} trajectories...")
    t0 = time.perf_counter()
    StochasticSolver(params, ntraj=ntraj).run()
    time_stoch = time.perf_counter() - t0
    print(f"Stochastic solver: {time_stoch:.2f}s")

    # Prepare results
    results = {
        'benchmark_type': 'time',
        'n_atoms': n_atoms,
        'hilbert_dim': hilbert_dim,
        'time_exact': float(time_exact) if time_exact is not None else None,
        'time_stochastic': float(time_stoch),
        'exact_failed': exact_failed,
        'exact_skipped': exact_skipped,
        'ntraj': ntraj,
        'params': {
            't_max': params.t_max,
            'n_steps': params.n_steps,
            'N': params.N,
        }
    }

    # Add runtime fields with old naming convention for compatibility
    results['runtime_exact'] = results['time_exact']
    results['runtime_stochastic'] = results['time_stochastic']

    # Save results
    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")

    return results


def run_purcell_benchmark(
    g_index: int,
    output_dir: Path,
    kappa: float = 1.0,
    gamma: float = 0.01,
    t_max: float = 50.0,
    n_steps: int = 400,
) -> dict[str, Any]:
    """Verify the Purcell enhancement against analytical theory.
    
    Parameters
    ----------
    g_index : int
        Coupling strength index (g = g_index/100 * kappa)
    output_dir : Path
        Directory to save results
    kappa : float, optional
        Cavity decay rate
    gamma : float, optional
        Atom free-space decay rate
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    g = (g_index / 100.0) * kappa
    
    params = SimParams()
    params.g = g
    params.kappa_in = kappa
    params.kappa_sc = 0.0
    params.gamma = gamma
    params.t_max = t_max
    params.n_steps = n_steps
    params.N = 3  # Small N is sufficient for decay from |e> in bad cavity
    params.wa = params.wc # On resonance

    print(f"Running Purcell physics benchmark with g={g:.4f} ({g_index}%)")
    
    t0 = time.perf_counter()
    solver = ExactSolver(params)
    res = solver.run()
    runtime = time.perf_counter() - t0
    
    times = res.times
    pe_numerical = np.array(res.expect[1], copy=True) # Prob of being in |e>
    
    # Analytical Purcell theory
    fp = purcell_factor(g, kappa, gamma)
    gamma_eff = cavity_enhanced_decay(gamma, fp)
    pe_analytical = np.exp(-gamma_eff * times)
    
    # Log-linear fit to find numerical decay rate
    mask = pe_numerical > 0.1
    if np.any(mask) and len(pe_numerical[mask]) > 5:
        coeffs = np.polyfit(times[mask], np.log(pe_numerical[mask] + 1e-15), 1)
        gamma_numerical = -coeffs[0]
    else:
        gamma_numerical = 0.0

    error_rate = abs(gamma_numerical - gamma_eff) / gamma_eff if gamma_eff > 0 else 0
    rmse = np.sqrt(np.mean((pe_numerical - pe_analytical) ** 2))

    print(f"Purcell Factor (Fp): {fp:.4f}")
    print(f"Analytical Rate: {gamma_eff:.6f}")
    print(f"Numerical Rate:  {gamma_numerical:.6f}")
    print(f"Rate Error:      {error_rate*100:.2f}%")

    results = {
        'benchmark_type': 'purcell',
        'g_index': g_index,
        'g': float(g),
        'fp': float(fp),
        'gamma_eff_analytical': float(gamma_eff),
        'gamma_eff_numerical': float(gamma_numerical),
        'rate_error': float(error_rate),
        'rmse': float(rmse),
        'runtime': runtime,
        'params': {
            'kappa': kappa,
            'gamma': gamma,
            't_max': t_max,
            'n_steps': n_steps
        }
    }

    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    np.savez(
        output_dir / 'purcell_data.npz',
        times=times,
        pe_numerical=pe_numerical,
        pe_analytical=pe_analytical
    )

    return results


def parse_benchmark_params(param_string: str) -> tuple[str, int]:
    """Parse parameter string from SLURM array.
    
    Parameters
    ----------
    param_string : str
        Parameter string in format "type:value" (e.g., "error:100" or "time:5")
        
    Returns:
    -------
    tuple
        (benchmark_type, parameter_value)
    """
    parts = param_string.split(':')
    if len(parts) != 2:
        raise ValueError(f"Invalid parameter string: {param_string}. Expected 'type:value'")

    bench_type = parts[0]
    if bench_type not in ['error', 'time', 'purcell']:
        raise ValueError(f"Invalid benchmark type: {bench_type}. Must be 'error', 'time', or 'purcell'")

    try:
        param_value = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid parameter value: {parts[1]}. Must be an integer")

    return bench_type, param_value


def run_benchmark_from_params(
    param_string: str,
    output_dir: Path,
    task_id: int | None = None,
) -> dict[str, Any]:
    """Run benchmark based on parameter string (convenience function for SLURM).
    
    Parameters
    ----------
    param_string : str
        Parameter string in format "type:value"
    output_dir : Path
        Base output directory for results
    task_id : int, optional
        SLURM array task ID (used for seeding)
        
    Returns:
    -------
    dict
        Results dictionary
    """
    bench_type, param_value = parse_benchmark_params(param_string)

    # Create task-specific output directory
    if task_id is not None:
        task_dir = output_dir / f"task_{task_id}_{bench_type}_{param_value}"
    else:
        task_dir = output_dir / f"{bench_type}_{param_value}"

    print("=" * 70)
    print(f"Benchmark Type: {bench_type}")
    print(f"Parameter Value: {param_value}")
    print(f"Output Directory: {task_dir}")
    print("=" * 70)

    if bench_type == 'error':
        seed = 42 + (task_id if task_id is not None else 0)
        results = run_error_benchmark(
            ntraj=param_value,
            output_dir=task_dir,
            seed=seed,
        )
    elif bench_type == 'time':
        results = run_time_benchmark(
            n_atoms=param_value,
            output_dir=task_dir,
        )
    else:  # bench_type == 'purcell'
        results = run_purcell_benchmark(
            g_index=param_value,
            output_dir=task_dir,
        )

    # Add task_id to results
    if task_id is not None:
        results['task_id'] = task_id

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run JC-SIM-OQP benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run error benchmark with 1000 trajectories
  uv run python benchmark_runner.py error:1000 results/

  # Run time benchmark with 5 atoms
  uv run python benchmark_runner.py time:5 results/

  # With task ID (for SLURM)
  uv run python benchmark_runner.py error:1000 results/ --task-id 5
        """,
    )

    parser.add_argument(
        'param_string',
        type=str,
        help='Parameter string (format: "type:value", e.g., "error:1000" or "time:5")',
    )
    parser.add_argument(
        'output_dir',
        type=Path,
        help='Output directory for results',
    )
    parser.add_argument(
        '--task-id',
        type=int,
        default=None,
        help='SLURM array task ID (optional, used for seeding)',
    )

    args = parser.parse_args()

    try:
        results = run_benchmark_from_params(
            param_string=args.param_string,
            output_dir=args.output_dir,
            task_id=args.task_id,
        )
        print("\n" + "=" * 70)
        print("Benchmark completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import sys
        sys.exit(1)
