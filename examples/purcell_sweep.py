#!/usr/bin/env python3
"""Dedicated runner for high-resolution Purcell Effect physics sweep.
Usage: uv run python purcell_sweep.py --task-id [0-29]
"""

import argparse
import json
import time
from pathlib import Path
import numpy as np
import sys

# Ensure local src is in path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root / "src"))

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.solvers import ExactSolver
from jc_sim_oqp.physics.purcell import purcell_factor, cavity_enhanced_decay

def run_purcell_sweep(
    task_id: int,
    output_base: Path,
    num_tasks: int = 30,
    fp_min: float = 1.0,
    fp_max: float = 1000.0,
    kappa: float = 1.0,
    gamma: float = 0.01,
    t_max: float = 50.0,
    n_steps: int = 500,
):
    # 1. Generate log-spaced Fp values
    fp_values = np.logspace(np.log10(fp_min), np.log10(fp_max), num_tasks)
    target_fp = fp_values[task_id]
    
    # 2. Map Fp back to g: Fp = 4*g^2 / (kappa*gamma) => g = sqrt(Fp * kappa * gamma / 4)
    g = np.sqrt(target_fp * kappa * gamma / 4.0)
    
    output_dir = output_base / f"task_{task_id:02d}_fp_{target_fp:.1f}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Setup Simulation
    params = SimParams()
    params.g = float(g)
    params.kappa_in = kappa
    params.kappa_sc = 0.0
    params.gamma = gamma
    params.t_max = t_max
    params.n_steps = n_steps
    params.N = 3 # Resonant, ground state of cavity, excited atom
    params.wa = params.wc # On resonance
    
    print(f"--- Task {task_id} ---")
    print(f"Target Fp: {target_fp:.4f}")
    print(f"Derived g: {g:.4f}")
    
    # 4. Run
    t0 = time.perf_counter()
    solver = ExactSolver(params)
    res = solver.run()
    runtime = time.perf_counter() - t0
    
    times = np.array(res.times)
    pe_numerical = np.array(res.expect[1], copy=True)
    
    # 5. Physics Analysis
    fp_actual = purcell_factor(g, kappa, gamma)
    gamma_eff = cavity_enhanced_decay(gamma, fp_actual)
    pe_analytical = np.exp(-gamma_eff * times)
    
    # Log-linear fit
    mask = (pe_numerical > 0.1) & (times < t_max / 2)
    if np.any(mask) and len(pe_numerical[mask]) > 5:
        coeffs = np.polyfit(times[mask], np.log(pe_numerical[mask] + 1e-15), 1)
        gamma_numerical = -coeffs[0]
    else:
        gamma_numerical = 0.0
        
    error_rate = abs(gamma_numerical - gamma_eff) / gamma_eff if gamma_eff > 0 else 0
    
    # 6. Save
    results = {
        'task_id': task_id,
        'fp_target': float(target_fp),
        'fp_actual': float(fp_actual),
        'g': float(g),
        'gamma_analytical': float(gamma_eff),
        'gamma_numerical': float(gamma_numerical),
        'rate_error': float(error_rate),
        'runtime': float(runtime)
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    np.savez(
        output_dir / 'purcell_trace.npz',
        times=times,
        pe_numerical=pe_numerical,
        pe_analytical=pe_analytical
    )
    
    print(f"Task {task_id} complete. Error: {error_rate*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="benchmark_purcell_sweep")
    args = parser.parse_args()
    
    run_purcell_sweep(args.task_id, Path(args.output_dir))
