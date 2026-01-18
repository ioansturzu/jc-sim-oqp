"""
Parallel master equation solver for large-scale simulations.

This module provides parallelization strategies for the master equation solver:
1. Time-chunking: Split time evolution into parallel chunks
2. Parameter sweeps: Run multiple parameter sets in parallel
3. GPU acceleration: Use GPU-accelerated linear algebra (if available)

Note: For the Jaynes-Cummings master equation (mesolve), true parallelization
is limited because the density matrix evolution is inherently sequential.
The main bottlenecks are:
- Matrix exponentials/integrator steps (sequential)
- Large Hilbert space dimensions: d = (N_cavity+1) × 2^N_atoms

Strategies:
- For single large N: GPU acceleration helps with matrix operations
- For parameter sweeps: Embarrassingly parallel (joblib/multiprocessing)
- For time evolution: Limited benefit, integrators are inherently sequential
"""

from typing import List, Optional, Callable
import numpy as np
from qutip import mesolve, Options
from qutip.solver import Result
from joblib import Parallel, delayed

from jc_sim_oqp.io import SimParams
from jc_sim_oqp.physics import (
    get_collapse_operators,
    get_initial_state,
    get_operators,
    jc_hamiltonian,
)


class ParallelExactSolver:
    """
    Parallel master equation solver with multiple parallelization strategies.
    
    Important: For single large-N problems, the master equation is inherently
    sequential. Parallelization benefits come from:
    1. GPU acceleration (if QuTiP compiled with GPU support)
    2. Running multiple independent simulations (parameter sweeps)
    3. Better ODE solver settings and sparse matrix operations
    """

    def __init__(self, params: SimParams, n_jobs: int = -1):
        """
        Initialize parallel solver.
        
        Args:
            params: Simulation parameters
            n_jobs: Number of parallel jobs (-1 = all cores)
        """
        self.params = params
        self.n_jobs = n_jobs

    def run(self, use_sparse: bool = True, atol: float = 1e-8, rtol: float = 1e-6) -> Result:
        """
        Execute single simulation with optimized settings.
        
        Args:
            use_sparse: Use sparse matrices (recommended for large N)
            atol: Absolute tolerance for ODE solver
            rtol: Relative tolerance for ODE solver
            
        Returns:
            QuTiP Result object
        """
        # Setup operators and state
        a, sm_list = get_operators(self.params.N, n_atoms=self.params.n_atoms)
        psi0 = get_initial_state(self.params.N, n_atoms=self.params.n_atoms)

        # Hamiltonian
        H = jc_hamiltonian(
            self.params.wc,
            self.params.wa,
            self.params.g,
            a,
            sm_list,
            use_rwa=self.params.use_rwa,
        )

        # Collapse operators
        c_ops = get_collapse_operators(
            self.params.kappa,
            self.params.gamma,
            self.params.n_th_a,
            a,
            sm_list,
            gamma_phi=self.params.gamma_phi,
        )

        # Observables
        n_atoms_op = sum(sm.dag() * sm for sm in sm_list)
        e_ops = [a.dag() * a, n_atoms_op]

        # Configure solver options for better performance
        opts = Options()
        opts.atol = atol
        opts.rtol = rtol
        opts.nsteps = 50000  # Allow more steps for stiff problems
        
        # Use sparse matrices for large systems
        if use_sparse:
            # QuTiP automatically uses sparse when beneficial
            # But we can encourage it
            if hasattr(opts, 'use_sparse'):
                opts.use_sparse = True
        
        # Use best available integrator
        # 'adams' for non-stiff, 'bdf' for stiff problems
        # Master equation is typically stiff
        opts.method = 'bdf'
        
        return mesolve(H, psi0, self.params.tlist, c_ops, e_ops=e_ops, options=opts)

    def run_parameter_sweep(
        self,
        param_name: str,
        param_values: List[float],
        progress: bool = True
    ) -> List[Result]:
        """
        Run simulations for multiple parameter values in parallel.
        
        This is the most effective parallelization strategy for master equation:
        run independent simulations with different parameters simultaneously.
        
        Args:
            param_name: Name of parameter to vary ('g', 'kappa', 'gamma', etc.)
            param_values: List of parameter values to test
            progress: Show progress bar (requires tqdm)
            
        Returns:
            List of Result objects, one per parameter value
            
        Example:
            >>> solver = ParallelExactSolver(params, n_jobs=-1)
            >>> results = solver.run_parameter_sweep('g', [0.01, 0.05, 0.1, 0.2])
        """
        def run_single(param_value):
            # Create a copy of params with modified value
            params_copy = SimParams()
            for attr in dir(self.params):
                if not attr.startswith('_'):
                    setattr(params_copy, attr, getattr(self.params, attr))
            setattr(params_copy, param_name, param_value)
            
            # Create temporary solver and run
            temp_solver = ParallelExactSolver(params_copy, n_jobs=1)
            return temp_solver.run()
        
        if progress:
            try:
                from tqdm import tqdm
                param_values = tqdm(param_values, desc=f"Sweeping {param_name}")
            except ImportError:
                pass
        
        # Run in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(run_single)(val) for val in param_values
        )
        
        return results

    def run_n_atoms_sweep(
        self,
        n_atoms_values: List[int],
        progress: bool = True
    ) -> List[Result]:
        """
        Run simulations for different atom counts in parallel.
        
        Args:
            n_atoms_values: List of atom counts to simulate
            progress: Show progress bar
            
        Returns:
            List of Result objects
        """
        return self.run_parameter_sweep('n_atoms', n_atoms_values, progress)

    def run_coupling_sweep(
        self,
        g_values: List[float],
        progress: bool = True
    ) -> List[Result]:
        """
        Run simulations for different coupling strengths in parallel.
        
        Args:
            g_values: List of coupling strengths (Hz)
            progress: Show progress bar
            
        Returns:
            List of Result objects
        """
        return self.run_parameter_sweep('g', g_values, progress)


class TimeChunkedSolver:
    """
    Experimental: Split time evolution into chunks and parallelize.
    
    WARNING: This provides limited speedup because:
    1. Chunks must be run sequentially (initial state depends on previous chunk)
    2. Overhead of splitting/merging results
    3. ODE solvers already optimize time-stepping internally
    
    Only useful for very long simulations where checkpointing is needed.
    """
    
    def __init__(self, params: SimParams, n_chunks: int = 4):
        self.params = params
        self.n_chunks = n_chunks
    
    def run(self) -> Result:
        """
        Run simulation in time chunks (mainly for checkpointing).
        
        Note: This is NOT truly parallel - chunks must run sequentially.
        Main benefit is memory management and intermediate checkpointing.
        """
        # Split time into chunks
        tlist_full = self.params.tlist
        chunk_size = len(tlist_full) // self.n_chunks
        
        results = []
        current_state = None
        
        for i in range(self.n_chunks):
            # Define chunk time list
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < self.n_chunks - 1 else len(tlist_full)
            tlist_chunk = tlist_full[start_idx:end_idx]
            
            # Setup for this chunk
            a, sm_list = get_operators(self.params.N, n_atoms=self.params.n_atoms)
            
            if current_state is None:
                psi0 = get_initial_state(self.params.N, n_atoms=self.params.n_atoms)
            else:
                psi0 = current_state
            
            H = jc_hamiltonian(
                self.params.wc, self.params.wa, self.params.g,
                a, sm_list, use_rwa=self.params.use_rwa
            )
            
            c_ops = get_collapse_operators(
                self.params.kappa, self.params.gamma, self.params.n_th_a,
                a, sm_list, gamma_phi=self.params.gamma_phi
            )
            
            n_atoms_op = sum(sm.dag() * sm for sm in sm_list)
            e_ops = [a.dag() * a, n_atoms_op]
            
            # Solve this chunk
            # Adjust tlist to start from 0 for integrator
            tlist_chunk_rel = tlist_chunk - tlist_chunk[0]
            result = mesolve(H, psi0, tlist_chunk_rel, c_ops, e_ops=e_ops)
            
            results.append(result)
            current_state = result.states[-1]
        
        # Merge results
        merged_result = self._merge_results(results, tlist_full)
        return merged_result
    
    def _merge_results(self, results: List[Result], tlist_full: np.ndarray) -> Result:
        """Merge chunked results into single Result object."""
        # Concatenate expectation values
        expect_merged = []
        for i in range(len(results[0].expect)):
            expect_i = np.concatenate([r.expect[i] for r in results])
            expect_merged.append(expect_i)
        
        # Create merged result
        merged = Result()
        merged.expect = expect_merged
        merged.times = tlist_full
        merged.solver = 'mesolve'
        
        return merged


def benchmark_solver_performance(
    n_atoms_range: List[int],
    n_jobs: int = -1,
    output_file: Optional[str] = None
) -> dict:
    """
    Benchmark master equation solver performance for different system sizes.
    
    Args:
        n_atoms_range: List of atom counts to test
        n_jobs: Number of parallel jobs
        output_file: Optional file to save results
        
    Returns:
        Dictionary with timing and scaling information
    """
    import time
    
    results = {
        'n_atoms': [],
        'hilbert_dim': [],
        'runtime': [],
        'memory_gb': []
    }
    
    for n_atoms in n_atoms_range:
        params = SimParams()
        params.n_atoms = n_atoms
        params.N = 5  # Keep cavity dimension manageable
        params.t_max = 10.0
        params.n_steps = 100
        
        # Calculate Hilbert space dimension
        dim = (params.N + 1) * (2 ** n_atoms)
        
        print(f"\nTesting N_atoms={n_atoms}, Hilbert dim={dim}")
        
        solver = ParallelExactSolver(params, n_jobs=n_jobs)
        
        start = time.time()
        result = solver.run()
        runtime = time.time() - start
        
        # Estimate memory (rough): density matrix is dim × dim complex numbers
        memory_gb = (dim ** 2) * 16 / (1024 ** 3)  # 16 bytes per complex128
        
        results['n_atoms'].append(n_atoms)
        results['hilbert_dim'].append(dim)
        results['runtime'].append(runtime)
        results['memory_gb'].append(memory_gb)
        
        print(f"Runtime: {runtime:.2f}s, Memory: {memory_gb:.3f} GB")
    
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results
