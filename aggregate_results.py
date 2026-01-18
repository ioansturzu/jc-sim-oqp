#!/usr/bin/env python3
"""Aggregate and analyze benchmark results from parallel SLURM runs (Version 2).
This version handles multiple trajectory counts in time benchmarks.
Usage: uv run aggregate_results_v2.py [results_directory]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr


def load_results(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Load all result JSON files from subdirectories."""
    results = {'error': [], 'time': []}

    for task_dir in sorted(results_dir.glob('task_*')):
        result_file = task_dir / 'results.json'
        if result_file.exists():
            with result_file.open() as f:
                data = json.load(f)
                bench_type = data['benchmark_type']
                # Store directory path for loading trajectory data
                data['task_dir'] = str(task_dir)
                results[bench_type].append(data)

    return results


def load_trajectory_data(task_dir: Path) -> dict[str, np.ndarray] | None:
    """Load trajectory data from npz file if it exists."""
    data_file = task_dir / 'trajectory_data.npz'
    if data_file.exists():
        npz = np.load(data_file)
        return {
            'times': npz['times'],
            'exact': npz['exact'],
            'stochastic': npz['stochastic']
        }
    return None


def compute_advanced_error_metrics(exact: np.ndarray, stochastic: np.ndarray) -> dict[str, float]:
    """Compute comprehensive error metrics beyond RMSE."""
    error = exact - stochastic
    abs_error = np.abs(error)

    return {
        'rmse': float(np.sqrt(np.mean(error**2))),
        'mae': float(np.mean(abs_error)),
        'max_error': float(np.max(abs_error)),
        'std_error': float(np.std(error)),
        'relative_rmse': float(np.sqrt(np.mean(error**2)) / (np.mean(np.abs(exact)) + 1e-10)),
        'correlation': float(pearsonr(exact, stochastic)[0])
    }


def fit_convergence_law(ntraj_list: list[int], rmse_list: list[float]) -> tuple[float, float, float]:
    """Fit 1/sqrt(N) convergence law and compute goodness of fit."""
    def model(n, c):
        return c / np.sqrt(n)

    try:
        popt, pcov = curve_fit(model, ntraj_list, rmse_list)
        C_fit = popt[0]
        # Compute R^2
        residuals = rmse_list - model(np.array(ntraj_list), C_fit)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((rmse_list - np.mean(rmse_list))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return C_fit, np.sqrt(pcov[0, 0]), r_squared
    except Exception:
        return 0.0, 0.0, 0.0


def plot_error_results(error_results: list[dict[str, Any]], output_dir: Path):
    """Plot comprehensive error convergence analysis with replicate aggregation."""
    plt.style.use('seaborn-v0_8-whitegrid')

    # Aggregate results
    data = defaultdict(list)
    for r in error_results:
        data[r['ntraj']].append(r['rmse'])

    unique_ntraj = np.array(sorted(data.keys()))
    means = np.array([np.mean(data[n]) for n in unique_ntraj])
    stds = np.array([np.std(data[n]) for n in unique_ntraj])

    # Fit convergence law to means
    C_fit, C_err, r_squared = fit_convergence_law(unique_ntraj, means)

    _, ax = plt.subplots(figsize=(11, 8), facecolor='#f8f9fa')
    ax.set_facecolor('white')

    # Plot individual replicates with high transparency
    for n in unique_ntraj:
        ax.loglog([n]*len(data[n]), data[n], 'o', color='#3498db', alpha=0.15, markersize=4, zorder=2)

    # Plot mean values
    ax.loglog(unique_ntraj, means, 'o-', color='#2980b9', linewidth=2.5, markersize=7,
              label='Mean RMSE across replicates', zorder=4)

    # Fill error region (mean ± std)
    ax.fill_between(unique_ntraj, means-stds, means+stds, color='#3498db', alpha=0.1, zorder=1)

    # Fitted 1/sqrt(N) line
    if C_fit > 0:
        ntraj_fit = np.logspace(np.log10(unique_ntraj[0]), np.log10(unique_ntraj[-1]), 100)
        fitted = C_fit / np.sqrt(ntraj_fit)
        ax.loglog(ntraj_fit, fitted, 'k--', linewidth=1.5, alpha=0.6,
                  label=f'Theory $C/\\sqrt{{N}}$ ($C={C_fit:.4f}$, $R^2={r_squared:.3f}$)',
                  zorder=3)

    ax.set_xlabel('Number of Trajectories ($N_{\\mathrm{traj}}$)', fontsize=13, fontweight='500')
    ax.set_ylabel('RMSE (Exact vs Stochastic)', fontsize=13, fontweight='500')
    ax.set_title('Quantum Trajectory Convergence', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, which="both", ls="-", alpha=0.15)
    ax.legend(fontsize=10, frameon=True, facecolor='white', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_convergence.png', dpi=300)
    plt.savefig(output_dir / 'error_convergence.pdf')
    print(f"Saved improved error convergence plot to {output_dir}")

    # Print detailed statistics
    print("\n" + "="*100)
    print("ERROR CONVERGENCE ANALYSIS")
    print("="*100)
    print(f"{'N_traj':>10} {'RMSE':>12} {'MAE':>12} {'Max Err':>12} {'Rel RMSE':>12} {'Correlation':>12}")
    print("-" * 100)

    for r in error_results:
        print(f"{r['ntraj']:>10} {r['rmse']:>12.6f} {r['mae']:>12.6f} "
              f"{r['max_error']:>12.6f} {r['relative_rmse']:>12.6f} {r['correlation']:>12.6f}")

    print(f"\nFitted convergence constant: C = {C_fit:.6f} ± {C_err:.6f}")
    print(f"Goodness of fit: R² = {r_squared:.6f}")
    print("="*100)


def plot_time_results_multi_trajectory(time_results: list[dict[str, Any]], output_dir: Path):
    """Plot timing results with support for multiple trajectory counts.
    Groups results by trajectory count and plots scaling analysis.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    # Group results by ntraj and n_atoms for aggregation
    data = defaultdict(lambda: defaultdict(list))
    for r in time_results:
        ntraj = r.get('ntraj', 100)
        n_atoms = r['n_atoms']
        if r.get('time_stochastic') is not None:
            data[ntraj][n_atoms].append(r['time_stochastic'])

    ntraj_values = sorted(data.keys())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(ntraj_values)))

    _, ax = plt.subplots(figsize=(13, 9), facecolor='#f8f9fa')
    ax.set_facecolor('white')

    # Plot stochastic results for each trajectory count
    for idx, ntraj in enumerate(ntraj_values):
        atom_counts = sorted(data[ntraj].keys())
        means = np.array([np.mean(data[ntraj][n]) for n in atom_counts])

        if len(atom_counts) < 2:
            continue

        # Fit exponential scaling
        try:
            log_t = np.log(means)
            coeffs = np.polyfit(atom_counts, log_t, 1)
            exp_rate = coeffs[0]
            base = np.exp(exp_rate)

            # Compute R^2
            predicted = coeffs[0] * np.array(atom_counts) + coeffs[1]
            ss_res = np.sum((log_t - predicted)**2)
            ss_tot = np.sum((log_t - np.mean(log_t))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            # Fallback for failed scaling fit
            r2, base = 0, 0

        # Plot mean points and line
        marker = ['o', 's', '^', 'D', 'v'][idx % 5]
        label = rf'{ntraj} traj: $T \propto {base:.2f}^N$ ($R^2={r2:.3f}$)'
        ax.semilogy(atom_counts, means, marker=marker, color=colors[idx], markersize=7,
                   linewidth=2, label=label, alpha=0.9)

        # Individual replicates with transparency
        for n in atom_counts:
            ax.semilogy([n]*len(data[ntraj][n]), data[ntraj][n], marker, color=colors[idx], alpha=0.15, markersize=3)

    # Plot exact results (unified comparison)
    exact_data = defaultdict(list)
    for r in time_results:
        if r.get('time_exact') is not None:
            exact_data[r['n_atoms']].append(r['time_exact'])

    if exact_data:
        ex_atoms = sorted(exact_data.keys())
        ex_means = np.array([np.mean(exact_data[n]) for n in ex_atoms])
        if len(ex_atoms) >= 2:
            try:
                log_t_ex = np.log(ex_means)
                coeffs_ex = np.polyfit(ex_atoms, log_t_ex, 1)
                ex_base = np.exp(coeffs_ex[0])
                ex_r2 = 1 - (np.sum((log_t_ex - (coeffs_ex[0]*np.array(ex_atoms) + coeffs_ex[1]))**2) / np.sum((log_t_ex - np.mean(log_t_ex))**2))
                ax.semilogy(ex_atoms, ex_means, 'o-', color='#e74c3c', linewidth=2.5, markersize=8,
                           label=rf'Exact ME: $T \propto {ex_base:.2f}^N$ ($R^2={ex_r2:.3f}$)')
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                # Fallback for failed exact scaling fit
                ax.semilogy(ex_atoms, ex_means, 'o-', color='#e74c3c', linewidth=2.5, markersize=8, label='Exact Master Eq')

    ax.set_xlabel('Number of Atoms ($N$)', fontsize=13, fontweight='500')
    ax.set_ylabel('Runtime (seconds)', fontsize=13, fontweight='500')
    ax.set_title('Computational Scaling vs Problem Size', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, which="both", alpha=0.15)
    ax.legend(fontsize=10, loc='upper left', frameon=True, facecolor='white')

    plt.tight_layout()
    plt.savefig(output_dir / 'time_scaling_multi_traj.png', dpi=300)
    plt.savefig(output_dir / 'time_scaling_multi_traj.pdf')
    print(f"Saved improved multi-trajectory timing plot to {output_dir}")

    # Print detailed statistics
    print("\n" + "="*100)
    print("TIMING BENCHMARK RESULTS (Multi-Trajectory)")
    print("="*100)
    print(f"{'N_traj':>8} {'N_atoms':>10} {'Stochastic (s)':>15} {'Exact (s)':>15} {'Speedup':>12}")
    print("-" * 100)

    for ntraj in ntraj_values:
        atom_counts = sorted(data[ntraj].keys())
        for n_atoms in atom_counts:
            # Display the first replicate if multiple exist for this point
            stoch_time = data[ntraj][n_atoms][0]
            # Exact is usually same for all ntraj
            exact_time = exact_data[n_atoms][0] if n_atoms in exact_data else None

            speedup = exact_time / stoch_time if exact_time and stoch_time else None
            stoch_str = f"{stoch_time:15.2f}"
            exact_str = f"{exact_time:15.2f}" if exact_time else "       Failed"
            speedup_str = f"{speedup:12.2f}x" if speedup else "         N/A"

            print(f"{ntraj:>8} {n_atoms:>10} {stoch_str} {exact_str} {speedup_str}")

    print("="*100)


def plot_trajectory_comparison(error_results: list[dict[str, Any]], output_dir: Path):
    """Create subplots comparing exact vs stochastic at representative trajectory counts.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    # Group by n_traj and pick one representative for each
    data_by_ntraj = {}
    for r in error_results:
        ntraj = r['ntraj']
        if ntraj not in data_by_ntraj:
            data_by_ntraj[ntraj] = r

    sorted_ntraj = sorted(data_by_ntraj.keys())
    if len(sorted_ntraj) < 2:
        return

    # Select up to 3 representative trajectory counts
    indices = [0, len(sorted_ntraj)//2, -1] if len(sorted_ntraj) >= 3 else [0, -1]
    selected_ntraj = [sorted_ntraj[i] for i in indices]

    _, axes = plt.subplots(1, len(selected_ntraj), figsize=(5.5*len(selected_ntraj), 5), facecolor='#f8f9fa')
    if len(selected_ntraj) == 1:
        axes = [axes]

    for idx, ntraj in enumerate(selected_ntraj):
        ax = axes[idx]
        result = data_by_ntraj[ntraj]
        task_dir = Path(result['task_dir'])
        traj_data = load_trajectory_data(task_dir)

        if traj_data is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        ax.set_facecolor('white')
        ax.plot(traj_data['times'], traj_data['exact'], 'k-', linewidth=2, label='Exact', alpha=0.9)
        ax.plot(traj_data['times'], traj_data['stochastic'], color='#3498db', linestyle='--',
                linewidth=1.5, label=f'Stoch ({ntraj} tr)', alpha=0.8)

        ax.set_xlabel('Time (ns)', fontsize=11)
        ax.set_ylabel('Population', fontsize=11)
        ax.set_title(f'$N_{{traj}}$ = {ntraj}\nRMSE = {result["rmse"]:.5f}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.15)
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'trajectory_comparison.png', dpi=300)
    plt.savefig(output_dir / 'trajectory_comparison.pdf')
    print(f"Saved improved trajectory comparison plot to {output_dir}")


def plot_time_dependent_error(error_results: list[dict[str, Any]], output_dir: Path):
    """Plot how error evolves over time for different trajectory counts with color gradient.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    # Filter to unique ntraj for cleaner plot (pick first representative of each)
    unique_results = {}
    for r in sorted(error_results, key=lambda x: x['ntraj']):
        if r['ntraj'] not in unique_results:
            unique_results[r['ntraj']] = r

    selected_ntraj = sorted(unique_results.keys())
    if len(selected_ntraj) > 6:
        indices = np.linspace(0, len(selected_ntraj)-1, 6, dtype=int)
        selected_ntraj = [selected_ntraj[i] for i in indices]

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor='#f8f9fa')
    colors = plt.cm.viridis(np.linspace(0.1, 0.8, len(selected_ntraj)))

    for idx, ntraj in enumerate(selected_ntraj):
        result = unique_results[ntraj]
        task_dir = Path(result['task_dir'])
        traj_data = load_trajectory_data(task_dir)

        if traj_data is None:
            continue

        times = traj_data['times']
        error = np.abs(traj_data['exact'] - traj_data['stochastic'])

        ax1.set_facecolor('white')
        ax1.plot(times, error, color=colors[idx], linewidth=1.2, alpha=0.8, label=f'$N_{{tr}}$={ntraj}')

        # Cumulative RMSE
        cumulative_rmse = np.sqrt(np.cumsum(error**2) / np.arange(1, len(error)+1))
        ax2.set_facecolor('white')
        ax2.plot(times, cumulative_rmse, color=colors[idx], linewidth=1.8, alpha=0.9, label=f'$N_{{tr}}$={ntraj}')

    ax1.set_xlabel('Time (ns)', fontsize=12)
    ax1.set_ylabel('Instantaneous |Error|', fontsize=12)
    ax1.set_title('Error Evolution Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.15)

    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.set_ylabel('Sequential RMSE', fontsize=12)
    ax2.set_title('Cumulative Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.15)
    ax2.set_yscale('log')
    ax2.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, facecolor='white')

    plt.tight_layout()
    plt.savefig(output_dir / 'time_dependent_error.png', dpi=300)
    plt.savefig(output_dir / 'time_dependent_error.pdf')
    print(f"Saved improved time-dependent error plot to {output_dir}")


def save_summary_stats(error_results: list[dict[str, Any]],
                      time_results: list[dict[str, Any]],
                      output_dir: Path):
    """Save comprehensive summary statistics to JSON and text files."""
    # Organize time results by trajectory count
    time_by_ntraj = defaultdict(list)
    for r in time_results:
        ntraj = r.get('ntraj', 100)
        time_by_ntraj[ntraj].append({
            'n_atoms': r['n_atoms'],
            'time_stochastic': r.get('time_stochastic'),
            'time_exact': r.get('time_exact')
        })

    summary = {
        'error_benchmarks': {
            'count': len(error_results),
            'trajectory_counts': [r['ntraj'] for r in error_results],
            'rmse_values': [r['rmse'] for r in error_results],
            'mae_values': [r['mae'] for r in error_results],
            'max_errors': [r['max_error'] for r in error_results],
            'correlations': [r['correlation'] for r in error_results]
        },
        'time_benchmarks': {
            'count': len(time_results),
            'trajectory_counts': list(time_by_ntraj.keys()),
            'by_trajectory_count': dict(time_by_ntraj)
        }
    }

    # Save JSON
    with (output_dir / 'summary_stats.json').open('w') as f:
        json.dump(summary, f, indent=2)

    # Save human-readable text
    with (output_dir / 'summary_report.txt').open('w') as f:
        f.write("="*100 + "\n")
        f.write("BENCHMARK SUMMARY REPORT\n")
        f.write("="*100 + "\n\n")

        f.write("ERROR BENCHMARKS:\n")
        f.write(f"  Total runs: {len(error_results)}\n")
        f.write(f"  Trajectory counts tested: {sorted([r['ntraj'] for r in error_results])}\n")
        f.write(f"  RMSE range: {min(summary['error_benchmarks']['rmse_values']):.6f} - "
                f"{max(summary['error_benchmarks']['rmse_values']):.6f}\n")
        f.write(f"  Best correlation: {max(summary['error_benchmarks']['correlations']):.6f}\n\n")

        f.write("TIME BENCHMARKS:\n")
        f.write(f"  Total runs: {len(time_results)}\n")
        f.write(f"  Trajectory counts tested: {sorted(time_by_ntraj.keys())}\n")
        for ntraj in sorted(time_by_ntraj.keys()):
            n_atoms_tested = [r['n_atoms'] for r in time_by_ntraj[ntraj]]
            f.write(f"  N_traj={ntraj}: tested {len(n_atoms_tested)} atom counts "
                   f"(N = {min(n_atoms_tested)} to {max(n_atoms_tested)})\n")

        f.write("\n" + "="*100 + "\n")

    print(f"Saved summary statistics to {output_dir}")


def main():
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        results_dir = Path.cwd() / 'benchmark_results'

    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} does not exist")
        sys.exit(1)

    print(f"Loading results from {results_dir}")
    all_results = load_results(results_dir)

    error_results = all_results['error']
    time_results = all_results['time']

    print(f"Found {len(error_results)} error benchmark results")
    print(f"Found {len(time_results)} time benchmark results")

    if not error_results and not time_results:
        print("No results found!")
        sys.exit(1)

    output_dir = results_dir / 'analysis'
    output_dir.mkdir(exist_ok=True)

    # Process error benchmarks
    if error_results:
        # Compute metrics for each result
        for result in error_results:
            task_dir = Path(result['task_dir'])
            traj_data = load_trajectory_data(task_dir)
            if traj_data:
                metrics = compute_advanced_error_metrics(traj_data['exact'], traj_data['stochastic'])
                result.update(metrics)

        # Filter out results without trajectory data
        error_results = [r for r in error_results if 'rmse' in r]

        if error_results:
            plot_error_results(error_results, output_dir)
            plot_trajectory_comparison(error_results, output_dir)
            plot_time_dependent_error(error_results, output_dir)

    # Process time benchmarks
    if time_results:
        plot_time_results_multi_trajectory(time_results, output_dir)

    # Save summary statistics
    save_summary_stats(error_results, time_results, output_dir)

    print(f"\nAll analysis complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
