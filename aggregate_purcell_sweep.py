#!/usr/bin/env python3
"""Aggregate and plot high-resolution Purcell physics sweep results.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps


def aggregate_sweep(results_dir: str = "benchmark_results_sweep"):
    res_path = Path(results_dir)
    tasks = sorted(res_path.glob("task_*"))
    
    fp_list = []
    error_list = []
    traces = []
    
    print(f"Loading {len(tasks)} result sets...")
    
    for task in tasks:
        # Load JSON
        res_file = task / 'results.json'
        if not res_file.exists():
            continue
            
        with open(res_file) as f:
            data = json.load(f)
            fp_list.append(data['fp_actual'])
            error_list.append(data['rate_error'] * 100) # Percent
            
        # Load Trace
        trace_file = task / 'purcell_trace.npz'
        if trace_file.exists():
            trace_data = np.load(trace_file)
            traces.append({
                'fp': data['fp_actual'],
                'times': trace_data['times'],
                'pe': trace_data['pe_numerical']
            })
        
    if not fp_list:
        print("No results found.")
        return

    # Sort by FP
    sort_idx = np.argsort(fp_list)
    fp_list = np.array(fp_list)[sort_idx]
    error_list = np.array(error_list)[sort_idx]
    traces = [traces[i] for i in sort_idx]
    
    # 1. Plot Rate Error vs Fp
    plt.figure(figsize=(10, 6))
    plt.plot(fp_list, error_list, 'o-', color='#e67e22', linewidth=2, markersize=8)
    plt.xscale('log')
    plt.xlabel('Purcell Factor ($F_p$)', fontsize=12)
    plt.ylabel('Decay Rate Error (%)', fontsize=12)
    plt.title('Approximation Error: Linear Purcell vs Full Simulation', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(res_path / 'fp_error_scaling.png', dpi=300)
    print(f"Saved error scaling plot to {res_path / 'fp_error_scaling.png'}")
    
    # 2. Plot Numerical Traces (Continuous Lines)
    plt.figure(figsize=(12, 7))
    cmap = colormaps['viridis']
    
    for i, trace in enumerate(traces):
        color = cmap(i / len(traces))
        # Label only every 5th trace to avoid legend clutter
        label = f"$F_p={trace['fp']:.1f}$" if i % 5 == 0 or i == len(traces)-1 else None
        plt.plot(trace['times'], trace['pe'], '-', color=color, alpha=0.7, linewidth=1.5, label=label)
        
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('Excited State Population $P_e$', fontsize=12)
    plt.title('Decay Trace Evolution: From Weak to Strong Coupling', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.1)
    plt.tight_layout()
    plt.savefig(res_path / 'purcell_traces.png', dpi=300)
    print(f"Saved combined traces plot to {res_path / 'purcell_traces.png'}")
    
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="benchmark_results_sweep")
    args = parser.parse_args()
    aggregate_sweep(args.dir)
