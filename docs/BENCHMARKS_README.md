# Benchmarking Documentation

This directory contains comprehensive documentation for the HPC benchmarking system.

## Documentation Structure

### User Guides

- **[benchmarks.md](benchmarks.md)** - Quick start guide for running benchmarks on SLURM clusters
  - Configuration and setup
  - Submitting jobs
  - Monitoring progress
  - Analyzing results

- **[benchmark_system.md](benchmark_system.md)** - System architecture and design
  - Technical architecture
  - Component descriptions
  - HPC optimization strategies
  - Configuration examples

### Integration with Main Docs

These benchmark guides are integrated into the main Sphinx documentation:

1. Listed in `index.md` table of contents
2. Linked from `tutorial.md` (Section 4)
3. Referenced in `README.md` (Section 4)

## Supporting Files (in repository root)

- `run_benchmark.batch` - SLURM job script (120 array tasks)
- `examples/benchmark_runner.py` - Core benchmark library
- `aggregate_results.py` - Results analysis and plotting
- `test_benchmark_runner.py` - Local validation suite
- `benchmark_helpers.sh` - Shell convenience functions
- `ARCHITECTURE.txt` - Visual system diagram

## Quick Navigation

- **Getting Started**: See {doc}`benchmarks` → Quick Start
- **System Design**: See {doc}`benchmark_system` → Architecture
- **Basic Tutorial**: See {doc}`tutorial` → HPC Benchmarking
- **Python API**: See {doc}`api`

## Building Documentation

To generate the complete HTML documentation including benchmarks:

```bash
# Ensure dependencies are installed
uv sync

# Build docs
uv run sphinx-build -b html docs docs/_build/html
# or simply
./build_docs.sh

# Open
xdg-open docs/_build/html/benchmarks.html
```

## For Developers

When modifying the benchmark system:

1. Update code in `examples/benchmark_runner.py`
2. Update usage guide in `docs/benchmarks.md`
3. Update architecture notes in `docs/benchmark_system.md`
4. Rebuild documentation to verify cross-references
5. Test locally with `uv run test_benchmark_runner.py`

## Performance Data

The benchmark system is designed to characterize:

- **Error convergence**: Monte Carlo accuracy vs trajectory count
- **Timing scaling**: Computational cost vs system size
- **Statistical robustness**: Multiple replicates for confidence intervals

Expected metrics:
- 120 concurrent jobs on 2×72-core cluster
- 30-60 minute total runtime
- ~500 MB storage for all results
