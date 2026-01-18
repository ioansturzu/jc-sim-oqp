#!/bin/bash
# Quick reference for HPC benchmark operations
# Usage: source benchmark_helpers.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_venv() {
    [[ -z "${VIRTUAL_ENV}" ]] && echo -e "${RED}No venv! Run: source .venv/bin/activate${NC}" && return 1
    echo -e "${GREEN}✓ venv active${NC}"
}

test_benchmark() { echo -e "${YELLOW}Testing...${NC}"; uv run test_benchmark_runner.py; }
run_single() { uv run python examples/benchmark_runner.py "${1:-error:100}" test_results/; }
submit_slurm() { echo -e "${YELLOW}Submitting 120 jobs...${NC}"; sbatch run_benchmark.batch; }
check_status() { squeue -u $USER; }
monitor_job() { watch -n 5 "squeue -j ${1}"; }
view_output() { tail -f "benchmark_${1}_${2}.out"; }
count_results() { 
    local count=$(find benchmark_results/task_* -name "results.json" 2>/dev/null | wc -l)
    echo -e "${GREEN}Completed: ${count}/120${NC}"
}
plot_results() { uv run aggregate_results.py && echo -e "${GREEN}✓ Plots in benchmark_results/plots/${NC}"; }
clean_results() { read -p "Delete all results? (yes/no): " c && [[ "$c" == "yes" ]] && rm -rf benchmark_results/ test_results/; }

show_help() {
    echo -e "${GREEN}HPC Benchmark Commands:${NC}"
    echo "  test_benchmark              - Local validation"
    echo "  submit_slurm                - Submit 120-task array"
    echo "  check_status                - Job queue status"
    echo "  count_results               - Completion progress (X/120)"
    echo "  plot_results                - Generate analysis plots"
    echo "  monitor_job JOBID           - Watch specific job"
    echo "  run_single error:1000       - Test single config"
    echo "  clean_results               - Delete all output"
    echo -e "${YELLOW}Workflow: test_benchmark → submit_slurm → count_results → plot_results${NC}"
}

echo -e "${GREEN}Benchmark helpers loaded.${NC} Type 'show_help'"
