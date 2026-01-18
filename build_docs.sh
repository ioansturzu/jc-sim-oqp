#!/bin/bash
# Quick documentation build and preview script

set -e

echo "Building JC-SIM-OQP documentation..."

# Clean old build
echo "Cleaning old build..."
rm -rf docs/_build/html

# Build documentation with uv
echo "Building HTML documentation..."
uv run sphinx-build -b html docs docs/_build/html

# Show result
echo ""
echo "✓ Documentation built successfully!"
echo ""
echo "Output: docs/_build/html/index.html"
echo ""
echo "View pages:"
echo "  - Index:      file://$(pwd)/docs/_build/html/index.html"
echo "  - Benchmarks: file://$(pwd)/docs/_build/html/benchmarks.html"
echo "  - System:     file://$(pwd)/docs/_build/html/benchmark_system.html"
echo "  - Tutorial:   file://$(pwd)/docs/_build/html/tutorial.html"
echo ""

# Try to open in browser
if command -v xdg-open &> /dev/null; then
    echo "Opening in browser..."
    xdg-open docs/_build/html/index.html
elif command -v open &> /dev/null; then
    echo "Opening in browser..."
    open docs/_build/html/index.html
else
    echo "Open manually: docs/_build/html/index.html"
fi
