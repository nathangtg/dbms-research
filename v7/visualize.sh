#!/bin/bash
# Quick visualization script
# Usage: ./visualize.sh

set -e

echo "=================================="
echo "ZGQ BENCHMARK VISUALIZATION"
echo "=================================="
echo ""

if [ ! -f "benchmark_results.json" ]; then
    echo "❌ Error: benchmark_results.json not found!"
    echo ""
    echo "Run the benchmark first:"
    echo "  python benchmarks/comprehensive_benchmark.py"
    exit 1
fi

echo "Generating visualizations from benchmark_results.json..."
echo ""

python3 visualize_results.py benchmark_results.json figures

echo ""
echo "✅ Done! Check the figures/ directory"
echo ""
echo "Generated files:"
ls -lh figures/*.png figures/*.txt | awk '{print "  " $9 " (" $5 ")"}'
echo ""
