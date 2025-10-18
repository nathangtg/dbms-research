#!/bin/bash

# Run complete algorithm comparison and generate all figures

echo "════════════════════════════════════════════════════════════════════════"
echo "  COMPREHENSIVE ANN ALGORITHM COMPARISON"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

# Navigate to benchmarks directory
cd "$(dirname "$0")"

# Step 1: Generate test data (if not exists)
if [ ! -f "data/vectors_10k.npy" ]; then
    echo "Step 1/3: Generating test data..."
    python generate_test_data.py
    echo ""
else
    echo "Step 1/3: Test data already exists ✓"
    echo ""
fi

# Step 2: Run benchmark
echo "Step 2/3: Running algorithm comparison benchmark..."
python compare_all_algorithms.py 2>&1 | tee algorithm_comparison.log
echo ""

# Step 3: Generate visualizations
echo "Step 3/3: Generating comparison figures..."
python visualize_algorithm_comparison.py
echo ""

echo "════════════════════════════════════════════════════════════════════════"
echo "  ✓ COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to:"
echo "  - algorithm_comparison_results.json"
echo "  - algorithm_comparison.log"
echo "  - ALGORITHM_COMPARISON.md"
echo "  - figures_algorithm_comparison/"
echo ""
echo "To view figures:"
echo "  cd figures_algorithm_comparison"
echo "  xdg-open 01_algorithm_comparison.png  # or use your image viewer"
echo ""
