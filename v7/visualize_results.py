#!/usr/bin/env python3
"""
Standalone script to generate visualizations from existing benchmark results.

Usage:
    python visualize_results.py                          # Uses benchmark_results.json
    python visualize_results.py my_results.json          # Uses custom file
    python visualize_results.py my_results.json figures  # Custom output dir
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from visualization import BenchmarkVisualizer


def main():
    """Generate visualizations from benchmark results."""
    # Parse arguments
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = 'benchmark_results.json'
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = 'figures'
    
    # Check file exists
    if not Path(results_file).exists():
        print(f"Error: Results file '{results_file}' not found!")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} [results_file] [output_dir]")
        print("\nExample:")
        print(f"  python {sys.argv[0]} benchmark_results.json figures")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("ZGQ BENCHMARK VISUALIZATION")
    print(f"{'='*70}")
    print(f"Results file: {results_file}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}\n")
    
    # Create visualizer
    visualizer = BenchmarkVisualizer(results_file, output_dir)
    
    # Generate all figures
    figures = visualizer.generate_all_figures()
    
    # Generate summary report
    report = visualizer.generate_summary_report()
    print("\n" + report)
    
    # Success message
    print(f"\n{'='*70}")
    print(f"✓ SUCCESS: Generated {len(figures)} visualizations")
    print(f"{'='*70}")
    print(f"\nOutput files in '{output_dir}/':")
    print("  • recall_latency_curve.png      - Main performance comparison")
    print("  • pareto_frontier.png            - Optimal trade-off curve")
    print("  • throughput_comparison.png      - QPS by recall range")
    print("  • latency_distribution.png       - p50/p95/p99 latencies")
    print("  • build_time_comparison.png      - Index construction time")
    print("  • memory_comparison.png          - Index size comparison")
    print("  • benchmark_summary.txt          - Text summary report")
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
