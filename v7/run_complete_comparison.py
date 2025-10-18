#!/usr/bin/env python3
"""
Complete Benchmark Pipeline: 10K vs 100K Comparison
====================================================

This script:
1. Generates test data (10K and 100K vectors)
2. Runs benchmarks on both datasets
3. Generates publication-quality comparison figures

Usage:
    python run_complete_comparison.py

Options:
    --skip-data-gen    Skip data generation if files already exist
    --skip-benchmarks  Skip benchmarks if results already exist
    --figures-only     Only generate figures from existing results
"""

import subprocess
import sys
from pathlib import Path
import argparse
import json

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80 + "\n")

def check_file_exists(filepath):
    """Check if a file exists."""
    return Path(filepath).exists()

def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"▶ {description}")
    print(f"  Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error: {description} failed with exit code {result.returncode}")
        return False
    
    print(f"✓ {description} completed successfully\n")
    return True

def main():
    parser = argparse.ArgumentParser(description='Run complete 10K vs 100K benchmark comparison')
    parser.add_argument('--skip-data-gen', action='store_true',
                       help='Skip data generation if files exist')
    parser.add_argument('--skip-benchmarks', action='store_true',
                       help='Skip benchmarks if results exist')
    parser.add_argument('--figures-only', action='store_true',
                       help='Only generate figures from existing results')
    
    args = parser.parse_args()
    
    # File paths
    base_path = Path(__file__).parent
    data_dir = base_path / 'data'
    benchmarks_dir = base_path / 'benchmarks'
    
    vectors_10k = data_dir / 'vectors_10k.npy'
    vectors_100k = data_dir / 'vectors_100k.npy'
    queries_100 = data_dir / 'queries_100.npy'
    results_10k = benchmarks_dir / 'algorithm_comparison_results_10k.json'
    results_100k = benchmarks_dir / 'algorithm_comparison_results_100k.json'
    
    print_section("🚀 ZGQ vs HNSW: COMPLETE BENCHMARK PIPELINE")
    
    # ========================================================================
    # Step 1: Generate test data
    # ========================================================================
    if not args.figures_only:
        print_section("STEP 1: Generate Test Data")
        
        # Check 10K data
        if args.skip_data_gen and vectors_10k.exists():
            print(f"✓ 10K data already exists: {vectors_10k}")
        else:
            if not run_command(
                "python benchmarks/generate_test_data.py --n_vectors 10000 --n_queries 100",
                "Generate 10K vector dataset"
            ):
                sys.exit(1)
        
        # Check 100K data
        if args.skip_data_gen and vectors_100k.exists():
            print(f"✓ 100K data already exists: {vectors_100k}")
        else:
            if not run_command(
                "python benchmarks/generate_test_data.py --n_vectors 100000 --n_queries 100",
                "Generate 100K vector dataset"
            ):
                sys.exit(1)
    
    # ========================================================================
    # Step 2: Run benchmarks
    # ========================================================================
    if not args.figures_only:
        print_section("STEP 2: Run Benchmarks")
        
        # Run 10K benchmark
        if args.skip_benchmarks and results_10k.exists():
            print(f"✓ 10K results already exist: {results_10k}")
        else:
            if not run_command(
                "python benchmarks/compare_all_algorithms.py --dataset 10k",
                "Run 10K benchmark (HNSW, IVF, IVF+PQ, ZGQ)"
            ):
                sys.exit(1)
        
        # Run 100K benchmark
        if args.skip_benchmarks and results_100k.exists():
            print(f"✓ 100K results already exist: {results_100k}")
        else:
            if not run_command(
                "python benchmarks/compare_all_algorithms.py --dataset 100k",
                "Run 100K benchmark (HNSW, IVF, IVF+PQ, ZGQ)"
            ):
                sys.exit(1)
    
    # ========================================================================
    # Step 3: Verify results exist
    # ========================================================================
    print_section("STEP 3: Verify Results")
    
    if not results_10k.exists():
        print(f"❌ Error: 10K results not found: {results_10k}")
        print("   Run without --figures-only to generate benchmarks")
        sys.exit(1)
    
    if not results_100k.exists():
        print(f"❌ Error: 100K results not found: {results_100k}")
        print("   Run without --figures-only to generate benchmarks")
        sys.exit(1)
    
    # Load and display results
    with open(results_10k, 'r') as f:
        data_10k = json.load(f)
    with open(results_100k, 'r') as f:
        data_100k = json.load(f)
    
    # Convert list to dict for easier access
    results_10k_dict = {item['name']: item for item in data_10k}
    results_100k_dict = {item['name']: item for item in data_100k}
    
    print("✓ Found benchmark results:")
    print(f"\n10K Results:")
    print(f"  HNSW: Recall={results_10k_dict['HNSW']['recall_at_10']:.1f}%, "
          f"Memory={results_10k_dict['HNSW']['memory_mb']:.1f}MB")
    print(f"  ZGQ:  Recall={results_10k_dict['ZGQ Unified']['recall_at_10']:.1f}%, "
          f"Memory={results_10k_dict['ZGQ Unified']['memory_mb']:.1f}MB")
    
    print(f"\n100K Results:")
    print(f"  HNSW: Recall={results_100k_dict['HNSW']['recall_at_10']:.1f}%, "
          f"Memory={results_100k_dict['HNSW']['memory_mb']:.1f}MB")
    print(f"  ZGQ:  Recall={results_100k_dict['ZGQ Unified']['recall_at_10']:.1f}%, "
          f"Memory={results_100k_dict['ZGQ Unified']['memory_mb']:.1f}MB")
    
    # ========================================================================
    # Step 4: Generate figures
    # ========================================================================
    print_section("STEP 4: Generate Publication Figures")
    
    if not run_command(
        "python benchmarks/generate_publication_figures.py",
        "Generate publication-quality figures"
    ):
        sys.exit(1)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print_section("✅ PIPELINE COMPLETE!")
    
    print("Generated Files:")
    print("\n📊 Benchmark Results:")
    print(f"  • {results_10k}")
    print(f"  • {results_100k}")
    
    print("\n📈 Publication Figures:")
    figures_dir = benchmarks_dir / 'figures_zgq_vs_hnsw'
    if figures_dir.exists():
        figures = sorted(figures_dir.glob('*.png'))
        for fig in figures:
            print(f"  • {fig}")
    
    print("\n📄 Documentation:")
    docs = [
        'ZGQ_VS_HNSW_ONLY.md',
        'GOOD_NEWS.md',
        'SCALING_ANALYSIS.md',
        'HONEST_RECOMMENDATIONS.md'
    ]
    for doc in docs:
        doc_path = benchmarks_dir / doc
        if doc_path.exists():
            print(f"  • {doc_path}")
    
    print("\n" + "="*80)
    print("Next Steps:")
    print("  1. Review figures in: benchmarks/figures_zgq_vs_hnsw/")
    print("  2. Read analysis in: benchmarks/ZGQ_VS_HNSW_ONLY.md")
    print("  3. Include figures in your research paper")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
