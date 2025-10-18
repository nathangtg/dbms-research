"""
Visualization module for ZGQ benchmark results.

Generates publication-quality figures comparing ZGQ against baseline methods.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

# Set style for publication-quality figures
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'


class BenchmarkVisualizer:
    """Generate visualizations from benchmark results."""
    
    def __init__(self, results_file: str, output_dir: str = "figures"):
        """
        Initialize visualizer.
        
        Args:
            results_file: Path to JSON file with benchmark results
            output_dir: Directory to save figures
        """
        self.results_file = results_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.results)
        
        # Extract dataset info
        if len(self.df) > 0:
            self.dataset_name = self.df['dataset_name'].iloc[0]
            self.n_vectors = self.df['n_vectors'].iloc[0]
            self.dimension = self.df['dimension'].iloc[0]
        
        print(f"Loaded {len(self.df)} benchmark results")
        print(f"Dataset: {self.dataset_name}")
        print(f"Methods: {self.df['method'].unique()}")
    
    def plot_recall_latency_curve(self, save: bool = True):
        """
        Plot recall vs latency trade-off curve.
        
        This is the primary metric for ANNS evaluation.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = self.df['method'].unique()
        colors = {'ZGQ': '#2E86AB', 'HNSW': '#A23B72', 'FAISS-IVF-PQ': '#F18F01'}
        markers = {'ZGQ': 'o', 'HNSW': 's', 'FAISS-IVF-PQ': '^'}
        
        for method in methods:
            method_df = self.df[self.df['method'] == method].sort_values('mean_latency_ms')
            
            # Filter out zero recall results
            method_df = method_df[method_df['recall_at_k'] > 0.01]
            
            if len(method_df) == 0:
                print(f"Warning: No valid results for {method}")
                continue
            
            ax.plot(
                method_df['mean_latency_ms'],
                method_df['recall_at_k'],
                marker=markers.get(method, 'o'),
                markersize=10,
                linewidth=2.5,
                label=method,
                color=colors.get(method, None),
                alpha=0.8
            )
            
            # Annotate points with parameters
            for idx, row in method_df.iterrows():
                if method == 'ZGQ':
                    label = f"n_probe={row['parameters']['n_probe']}"
                elif method == 'HNSW':
                    label = f"ef={row['parameters']['ef']}"
                elif method == 'FAISS-IVF-PQ':
                    label = f"nprobe={row['parameters']['nprobe']}"
                else:
                    label = ""
                
                ax.annotate(
                    label,
                    xy=(row['mean_latency_ms'], row['recall_at_k']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )
        
        ax.set_xlabel('Mean Query Latency (ms)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Recall@10', fontsize=14, fontweight='bold')
        ax.set_title(f'Recall-Latency Trade-off\n{self.dataset_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        
        if save:
            filepath = self.output_dir / 'recall_latency_curve.png'
            plt.savefig(filepath)
            print(f"Saved: {filepath}")
        
        plt.tight_layout()
        return fig
    
    def plot_throughput_comparison(self, save: bool = True):
        """Plot throughput (QPS) comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group by method and get best throughput for each recall range
        recall_bins = [(0, 0.4), (0.4, 0.7), (0.7, 0.9), (0.9, 1.0)]
        
        data = []
        for method in self.df['method'].unique():
            method_df = self.df[self.df['method'] == method]
            method_df = method_df[method_df['recall_at_k'] > 0.01]  # Filter zero recall
            
            for recall_min, recall_max in recall_bins:
                subset = method_df[
                    (method_df['recall_at_k'] >= recall_min) & 
                    (method_df['recall_at_k'] < recall_max)
                ]
                if len(subset) > 0:
                    best = subset.loc[subset['throughput_qps'].idxmax()]
                    data.append({
                        'method': method,
                        'recall_range': f'{int(recall_min*100)}-{int(recall_max*100)}%',
                        'throughput': best['throughput_qps'],
                        'recall': best['recall_at_k']
                    })
        
        if not data:
            print("No valid throughput data to plot")
            return None
        
        plot_df = pd.DataFrame(data)
        
        # Create grouped bar chart
        x = np.arange(len(recall_bins))
        width = 0.25
        methods = plot_df['method'].unique()
        
        colors = {'ZGQ': '#2E86AB', 'HNSW': '#A23B72', 'FAISS-IVF-PQ': '#F18F01'}
        
        for i, method in enumerate(methods):
            method_data = plot_df[plot_df['method'] == method]
            throughputs = []
            for recall_min, recall_max in recall_bins:
                range_str = f'{int(recall_min*100)}-{int(recall_max*100)}%'
                subset = method_data[method_data['recall_range'] == range_str]
                throughputs.append(subset['throughput'].values[0] if len(subset) > 0 else 0)
            
            ax.bar(x + i * width, throughputs, width, 
                  label=method, color=colors.get(method, None), alpha=0.8)
        
        ax.set_xlabel('Recall Range', fontsize=14, fontweight='bold')
        ax.set_ylabel('Throughput (Queries/sec)', fontsize=14, fontweight='bold')
        ax.set_title(f'Throughput Comparison by Recall Range\n{self.dataset_name}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x + width)
        ax.set_xticklabels([f'{int(r[0]*100)}-{int(r[1]*100)}%' for r in recall_bins])
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        
        if save:
            filepath = self.output_dir / 'throughput_comparison.png'
            plt.savefig(filepath)
            print(f"Saved: {filepath}")
        
        plt.tight_layout()
        return fig
    
    def plot_build_time_comparison(self, save: bool = True):
        """Plot index build time comparison."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get unique build times per method
        build_times = self.df.groupby('method')['build_time_sec'].first()
        
        colors = {'ZGQ': '#2E86AB', 'HNSW': '#A23B72', 'FAISS-IVF-PQ': '#F18F01'}
        method_colors = [colors.get(m, '#888888') for m in build_times.index]
        
        bars = ax.bar(range(len(build_times)), build_times.values, 
                     color=method_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, build_times.values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.2f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Build Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_title(f'Index Build Time Comparison\n{self.dataset_name}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(range(len(build_times)))
        ax.set_xticklabels(build_times.index, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        if save:
            filepath = self.output_dir / 'build_time_comparison.png'
            plt.savefig(filepath)
            print(f"Saved: {filepath}")
        
        plt.tight_layout()
        return fig
    
    def plot_memory_comparison(self, save: bool = True):
        """Plot memory usage comparison."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get unique index sizes per method
        index_sizes = self.df.groupby('method')['index_size_mb'].first()
        
        colors = {'ZGQ': '#2E86AB', 'HNSW': '#A23B72', 'FAISS-IVF-PQ': '#F18F01'}
        method_colors = [colors.get(m, '#888888') for m in index_sizes.index]
        
        bars = ax.bar(range(len(index_sizes)), index_sizes.values,
                     color=method_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, index_sizes.values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{value:.2f} MB', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Index Size (MB)', fontsize=14, fontweight='bold')
        ax.set_title(f'Memory Usage Comparison\n{self.dataset_name}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(range(len(index_sizes)))
        ax.set_xticklabels(index_sizes.index, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        if save:
            filepath = self.output_dir / 'memory_comparison.png'
            plt.savefig(filepath)
            print(f"Saved: {filepath}")
        
        plt.tight_layout()
        return fig
    
    def plot_latency_distribution(self, save: bool = True):
        """Plot latency distribution (p50, p95, p99)."""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Get best configuration for each method at ~0.7 recall
        target_recall = 0.7
        data = []
        
        for method in self.df['method'].unique():
            method_df = self.df[self.df['method'] == method]
            method_df = method_df[method_df['recall_at_k'] > 0.01]
            
            if len(method_df) == 0:
                continue
            
            # Find closest to target recall
            method_df['recall_diff'] = abs(method_df['recall_at_k'] - target_recall)
            best = method_df.loc[method_df['recall_diff'].idxmin()]
            
            data.append({
                'method': method,
                'p50': best['median_latency_ms'],
                'p95': best['p95_latency_ms'],
                'p99': best['p99_latency_ms'],
                'recall': best['recall_at_k']
            })
        
        if not data:
            print("No valid latency data to plot")
            return None
        
        plot_df = pd.DataFrame(data)
        
        x = np.arange(len(plot_df))
        width = 0.25
        
        ax.bar(x - width, plot_df['p50'], width, label='p50 (Median)', 
              color='#2E86AB', alpha=0.8)
        ax.bar(x, plot_df['p95'], width, label='p95',
              color='#A23B72', alpha=0.8)
        ax.bar(x + width, plot_df['p99'], width, label='p99',
              color='#F18F01', alpha=0.8)
        
        ax.set_xlabel('Method', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontsize=14, fontweight='bold')
        ax.set_title(f'Latency Distribution (at ~{target_recall:.0%} Recall)\n{self.dataset_name}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['method']}\n(R={row['recall']:.2f})" 
                           for _, row in plot_df.iterrows()], fontsize=11)
        ax.legend(fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_yscale('log')
        
        if save:
            filepath = self.output_dir / 'latency_distribution.png'
            plt.savefig(filepath)
            print(f"Saved: {filepath}")
        
        plt.tight_layout()
        return fig
    
    def plot_pareto_frontier(self, save: bool = True):
        """
        Plot Pareto frontier showing optimal recall-latency trade-offs.
        
        This highlights which methods dominate in different regions.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = self.df['method'].unique()
        colors = {'ZGQ': '#2E86AB', 'HNSW': '#A23B72', 'FAISS-IVF-PQ': '#F18F01'}
        markers = {'ZGQ': 'o', 'HNSW': 's', 'FAISS-IVF-PQ': '^'}
        
        all_points = []
        
        for method in methods:
            method_df = self.df[self.df['method'] == method].sort_values('mean_latency_ms')
            method_df = method_df[method_df['recall_at_k'] > 0.01]
            
            if len(method_df) == 0:
                continue
            
            # Plot all points
            ax.scatter(
                method_df['mean_latency_ms'],
                method_df['recall_at_k'],
                marker=markers.get(method, 'o'),
                s=150,
                label=method,
                color=colors.get(method, None),
                alpha=0.7,
                edgecolors='black',
                linewidths=1.5
            )
            
            for _, row in method_df.iterrows():
                all_points.append({
                    'latency': row['mean_latency_ms'],
                    'recall': row['recall_at_k'],
                    'method': method
                })
        
        # Compute and plot Pareto frontier
        pareto_df = pd.DataFrame(all_points)
        pareto_df = pareto_df.sort_values('latency')
        
        pareto_points = []
        max_recall = 0
        
        for _, row in pareto_df.iterrows():
            if row['recall'] > max_recall:
                pareto_points.append(row)
                max_recall = row['recall']
        
        if pareto_points:
            pareto_frontier = pd.DataFrame(pareto_points)
            ax.plot(
                pareto_frontier['latency'],
                pareto_frontier['recall'],
                'k--',
                linewidth=2,
                label='Pareto Frontier',
                alpha=0.5
            )
        
        ax.set_xlabel('Mean Query Latency (ms)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Recall@10', fontsize=14, fontweight='bold')
        ax.set_title(f'Pareto Frontier Analysis\n{self.dataset_name}',
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1.05)
        
        if save:
            filepath = self.output_dir / 'pareto_frontier.png'
            plt.savefig(filepath)
            print(f"Saved: {filepath}")
        
        plt.tight_layout()
        return fig
    
    def generate_all_figures(self):
        """Generate all visualization figures."""
        print(f"\n{'='*60}")
        print("Generating All Visualization Figures")
        print(f"{'='*60}\n")
        
        figures = []
        
        try:
            print("1. Recall-Latency Curve...")
            fig = self.plot_recall_latency_curve()
            figures.append(('recall_latency', fig))
        except Exception as e:
            print(f"  Error: {e}")
        
        try:
            print("\n2. Throughput Comparison...")
            fig = self.plot_throughput_comparison()
            if fig:
                figures.append(('throughput', fig))
        except Exception as e:
            print(f"  Error: {e}")
        
        try:
            print("\n3. Build Time Comparison...")
            fig = self.plot_build_time_comparison()
            figures.append(('build_time', fig))
        except Exception as e:
            print(f"  Error: {e}")
        
        try:
            print("\n4. Memory Usage Comparison...")
            fig = self.plot_memory_comparison()
            figures.append(('memory', fig))
        except Exception as e:
            print(f"  Error: {e}")
        
        try:
            print("\n5. Latency Distribution...")
            fig = self.plot_latency_distribution()
            if fig:
                figures.append(('latency_dist', fig))
        except Exception as e:
            print(f"  Error: {e}")
        
        try:
            print("\n6. Pareto Frontier...")
            fig = self.plot_pareto_frontier()
            figures.append(('pareto', fig))
        except Exception as e:
            print(f"  Error: {e}")
        
        print(f"\n{'='*60}")
        print(f"Generated {len(figures)} figures in '{self.output_dir}/'")
        print(f"{'='*60}\n")
        
        return figures
    
    def generate_summary_report(self, save: bool = True):
        """Generate text summary of benchmark results."""
        report = []
        report.append("="*70)
        report.append("BENCHMARK SUMMARY REPORT")
        report.append("="*70)
        report.append(f"\nDataset: {self.dataset_name}")
        report.append(f"Vectors: {self.n_vectors:,}, Dimension: {self.dimension}")
        report.append(f"\n{'-'*70}\n")
        
        for method in self.df['method'].unique():
            method_df = self.df[self.df['method'] == method]
            method_df = method_df[method_df['recall_at_k'] > 0.01]
            
            if len(method_df) == 0:
                report.append(f"\n{method}: No valid results (all recall = 0)")
                continue
            
            report.append(f"\n{method}:")
            report.append(f"  Build Time: {method_df['build_time_sec'].iloc[0]:.2f}s")
            report.append(f"  Index Size: {method_df['index_size_mb'].iloc[0]:.2f} MB")
            
            best_recall = method_df.loc[method_df['recall_at_k'].idxmax()]
            report.append(f"\n  Best Recall Configuration:")
            report.append(f"    Recall@10: {best_recall['recall_at_k']:.4f}")
            report.append(f"    Latency: {best_recall['mean_latency_ms']:.3f}ms")
            report.append(f"    Throughput: {best_recall['throughput_qps']:.1f} QPS")
            report.append(f"    Parameters: {best_recall['parameters']}")
            
            best_latency = method_df.loc[method_df['mean_latency_ms'].idxmin()]
            report.append(f"\n  Fastest Configuration:")
            report.append(f"    Latency: {best_latency['mean_latency_ms']:.3f}ms")
            report.append(f"    Recall@10: {best_latency['recall_at_k']:.4f}")
            report.append(f"    Throughput: {best_latency['throughput_qps']:.1f} QPS")
            
            report.append(f"\n{'-'*70}")
        
        report_text = "\n".join(report)
        
        if save:
            filepath = self.output_dir / 'benchmark_summary.txt'
            with open(filepath, 'w') as f:
                f.write(report_text)
            print(f"\nSummary report saved to: {filepath}")
        
        return report_text


def main():
    """Run visualization on benchmark results."""
    import sys
    
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = 'benchmark_results.json'
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = 'figures'
    
    print(f"Loading results from: {results_file}")
    
    visualizer = BenchmarkVisualizer(results_file, output_dir)
    
    # Generate all figures
    visualizer.generate_all_figures()
    
    # Generate summary report
    report = visualizer.generate_summary_report()
    print("\n" + report)
    
    print("\nVisualization complete! Check the figures directory.")


if __name__ == '__main__':
    main()
