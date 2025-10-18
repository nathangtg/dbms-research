"""
Visualization Module for ZGQ V6 Benchmarking
Generates publication-quality charts comparing ZGQ versions and baselines

Charts:
1. Recall-Latency Curves (Pareto frontier)
2. Memory Usage Comparison
3. Build Time Comparison
4. Throughput vs Recall
5. Algorithm Evolution (V1-V6)
6. Ablation Study (ZGQ components)

Reference: Standard ANNS visualization practices
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
from dataclasses import dataclass


# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['figure.dpi'] = 300


@dataclass
class AlgorithmResult:
    """Results for a single algorithm configuration."""
    name: str
    version: str  # e.g., "V1", "V2", "V6", "baseline"
    recall_at_10: float
    latency_ms: float
    memory_mb: float
    build_time_s: float
    qps: float
    config: Dict[str, Any]
    
    # Optional for detailed analysis
    recall_at_k: Optional[Dict[int, float]] = None
    p50_latency: Optional[float] = None
    p95_latency: Optional[float] = None
    p99_latency: Optional[float] = None


class ZGQVisualizer:
    """
    Comprehensive visualization suite for ZGQ benchmarking.
    
    Generates publication-quality figures comparing:
    - ZGQ versions (V1 through V6)
    - Baseline algorithms (HNSW, IVF, IVF+PQ)
    - Ablation studies (ZGQ with/without components)
    """
    
    # Color schemes
    COLORS = {
        'V1': '#8B4513',  # Brown - basic
        'V2': '#FF8C00',  # Dark orange - improved
        'V3': '#FFD700',  # Gold - better
        'V4': '#32CD32',  # Lime green - good
        'V5': '#4169E1',  # Royal blue - very good
        'V6': '#DC143C',  # Crimson - best (ZGQ)
        'HNSW': '#696969',  # Dim gray
        'IVF': '#A9A9A9',  # Dark gray
        'IVF+PQ': '#C0C0C0',  # Silver
        'ZGQ': '#DC143C',  # Crimson (alias for V6)
    }
    
    MARKERS = {
        'V1': 'v',  # Triangle down
        'V2': '^',  # Triangle up
        'V3': '<',  # Triangle left
        'V4': '>',  # Triangle right
        'V5': 's',  # Square
        'V6': '*',  # Star (ZGQ)
        'HNSW': 'o',  # Circle
        'IVF': 'D',  # Diamond
        'IVF+PQ': 'P',  # Plus
        'ZGQ': '*',  # Star
    }
    
    def __init__(self, output_dir: str = "./figures"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[AlgorithmResult] = []
    
    def add_result(self, result: AlgorithmResult):
        """Add a result to visualize."""
        self.results.append(result)
    
    def add_results_from_json(self, filepath: str):
        """Load results from benchmark JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for result_dict in data.get('results', []):
            algo_name = result_dict['algorithm']
            
            # Determine version
            if 'ZGQ' in algo_name:
                version = 'V6' if 'V6' not in algo_name else algo_name.split('V')[-1][0]
            elif 'HNSW' in algo_name:
                version = 'baseline'
            elif 'IVF' in algo_name:
                version = 'baseline'
            else:
                version = 'unknown'
            
            result = AlgorithmResult(
                name=algo_name,
                version=version,
                recall_at_10=result_dict.get('recall_at_k', {}).get(10, 0.0),
                latency_ms=result_dict.get('mean_latency_ms', 0.0),
                memory_mb=result_dict.get('memory_mb', 0.0),
                build_time_s=result_dict.get('build_time_s', 0.0),
                qps=result_dict.get('qps', 0.0),
                config=result_dict.get('config', {}),
                recall_at_k=result_dict.get('recall_at_k'),
                p50_latency=result_dict.get('median_latency_ms'),
                p95_latency=result_dict.get('p95_latency_ms'),
                p99_latency=result_dict.get('p99_latency_ms')
            )
            
            self.add_result(result)
    
    def plot_recall_latency_curve(
        self,
        k: int = 10,
        title: str = None,
        filename: str = "recall_latency.png",
        figsize: Tuple[float, float] = (8, 6)
    ):
        """
        Plot recall@k vs latency (Pareto frontier).
        
        This is the most important ANNS benchmark plot showing
        the quality-speed tradeoff.
        
        Args:
            k: k value for recall@k
            title: Plot title
            filename: Output filename
            figsize: Figure size in inches
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group by algorithm type
        groups = {}
        for result in self.results:
            key = result.name.split('_')[0]  # Get base name
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        # Plot each group
        for algo_name, group_results in groups.items():
            # Sort by latency
            group_results.sort(key=lambda r: r.latency_ms)
            
            recalls = [r.recall_at_10 for r in group_results]
            latencies = [r.latency_ms for r in group_results]
            
            # Determine color and marker
            color = self.COLORS.get(algo_name, '#000000')
            marker = self.MARKERS.get(algo_name, 'o')
            
            # Plot line and points
            ax.plot(recalls, latencies, 
                   color=color, linewidth=2, alpha=0.7,
                   label=algo_name)
            ax.scatter(recalls, latencies,
                      color=color, marker=marker, s=100, 
                      edgecolors='black', linewidth=1, zorder=10)
        
        ax.set_xlabel(f'Recall@{k}', fontweight='bold')
        ax.set_ylabel('Latency (ms)', fontweight='bold')
        ax.set_title(title or f'Recall@{k} vs Latency - ZGQ Evolution', 
                    fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9)
        
        # Add annotations for best points
        if len(self.results) > 0:
            # Highlight ZGQ V6
            zgq_results = [r for r in self.results if 'V6' in r.name or 'ZGQ' in r.name]
            if zgq_results:
                best_zgq = max(zgq_results, key=lambda r: r.recall_at_10)
                ax.annotate('ZGQ V6',
                           xy=(best_zgq.recall_at_10, best_zgq.latency_ms),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved recall-latency curve to {output_path}")
    
    def plot_memory_comparison(
        self,
        title: str = None,
        filename: str = "memory_comparison.png",
        figsize: Tuple[float, float] = (10, 6)
    ):
        """
        Plot memory usage comparison across algorithms.
        
        Args:
            title: Plot title
            filename: Output filename
            figsize: Figure size
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by memory
        sorted_results = sorted(self.results, key=lambda r: r.memory_mb)
        
        names = [r.name for r in sorted_results]
        memories = [r.memory_mb for r in sorted_results]
        colors = [self.COLORS.get(r.name.split('_')[0], '#888888') 
                 for r in sorted_results]
        
        bars = ax.barh(names, memories, color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, mem) in enumerate(zip(bars, memories)):
            ax.text(mem, i, f' {mem:.1f} MB', 
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Memory Usage (MB)', fontweight='bold')
        ax.set_title(title or 'Memory Usage Comparison', 
                    fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved memory comparison to {output_path}")
    
    def plot_build_time_comparison(
        self,
        title: str = None,
        filename: str = "build_time_comparison.png",
        figsize: Tuple[float, float] = (10, 6)
    ):
        """
        Plot build time comparison across algorithms.
        
        Args:
            title: Plot title
            filename: Output filename
            figsize: Figure size
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by build time
        sorted_results = sorted(self.results, key=lambda r: r.build_time_s)
        
        names = [r.name for r in sorted_results]
        build_times = [r.build_time_s for r in sorted_results]
        colors = [self.COLORS.get(r.name.split('_')[0], '#888888') 
                 for r in sorted_results]
        
        bars = ax.barh(names, build_times, color=colors, edgecolor='black', linewidth=1)
        
        # Add value labels
        for i, (bar, time) in enumerate(zip(bars, build_times)):
            ax.text(time, i, f' {time:.1f}s', 
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Build Time (seconds)', fontweight='bold')
        ax.set_title(title or 'Index Build Time Comparison', 
                    fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved build time comparison to {output_path}")
    
    def plot_throughput_vs_recall(
        self,
        k: int = 10,
        title: str = None,
        filename: str = "throughput_recall.png",
        figsize: Tuple[float, float] = (8, 6)
    ):
        """
        Plot throughput (QPS) vs recall@k.
        
        Args:
            k: k value for recall
            title: Plot title
            filename: Output filename
            figsize: Figure size
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group by algorithm
        groups = {}
        for result in self.results:
            key = result.name.split('_')[0]
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        # Plot each group
        for algo_name, group_results in groups.items():
            recalls = [r.recall_at_10 for r in group_results]
            qps = [r.qps for r in group_results]
            
            color = self.COLORS.get(algo_name, '#000000')
            marker = self.MARKERS.get(algo_name, 'o')
            
            ax.scatter(recalls, qps, 
                      color=color, marker=marker, s=150,
                      edgecolors='black', linewidth=1.5,
                      label=algo_name, alpha=0.7, zorder=10)
        
        ax.set_xlabel(f'Recall@{k}', fontweight='bold')
        ax.set_ylabel('Throughput (Queries/sec)', fontweight='bold')
        ax.set_title(title or f'Throughput vs Recall@{k}', 
                    fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', framealpha=0.9)
        
        # Log scale for QPS if large range
        if max([r.qps for r in self.results]) / min([r.qps for r in self.results]) > 100:
            ax.set_yscale('log')
        
        plt.tight_layout()
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved throughput vs recall to {output_path}")
    
    def plot_version_evolution(
        self,
        title: str = None,
        filename: str = "zgq_evolution.png",
        figsize: Tuple[float, float] = (12, 8)
    ):
        """
        Plot ZGQ algorithm evolution across versions (V1-V6).
        
        Shows improvements in recall, latency, memory across versions.
        
        Args:
            title: Plot title
            filename: Output filename
            figsize: Figure size
        """
        # Filter for ZGQ versions only
        version_results = {}
        for result in self.results:
            if result.version.startswith('V'):
                version_results[result.version] = result
        
        if not version_results:
            print("No version results found")
            return
        
        # Sort versions
        versions = sorted(version_results.keys(), key=lambda v: int(v[1:]))
        
        if len(versions) < 2:
            print("Need at least 2 versions to plot evolution")
            return
        
        # Create subplots
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Recall progression
        ax1 = fig.add_subplot(gs[0, 0])
        recalls = [version_results[v].recall_at_10 for v in versions]
        bars = ax1.bar(versions, recalls, color=[self.COLORS[v] for v in versions],
                      edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Recall@10', fontweight='bold')
        ax1.set_title('Recall Improvement', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add percentage labels
        for i, (bar, recall) in enumerate(zip(bars, recalls)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{recall:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Latency progression
        ax2 = fig.add_subplot(gs[0, 1])
        latencies = [version_results[v].latency_ms for v in versions]
        bars = ax2.bar(versions, latencies, color=[self.COLORS[v] for v in versions],
                      edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Latency (ms)', fontweight='bold')
        ax2.set_title('Latency Reduction', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, lat) in enumerate(zip(bars, latencies)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{lat:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Memory progression
        ax3 = fig.add_subplot(gs[1, 0])
        memories = [version_results[v].memory_mb for v in versions]
        bars = ax3.bar(versions, memories, color=[self.COLORS[v] for v in versions],
                      edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Memory (MB)', fontweight='bold')
        ax3.set_title('Memory Efficiency', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, mem) in enumerate(zip(bars, memories)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{mem:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Overall score (normalized)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Normalize metrics (higher is better)
        norm_recalls = np.array(recalls) / max(recalls)
        norm_latencies = min(latencies) / np.array(latencies)  # Invert (lower is better)
        norm_memories = min(memories) / np.array(memories)  # Invert
        
        overall_scores = (norm_recalls + norm_latencies + norm_memories) / 3
        
        bars = ax4.bar(versions, overall_scores, color=[self.COLORS[v] for v in versions],
                      edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Overall Score (Normalized)', fontweight='bold')
        ax4.set_title('Overall Performance', fontweight='bold')
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, score) in enumerate(zip(bars, overall_scores)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        fig.suptitle(title or 'ZGQ Algorithm Evolution (V1 → V6)', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved version evolution to {output_path}")
    
    def plot_comprehensive_comparison(
        self,
        title: str = None,
        filename: str = "comprehensive_comparison.png",
        figsize: Tuple[float, float] = (14, 10)
    ):
        """
        Create comprehensive comparison dashboard with multiple subplots.
        
        Args:
            title: Main title
            filename: Output filename
            figsize: Figure size
        """
        if not self.results:
            print("No results to plot")
            return
        
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
        
        # 1. Recall vs Latency
        ax1 = fig.add_subplot(gs[0, :])
        groups = {}
        for result in self.results:
            key = result.name.split('_')[0]
            if key not in groups:
                groups[key] = []
            groups[key].append(result)
        
        for algo_name, group_results in groups.items():
            group_results.sort(key=lambda r: r.latency_ms)
            recalls = [r.recall_at_10 for r in group_results]
            latencies = [r.latency_ms for r in group_results]
            color = self.COLORS.get(algo_name, '#000000')
            marker = self.MARKERS.get(algo_name, 'o')
            
            ax1.plot(recalls, latencies, color=color, linewidth=2, alpha=0.7, label=algo_name)
            ax1.scatter(recalls, latencies, color=color, marker=marker, s=100,
                       edgecolors='black', linewidth=1, zorder=10)
        
        ax1.set_xlabel('Recall@10', fontweight='bold')
        ax1.set_ylabel('Latency (ms)', fontweight='bold')
        ax1.set_title('Recall-Latency Tradeoff', fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='best', framealpha=0.9, ncol=2)
        
        # 2. Memory Usage
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_results = sorted(self.results, key=lambda r: r.memory_mb)[:8]  # Top 8
        names = [r.name for r in sorted_results]
        memories = [r.memory_mb for r in sorted_results]
        colors = [self.COLORS.get(r.name.split('_')[0], '#888888') for r in sorted_results]
        
        bars = ax2.barh(names, memories, color=colors, edgecolor='black', linewidth=1)
        for i, (bar, mem) in enumerate(zip(bars, memories)):
            ax2.text(mem, i, f' {mem:.1f}', va='center', fontsize=8)
        ax2.set_xlabel('Memory (MB)', fontweight='bold')
        ax2.set_title('Memory Usage', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Build Time
        ax3 = fig.add_subplot(gs[1, 1])
        sorted_results = sorted(self.results, key=lambda r: r.build_time_s)[:8]
        names = [r.name for r in sorted_results]
        times = [r.build_time_s for r in sorted_results]
        colors = [self.COLORS.get(r.name.split('_')[0], '#888888') for r in sorted_results]
        
        bars = ax3.barh(names, times, color=colors, edgecolor='black', linewidth=1)
        for i, (bar, time) in enumerate(zip(bars, times)):
            ax3.text(time, i, f' {time:.1f}s', va='center', fontsize=8)
        ax3.set_xlabel('Build Time (s)', fontweight='bold')
        ax3.set_title('Index Construction Time', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Throughput
        ax4 = fig.add_subplot(gs[2, :])
        sorted_results = sorted(self.results, key=lambda r: r.qps, reverse=True)[:10]
        names = [r.name for r in sorted_results]
        qps = [r.qps for r in sorted_results]
        colors = [self.COLORS.get(r.name.split('_')[0], '#888888') for r in sorted_results]
        
        bars = ax4.bar(names, qps, color=colors, edgecolor='black', linewidth=1)
        for bar, q in zip(bars, qps):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{int(q)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax4.set_ylabel('Throughput (QPS)', fontweight='bold')
        ax4.set_title('Query Throughput', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.suptitle(title or 'Comprehensive Algorithm Comparison', 
                    fontsize=15, fontweight='bold', y=0.995)
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comprehensive comparison to {output_path}")
    
    def generate_all_plots(self):
        """Generate all available plots."""
        print(f"\n{'='*80}")
        print(f"Generating all visualizations...")
        print(f"{'='*80}")
        
        self.plot_recall_latency_curve()
        self.plot_memory_comparison()
        self.plot_build_time_comparison()
        self.plot_throughput_vs_recall()
        self.plot_version_evolution()
        self.plot_comprehensive_comparison()
        
        print(f"\n✓ All visualizations saved to {self.output_dir}")


# Demonstration/Testing
if __name__ == "__main__":
    print("="*80)
    print("ZGQ Visualization Module - Demonstration")
    print("="*80)
    
    # Create visualizer
    viz = ZGQVisualizer(output_dir="./figures")
    
    # Add mock results simulating ZGQ evolution
    print("\nGenerating mock results for demonstration...")
    
    # V1: Basic implementation
    viz.add_result(AlgorithmResult(
        name="V1", version="V1",
        recall_at_10=0.45, latency_ms=15.0, memory_mb=120.0,
        build_time_s=45.0, qps=67, config={}
    ))
    
    # V2: With partitioning
    viz.add_result(AlgorithmResult(
        name="V2", version="V2",
        recall_at_10=0.58, latency_ms=12.0, memory_mb=100.0,
        build_time_s=38.0, qps=83, config={}
    ))
    
    # V3: With HNSW
    viz.add_result(AlgorithmResult(
        name="V3", version="V3",
        recall_at_10=0.72, latency_ms=8.5, memory_mb=95.0,
        build_time_s=42.0, qps=118, config={}
    ))
    
    # V4: With PQ
    viz.add_result(AlgorithmResult(
        name="V4", version="V4",
        recall_at_10=0.78, latency_ms=6.2, memory_mb=45.0,
        build_time_s=50.0, qps=161, config={}
    ))
    
    # V5: Optimizations
    viz.add_result(AlgorithmResult(
        name="V5", version="V5",
        recall_at_10=0.85, latency_ms=4.8, memory_mb=38.0,
        build_time_s=48.0, qps=208, config={}
    ))
    
    # V6: ZGQ complete (current)
    viz.add_result(AlgorithmResult(
        name="V6", version="V6",
        recall_at_10=0.92, latency_ms=2.4, memory_mb=11.4,
        build_time_s=12.8, qps=413, config={}
    ))
    
    # Baselines
    viz.add_result(AlgorithmResult(
        name="HNSW", version="baseline",
        recall_at_10=0.88, latency_ms=3.5, memory_mb=65.0,
        build_time_s=25.0, qps=286, config={}
    ))
    
    viz.add_result(AlgorithmResult(
        name="IVF", version="baseline",
        recall_at_10=0.75, latency_ms=5.0, memory_mb=52.0,
        build_time_s=8.0, qps=200, config={}
    ))
    
    viz.add_result(AlgorithmResult(
        name="IVF+PQ", version="baseline",
        recall_at_10=0.68, latency_ms=4.2, memory_mb=18.0,
        build_time_s=15.0, qps=238, config={}
    ))
    
    # Generate all plots
    print(f"\nGenerating visualizations...")
    viz.generate_all_plots()
    
    print(f"\n{'='*80}")
    print("✓ Demonstration complete!")
    print(f"Check the './figures' directory for generated plots")
    print("="*80)
