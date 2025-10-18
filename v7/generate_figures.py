"""
Generate comparison figures for ZGQ optimization results.

This script creates comprehensive visualizations comparing:
1. Performance comparison (latency, throughput, recall)
2. Memory consumption
3. Optimization journey
4. Scaling projections
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

# Create output directory
output_dir = Path('figures_optimization')
output_dir.mkdir(exist_ok=True)


def plot_performance_comparison():
    """Figure 1: Performance comparison bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ZGQ Optimization Results - Performance Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    methods = ['HNSW', 'ZGQ\nUnified', 'ZGQ\nMulti-1', 'ZGQ\nMulti-3']
    latencies = [0.071, 0.053, 0.090, 0.167]  # ms
    recalls = [64.6, 64.3, 8.1, 19.8]  # %
    throughputs = [13986, 18933, 11062, 6006]  # QPS
    build_times = [0.275, 0.476, 0.766, 0.766]  # seconds
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # Latency
    ax1 = axes[0, 0]
    bars1 = ax1.bar(methods, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Latency (ms)', fontweight='bold')
    ax1.set_title('Query Latency (Lower is Better)', fontweight='bold')
    ax1.axhline(y=latencies[0], color='gray', linestyle='--', alpha=0.5, label='HNSW baseline')
    
    # Add value labels and speedup
    for i, (bar, lat) in enumerate(zip(bars1, latencies)):
        height = bar.get_height()
        speedup = latencies[0] / lat
        label = f'{lat:.3f}ms\n({speedup:.2f}x)'
        color = 'green' if speedup > 1.0 else 'red'
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom', fontweight='bold',
                color=color, fontsize=10)
    
    # Throughput
    ax2 = axes[0, 1]
    bars2 = ax2.bar(methods, np.array(throughputs)/1000, color=colors, alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Throughput (K QPS)', fontweight='bold')
    ax2.set_title('Query Throughput (Higher is Better)', fontweight='bold')
    ax2.axhline(y=throughputs[0]/1000, color='gray', linestyle='--', alpha=0.5)
    
    for i, (bar, tput) in enumerate(zip(bars2, throughputs)):
        height = bar.get_height()
        speedup = tput / throughputs[0]
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{tput/1000:.1f}K\n({speedup:.2f}x)',
                ha='center', va='bottom', fontweight='bold',
                color='green' if speedup > 1.0 else 'red', fontsize=10)
    
    # Recall
    ax3 = axes[1, 0]
    bars3 = ax3.bar(methods, recalls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Recall@10 (%)', fontweight='bold')
    ax3.set_title('Search Recall (Higher is Better)', fontweight='bold')
    ax3.axhline(y=recalls[0], color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylim([0, 100])
    
    for i, (bar, rec) in enumerate(zip(bars3, recalls)):
        height = bar.get_height()
        diff = rec - recalls[0]
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{rec:.1f}%\n({diff:+.1f}%)',
                ha='center', va='bottom', fontweight='bold',
                color='green' if diff >= -1 else 'red', fontsize=10)
    
    # Build time
    ax4 = axes[1, 1]
    bars4 = ax4.bar(methods, build_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Build Time (seconds)', fontweight='bold')
    ax4.set_title('Index Build Time (Lower is Better)', fontweight='bold')
    ax4.axhline(y=build_times[0], color='gray', linestyle='--', alpha=0.5)
    
    for i, (bar, bt) in enumerate(zip(bars4, build_times)):
        height = bar.get_height()
        ratio = bt / build_times[0]
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{bt:.2f}s\n({ratio:.2f}x)',
                ha='center', va='bottom', fontweight='bold',
                color='red' if ratio > 1.5 else 'black', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 01_performance_comparison.png")
    plt.close()


def plot_memory_comparison():
    """Figure 2: Memory consumption comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Memory Consumption Comparison', fontsize=16, fontweight='bold')
    
    methods = ['HNSW', 'ZGQ\nUnified', 'ZGQ Multi\n(no PQ)', 'ZGQ Multi\n(PQ)']
    memory_10k = [10.9, 17.9, 328.2, 263.4]  # MB
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    # 10K vectors - bar chart
    ax1 = axes[0]
    bars = ax1.bar(methods, memory_10k, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Memory (MB)', fontweight='bold')
    ax1.set_title('Memory Usage (10K vectors, 128 dims)', fontweight='bold')
    ax1.axhline(y=memory_10k[0], color='gray', linestyle='--', alpha=0.5, label='HNSW baseline')
    
    for i, (bar, mem) in enumerate(zip(bars, memory_10k)):
        height = bar.get_height()
        ratio = mem / memory_10k[0]
        color_text = 'green' if ratio < 2.0 else 'red'
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f} MB\n({ratio:.1f}x)',
                ha='center', va='bottom', fontweight='bold',
                color=color_text, fontsize=10)
    
    # Scaling - line chart
    ax2 = axes[1]
    
    n_vectors = np.array([10000, 100000, 1000000, 10000000])
    dim = 128
    M = 16
    
    # Calculate theoretical memory
    def calc_memory(n, method_type):
        vectors_mb = (n * dim * 4) / (1024 * 1024)
        graph_mb = (n * M * 2 * 4) / (1024 * 1024)
        centroids_mb = (100 * dim * 4) / (1024 * 1024)
        metadata_mb = (n * 4) / (1024 * 1024)
        
        if method_type == 'hnsw':
            return vectors_mb + graph_mb
        elif method_type == 'unified':
            return vectors_mb + graph_mb + centroids_mb + metadata_mb
        elif method_type == 'multi_pq':
            pq_mb = (n * 8) / (1024 * 1024)
            codebooks_mb = (8 * 256 * (dim // 8) * 4) / (1024 * 1024)
            return pq_mb + codebooks_mb + graph_mb + centroids_mb + metadata_mb
    
    mem_hnsw = [calc_memory(n, 'hnsw') for n in n_vectors]
    mem_unified = [calc_memory(n, 'unified') for n in n_vectors]
    mem_pq = [calc_memory(n, 'multi_pq') for n in n_vectors]
    
    ax2.plot(n_vectors, mem_hnsw, 'o-', label='HNSW', color='#3498db', linewidth=2, markersize=8)
    ax2.plot(n_vectors, mem_unified, 's-', label='ZGQ Unified', color='#2ecc71', linewidth=2, markersize=8)
    ax2.plot(n_vectors, mem_pq, '^-', label='ZGQ Multi (PQ)', color='#f39c12', linewidth=2, markersize=8)
    
    ax2.set_xlabel('Number of Vectors', fontweight='bold')
    ax2.set_ylabel('Memory (MB)', fontweight='bold')
    ax2.set_title('Memory Scaling (Theoretical)', fontweight='bold')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(loc='upper left', frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    for i, n in enumerate(n_vectors):
        overhead = ((mem_unified[i] - mem_hnsw[i]) / mem_hnsw[i]) * 100
        if i in [0, 3]:  # First and last
            ax2.annotate(f'+{overhead:.1f}%',
                        xy=(n, mem_unified[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=9, color='green',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_memory_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 02_memory_comparison.png")
    plt.close()


def plot_optimization_journey():
    """Figure 3: Optimization journey timeline."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    versions = ['ZGQ v6\nOriginal', 'ZGQ v6\nBaseline', 'Opt v1\nThreading',
                'Opt v2\nLightweight', 'Opt v3\nFast Zones', 'ZGQ\nUnified']
    latencies = [2.906, 1.069, 5.765, 1.121, 1.001, 0.053]
    recalls = [98.9, 70.9, 71.0, 71.1, 71.1, 64.3]
    statuses = ['❌ Too Slow', '⚠️ Slow', '❌ Regression', '⚠️ Still Slow', '⚠️ Slow', '✅ Winner!']
    colors = ['#e74c3c', '#f39c12', '#e74c3c', '#f39c12', '#f39c12', '#2ecc71']
    
    hnsw_latency = 0.071
    
    # Plot bars
    bars = ax.bar(range(len(versions)), latencies, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    
    # HNSW baseline
    ax.axhline(y=hnsw_latency, color='#3498db', linestyle='--', linewidth=2, 
               label='HNSW Target (0.071ms)', alpha=0.7)
    
    ax.set_ylabel('Query Latency (ms)', fontweight='bold', fontsize=12)
    ax.set_xlabel('Optimization Version', fontweight='bold', fontsize=12)
    ax.set_title('ZGQ Optimization Journey: From 17x Slower to 35% Faster than HNSW',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(versions, fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    
    # Add value labels and status
    for i, (bar, lat, status, recall) in enumerate(zip(bars, latencies, statuses, recalls)):
        height = bar.get_height()
        speedup = hnsw_latency / lat
        
        # Main label
        label = f'{lat:.3f}ms\n'
        if speedup > 1.0:
            label += f'{speedup:.2f}x faster ✓'
            label_color = 'darkgreen'
        else:
            label += f'{1/speedup:.2f}x slower'
            label_color = 'darkred'
        
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.3,
               label, ha='center', va='bottom',
               fontweight='bold', fontsize=10, color=label_color,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=label_color, linewidth=1.5, alpha=0.9))
        
        # Status label
        ax.text(bar.get_x() + bar.get_width()/2., height * 0.3,
               status, ha='center', va='center',
               fontweight='bold', fontsize=9, color='white')
    
    # Add arrows showing progression
    for i in range(len(versions) - 1):
        if i == 4:  # Special arrow for breakthrough
            ax.annotate('', xy=(i+1, latencies[i+1]*2), xytext=(i, latencies[i]*0.8),
                       arrowprops=dict(arrowstyle='->', lw=3, color='green', alpha=0.7))
            ax.text((i+i+1)/2, latencies[i]*0.5, 'BREAKTHROUGH!',
                   ha='center', fontsize=11, fontweight='bold', color='green',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_optimization_journey.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 03_optimization_journey.png")
    plt.close()


def plot_speed_vs_memory_scatter():
    """Figure 4: Speed vs Memory tradeoff scatter plot."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = ['HNSW', 'ZGQ Unified', 'ZGQ Multi-1', 'ZGQ Multi-3', 'ZGQ Multi (PQ)']
    memory = [10.9, 17.9, 328.2, 328.2, 263.4]
    latency = [0.071, 0.053, 0.090, 0.167, 0.976]
    recalls = [64.6, 64.3, 8.1, 19.8, 67.3]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    sizes = [r * 30 for r in recalls]  # Size by recall
    
    # Plot scatter
    for i, (m, l, s, c, method, rec) in enumerate(zip(memory, latency, sizes, colors, methods, recalls)):
        ax.scatter(m, l, s=s, color=c, alpha=0.7, edgecolor='black', linewidth=2, label=method)
        
        # Add method label
        offset_x = 20 if i == 1 else -20
        offset_y = 0.002 if i == 1 else -0.002
        ax.annotate(method + f'\n({rec:.1f}% recall)',
                   xy=(m, l), xytext=(offset_x, offset_y),
                   textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor=c, alpha=0.3),
                   ha='left' if i == 1 else 'right')
    
    # Mark the winner
    winner_idx = 1  # ZGQ Unified
    ax.scatter(memory[winner_idx], latency[winner_idx], s=1500, 
              facecolors='none', edgecolors='gold', linewidth=4, 
              marker='*', zorder=10)
    ax.annotate('🏆 WINNER!',
               xy=(memory[winner_idx], latency[winner_idx]),
               xytext=(0, -30), textcoords='offset points',
               fontsize=14, fontweight='bold', color='gold',
               ha='center',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen', 
                        edgecolor='gold', linewidth=3, alpha=0.9))
    
    ax.set_xlabel('Memory (MB) - Log Scale', fontweight='bold', fontsize=12)
    ax.set_ylabel('Query Latency (ms) - Log Scale', fontweight='bold', fontsize=12)
    ax.set_title('Speed vs Memory Tradeoff (bubble size = recall)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add ideal region
    ax.axvline(x=50, color='green', linestyle=':', linewidth=2, alpha=0.3, label='Low memory')
    ax.axhline(y=0.1, color='green', linestyle=':', linewidth=2, alpha=0.3, label='Fast query')
    ax.fill_between([0, 50], [0, 0], [0.1, 0.1], color='green', alpha=0.05)
    ax.text(15, 0.055, 'IDEAL ZONE', fontsize=12, fontweight='bold', 
           color='green', alpha=0.5, rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_speed_vs_memory.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 04_speed_vs_memory.png")
    plt.close()


def plot_nprobe_tuning():
    """Figure 5: n_probe parameter tuning results."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('ZGQ Multi-Graph: n_probe Parameter Tuning', fontsize=16, fontweight='bold')
    
    n_probes = [3, 5, 8, 10, 15, 20, 30, 40]
    latencies = [0.288, 0.348, 0.487, 0.565, 0.823, 0.976, 1.330, 1.696]
    recalls = [19.8, 26.9, 37.6, 43.5, 56.4, 67.3, 81.4, 89.8]
    
    hnsw_latency = 0.056
    hnsw_recall = 64.3
    
    # Latency vs n_probe
    ax1 = axes[0]
    ax1.plot(n_probes, latencies, 'o-', color='#e74c3c', linewidth=2, markersize=8, label='ZGQ Multi')
    ax1.axhline(y=hnsw_latency, color='#3498db', linestyle='--', linewidth=2, label='HNSW', alpha=0.7)
    ax1.set_xlabel('n_probe (number of zones searched)', fontweight='bold')
    ax1.set_ylabel('Query Latency (ms)', fontweight='bold')
    ax1.set_title('Latency vs n_probe', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, shadow=True)
    
    # Mark best speed point
    best_speed_idx = 0
    ax1.scatter(n_probes[best_speed_idx], latencies[best_speed_idx], 
               s=200, facecolors='none', edgecolors='green', linewidth=3, zorder=10)
    ax1.annotate('Best Speed\n(but low recall)',
                xy=(n_probes[best_speed_idx], latencies[best_speed_idx]),
                xytext=(10, 20), textcoords='offset points',
                fontsize=9, color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Recall vs n_probe
    ax2 = axes[1]
    ax2.plot(n_probes, recalls, 's-', color='#2ecc71', linewidth=2, markersize=8, label='ZGQ Multi')
    ax2.axhline(y=hnsw_recall, color='#3498db', linestyle='--', linewidth=2, label='HNSW', alpha=0.7)
    ax2.set_xlabel('n_probe (number of zones searched)', fontweight='bold')
    ax2.set_ylabel('Recall@10 (%)', fontweight='bold')
    ax2.set_title('Recall vs n_probe', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, shadow=True)
    ax2.set_ylim([0, 100])
    
    # Mark HNSW-match point
    hnsw_match_idx = np.argmin(np.abs(np.array(recalls) - hnsw_recall))
    ax2.scatter(n_probes[hnsw_match_idx], recalls[hnsw_match_idx], 
               s=200, facecolors='none', edgecolors='orange', linewidth=3, zorder=10)
    ax2.annotate('HNSW-level recall\n(but slow)',
                xy=(n_probes[hnsw_match_idx], recalls[hnsw_match_idx]),
                xytext=(10, -30), textcoords='offset points',
                fontsize=9, color='orange',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / '05_nprobe_tuning.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 05_nprobe_tuning.png")
    plt.close()


def plot_summary_table():
    """Figure 6: Summary comparison table as image."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Data
    headers = ['Method', 'Latency\n(ms)', 'vs HNSW', 'Recall@10\n(%)', 'Memory\n(MB)', 
               'Build Time\n(s)', 'Status']
    
    data = [
        ['HNSW\n(Baseline)', '0.071', '1.00x', '64.6', '10.9', '0.28', '✓ Standard'],
        ['ZGQ Unified\n(NEW)', '0.053', '1.35x', '64.3', '17.9', '0.48', '🏆 WINNER'],
        ['ZGQ Multi\n(n_probe=1)', '0.090', '0.79x', '8.1', '328', '0.77', '❌ Poor recall'],
        ['ZGQ Multi\n(n_probe=3)', '0.167', '0.43x', '19.8', '328', '0.77', '❌ Slow'],
        ['ZGQ Multi\n(n_probe=20, PQ)', '0.976', '0.07x', '67.3', '263', '0.77', '❌ Very slow'],
    ]
    
    # Color rows
    cell_colors = []
    for i, row in enumerate(data):
        if i == 1:  # ZGQ Unified
            cell_colors.append(['#2ecc71'] * len(row))
        elif i == 0:  # HNSW
            cell_colors.append(['#3498db'] * len(row))
        else:
            cell_colors.append(['#e74c3c'] * len(row))
    
    table = ax.table(cellText=data, colLabels=headers,
                    cellLoc='center', loc='center',
                    cellColours=cell_colors,
                    colColours=['#34495e'] * len(headers))
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)
    
    # Style cells
    for i in range(1, len(data) + 1):
        for j in range(len(headers)):
            table[(i, j)].set_text_props(weight='bold', fontsize=11)
            if i == 2:  # Winner row
                table[(i, j)].set_text_props(color='white')
            else:
                table[(i, j)].set_text_props(color='white' if i == 1 else 'white')
    
    plt.title('Comprehensive Performance Comparison Summary',
             fontsize=18, fontweight='bold', pad=30)
    
    # Add footer note
    footer_text = (
        "Key Findings:\n"
        "• ZGQ Unified beats HNSW: 35% faster with same recall\n"
        "• Memory overhead: Only +64% for small data, <1% at scale\n"
        "• Multi-graph approach has high overhead and requires many zones for good recall"
    )
    plt.figtext(0.5, 0.02, footer_text, ha='center', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.savefig(output_dir / '06_summary_table.png', dpi=300, bbox_inches='tight')
    print("✓ Generated: 06_summary_table.png")
    plt.close()


def main():
    """Generate all figures."""
    print("=" * 70)
    print("GENERATING ZGQ OPTIMIZATION FIGURES")
    print("=" * 70)
    print()
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    print("Generating figures...")
    print()
    
    try:
        plot_performance_comparison()
        plot_memory_comparison()
        plot_optimization_journey()
        plot_speed_vs_memory_scatter()
        plot_nprobe_tuning()
        plot_summary_table()
        
        print()
        print("=" * 70)
        print("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print(f"Location: {output_dir.absolute()}")
        print()
        print("Generated figures:")
        print("  1. 01_performance_comparison.png - Bar charts of latency, throughput, recall, build time")
        print("  2. 02_memory_comparison.png - Memory usage comparison and scaling")
        print("  3. 03_optimization_journey.png - Timeline showing optimization progress")
        print("  4. 04_speed_vs_memory.png - Scatter plot of speed vs memory tradeoff")
        print("  5. 05_nprobe_tuning.png - n_probe parameter tuning results")
        print("  6. 06_summary_table.png - Comprehensive comparison table")
        print()
        
    except Exception as e:
        print(f"❌ Error generating figures: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
