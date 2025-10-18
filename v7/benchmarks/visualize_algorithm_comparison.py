"""
Generate comparison figures for ZGQ vs HNSW vs IVF vs IVF+PQ
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Load results
results_file = Path(__file__).parent / 'algorithm_comparison_results.json'
with open(results_file, 'r') as f:
    results = json.load(f)

# Create output directory
output_dir = Path('figures_algorithm_comparison')
output_dir.mkdir(exist_ok=True)

# Extract data
algorithms = [r['name'] for r in results]
latencies = [r['avg_latency_per_query_ms'] for r in results]
throughputs = [r['throughput_qps'] for r in results]
recalls = [r['recall_at_10'] for r in results]
memories = [r['memory_mb'] for r in results]
build_times = [r['build_time'] for r in results]

colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']

# Figure 1: Multi-metric comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comprehensive ANN Algorithm Comparison', fontsize=18, fontweight='bold')

# 1. Latency
ax = axes[0, 0]
bars = ax.bar(algorithms, latencies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Query Latency (ms)', fontweight='bold')
ax.set_title('Query Latency (Lower is Better)', fontweight='bold')
ax.set_yscale('log')
for bar, lat in zip(bars, latencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{lat:.4f}ms', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 2. Throughput
ax = axes[0, 1]
bars = ax.bar(algorithms, np.array(throughputs)/1000, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Throughput (K QPS)', fontweight='bold')
ax.set_title('Query Throughput (Higher is Better)', fontweight='bold')
for bar, tput in zip(bars, throughputs):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{tput/1000:.1f}K', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 3. Recall
ax = axes[0, 2]
bars = ax.bar(algorithms, recalls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Recall@10 (%)', fontweight='bold')
ax.set_title('Search Recall (Higher is Better)', fontweight='bold')
ax.set_ylim([0, 100])
for bar, rec in zip(bars, recalls):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{rec:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 4. Memory
ax = axes[1, 0]
bars = ax.bar(algorithms, memories, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Memory (MB)', fontweight='bold')
ax.set_title('Memory Usage (Lower is Better)', fontweight='bold')
for bar, mem in zip(bars, memories):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{mem:.1f}MB', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 5. Build Time
ax = axes[1, 1]
bars = ax.bar(algorithms, build_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Build Time (seconds)', fontweight='bold')
ax.set_title('Index Build Time (Lower is Better)', fontweight='bold')
for bar, bt in zip(bars, build_times):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{bt:.2f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)

# 6. Speedup vs HNSW
ax = axes[1, 2]
speedups = [latencies[0] / lat for lat in latencies]
bars = ax.bar(algorithms, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Speedup', fontweight='bold')
ax.set_title('Speed vs HNSW (Higher is Better)', fontweight='bold')
ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.5)
for bar, sp in zip(bars, speedups):
    height = bar.get_height()
    color_text = 'green' if sp >= 1.0 else 'red'
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{sp:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=9, color=color_text)

plt.tight_layout()
plt.savefig(output_dir / '01_algorithm_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 01_algorithm_comparison.png")
plt.close()

# Figure 2: Speed vs Recall tradeoff
fig, ax = plt.subplots(figsize=(12, 8))

sizes = [mem * 20 for mem in memories]  # Size by memory

for i, (alg, lat, rec, mem, s, c) in enumerate(zip(algorithms, latencies, recalls, memories, sizes, colors)):
    ax.scatter(rec, 1/lat, s=s, color=c, alpha=0.7, edgecolor='black', linewidth=2, label=alg)
    
    offset_x = 2 if i != 1 else -5
    offset_y = 5000 if i != 1 else -5000
    ax.annotate(f'{alg}\n({mem:.1f} MB)',
               xy=(rec, 1/lat), xytext=(offset_x, offset_y),
               textcoords='offset points',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=c, alpha=0.3),
               ha='center')

ax.set_xlabel('Recall@10 (%)', fontweight='bold', fontsize=14)
ax.set_ylabel('Throughput (QPS)', fontweight='bold', fontsize=14)
ax.set_title('Speed vs Recall Tradeoff\n(bubble size = memory usage)',
            fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

# Add ideal region
ax.axvline(x=50, color='green', linestyle=':', linewidth=2, alpha=0.3)
ax.axhline(y=10000, color='green', linestyle=':', linewidth=2, alpha=0.3)
ax.fill_between([50, 100], [10000, 10000], [1e6, 1e6], color='green', alpha=0.05)
ax.text(65, 50000, 'IDEAL ZONE\n(High Recall + Fast)', 
       fontsize=12, fontweight='bold', color='green', alpha=0.7)

plt.tight_layout()
plt.savefig(output_dir / '02_speed_vs_recall.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 02_speed_vs_recall.png")
plt.close()

# Figure 3: Radar chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

categories = ['Speed\n(QPS/1000)', 'Recall\n(%)', 'Memory Eff.\n(1/MB)', 
              'Build Speed\n(1/s)', 'Overall']

# Normalize metrics (0-1, higher is better)
def normalize_metrics(results):
    metrics = []
    for r in results:
        speed_norm = (r['throughput_qps'] / 1000) / max(res['throughput_qps'] / 1000 for res in results)
        recall_norm = r['recall_at_10'] / max(res['recall_at_10'] for res in results)
        memory_norm = min(res['memory_mb'] for res in results) / r['memory_mb']
        build_norm = min(res['build_time'] for res in results) / r['build_time']
        overall = (speed_norm + recall_norm + memory_norm + build_norm) / 4
        metrics.append([speed_norm, recall_norm, memory_norm, build_norm, overall])
    return metrics

normalized = normalize_metrics(results)

angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for i, (alg, metrics, c) in enumerate(zip(algorithms, normalized, colors)):
    values = metrics + [metrics[0]]  # Complete the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=alg, color=c)
    ax.fill(angles, values, alpha=0.15, color=c)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True, shadow=True)
ax.set_title('Multi-Dimensional Performance Comparison', 
            fontsize=16, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig(output_dir / '03_radar_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 03_radar_comparison.png")
plt.close()

# Figure 4: Summary table
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

headers = ['Algorithm', 'Latency\n(ms)', 'Throughput\n(QPS)', 'Recall@10\n(%)', 
          'Memory\n(MB)', 'Build\n(s)', 'Speedup\nvs HNSW']

data = []
for i, r in enumerate(results):
    speedup = latencies[0] / r['avg_latency_per_query_ms']
    data.append([
        r['name'],
        f"{r['avg_latency_per_query_ms']:.4f}",
        f"{r['throughput_qps']:,.0f}",
        f"{r['recall_at_10']:.1f}",
        f"{r['memory_mb']:.1f}",
        f"{r['build_time']:.2f}",
        f"{speedup:.2f}x"
    ])

# Color rows
cell_colors = []
for i, row in enumerate(data):
    cell_colors.append([colors[i]] * len(row))

table = ax.table(cellText=data, colLabels=headers,
                cellLoc='center', loc='center',
                cellColours=cell_colors,
                colColours=['#34495e'] * len(headers))

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#34495e')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=12)

# Style cells
for i in range(1, len(data) + 1):
    for j in range(len(headers)):
        table[(i, j)].set_text_props(weight='bold', fontsize=11, color='white')

plt.title('Comprehensive Algorithm Comparison Summary',
         fontsize=18, fontweight='bold', pad=30)

footer_text = (
    "Key Findings:\n"
    "• HNSW: Best overall performance - fastest queries with good recall\n"
    "• IVF: Moderate speed, poor recall, good memory efficiency\n"
    "• IVF+PQ: Very slow queries, poor recall, but excellent compression (32x)\n"
    "• ZGQ Unified: Competitive speed with HNSW-level recall, lowest memory"
)
plt.figtext(0.5, 0.02, footer_text, ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(output_dir / '04_summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Generated: 04_summary_table.png")
plt.close()

print("\n" + "="*70)
print("✓ ALL COMPARISON FIGURES GENERATED!")
print("="*70)
print(f"\nLocation: {output_dir.absolute()}")
print("\nGenerated figures:")
print("  1. 01_algorithm_comparison.png - Multi-metric bar charts")
print("  2. 02_speed_vs_recall.png - Speed vs recall tradeoff scatter")
print("  3. 03_radar_comparison.png - Multi-dimensional radar chart")
print("  4. 04_summary_table.png - Comprehensive summary table")
print()
