"""
Visualize Scaling Behavior: 10K vs 100K Comparison
===================================================

Shows how each algorithm's performance changes as dataset grows 10x.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 11

# Data from benchmarks
results_10k = {
    'HNSW': {
        'latency_ms': 0.0128,
        'throughput_qps': 78138,
        'recall': 54.7,
        'memory_mb': 6.1,
        'build_time_s': 0.850
    },
    'IVF': {
        'latency_ms': 0.8398,
        'throughput_qps': 1191,
        'recall': 37.6,
        'memory_mb': 4.9,
        'build_time_s': 0.052
    },
    'IVF+PQ': {
        'latency_ms': 6.1456,
        'throughput_qps': 163,
        'recall': 19.0,
        'memory_mb': 5.2,
        'build_time_s': 1.631
    },
    'ZGQ Unified': {
        'latency_ms': 0.0582,
        'throughput_qps': 17169,
        'recall': 55.1,
        'memory_mb': 4.9,
        'build_time_s': 0.901
    }
}

results_100k = {
    'HNSW': {
        'latency_ms': 0.0453,
        'throughput_qps': 22066,
        'recall': 17.7,
        'memory_mb': 61.0,
        'build_time_s': 8.422
    },
    'IVF': {
        'latency_ms': 7.5059,
        'throughput_qps': 133,
        'recall': 34.4,
        'memory_mb': 48.9,
        'build_time_s': 0.512
    },
    'IVF+PQ': {
        'latency_ms': 34.7259,
        'throughput_qps': 29,
        'recall': 12.7,
        'memory_mb': 50.5,
        'build_time_s': 15.448
    },
    'ZGQ Unified': {
        'latency_ms': 0.1397,
        'throughput_qps': 7160,
        'recall': 21.2,
        'memory_mb': 48.9,
        'build_time_s': 8.866
    }
}

algorithms = list(results_10k.keys())
colors = {'HNSW': '#2E86AB', 'IVF': '#A23B72', 'IVF+PQ': '#F18F01', 'ZGQ Unified': '#06A77D'}

# Create figure
fig = plt.figure(figsize=(18, 12))

# ============================================================================
# 1. Recall Scaling
# ============================================================================
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(algorithms))
width = 0.35

recall_10k = [results_10k[alg]['recall'] for alg in algorithms]
recall_100k = [results_100k[alg]['recall'] for alg in algorithms]

bars1 = ax1.bar(x - width/2, recall_10k, width, label='10K vectors', alpha=0.8)
bars2 = ax1.bar(x + width/2, recall_100k, width, label='100K vectors', alpha=0.8)

# Color bars
for i, bar in enumerate(bars1):
    bar.set_color(colors[algorithms[i]])
    bar.set_alpha(0.6)
for i, bar in enumerate(bars2):
    bar.set_color(colors[algorithms[i]])
    bar.set_alpha(1.0)

# Add value labels
for i, (v1, v2) in enumerate(zip(recall_10k, recall_100k)):
    ax1.text(i - width/2, v1 + 1, f'{v1:.1f}%', ha='center', va='bottom', fontsize=9)
    ax1.text(i + width/2, v2 + 1, f'{v2:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Show drop
    drop = v2 - v1
    drop_pct = (drop / v1) * 100 if v1 > 0 else 0
    ax1.annotate('', xy=(i + width/2, v2), xytext=(i - width/2, v1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.4, lw=1))
    ax1.text(i, max(v1, v2) + 3, f'{drop_pct:+.0f}%', ha='center', va='bottom', 
            fontsize=8, color='red', fontweight='bold')

ax1.set_ylabel('Recall@10 (%)', fontweight='bold')
ax1.set_xlabel('Algorithm', fontweight='bold')
ax1.set_title('🚨 RECALL DEGRADATION AT SCALE', fontsize=14, fontweight='bold', color='darkred')
ax1.set_xticks(x)
ax1.set_xticklabels(algorithms, rotation=15, ha='right')
ax1.legend(loc='upper right')
ax1.grid(axis='y', alpha=0.3)

# ============================================================================
# 2. Latency Scaling
# ============================================================================
ax2 = plt.subplot(2, 3, 2)

latency_10k = [results_10k[alg]['latency_ms'] for alg in algorithms]
latency_100k = [results_100k[alg]['latency_ms'] for alg in algorithms]

bars1 = ax2.bar(x - width/2, latency_10k, width, label='10K vectors', alpha=0.8)
bars2 = ax2.bar(x + width/2, latency_100k, width, label='100K vectors', alpha=0.8)

for i, bar in enumerate(bars1):
    bar.set_color(colors[algorithms[i]])
    bar.set_alpha(0.6)
for i, bar in enumerate(bars2):
    bar.set_color(colors[algorithms[i]])
    bar.set_alpha(1.0)

# Add value labels and slowdown factors
for i, (v1, v2) in enumerate(zip(latency_10k, latency_100k)):
    ax2.text(i - width/2, v1 + 0.5, f'{v1:.3f}', ha='center', va='bottom', fontsize=8)
    ax2.text(i + width/2, v2 + 0.5, f'{v2:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    slowdown = v2 / v1 if v1 > 0 else 0
    ax2.text(i, max(v1, v2) + 2, f'{slowdown:.1f}x', ha='center', va='bottom', 
            fontsize=9, color='orange', fontweight='bold')

ax2.set_ylabel('Query Latency (ms)', fontweight='bold')
ax2.set_xlabel('Algorithm', fontweight='bold')
ax2.set_title('Query Latency Scaling (10x data)', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(algorithms, rotation=15, ha='right')
ax2.legend(loc='upper left')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3)

# ============================================================================
# 3. Memory Scaling
# ============================================================================
ax3 = plt.subplot(2, 3, 3)

memory_10k = [results_10k[alg]['memory_mb'] for alg in algorithms]
memory_100k = [results_100k[alg]['memory_mb'] for alg in algorithms]

bars1 = ax3.bar(x - width/2, memory_10k, width, label='10K vectors', alpha=0.8)
bars2 = ax3.bar(x + width/2, memory_100k, width, label='100K vectors', alpha=0.8)

for i, bar in enumerate(bars1):
    bar.set_color(colors[algorithms[i]])
    bar.set_alpha(0.6)
for i, bar in enumerate(bars2):
    bar.set_color(colors[algorithms[i]])
    bar.set_alpha(1.0)

# Add value labels and scale factors
for i, (v1, v2) in enumerate(zip(memory_10k, memory_100k)):
    ax3.text(i - width/2, v1 + 1, f'{v1:.1f}', ha='center', va='bottom', fontsize=9)
    ax3.text(i + width/2, v2 + 1, f'{v2:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    scale = v2 / v1 if v1 > 0 else 0
    color = 'green' if scale < 11 else 'orange'  # Linear = 10x
    ax3.text(i, max(v1, v2) + 3, f'{scale:.1f}x', ha='center', va='bottom', 
            fontsize=9, color=color, fontweight='bold')

ax3.set_ylabel('Memory Usage (MB)', fontweight='bold')
ax3.set_xlabel('Algorithm', fontweight='bold')
ax3.set_title('Memory Scaling (expected: 10x linear)', fontsize=13, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(algorithms, rotation=15, ha='right')
ax3.legend(loc='upper left')
ax3.grid(axis='y', alpha=0.3)

# ============================================================================
# 4. Recall Drop Analysis
# ============================================================================
ax4 = plt.subplot(2, 3, 4)

recall_drops = [(results_100k[alg]['recall'] - results_10k[alg]['recall']) for alg in algorithms]
recall_drops_pct = [(drop / results_10k[alg]['recall']) * 100 for alg, drop in zip(algorithms, recall_drops)]

bars = ax4.barh(algorithms, recall_drops_pct, color=[colors[alg] for alg in algorithms], alpha=0.8)

# Color code by severity
for i, (bar, pct) in enumerate(zip(bars, recall_drops_pct)):
    if pct > -20:
        bar.set_color('green')
    elif pct > -50:
        bar.set_color('orange')
    else:
        bar.set_color('red')
    
    ax4.text(pct - 2, i, f'{pct:.1f}%', ha='right', va='center', fontsize=10, fontweight='bold', color='white')

ax4.set_xlabel('Relative Recall Change (%)', fontweight='bold')
ax4.set_title('🚨 Recall Degradation (10K → 100K)', fontsize=13, fontweight='bold', color='darkred')
ax4.axvline(x=0, color='black', linestyle='--', alpha=0.3)
ax4.grid(axis='x', alpha=0.3)

# Add severity legend
ax4.text(0.98, 0.98, '🟢 Good: >-20%\n🟠 Moderate: -20% to -50%\n🔴 Severe: <-50%', 
        transform=ax4.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)

# ============================================================================
# 5. Throughput Scaling
# ============================================================================
ax5 = plt.subplot(2, 3, 5)

throughput_10k = [results_10k[alg]['throughput_qps'] for alg in algorithms]
throughput_100k = [results_100k[alg]['throughput_qps'] for alg in algorithms]

bars1 = ax5.bar(x - width/2, throughput_10k, width, label='10K vectors', alpha=0.8)
bars2 = ax5.bar(x + width/2, throughput_100k, width, label='100K vectors', alpha=0.8)

for i, bar in enumerate(bars1):
    bar.set_color(colors[algorithms[i]])
    bar.set_alpha(0.6)
for i, bar in enumerate(bars2):
    bar.set_color(colors[algorithms[i]])
    bar.set_alpha(1.0)

# Add value labels
for i, (v1, v2) in enumerate(zip(throughput_10k, throughput_100k)):
    if v1 > 1000:
        ax5.text(i - width/2, v1 + 1000, f'{v1/1000:.1f}K', ha='center', va='bottom', fontsize=8)
    else:
        ax5.text(i - width/2, v1 + 5, f'{v1:.0f}', ha='center', va='bottom', fontsize=8)
    
    if v2 > 1000:
        ax5.text(i + width/2, v2 + 1000, f'{v2/1000:.1f}K', ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        ax5.text(i + width/2, v2 + 5, f'{v2:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax5.set_ylabel('Throughput (QPS)', fontweight='bold')
ax5.set_xlabel('Algorithm', fontweight='bold')
ax5.set_title('Throughput Scaling', fontsize=13, fontweight='bold')
ax5.set_xticks(x)
ax5.set_xticklabels(algorithms, rotation=15, ha='right')
ax5.legend(loc='upper right')
ax5.set_yscale('log')
ax5.grid(axis='y', alpha=0.3)

# ============================================================================
# 6. Winner Analysis at Each Scale
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

title_text = "🏆 SCALING ANALYSIS SUMMARY"
ax6.text(0.5, 0.95, title_text, ha='center', va='top', fontsize=14, fontweight='bold',
        transform=ax6.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

summary = f"""
AT 10K VECTORS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 Fastest:       HNSW (0.0128ms)
🏆 Best Recall:   ZGQ Unified (55.1%) ✨
🏆 Least Memory:  IVF & ZGQ (4.9 MB)
🏆 Best Throughput: HNSW (78K QPS)

Key Finding: ZGQ beats HNSW on recall!

AT 100K VECTORS (10x scale):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 Fastest:       HNSW (0.0453ms)
🏆 Best Recall:   IVF (34.4%) ⭐
🏆 Least Memory:  IVF & ZGQ (48.9 MB)
🏆 Best Throughput: HNSW (22K QPS)

Key Finding: ZGQ recall advantage LOST!

RECALL DEGRADATION (10K → 100K):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IVF:        -9%   (37.6% → 34.4%) ✅ BEST
IVF+PQ:    -33%   (19.0% → 12.7%)
ZGQ:       -62%   (55.1% → 21.2%) 🚨 WORST
HNSW:      -68%   (54.7% → 17.7%)

CRITICAL ISSUE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ ZGQ recall drops 62% (10K → 100K)
❌ Projected 1M recall: ~8-12%
❌ Projected 1B recall: <5% (unusable)
❌ "Billion-scale" claims INVALID

MEMORY EFFICIENCY (100K):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ ZGQ: 48.9 MB (20% less than HNSW)
✅ Savings: 12.1 MB per 100K vectors
✅ Scales linearly: ~121 MB at 1M

RECOMMENDATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IVF shows best scaling (only -9% recall drop)
ZGQ needs fundamental redesign for scale
HNSW still best for speed+recall balance
"""

ax6.text(0.05, 0.85, summary, ha='left', va='top', fontsize=9.5, family='monospace',
        transform=ax6.transAxes)

# ============================================================================
# Final touches
# ============================================================================
plt.suptitle('ZGQ SCALING ANALYSIS: 10K vs 100K Vectors\n' + 
             '⚠️  Critical Finding: Recall Advantage Disappears at Scale',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.985])

# Save
output_dir = Path('figures_scaling_analysis')
output_dir.mkdir(exist_ok=True)

output_path = output_dir / '01_scaling_comparison_10k_vs_100k.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")

plt.close()

print("\n" + "="*80)
print("SCALING VISUALIZATION COMPLETE")
print("="*80)
